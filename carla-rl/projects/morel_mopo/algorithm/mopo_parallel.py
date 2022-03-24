# morel imports
import sys, os
import numpy as np
from tqdm import tqdm
from collections import deque
import copy
import time
import torch
import random
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import projects.morel_mopo.algorithm.dist_utils as dist

from environment.env import CarlaEnv

# Stable baselines PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3 import sac

# Environment
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

from projects.morel_mopo.scripts.collect_data import DataCollector

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig

'''
1. Learn approx dynamics model from OfflineCarlaDataset
3. Construct pessimistic MDP FakeEnv with USAD detector
4. Train policy in FakeEnv
'''

class SIG:
    EXP_PUSH = 0
    PARAM_REQ = 1
    QUERY = 2
    EXP = 3
    PARAM = 4

class MOPO():
    log_dir = "mopo"

    def __init__(self, config,
                    logger,
                    load_data = True):
        self.config = config
        self.dynamics_config = self.config.dynamics_config
        self.fake_env_config = self.config.fake_env_config
        self.eval_env_config = self.config.eval_env_config

        self.policy_epochs = 10_000_000

        self.logger = logger

        self.dynamics = None
        self.load_data = load_data

        self.fake_env = None

        # self.policy_algo = self.config.policy_algorithm
        self.policy_algo = sac.SAC
        print(51, self.policy_algo)
        self.policy  = None
        self.total_equivalent_glb_steps = 10000

        comm_vars = dist.init_param_server_comm()
        self.rank, self.world_size = comm_vars[:2]
        self.server_list, self.worker_list = comm_vars[2:4]
        self.server_group, self.worker_group = comm_vars[4:]
        self.num_servers = len(self.server_list)
        self.num_workers = len(self.worker_list)

        # Save config to load in the future
        if logger is not None:
            self.logger.pickle_save(self.config, MOPO.log_dir, "config.pkl")

    def serve(self):
        self.recv_info_len = 5
        self.glb_update_freq = 1
        checkpoint_callback = CheckpointCallback(save_freq=self.policy_epochs//10, save_path=os.path.join(self.logger.log_dir, "policy", "models"),
                                                name_prefix='policy_checkpoint_')

        # Log MOPO hyperparameters
        self.logger.log_hyperparameters({
            "mopo/uncertainty_penalty" : self.fake_env_config.uncertainty_coeff,
            "mopo/rollout_length" : self.fake_env_config.timeout_steps,
            "mopo/policy_algorithm" : str(self.policy_algo),

            "mopo/policy_weight_decay" : 0.0
        })

        self.num_online_loops = 1
        self.num_online_samples = 50000

        self.steps_per_loop = self.policy_epochs // self.num_online_loops
        for i in range(self.num_online_loops):
            if(self.config.pretrained_dynamics_model_config is None):
                if(i == 0):
                    print("MOPO: Beginning Dynamics Training")
                    self.dynamics.train_model(self.dynamics_epochs)
                else:
                    print("MOPO: Beginning Dynamics Training")
                    self.dynamics.lr = self.dynamics_config.lr / 10
                    self.dynamics.train_model(25)

        print("MOPO: Constructing Fake Env")

        fake_env = self.dynamics_config.fake_env_type(self.dynamics,
                        config = self.fake_env_config,
                        logger = self.logger)
        print("MOPO: Beginning Policy Training")

        self.policy = self.policy_algo("MlpPolicy",
            fake_env,
            verbose=1,
            carla_logger = self.logger,
            device = self.dynamics.device,
        )
        ################################## Server ##################################
        self.model_len = len(parameters_to_vector(self.policy.policy.parameters()))
        print(164, self.model_len)


        self.glb_num_steps = 0
        self.last_save_steps = 0
        self.glb_num_episodes = 0
        self.num_steps_since_update = 0
        self.save_freq = 5
        # self.N_S = self.policy.observation_space.shape[0]
        # self.N_A = self.policy.action_space.shape[0]

        self.N_S = 6
        self.N_A = 2

        while self.glb_num_steps < self.total_equivalent_glb_steps:
            sender, num_steps_added, buffer_len, num_eps_added, signal = dist.recv(
                self.recv_info_len, tag=SIG.QUERY)
            print('server', self.rank, 'QUERY', sender,
                num_steps_added, 'buffer_len', buffer_len, signal)
            if signal == SIG.EXP_PUSH:
                total_len = (2 * self.N_S + self.N_A + 2) * buffer_len
                print(185, 'total_len', total_len)
                _, vec_buf = dist.recv(self.recv_info_len, total_len,
                    src=sender, tag=SIG.EXP, device='cpu')
                vec_buf = vec_buf.reshape(buffer_len, -1).numpy()
                # disintegrate them into buffer items derived from buffer_len
                for _buf in vec_buf:
                    _state = _buf[:self.N_S]
                    _action = _buf[self.N_S:self.N_S + self.N_A]
                    _reward = _buf[-self.N_S - 2:-self.N_S - 1]
                    _next_state = _buf[-self.N_S - 1:-1]
                    _done = _buf[-1:]
                    self.policy.replay_buffer.add(_state, _next_state, _action, _reward, _done, [{}])
                self.glb_num_steps += num_steps_added
                self.num_steps_since_update += num_steps_added
                self.glb_num_episodes += num_eps_added
            elif signal == SIG.PARAM_REQ:
                print('server', self.rank, 'send param', sender,
                    num_steps_added, signal)
                dist.isend(
                    [self.glb_num_steps, self.glb_num_episodes],
                    self.policy.policy.parameters(),
                    dst=sender, tag=SIG.PARAM
                ).wait()
            else:
                raise ValueError('signal not seen')
            if self.num_steps_since_update >= self.glb_update_freq:
                print('[server rank {}][glb ep {}][glb step {}] updating ...'.format(
                    self.rank, self.glb_num_episodes, self.glb_num_steps,
                ))
                self.policy.train(1)
                self.num_steps_since_update = 0

            # save checkpoint
            if self.glb_num_steps - self.last_save_steps >= self.save_freq:
                self.last_save_steps = self.glb_num_steps
                self.policy.save(os.path.join(self.logger.log_dir, "policy", "models", "final_policy"))

    def work(self):
        self.recv_info_len = 3
        self.num_steps_since_update = 0
        self.num_eps_since_update = 0
        self.local_policy_timestamp = 0
        self.server_rank = (self.rank - self.num_servers) % self.num_servers
        # Setup dynamics model
        # If we are using a pretrained dynamics model, we need to load it
        # Else, we need to train a new dynamics model
        if(self.config.pretrained_dynamics_model_config is not None):
            print("MOPO: Using pretrained dynamics model")

            # Construct a logger temporarily to load the dynamics model
            logger_conf = ExistingCometLoggerConfig()
            logger_conf.experiment_key = self.config.pretrained_dynamics_model_config.key
            logger_conf.clear_temp_logs = False
            temp_logger = CometLogger(logger_conf)

            # Load dynamics config
            temp_config = temp_logger.pickle_load("mopo", "config.pkl")
            self.dynamics_config = temp_config.dynamics_config

        self.num_online_loops = 1
        self.num_online_samples = 50000

        self.steps_per_loop = self.policy_epochs // self.num_online_loops
        for i in range(self.num_online_loops):
            if(self.config.pretrained_dynamics_model_config is None):
                if(i == 0):
                    print("MOPO: Beginning Dynamics Training")
                    self.dynamics.train_model(self.dynamics_epochs)
                else:
                    print("MOPO: Beginning Dynamics Training")
                    self.dynamics.lr = self.dynamics_config.lr / 10
                    self.dynamics.train_model(25)

            print("MOPO: Constructing Fake Env")


            fake_env = self.dynamics_config.fake_env_type(self.dynamics,
                            config = self.fake_env_config,
                            logger = self.logger)

            print("MOPO: Constructing Real Env for evaluation")

            print("MOPO: Beginning Policy Training")

        ################################## Worker ##################################
        self.policy = self.policy_algo("MlpPolicy",
            fake_env,
            verbose=1,
            carla_logger = self.logger,
            device = self.dynamics.device,
        )
        policy_collections = deque(maxlen=100)

        self.N_S = self.policy.observation_space.shape[1]
        self.N_A = self.policy.action_space.shape[0]
        # self.N_S = 6
        # self.N_A = 2

        self.model_len = len(parameters_to_vector(self.policy.policy.parameters()))
        self.worker_buffer = []
        print(302, self.model_len)
        # total_timesteps, callback = self.policy._setup_learn(
        #     self.steps_per_loop, None, None, -1, 5, None, True, 'OffPolicyAlgorithm'
        # )
        glb_stats = self.update_parameters()
        print(312, glb_stats)

        # policy_collections.append(copy.deepcopy(self.policy))
        policy_collections.append(copy.deepcopy(self.policy.get_parameters()))

        prev_obs = fake_env.reset()
        for i in range(10):
            self.num_steps_since_update += 1
            # actions = torch.zeros(prev_obs.shape[0], 2)
            # get actions from sampling old policies
            action_list = []
            for idx in range(prev_obs.shape[0]):
                _policy_param = random.choice(policy_collections)
                self.policy.set_parameters(_policy_param)
                # print(341, self.policy.predict(prev_obs[idx, None]))
                action_list.append(torch.tensor(self.policy.predict(prev_obs[idx, None])[0]))
            actions = torch.vstack(action_list)
            print(340, prev_obs.shape, actions.shape)
            start_time = time.time()
            curr_obs, rewards, dones, _ = fake_env.step(actions)
            # print(273, type(rewards), type(dones), curr_obs.shape, rewards.shape, dones.shape)
            # print(274, self.N_A, self.N_S)
            print(f"Time taken: {time.time() - start_time}")

            # # add to buffer
            # for j in range(prev_obs.shape[0]):
            #     # print(278, rewards, rewards[:, j], rewards[j], dones[j])
            #     self.policy.replay_buffer.add(
            #         prev_obs[j],
            #         curr_obs[j],
            #         actions[j],
            #         rewards[0, j],
            #         dones[0, j],
            #         [{}],
            #     )

            # self.worker_buffer.append(np.hstack((prev_obs, actions, rewards[:, [0]], curr_obs, dones[:, [0]])))
            self.worker_buffer.append(np.hstack((prev_obs, actions, rewards[[0]], curr_obs, dones[[0]])))
            print(336, [x.shape for x in self.worker_buffer])

            prev_obs = curr_obs

            self.send_buffer()
            glb_stats = self.update_parameters()

            if dones.all():
                prev_obs = fake_env.reset()



    def update_parameters(self):
        print(335, 'rank', self.rank, 'update_parameters')
        # overhead = [self.rank, self.num_steps_since_update,
        #     self.num_eps_since_update, SIG.PARAM_REQ]
        overhead = [self.rank, self.num_steps_since_update,
            self.local_policy_timestamp, self.num_eps_since_update, SIG.PARAM_REQ]
        dist.isend(overhead, dst=self.server_rank, tag=SIG.QUERY).wait()
        glb_stats, vec_param = dist.recv(self.recv_info_len, self.model_len,
            src=self.server_rank, tag=SIG.PARAM, device=self.policy.device)
        vector_to_parameters(vec_param, self.policy.policy.parameters())
        self.num_steps_since_update = 0
        self.local_policy_timestamp += 1
        return glb_stats

    def send_buffer(self):
        print(335, 'rank', self.rank, 'send_buffer')
        exp_buf = torch.from_numpy(np.vstack(self.worker_buffer))
        assert len(exp_buf.shape) == 2, exp_buf.shape
        overhead = [self.rank, self.num_steps_since_update,
            len(exp_buf), self.num_eps_since_update, SIG.EXP_PUSH]
        dist.isend(overhead, dst=self.server_rank, tag=SIG.QUERY).wait()
        print(369, 'buffer_size', len(exp_buf), len(exp_buf.reshape(-1)))
        dist.isend(overhead, exp_buf.reshape(-1), dst=self.server_rank, tag=SIG.EXP).wait()
        self.num_steps_since_update = 0
        # reset buffer
        self.worker_buffer = []

    def save(self, save_dir):
        if(not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        self.policy.save(save_dir)
        self.dynamics.save(save_dir)

    def policy_predict(self, obs, deterministic = True):
        action, _ = self.policy.predict(obs, deterministic = False)
        return action

    @classmethod
    def get_dynamics_model(cls, config, load_data = True, logger = None):
        print(f"MOPO: Loading dynamics model {self.config.pretrained_dynamics_model_config.name} from experiment {self.config.pretrained_dynamics_model_config.key}")


        # Setup dynamics model
        # If we are using a pretrained dynamics model, we need to load it
        # Else, we need to train a new dynamics model
        if(config.pretrained_dynamics_model_config is not None):
            print("MOPO: Using pretrained dynamics model")

            # Construct a logger temporarily to load the dynamics model
            logger_conf = ExistingCometLoggerConfig()
            logger_conf.experiment_key = config.pretrained_dynamics_model_config.key
            logger_conf.clear_temp_logs = False
            temp_logger = CometLogger(logger_conf)

            # Load dynamics config
            temp_config = temp_logger.pickle_load("mopo", "config.pkl")
            dynamics_config = temp_config.dynamics_config

            print(f"MOPO: Loading dynamics model {config.pretrained_dynamics_model_config.name} from experiment {config.pretrained_dynamics_model_config.key}")
            # Load dataset, only if we need it
            #TODO: We need to be able to pass the norm statistics, in case the datasets are not exactly the same
            if(load_data):
                dynamics = dynamics_config.dynamics_model_type.load(
                            logger = temp_logger,
                            model_name = config.pretrained_dynamics_model_config.name,
                            gpu = dynamics_config.gpu,
                            data_config = dynamics_config.dataset_config)


            else:
                print("MOPO: Skipping Loading dataset")

                dynamics = self.dynamics_config.dynamics_model_type.load(
                            logger = temp_logger,
                            model_name = self.config.pretrained_dynamics_model_config.name,
                            gpu = self.dynamics_config.gpu)


        # If we are not using a pretrained dynamics model, we need to train a new dynamics model
        # Initialize a new one
        else:
            print("MOPO: Loading dataset")
            # We have to load the dataset here, because we need to train the dynamics model
            dynamics_config = config.dynamics_config
            data_module = dynamics_config.dataset_config.dataset_type(dynamics_config.dataset_config)
            print("MOPO: Initializing new dynamics model")
            dynamics = dynamics_config.dynamics_model_type(
                    config = dynamics_config.dynamics_model_config,
                    data_module = data_module,
                    logger = logger)

        return dynamics


    @classmethod
    def load(cls, logger, policy_model_name, gpu, policy_only = True, dynamics_model_name = None):
        # To load the model, we first need to build an instance of this class
        # We want to keep the same config parameters, so we will build it from the pickled config
        # Also, we will load the dimensional parameters of the model from the saved dimensions
        # This allows us to avoid loading a dataloader every time we want to do inference

        print("MOPO: Loading policy {}".format(policy_model_name))
        # Get config from pickle first
        config = logger.pickle_load(MOPO.log_dir, "config.pkl")

        # Create a configured dynamics ensemble object
        mopo = cls(config = config,
                    logger = logger,
                    load_data = False)

        device = f"cuda:{gpu}"

        mopo.policy = mopo.policy_algo.load(
                logger.other_load("policy/models", policy_model_name),
                device = device)

        if(not policy_only):
            mopo.dynamics = config.dynamics_config.dynamics_model_type.load(
                    logger = logger,
                    model_name = dynamics_model_name,
                    gpu = gpu)

            mopo.fake_env = config.dynamics_config.fake_env_type(mopo.dynamics,
                        config = config.fake_env_config,
                        logger = logger)

        return mopo



