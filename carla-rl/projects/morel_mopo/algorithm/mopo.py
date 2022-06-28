# morel imports
import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../../../')))

from environment.env import CarlaEnv

# Stable baselines PPO
from stable_baselines3.common.env_util import DummyVecEnv

# Environment
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from projects.morel_mopo.scripts.collect_data import DataCollector

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig

from stable_baselines3.sac import SAC




'''
1. Learn approx dynamics model from OfflineCarlaDataset
3. Construct pessimistic MDP FakeEnv with USAD detector
4. Train policy in FakeEnv
'''
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

        self.policy_algo = self.config.policy_algorithm

        # Retrieve the policy hyperparameters as a dictionary to pass to the policy constructor
        self.policy_hp = vars(self.config.policy_hyperparameters)
        self.policy  = None

        # Save config to load in the future
        if logger is not None:
            self.logger.pickle_save(self.config, MOPO.log_dir, "config.pkl")

    def train(self):
        # Construct env first to reserve GPU space
        env = CarlaEnv(config = self.eval_env_config, logger = self.logger, log_dir = self.logger.log_dir)


        ## Construct evaluation environment to periodically test policy
        eval_env = env.get_eval_env(eval_frequency = 100000)
        dummy_eval_env = DummyVecEnv([lambda: eval_env])

        eval_callback = EvalCallback(dummy_eval_env, best_model_save_path=os.path.join(self.logger.log_dir, "policy", "models"),
                                    log_path=os.path.join(self.logger.log_dir, "policy"), eval_freq=100000,
                                    deterministic=False, render=False,
                                    n_eval_episodes=self.eval_env_config.scenario_config.num_episodes)

        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(save_freq=self.policy_epochs//10, save_path=os.path.join(self.logger.log_dir, "policy", "models"),
                                                name_prefix='policy_checkpoint_')

        hyperparameters_to_log = {
            "mopo/uncertainty_penalty" : self.fake_env_config.uncertainty_coeff,
            "mopo/rollout_length" : self.fake_env_config.timeout_steps,
            "mopo/policy_algorithm" : str(self.policy_algo),
            "mopo/policy_weight_decay" : 0.0
        }
        # Add policy hyperparameters
        for key, value in self.policy_hp.items():
            # Prepend policy/ to the key
            hyperparameters_to_log["policy/" + key] = value

        # Log MOPO hyperparameters
        self.logger.log_hyperparameters(hyperparameters_to_log)

        # Setup dynamics model
        # If we are using a pretrained dynamics model, we need to load it
        # Else, we need to train a new dynamics model
        if(self.config.pretrained_dynamics_model_config is not None):
            print("MOPO: Using pretrained dynamics model")

            # Construct a logger temporarily to load the dynamics model
            logger_conf = ExistingCometLoggerConfig()
            logger_conf.experiment_key = self.config.pretrained_dynamics_model_config.key
            temp_logger = CometLogger(logger_conf)

            # Load dynamics config
            temp_config = temp_logger.pickle_load("mopo", "config.pkl")
            self.dynamics_config = temp_config.dynamics_config

            print(f"MOPO: Loading dynamics model {self.config.pretrained_dynamics_model_config.name} from experiment {self.config.pretrained_dynamics_model_config.key}")
            # Load dataset, only if we need it
            #TODO: We need to be able to pass the norm statistics, in case the datasets are not exactly the same
            if(self.load_data):
                self.dynamics_config.dataset_config
                self.dynamics = self.dynamics_config.dynamics_model_type.load(
                            logger = temp_logger,
                            model_name = self.config.pretrained_dynamics_model_config.name,
                            gpu = self.dynamics_config.gpu,
                            data_config = self.dynamics_config.dataset_config)
                self.data_module = self.dynamics.data_module #self.dynamics_config.dataset_config.dataset_type(self.dynamics_config.dataset_config)
            else:
                print("MOPO: Skipping Loading dataset")

        # If we are not using a pretrained dynamics model, we need to train a new dynamics model
        # Initialize a new one
        else:
            print("MOPO: Loading dataset")
            # We have to load the dataset here, because we need to train the dynamics model
            self.data_module = self.dynamics_config.dataset_config.dataset_type(self.dynamics_config.dataset_config)
            print("MOPO: Initializing new dynamics model")
            self.dynamics_epochs = self.config.dynamics_config.train_epochs
            self.dynamics = self.dynamics_config.dynamics_model_type(
                    config = self.dynamics_config.dynamics_model_config,
                    data_module = self.data_module,
                    logger = self.logger)

            print("MOPO: Beginning Dynamics Training")
            self.dynamics.train_model(self.dynamics_epochs)

        print("MOPO: Constructing Fake Env")


        fake_env = self.dynamics_config.fake_env_type(self.dynamics,
                        config = self.fake_env_config,
                        logger = self.logger)

        print("MOPO: Constructing Real Env for evaluation")

        print("MOPO: Beginning Policy Training")

        self.policy = self.policy_algo("MlpPolicy",
            fake_env,
            verbose=1,
            carla_logger = self.logger,
            device = self.dynamics.device,
            **self.policy_hp
        )
        self.policy.learn(total_timesteps=self.policy_epochs, callback = [eval_callback, checkpoint_callback])

        self.policy.save(os.path.join(self.logger.log_dir, "policy", "models", "final_policy"))


    def save(self, save_dir):
        if(not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        self.policy.save(save_dir)
        self.dynamics.save(save_dir)

    def policy_predict(self, obs, deterministic = True):
        action, _ = self.policy.predict(obs, deterministic = deterministic)
        return action

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


