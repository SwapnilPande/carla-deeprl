
# morel imports
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os

from gym.core import ObservationWrapper

# Add to path for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

# Logger
from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig

from projects.morel_mopo.config.fake_env_config import DefaultFakeEnvConfig
from fake_env import FakeEnv

from projects.morel_mopo.scripts.collect_data import mbpo_collect_data
from projects.morel_mopo.algorithm.dynamics_init import DynamicsEnsemble

# Gym
import gym
import gym.spaces
from gym.core import ObservationWrapper

# Pytorch
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

# Stable baselines PPO
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
# Dynamics 
from projects.morel_mopo.config.dynamics_ensemble_config import MBPODynamicsEnsembleConfig, BaseDynamicsEnsembleConfig, MBPODynamicsModuleConfig, BaseDynamicsModuleConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble
# Datamodule
from data_modules import OfflineCarlaDataModule
import copy




class CarlaEnvSampler():
    def __init__(self, env):
        self.env = env
        self.obs = None

    ''' Take step in env with action suggested by policy '''
    def sample(self, policy):
        if self.obs is None:
            # print("Resetting env in CarlaEnvSampler")
            self.obs = self.env.reset()

        obs = self.obs
        action, _ = policy.predict(self.obs)
        next_obs, reward, done, info = self.env.step(action)

        if done:
            self.obs = None
        else:
            self.obs = next_obs

        return obs, next_obs, action, reward, done, info




class MBPO:
    def __init__(self,
                logger,
                n_epochs = 400,\
                n_models = 7,\
                n_timesteps = 1000,\
                n_rollouts = 400,\
                n_grad_updates = -1,\
                batch_size = 256,\
                buffer_size = 1000,
                gpu = 2):

        print('------------------Initializing MBPO Parameters-----------------')
        # hyperparams
        self.n_epochs = n_epochs                # N
        self.n_models = n_models                # ensemble size
        self.n_timesteps = n_timesteps          # environment steps per epoch
        self.n_rollouts = n_rollouts            # number model rollouts per environment step
        self.n_grad_updates = n_grad_updates    # same as number of steps
        self.batch_size = batch_size            # Batch size
        self.buffer_size = buffer_size          # replay buffer
        self.gpu = gpu
        self.dynamics_epochs = 10 # TODO 

        print('-----------------------Initializing Dynamics------------------')

        # datasets 
        self.env_dataset = None 
        self.model_dataset = None
        self.logger = logger
        self.log_dir = "/home/scratch/vccheng/carla_test"
        self.device = "cuda:{}".format(self.gpu)
        
        print('--------------------------Initializing Env-------------------')
        carla_env_config = DefaultMainConfig()
        carla_env_config.populate_config(
            observation_config = "LowDimObservationConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config = "Simple2RewardConfig",
            scenario_config = "NoCrashEmptyTown01Config",
            testing = False,
            carla_gpu = self.gpu,
        )


        self.env = CarlaEnv(config = carla_env_config, \
                            logger = self.logger,
                            log_dir = self.log_dir)

                            

        # # fake env 
        # fake_env_config = DefaultFakeEnvConfig()
        # fake_env_config.populate_config()
        # self.env = FakeEnv(self.dynamics, fake_env_config)

        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space

        print('---------------------Initializing Replay Memory-----------------')

        # replay buffer for env dataset
        self.env_buffer = ReplayBuffer(buffer_size = self.buffer_size,\
                                       observation_space = self.obs_space,
                                       action_space = self.action_space,
                                       device = self.device)
        # replay buffer for model dataset
        self.model_buffer = ReplayBuffer(buffer_size = self.buffer_size,\
                                       observation_space = self.obs_space,
                                       action_space = self.action_space,
                                       device = self.device)

        print('--------------------------Initializing Policy-----------------')

        # policy 

        # import pdb;pdb.set_trace();
        self.policy = SAC(policy='MlpPolicy', 
                          env_buffer = self.env_buffer, \
                          replay_buffer = self.model_buffer, \
                          env = self.env, \
                          batch_size = self.batch_size,\
                          buffer_size = self.buffer_size,
                          learning_starts = 10,\
                          gradient_steps=0,
                          train_freq=(1, "episode"),\
                          carla_logger = self.logger) 
                        #   ,\
                        #   gradient_steps = 0)

        print('---------------------Initializing CarlaEnv Sampler-----------------')

        self.carla_env_sampler = CarlaEnvSampler(self.env)
        self.collect_data_path = '/zfsauton/datasets/ArgoRL/vccheng/collect_data_mbpo' 

        print('---------------------Initializing Callbacks---------------')
 
        # self.vec_env = make_vec_env(self.env, n_envs = 1)

  

        # Parallel environments
        # self.dummy_env = DummyVecEnv([lambda: self.env])
        # eval_env = self.env.get_eval_env(eval_frequency = 500)
        # dummy_eval_env = DummyVecEnv([lambda: eval_env])
        # self.eval_callback = EvalCallback(dummy_eval_env, \
        #                             best_model_save_path=os.path.join(self.log_dir, "policy", "models"),
        #                             log_path=os.path.join(self.log_dir, "policy"), \
        #                             eval_freq=500,
        #                             deterministic=False, render=False,
        #                             n_eval_episodes=5)

        # self.eval_callback.init_callback(self.policy)


        self.dummy_env = DummyVecEnv([lambda: self.env])
        # eval_env = self.env.get_eval_env(eval_frequency = 500)

        self.callback = CheckpointCallback(save_freq = 5000,\
                                         save_path = os.path.join(self.log_dir, "policy", "models"),\
                                         name_prefix = 'mbpo')


    # Dynamics
    def setup_dynamics(self):

        class TempDataModuleConfig():
            def __init__(self, collect_data_paths):
                self.dataset_paths = collect_data_paths
                self.batch_size = 8
                self.frame_stack = 2
                self.num_workers = 1
                self.train_val_split = 0.8
        # data config
        data_config = TempDataModuleConfig([self.collect_data_path])
        print('------------------------Creating CarlaDataModule-------------')
        self.data_module = OfflineCarlaDataModule(data_config, normalize_data=False)      

        # dynamics ensemble
        dyn_ensemble_config = MBPODynamicsEnsembleConfig()
        dyn_module_config = MBPODynamicsModuleConfig()

        print('--------------------Initializing Dynamics Ensemble------------')
        self.dynamics = DynamicsEnsemble(
            config=dyn_ensemble_config,
            data_module = self.data_module,
            state_dim_in = dyn_module_config.state_dim_in,
            state_dim_out = dyn_module_config.state_dim_out,
            action_dim = 2,
            frame_stack = dyn_module_config.frame_stack,
            norm_stats = self.data_module.normalization_stats,
            gpu = self.gpu,
            logger = self.logger,
            log_freq = 100)
        return self.dynamics
       

    ''' MBPO algorithm'''
    def train(self):

        self.rollout_len = 1
        total_steps = 0
        total_rewards = 0

        # initial exploration in environment
        self.env_buffer = mbpo_collect_data(self.env, \
                                             self.policy, \
                                             self.collect_data_path,\
                                             n_samples = 50, 
                                             carla_gpu = self.gpu, \
                                             replay_buffer = self.env_buffer)

        print('-------------2: Epochs-------------------')

        for epoch in range(self.n_epochs):
            print(f'Epoch {epoch}')
            
            print('---------------Setting up dynamics--------------')
            self.dynamics = self.setup_dynamics()
                
            # train a predictive model on env_dataset via max likelihood 
            # (can also train once per F steps instead of once per epoch)
            print('-------------3: Training Dynamics-------------------')
            self.dynamics.train(self.dynamics_epochs)

            # Dynamics Model interacts with policy
            print('-------------4: Timesteps------------------')
            for step in range(self.n_timesteps):

                print('-------------5: Step in env, add to env_buffer-------------')
                # import pdb; pdb.set_trace()

                obs, next_obs, action, reward, done, info = self.carla_env_sampler.sample(self.policy)
                self.env_buffer.add(obs, next_obs, action, reward, done, [info])

                # for M model rollouts:
                #   sample st from env_buffer uniformly
                #   k-step rollout using policy from st
                # for G grad:
                #   update update params

                print('-------------6-10: Rollouts, Gradient Updates -------------------')
                
                self.policy.learn(total_timesteps=100,
                                  n_rollouts = 10,\
                                  rollout_len = 1,\
                                  callback = None)
                
                total_steps += 1


'''
PPO is model free, on-policy
MBPO: model-based
SAC: off-policy, model-free (improvement from PPO, can 
explore more because it can use previous data)
'''

def main():
    # import pdb; pdb.set_trace()
    EXPERIMENT_NAME = "vivian_mbpo_6-29"

    logger_conf = CometLoggerConfig()
    logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = ["MBPO"])

    logger = CometLogger(logger_conf)
    m = MBPO(logger=logger)
    m.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=400)
    parser.add_argument('--n_models', type=int, default=7)
    parser.add_argument('--n_timesteps', type=int, default=1000)
    parser.add_argument('--n_rollouts', type=int, default=400)
    parser.add_argument('--n_grad_updates', type=int, default=-1)
    parser.add_argument('--n_batch_size', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=1000)
    args = parser.parse_args()
    main()