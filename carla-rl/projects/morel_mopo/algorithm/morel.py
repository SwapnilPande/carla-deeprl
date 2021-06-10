# morel imports
# from offline_rl.ppo2 import PPO2
#from offline_rl.fake_env import FakeEnv
from fake_env import FakeEnv

import hydra

import numpy as np
from tqdm import tqdm

import gym

# torch imports
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F


#TODO import PPO here
import sys
import os
sys.path.append(os.path.abspath(os.path.join('../../../')))
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
# Environment
sys.path.append(os.path.abspath(os.path.join('../../../')))
from environment.config.config import DefaultMainConfig
from environment.env import CarlaEnv
sys.path.append(os.path.abspath(os.path.join('')))
from data_modules import OfflineCarlaDataModule
#debug
from pprint import pprint



class Morel():
    def __init__(self, obs_dim,
                        action_dim,
                        uncertainty_threshold,
                        uncertainty_penalty,
                        dynamics_cfg,
                        policy_cfg,
                        dynamics_epochs,
                        policy_epochs):

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_penalty = uncertainty_penalty


        #self.dynamics = hydra.utils.instantiate(dynamics_cfg)
        #self.policy = hydra.utils.instantiate(policy_cfg)
        #set the environment
        self.dynamicsEnv = CarlaEnv(config = dynamics_cfg, log_dir = "/home/scratch/swapnilp/carla_test")
        self.policyEnv = CarlaEnv(config = policy_cfg, log_dir = "/home/scratch/swapnilp/carla_test")
        


    def train(self,
            offline_data_module_cfg,
            online_data_module_cfg,
            logger,
            gpu_number = None,
            precision = 16):

        if(precision != 16 and precision != 32):
            raise Exception("Precision must be 16 or 32")

        print("---------------- Beginning Logger ----------------")
        logger.step = 0
        logger.log_hyperparams({
            "uncertainty_threshold (morel)" : self.uncertainty_threshold,
            "uncertainty_penatly (morel)" : self.uncertainty_penalty,
            "offline_batch_size (morel)" : offline_data_module_cfg.batch_size,
            "offline_buffer_size (morel)" : offline_data_module_cfg.buffer_size,
            "offline_epochs_per_experience (morel)"  : offline_data_module_cfg.epochs_per_experience,
            "online_batch_size (morel)" : online_data_module_cfg.batch_size,
            "online_buffer_size (morel)" : online_data_module_cfg.buffer_size,
            "online_epochs_per_experience (morel)"  : online_data_module_cfg.epochs_per_experience
        })

        print("---------------- Ending Logger ----------------")

        print("---------------- Beginning Dynamics Training ----------------")
        self.dynamics = PPO("MlpPolicy", self.dynamicsEnv, verbose=1, policy_epochs = dynamics_epochs, batch_size = offline_data_module_cfg.batch_size, n_epochs = offline_data_module_cfg.epochs_per_experience, device = device(type='gpu', index=gpu_number))
        self.dynamics.learn(total_timesteps=25000)



        # self.dynamics.train(
        #     offline_data_module_cfg,
        #     logger = logger,
        #     gpu_number = gpu_number,
        #     precision = precision
        # )
        # (cfg.data_module, logger, gpu_number = cfg.gpu, precision = cfg.trainer.precision, epochs = cfg.epochs)
        print("---------------- Ending Dynamics Training ----------------")

        print("---------------- Beginning Dynamics Analysis ----------------")
        env = FakeEnv(
            self.dynamics,
            logger,
            uncertainty_threshold = self.uncertainty_threshold,
            uncertain_penalty = self.uncertainty_penalty,
            timeout_steps = 1
        )
        print("---------------- Ending Dynamics Analysis ----------------")

        print("---------------- Beginning Policy Training ----------------")
        self.policy = PPO("MlpPolicy", self.policyEnv, verbose=1, policy_epochs = policy_epochs, batch_size = online_data_module_cfg.batch_size, n_epochs = online_data_module_cfg.epochs_per_experience, device = device(type='gpu', index=gpu_number))
        self.policy.learn(total_timesteps=25000)
        # self.policy.train(
        #     data_module_cfg = online_data_module_cfg,
        #     env = env,
        #     logger = logger,
        #     gpu_number = gpu_number,
        #     precision = precision
        # )
        print("---------------- Ending Policy Training ----------------")


    def save(self, save_dir):
        if(not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        self.policy.save(save_dir)
        self.dynamics.save(save_dir)

    def load(self, load_dir):
        self.policy.load(load_dir)
        # self.dynamics.load(load_dir)

#testing
obs_dim = gym.spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))
action_dim = gym.spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))
uncertainty_threshold = 1
uncertainty_penalty = 1

dynamics_cfg = DefaultMainConfig()
dynamics_cfg.populate_config(observation_config = "LowDimObservationConfig",
    action_config = "MergedSpeedScaledTanhConfig",
    reward_config = "Simple2RewardConfig",
    scenario_config = "NoCrashEmptyTown01Config",
    testing = False,
    carla_gpu = 0)

policy_cfg = DefaultMainConfig()
policy_cfg.populate_config(observation_config = "LowDimObservationConfig",
    action_config = "MergedSpeedScaledTanhConfig",
    reward_config = "Simple2RewardConfig",
    scenario_config = "NoCrashEmptyTown01Config",
    testing = False,
    carla_gpu = 0)

dynamics_epochs = 10
policy_epochs = 10



obj = Morel(obs_dim,
            action_dim,
            uncertainty_threshold,
            uncertainty_penalty,
            dynamics_cfg,
            policy_cfg,
            dynamics_epochs,
            policy_epochs)
off_data = OfflineCarlaDataModule(dynamics_cfg)
#on_data = OnlineCarlaDataset(policy_cfg)
#obj.train(off_data, on_data, obj.logger)


