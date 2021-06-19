# morel imports
import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join('../../../')))

# Environment
sys.path.append(os.path.abspath(os.path.join('../../../')))
from projects.morel_mopo.config.fake_env_config import DefaultFakeEnvConfig
import gym
import gym.spaces
from fake_env import FakeEnv

# torch imports
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

# Stable baselines PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

# Dynamics 
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig, BaseDynamicsEnsembleConfig, DefaultDynamicsModuleConfig, BaseDynamicsModuleConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble

# Data 
sys.path.append(os.path.abspath(os.path.join('')))
from data_modules import OfflineCarlaDataModule
# Debug
from pprint import pprint



''' 
1. Learn approx dynamics model from OfflineCarlaDataset 
3. Construct pessimistic MDP FakeEnv with USAD detector
4. Train policy in FakeEnv
'''
class Morel():
    def __init__(self,  uncertainty_threshold,
                        uncertainty_penalty,
                        dynamics_epochs,
                        policy_epochs,
                        logger):


        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_penalty = uncertainty_penalty
        
        self.dynamics_epochs = dynamics_epochs
        self.policy_epochs = policy_epochs

        self.logger = logger

        self.dynamics = None
        self.env = None
        self.policy  = None

    def train(self,
            precision = 16):

        if(precision != 16 and precision != 32):
            raise Exception("Precision must be 16 or 32")



        print("---------------- Instantiate Data module ----------------")
        
        class TempDataModuleConfig():
            def __init__(self):
                self.dataset_paths = ["/zfsauton/datasets/ArgoRL/swapnilp/new_state_space"]
                self.batch_size = 512
                self.frame_stack = 2
                self.num_workers = 2
                self.train_val_split = 0.95
        # data config
        data_config = TempDataModuleConfig()
        self.data_module = OfflineCarlaDataModule(data_config)      





        print("---------------- Beginning Logger ----------------")
        self.logger.step = 0
        self.logger.log_hyperparameters({
            "uncertainty_threshold (morel)" : self.uncertainty_threshold,
            "uncertainty_penalty (morel)" : self.uncertainty_penalty,
            "offline_batch_size (morel)" : self.data_module.batch_size,
        })





        print("---------------- Beginning Dynamics Training ----------------")
        dyn_ensemble_config = DefaultDynamicsEnsembleConfig()
        dyn_module_config = DefaultDynamicsModuleConfig()
        print('gpu', dyn_ensemble_config.gpu)

        self.dynamics = DynamicsEnsemble(
            config=dyn_ensemble_config,
            gpu=dyn_ensemble_config.gpu,
            data_module = self.data_module,
            state_dim_in = dyn_module_config.state_dim_in,
            state_dim_out = dyn_module_config.state_dim_out,
            action_dim = 2,
            frame_stack = dyn_module_config.frame_stack,
            logger = self.logger,
            log_freq = 100)

        self.dynamics.train(self.dynamics_epochs)

        print("----------------End Dynamics Training ----------------")




        print("---------------- Construct pessimistic MDP  ----------------")

        fake_env_config = DefaultFakeEnvConfig()
        fake_env_config.populate_config(\
            observation_config = "DefaultObservationConfig", \
            action_config = "DefaultActionConfig",\
            reward_config="DefaultRewardConfig",\
            uncertainty_config="DefaultUncertaintyConfig")
            
        self.env = FakeEnv(self.dynamics,
                        config=fake_env_config,
                        logger = self.logger)
                      




        print("---------------- Beginning Policy Training ----------------")

        self.policy = PPO("MlpPolicy", self.env, verbose=1, carla_logger = self.logger)# , policy_epochs = self.policy_epochs)#, \
        
        self.policy.learn(total_timesteps=10000000)
        print("---------------- End Policy Training  ----------------")


    def save(self, save_dir):
        if(not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        self.policy.save(save_dir)
        self.dynamics.save(save_dir)




    def load(self, load_dir):
        self.policy.load(load_dir)
        # self.dynamics.load(load_dir)
