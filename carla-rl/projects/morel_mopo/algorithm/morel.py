# morel imports
import hydra

import numpy as np
from tqdm import tqdm

import gym
import gym.spaces

from fake_env import FakeEnv

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

# Policy
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Environment
sys.path.append(os.path.abspath(os.path.join('../../../')))
from projects.morel_mopo.config.fake_env_config import DefaultFakeEnvConfig

# Dynamics 
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig, BaseDynamicsEnsembleConfig, DefaultDynamicsModuleConfig, BaseDynamicsModuleConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble

# Data 
sys.path.append(os.path.abspath(os.path.join('')))
from data_modules import OfflineCarlaDataModule
#debug
from pprint import pprint



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

        #self.dynamics = hydra.utils.instantiate(dynamics_cfg)
        #self.policy = hydra.utils.instantiate(policy_cfg)


        print("---------------  Dynamics Model ----------------")
        dyn_ensemble_config = DefaultDynamicsEnsembleConfig()
        dyn_module_config = DefaultDynamicsModuleConfig()
        self.dynamics = DynamicsEnsemble(
            config=dyn_ensemble_config,
            gpu=dyn_ensemble_config.gpu,
            data_module = self.data_module,
            state_dim_in = dyn_module_config.state_dim_in,
            state_dim_out = dyn_module_config.state_dim_out,
            action_dim = 2,
            frame_stack = dyn_module_config.frame_stack,
            logger = logger,
            log_freq = 100)


        print("---------------- Instantiate Fake environment ----------------")
        
        fake_env_config = DefaultFakeEnvConfig()
        fake_env_config.populate_config(\
            observation_config = "DefaultObservationConfig", \
            action_config = "DefaultActionConfig",\
            reward_config="DefaultRewardConfig",\
            uncertainty_config="DefaultUncertaintyConfig")
            
        self.fake_env = FakeEnv(self.dynamics,
                        config=fake_env_config,
                        logger = self.logger)
                      
        # check_env(self.fake_env)

        print("---------------- Instantiate PPO Policy  ----------------")

        # POLICY (from stable baselines)

        self.policy = PPO("MlpPolicy", self.fake_env, verbose=1)# , policy_epochs = self.policy_epochs)#, \
                    # , batch_size = online_data_module_cfg.batch_size,\
                    #  n_epochs = online_data_module_cfg.epochs_per_experience, \
                    #  device = torch.device(type='gpu', index=dyn_ensemble_config.gpu))



    def train(self,
            precision = 16):

        if(precision != 16 and precision != 32):
            raise Exception("Precision must be 16 or 32")

        print("---------------- Beginning Logger ----------------")
        # self.logger.step = 0
        # self.logger.log_hyperparameters({
        #     "uncertainty_threshold (morel)" : self.uncertainty_threshold,
        #     "uncertainty_penalty (morel)" : self.uncertainty_penalty,
        #     "offline_batch_size (morel)" : self.data_module.data_cfg.batch_size,
        # })

        print("---------------- Beginning Dynamics Training ----------------")
        # dynamics is DynamicsEnsembleModule
        self.dynamics.train(self.dynamics_epochs)

        # self.dynamics.train(
        #     offline_data_module_cfg,
        #     logger = logger,
        #     gpu_number = gpu_number,
        #     precision = precision
        # )
        # (cfg.data_module, logger, gpu_number = cfg.gpu, precision = cfg.trainer.precision, epochs = cfg.epochs)


        print("---------------- Beginning Policy Training ----------------")
        self.policy.learn(total_timesteps=25000)
        # self.policy.train(
        #     data_module_cfg = online_data_module_cfg,
        #     env = env,
        #     logger = logger,
        #     gpu_number = gpu_number,
        #     precision = precision
        # )

        print("---------------- Beginning Dynamics Analysis ----------------")

        # dynamics: predicting the change in state/vehicle pose
        # policy: maps obs -> action
        obs = self.fake_env.reset()
        for i in range(self.policy_epochs):
            while True:
                # policy maps input obs to action
                action, _states  = self.policy.predict(obs)
                # step in fake env to advance timestep 
                obs, reward, done, info = self.fake_env.step(torch.from_numpy(action))
                # termination
                if done: break



    def save(self, save_dir):
        if(not os.path.isdir(save_dir)):
            os.mkdir(save_dir)

        self.policy.save(save_dir)
        self.dynamics.save(save_dir)

    def load(self, load_dir):
        self.policy.load(load_dir)
        # self.dynamics.load(load_dir)
