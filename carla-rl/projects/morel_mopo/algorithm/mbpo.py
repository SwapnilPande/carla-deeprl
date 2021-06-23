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
from stable_baselines3 import SAC

# Dynamics 
from projects.morel_mopo.config.dynamics_ensemble_config import DefaultDynamicsEnsembleConfig, BaseDynamicsEnsembleConfig, DefaultDynamicsModuleConfig, BaseDynamicsModuleConfig
from projects.morel_mopo.algorithm.dynamics_ensemble_module import DynamicsEnsemble

# Data 
sys.path.append(os.path.abspath(os.path.join('')))
from data_modules import OfflineCarlaDataModule
# Debug
from pprint import pprint


'''
collect dataset Denv from policy π_phi
for N epochs:
    train dynamics on Denv with max likelihood
'''
class MBPO:
    def __init__(self, n_epochs, n_models, k):

        # hyperparams
        self.k = k  # 1 to 15
        self.n_epochs = n_epochs        # N
        self.n_models = n_models        # M
        self.n_timesteps = n_timesteps  # E
        self.n_grad_updates = n_grad_updates # G

        # datasets 
        self.D_env = None # TODO: collect data
        self.D_model = None
        self.logger = logger

        # probabilistic NN dynamics 
        self.dynamics = mbpo_dynamics()
        self.env = FakeEnv(self.dynamics)
        # policy 
        self.policy = SAC('MlpPolicy', 
                          env = self.env,\
                          gradient_steps = self.n_grad_updates,\
                          batch_size = batch_size,\
                          buffer_size = buffer_size)


    ''' MBPO algorithm'''
    def train(self):
        for epoch in range(self.n_epochs):
            # train a dynamics model on D_env via max likelihood 
            self.dynamics.train(self.D_env)

            # 
            for step in range(self.n_timesteps):
                random_state =  self.env.sample()
            
                action = self.policy.predict(random_state)
                self.D_env.append(self.env.step(action))

                for model in range(self.n_models):

                    # samples {s,a,r',s'} uniformly randomly chosen

                    state, action, reward, next_state = self.D_env.sample() 
                    # start k step rollout at state 
                    self.k_step_rollout(state)

                for grad in range(self.n_grad_updates):
                    update_policy_params


    ''' performs k step rollout beginning at state '''
    def k_step_rollout(self, state):
        for k in range(self.k):
            # step with action recommended by policy
            action = self.policy.predict(obs)
            # dynamics updates next state
            next_obs, reward, done, info = self.env.step(obs, action)
            if done:
                print('Reached done condition')
                break
            # add to D_model
            self.D_model.append([state, action, reward, new_state])
            # update state
            obs = next_obs



    self.policy.learn(total_timesteps=10000, log_interval=4)
