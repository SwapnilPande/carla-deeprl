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
def learn():
    while timestep < timesteps:
        collect rollouts in env with replay buffer

        if don't continue rollout, break
        else:
            self.train(gradient_steps)

Train: sample replay buffer, do updates (gradient)
train(grad_steps, batch_size)



***************************************************
Collect Rollouts: collect experiences, store into ReplayBuffer
env, callback, train_freq, replay_buffer, ...

while should collect more steps,
while not done:
        sample action from buffer according to policy
        obs, rewrd, done, info = env.step(action)
        update reward, timesteps
        store into replay buffer D_model
    if done:
        append ep rewards, ep timesteps
get mean reward, 
return rolloutreturn


Train: self. gradientsteps, batch_size
    optimizers = self.actor.optim, self.critic.optim
    update learning rate
    entropy coef losses, [], []
    actor, critic losses = [], []
    for g in gradient steps:
        state = sample replay buffer
        action, log_prob =  by actor for sampled state 
    ent coef 
    with no grad:
        # actor select action according to policy 
        next act, next log prob = actor
        next q = 
        next q += entropy term
        target q 
    current q = critic estimates using action from replay buffer
    compute critic loss
    optimizer critic
    compute actor loss
    optimize actor
    update target networks 


***************************************************

# ON POLICY

reset rollout buffer
assert self.last_obs not none 
while < n_rollout:
    actions, values,  = self.policy.predict(obs)
    actions.tocpu()
    clip actions
    nextobs, rew, done, info = env.setp(clipped actions)
    timesteps += 
    add last obs, actions, rewards, to rollout buffer
    lastobs = newobs
    lastepisodestart = dones

    return True


def learn:
    while < total timesteps:
        continue_train = collect rollouts
    self.train()
'''
'''
collect dataset Denv from policy π_phi
for N epochs:
    train dynamics on Denv with max likelihood
'''
class MBPO:
    def __init__(self, n_epochs, n_models, n):

        # hyperparams
        self.n_epochs = 400       # N
        self.n_models = 7    # ensemble size
        self.n_timesteps = 1000 # environment steps per epoch
        self.n_rollouts = 400 # number model rollouts per environment step
        self.n_grad_updates = -1 # same as number of steps

        # datasets 
        self.D_env = None # TODO: collect data
        self.D_model = None
        self.logger = logger

        # probabilistic NN dynamics 


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


        self.env = FakeEnv(self.dynamics)
        # policy 
        self.policy = SAC('MlpPolicy', 
                          env = self.env,\
                          gradient_steps = self.n_grad_updates, \
                          batch_size = batch_size,\
                          buffer_size = buffer_size)
        self.save_dir = '/zfsauton/datasets/ArgoRL/vccheng/collect_mbpo'

    ''' MBPO algorithm'''
    def train(self):
        # collect data into DEnv, 5000
        collect_trajectory(self.env, self.save_dir, self.policy, max_path_length=5000):
        
        for epoch in range(self.n_epochs):
            
            # train a dynamics model on D_env via max likelihood 
            # (can also train once per F steps instead of once per epoch)
            self.dynamics.train(self.dynamics_epochs)

            # Dynamics Model interacts with policy
            for step in range(self.n_timesteps):
                # dynamics model feeds st, rt to policy
                random_state =  self.env.sample()
                # policy predict action to take
                action = self.policy.predict(random_state)
                # add to Denv obs, reward, done, info
                self.D_env.append(self.env.step(action))

                for rollout in range(self.n_rollouts):
                    # stores ep rewards, total timesteps, ...
                    self.policy.collect_rollouts(self.env,\
                                                 self.train_freq,\
                                                 self.replay_buffer)

                    self.policy.train()