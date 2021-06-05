# morel imports
# from offline_rl.ppo2 import PPO2
from offline_rl.fake_env import FakeEnv

import hydra

import numpy as np
from tqdm import tqdm

# torch imports
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

#TODO import PPO here

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

        self.dynamics = hydra.utils.instantiate(dynamics_cfg)
        self.policy = hydra.utils.instantiate(policy_cfg)


    def train(self,
            offline_data_module_cfg,
            online_data_module_cfg,
            logger,
            gpu_number = None,
            precision = 16):

        logger.log_hyperparams({
            "uncertainty_threshold (morel)" : self.uncertainty_threshold,
            "uncertainty_penatly (morel)" : self.uncertainty_penalty
        })

        print("---------------- Beginning Dynamics Training ----------------")
        self.dynamics.train(
            offline_data_module_cfg,
            logger = logger,
            gpu_number = gpu_number,
            precision = precision
        )
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
        self.policy.train(
            data_module_cfg = online_data_module_cfg,
            env = env,
            logger = logger,
            gpu_number = gpu_number,
            precision = precision
        )
        print("---------------- Ending Policy Training ----------------")


    # def save(self, save_dir):
    #     if(not os.path.isdir(save_dir)):
    #         os.mkdir(save_dir)

    #     self.policy.save(save_dir)
    #     self.dynamics.save(save_dir)

    # def load(self, load_dir):
    #     self.policy.load(load_dir)
    #     # self.dynamics.load(load_dir)



