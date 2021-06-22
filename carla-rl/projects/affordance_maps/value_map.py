""" Copied from LBC """

import os
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, griddata, Rbf
from sklearn.neighbors import KDTree
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from omegaconf import DictConfig, OmegaConf
import hydra

from spatial_data import SpatialDataset
from train_ego import EgoModel
from utils import CALIBRATION


ACTIONS = torch.tensor(np.stack(np.meshgrid(np.linspace(-.5,.5,5), np.array([-.5,.5,1])), axis=-1)).reshape(15,2).float()
YAWS = torch.tensor(np.linspace(-.8,.8,5))
SPEEDS = torch.tensor(np.linspace(0,10,4))


def solve_for_value_function(rewards, world_pts, model, discount_factor=.9):
    """
    Solves for value function using Bellman updates and dynamic programming

    Expects list of RewardMap objects
    """

    num_steps = len(rewards)
    num_yaws = len(YAWS)
    num_spds = len(SPEEDS)
    num_acts = len(ACTIONS)

    Q = torch.zeros((num_steps,64,64,num_spds,num_yaws,num_acts))
    V = Q.clone().detach()[:,:,:,:,:,0]

    start = time.time()

    for t in range(num_steps-1,-1,-1):
        reward = (rewards[t] == 0).float()-1

        if t == num_steps-1:
            Q[t] = reward.reshape(1,64,64,1,1,1)
            continue

        yaws = YAWS.clone()
        spds = SPEEDS.clone()
        acts = ACTIONS.clone()

        locs = world_pts[t]
        for s, spd in enumerate(SPEEDS):
            for y, yaw in enumerate(YAWS):
                next_locs, next_yaws, next_spds = model.forward(
                    locs[:,None,:].repeat(1,num_acts,1).reshape(-1,2),
                    yaw[None,None].repeat(64*64,num_acts,1).reshape(-1,1),
                    spd[None,None].repeat(64*64,num_acts,1).reshape(-1,1),
                    ACTIONS[None].repeat(64*64,1,1).reshape(-1,2))

                pos_idx = KDTree(world_pts[t+1], leaf_size=5).query(next_locs.detach())[1]
                next_Vs = V[t+1].reshape(-1,num_spds,num_yaws)[pos_idx,s,y]

                # Bellman backup
                Q[t,:,:,s,y] = reward[...,None] + (discount_factor * next_Vs.reshape(64,64,num_acts))

        # max over all actions
        V[t] = Q[t].max(dim=-1)[0]

    V = V.detach().numpy()

    end = time.time()
    print('total: {}'.format(end - start))

    # fig, axs = plt.subplots(num_spds, num_yaws)
    # for s in range(num_spds):
    #     for y in range(num_yaws):
    #         axs[s,y].imshow(V[0,:,:,s,y])
    # plt.show()


def main():
    dataset_paths = [
        '/home/brian/carla-rl/carla-rl/projects/affordance_maps/sample_data/'
    ]
    # val_path = '/media/brian/linux-data/reward_maps_val/'
    dataset = SpatialDataset(dataset_paths[0])

    # images, rewards, world_pts = dataset[100]
    model = torch.load('/home/brian/carla-rl/carla-rl/projects/affordance_maps/ego_model.th')

    for (images, rewards, world_pts) in dataset:
        solve_for_value_function(rewards, world_pts, model)


if __name__ == '__main__':
    main()
