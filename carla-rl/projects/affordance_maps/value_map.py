""" Copied from LBC """

import os
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
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


ACTIONS = torch.tensor(np.stack(np.meshgrid(np.linspace(-.5,.5,5), np.linspace(-1,1,5)), axis=-1)).reshape(25,2).float()
YAWS = torch.tensor(np.linspace(0,2*np.pi,5))
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

    for t in range(num_steps-1,-1,-1):
        reward = (rewards[t] == 0).float()-1

        if t == num_steps-1:
            Q[t] = reward.reshape(1,64,64,1,1,1)
            continue

        # locs = torch.tensor(np.stack(np.meshgrid(np.arange(64), np.arange(64)), axis=-1)).reshape(64*64,2).long()
        # _locs = (locs.float() - 32) * 18 / 32 # transform to world coordinates

        yaws = YAWS.clone()
        spds = SPEEDS.clone()
        acts = ACTIONS.clone()

        locs = world_pts[t]
        for a, act in enumerate(ACTIONS):
            for s, spd in enumerate(SPEEDS):
                for y, yaw in enumerate(YAWS):
                    next_locs, next_yaws, next_spds = model.forward(
                        locs, 
                        yaw[None].repeat(64*64,1),
                        spd[None].repeat(64*64,1),
                        act[None].repeat(64*64,1))

                    # construct reference points
                    pos_pts = world_pts[t+1][:,None,None,:].repeat(1,num_spds,num_yaws,1).reshape(-1,2)
                    yaw_pts = YAWS[None,None,:,None].repeat(64*64,num_spds,1,1).reshape(-1,1)
                    spd_pts = SPEEDS[None,:,None,None].repeat(64*64,1,num_yaws,1).reshape(-1,1)
                    ref_pts = torch.cat([pos_pts, yaw_pts, spd_pts], dim=1) # should be n x 4
                    V_pts = V[t+1].flatten() # reward[:,:,None,None].repeat(1,1,num_spds,num_yaws).flatten()

                    # construct query points
                    query_pts = torch.cat([next_locs, next_yaws, next_spds], dim=1).detach() # should be n x 4

                    # perform Bellman backup
                    next_Vs = griddata(ref_pts, V_pts, query_pts, method='nearest')
                    Q[t,:,:,s,y,a] = reward + (discount_factor * next_Vs.reshape(64,64))

                    print(a,s,y)

        # max over all actions
        V[t] = Q[t].max(dim=-1)[0]

    V = V.detach().numpy()

    import matplotlib.pyplot as plt
    plt.imshow(V[0,:,:,0,-1])
    plt.show()

    import ipdb; ipdb.set_trace()


def main():
    dataset_paths = [
        '/home/brian/carla-rl/carla-rl/projects/affordance_maps/sample_data/'
    ]
    val_path = '/media/brian/linux-data/reward_maps_val/'
    dataset = SpatialDataset(dataset_paths[0])

    images, rewards, world_pts = dataset[100]
    model = torch.load('/home/brian/carla-rl/carla-rl/projects/affordance_maps/ego_model.th')

    solve_for_value_function(rewards, world_pts, model)


if __name__ == '__main__':
    main()
