""" Copied from LBC """

import os
import math
import time
import glob
import traceback
import json

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
# from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.multiprocessing as mp
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
import hydra

from spatial_data import SpatialDataset
from train_ego import EgoModel
from utils import CALIBRATION
from common.utils import preprocess_rgb


ACTIONS = np.array([
    [-1, 1/3],  [-1, 2/3], [-1, 1],
    [-.75,1/3], [-.75,2/3],[-.75,1],
    [-.5,1/3],  [-.5,2/3], [-.5,1],
    [-.25,1/3], [-.25,2/3],[-.25,1],
    [0, 1/3],   [0, 2/3],  [0, 1],
    [.25,1/3], [.25,2/3],[.25,1],
    [.5,1/3],  [.5,2/3], [.5,1],
    [.75,1/3], [.75,2/3],[.75,1],
    [1, 1/3],  [1, 2/3], [1, 1],
    [0, -1]
], dtype=np.float32).reshape(28,2)


# torch.tensor(np.stack(np.meshgrid(np.linspace(-.5,.5,5), np.array([-.5,.5,1])), axis=-1)).reshape(15,2).float()
YAWS = np.linspace(-1.0,1.0,5)
SPEEDS = np.linspace(0,8,4)


num_yaws = len(YAWS)
num_spds = len(SPEEDS)
num_acts = len(ACTIONS)


def solve_for_value_function(rewards, locs, model, next_locs, prev_V, curr_yaw, discount_factor=.9):
    """
    Solves for value function using Bellman updates and dynamic programming

    Expects list of RewardMap objects
    """

    with torch.no_grad():
        start = time.time()

        reward = (rewards == 0).float()-1

        yaws = torch.tensor(YAWS)
        spds = torch.tensor(SPEEDS)
        acts = torch.tensor(ACTIONS)

        # normalize grid so we can use grid interpolation
        offset = next_locs[0]
        _next_locs = next_locs - offset

        theta = np.arctan2(_next_locs[-1][1], _next_locs[-1][0])
        _next_locs = rotate_pts(_next_locs, (np.pi/4)-theta)

        min_x, min_y = np.min(_next_locs, axis=0)
        max_x, max_y = np.max(_next_locs, axis=0)

        # set up grid interpolator
        xs, ys = np.linspace(min_x, max_x, 64), np.linspace(min_y, max_y, 64)
        values = np.array(prev_V)
        values = np.moveaxis(values, 0, 1) # because indexing=ij, for more: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        grid_interpolator = RegularGridInterpolator((xs, ys, spds, yaws), values, bounds_error=False, fill_value=None)

        Q = torch.zeros((64,64,num_spds,num_yaws,num_acts))

        print(curr_yaw)

        for s, spd in enumerate(spds):
            for y, yaw in enumerate(yaws):
                # predict next states
                pred_locs, pred_yaws, pred_spds = model.forward(
                    locs[:,None,:].repeat(1,num_acts,1).reshape(-1,2),
                    yaw[None,None].repeat(64*64,num_acts,1).reshape(-1,1),
                    spd[None,None].repeat(64*64,num_acts,1).reshape(-1,1),
                    acts[None].repeat(64*64,1,1).reshape(-1,2))

                # convert locs to normalized grid coordinates and interpolate next Vs
                _pred_locs = pred_locs - offset
                _pred_locs = rotate_pts(_pred_locs, (np.pi/4)-theta)
                pred_pts = np.concatenate([_pred_locs, pred_spds, pred_yaws], axis=1)
                next_Vs = grid_interpolator(pred_pts, method='linear')

                # Bellman backup
                terminal = (reward < 0)[...,None]
                Q_target = reward[...,None] + (discount_factor * ~terminal * next_Vs.reshape(64,64,num_acts))
                Q_target[torch.isnan(Q_target)] = -1
                Q[:,:,s,y] = torch.clamp(Q_target, -1, 0)

        # max over all actions
        V = Q.max(dim=-1)[0]
        V = V.detach().numpy()

    end = time.time()
    print('total: {}'.format(end - start))
    return V


def rotate_pts(pts, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R.dot(pts.T).T


def load_trajectory(path):
    rgb_paths = sorted(glob.glob('{}/topdown/*.png'.format(path)))[::-1]
    measurement_paths = sorted(glob.glob('{}/measurements/*.json'.format(path)))[::-1]
    reward_paths = sorted([r for r in glob.glob('{}/reward/*.png'.format(path)) if 'value' not in r])[::-1]
    world_paths = sorted(glob.glob('{}/world/*.npy'.format(path)))[::-1]
    assert len(measurement_paths) == len(reward_paths) == len(world_paths), 'Uneven number of reward/world/rgb paths'

    for rgb_path, measurement_path, reward_path, world_path in zip(rgb_paths, measurement_paths, reward_paths, world_paths):
        rgb = cv2.imread(rgb_path)
        with open(measurement_path, 'r') as f:
            m = json.load(f)
            yaw = m['actor_tf'][4]

        reward = (preprocess_rgb(cv2.imread(reward_path), image_size=(64,64)) * 255)[0]
        reward = (reward == 0).float()-1
        world_pts = torch.FloatTensor(np.load(world_path))

        value_path = reward_path.split('/')
        value_path[-1] = value_path[-1][:-4] + '_value'
        value_path = '/'.join(value_path)

        yield rgb, reward, world_pts, value_path, yaw


def labeler_worker(worker_id, queue, model):
    try:
        while True:
            trajectory_path = queue.get(timeout=60)
            trajectory = load_trajectory(trajectory_path)
            for i, (reward, world_pts, value_path, yaw) in enumerate(trajectory):
                if i == 0:
                    V = reward.reshape(64,64,1,1).repeat(1,1,num_spds,num_yaws)
                else:
                    V = solve_for_value_function(reward, world_pts, model, next_world_pts, V, yaw)

                np.save(value_path, V)
                print('Worker {} saving to {}'.format(worker_id, value_path))
                next_world_pts = world_pts
    except Exception as e:
        print(worker_id)
        traceback.print_exc(e)
        return


def main():
    dataset_paths = [
        '/zfsauton/datasets/ArgoRL/brianyan/expert_data/',
        # '/zfsauton/datasets/ArgoRL/brianyan/expert_data_ignorecars/',
        '/zfsauton/datasets/ArgoRL/brianyan/bad_data/'
        # '/home/brian/carla-rl/carla-rl/projects/affordance_maps/sample_data/'
    ]

    trajectory_paths = []
    for dataset_path in dataset_paths:
        paths = glob.glob(dataset_path + '/*')
        trajectory_paths.extend(paths)

    model_weights = torch.load('ego_model.th')
    model = EgoModel()
    model.load_state_dict(model_weights)
    model.eval()

    plt.ion()
    fig, axs = plt.subplots(5,5)

    V = None
    for (rgb, reward, world_pts, value_path, yaw) in load_trajectory('/home/brian/carla-rl/carla-rl/projects/affordance_maps/sample_data/06_29_16_27_29_89/'):
        if isinstance(V, type(None)):
            V = reward.reshape(64,64,1,1).repeat(1,1,num_spds,num_yaws)
        else:
            V = solve_for_value_function(reward, world_pts, model, next_world_pts, V, yaw)
        next_world_pts = world_pts

        for s in range(4):
            for y in range(5):
                axs[s,y].imshow(V[:,:,s,y])
        axs[4,2].imshow(rgb)
        axs[4,1].imshow(reward)
        plt.show()
        plt.pause(.001)

    # model.share_memory()

    # NUM_PROCESSES = 30
    # processes = []
    # sample_queue = mp.Queue(maxsize=len(trajectory_paths))

    # print('Spawning {} labelers'.format(NUM_PROCESSES))

    # # spawn labeling workers
    # for pid in range(NUM_PROCESSES):
    #     p = mp.Process(target=labeler_worker, args=(pid, sample_queue, model))
    #     p.start()
    #     processes.append(p)

    # print('Populating queue')

    # # fill queue as needed
    # for traj in tqdm(trajectory_paths):
    #     sample_queue.put(traj)

    #     while sample_queue.qsize() >= NUM_PROCESSES:
    #         time.sleep(.1)

    # print('Finished populating queue. Waiting for queue to empty')

    # # stall until queue is empty
    # while sample_queue.qsize() > 0:
    #     time.sleep(.1)

    # print('Killing labeler processes')

    # # kill processes
    # for p in processes:
    #     p.join()

    # print('Done')


if __name__ == '__main__':
    main()
