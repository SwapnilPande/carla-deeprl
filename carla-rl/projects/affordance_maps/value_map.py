# """ Copied from LBC """

# import os
# import math
# import time

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import RegularGridInterpolator, griddata, Rbf
# from sklearn.neighbors import KDTree
# import torch
# import torch.nn.functional as F
# from torch import nn
# import torch.optim as optim
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint

# from omegaconf import DictConfig, OmegaConf
# import hydra

# from spatial_data import SpatialDataset
# from train_ego import EgoModel
# from utils import CALIBRATION


# ACTIONS = torch.tensor(np.stack(np.meshgrid(np.linspace(-.5,.5,5), np.array([-.5,.5,1])), axis=-1)).reshape(15,2).float()
# YAWS = torch.tensor(np.linspace(-.8,.8,5))
# SPEEDS = torch.tensor(np.linspace(0,10,4))


# def solve_for_value_function(rewards, world_pts, model, discount_factor=.9):
#     """
#     Solves for value function using Bellman updates and dynamic programming

#     Expects list of RewardMap objects
#     """

#     num_steps = len(rewards)
#     num_yaws = len(YAWS)
#     num_spds = len(SPEEDS)
#     num_acts = len(ACTIONS)

#     Q = torch.zeros((num_steps,64,64,num_spds,num_yaws,num_acts))
#     V = Q.clone().detach()[:,:,:,:,:,0]

#     start = time.time()

#     for t in range(num_steps-1,-1,-1):
#         reward = (rewards[t] == 0).float()-1

#         if t == num_steps-1:
#             Q[t] = reward.reshape(1,64,64,1,1,1)
#             continue

#         yaws = YAWS.clone()
#         spds = SPEEDS.clone()
#         acts = ACTIONS.clone()

#         locs = world_pts[t]
#         for s, spd in enumerate(SPEEDS):
#             for y, yaw in enumerate(YAWS):
#                 next_locs, next_yaws, next_spds = model.forward(
#                     locs[:,None,:].repeat(1,num_acts,1).reshape(-1,2),
#                     yaw[None,None].repeat(64*64,num_acts,1).reshape(-1,1),
#                     spd[None,None].repeat(64*64,num_acts,1).reshape(-1,1),
#                     ACTIONS[None].repeat(64*64,1,1).reshape(-1,2))

#                 pos_idx = KDTree(world_pts[t+1], leaf_size=5).query(next_locs.detach())[1]
#                 next_Vs = V[t+1].reshape(-1,num_spds,num_yaws)[pos_idx,s,y]

#                 # Bellman backup
#                 Q[t,:,:,s,y] = reward[...,None] + (discount_factor * next_Vs.reshape(64,64,num_acts))

#         # max over all actions
#         V[t] = Q[t].max(dim=-1)[0]

#     V = V.detach().numpy()

#     end = time.time()
#     print('total: {}'.format(end - start))

#     # fig, axs = plt.subplots(num_spds, num_yaws)
#     # for s in range(num_spds):
#     #     for y in range(num_yaws):
#     #         axs[s,y].imshow(V[0,:,:,s,y])
#     # plt.show()


# def main():
#     dataset_paths = [
#         '/home/brian/carla-rl/carla-rl/projects/affordance_maps/sample_data/'
#     ]
#     # val_path = '/media/brian/linux-data/reward_maps_val/'
#     dataset = SpatialDataset(dataset_paths[0])

#     # images, rewards, world_pts = dataset[100]
#     model = torch.load('/home/brian/carla-rl/carla-rl/projects/affordance_maps/ego_model.th')

#     for (images, rewards, world_pts) in dataset:
#         solve_for_value_function(rewards, world_pts, model)


# if __name__ == '__main__':
#     main()


""" Copied from LBC """

import os
import math
import time
import queue

import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator, griddata, Rbf
from sklearn.neighbors import KDTree
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


ACTIONS = np.array([
    [-1, 1/3],  [-1, 2/3], [-1, 1],
    # [-.75,1/3], [-.75,2/3],[-.75,1],
    [-.5,1/3],  [-.5,2/3], [-.5,1],
    # [-.25,1/3], [-.25,2/3],[-.25,1],
    [0, 1/3],   [0, 2/3],  [0, 1],
    # [.25,1/3], [.25,2/3],[.25,1],
    [.5,1/3],  [.5,2/3], [.5,1],
    # [.75,1/3], [.75,2/3],[.75,1],
    [1, 1/3],  [1, 2/3], [1, 1],
    [0, -1]
], dtype=np.float32).reshape(16,2)


# torch.tensor(np.stack(np.meshgrid(np.linspace(-.5,.5,5), np.array([-.5,.5,1])), axis=-1)).reshape(15,2).float()
YAWS = np.linspace(-1.,1.,3)
SPEEDS = np.linspace(0,8,3)


num_yaws = len(YAWS)
num_spds = len(SPEEDS)
num_acts = len(ACTIONS)


def solve_for_value_function_mp(*args, num_attempts=5):
    for i in range(num_attempts):
        try:
            V = solve_for_value_function(*args)
            return V
        except Exception as e:
            print(e)

    raise Exception('num_attempts exceeded')


def solve_for_value_function(rewards, world_pts, model, discount_factor=.9):
    """
    Solves for value function using Bellman updates and dynamic programming

    Expects list of RewardMap objects
    """

    with torch.no_grad():
        num_steps = len(rewards)

        Q = torch.zeros((num_steps,64,64,num_spds,num_yaws,num_acts))
        V = torch.zeros((num_steps,64,64,num_spds,num_yaws))

        start = time.time()

        for t in range(num_steps-1,-1,-1):
            reward = (rewards[t] == 0).float()-1

            if t == num_steps-1:
                Q[t] = reward.reshape(1,64,64,1,1,1)
            else:
                yaws = torch.tensor(YAWS)
                spds = torch.tensor(SPEEDS)
                acts = torch.tensor(ACTIONS)

                locs = world_pts[t]
                for s, spd in enumerate(spds):
                    for y, yaw in enumerate(yaws):
                        next_locs, next_yaws, next_spds = model.forward(
                            locs[:,None,:].repeat(1,num_acts,1).reshape(-1,2),
                            spd[None,None].repeat(64*64,num_acts,1).reshape(-1,1),
                            yaw[None,None].repeat(64*64,num_acts,1).reshape(-1,1),
                            acts[None].repeat(64*64,1,1).reshape(-1,2))

                        pos_idx = KDTree(world_pts[t+1], leaf_size=5).query(next_locs.detach())[1]
                        spd_idx = torch.cdist(spds[:,None], next_spds).argmin(dim=0)[:,None]
                        yaw_idx = torch.cdist(yaws[:,None], next_yaws).argmin(dim=0)[:,None]
                        next_Vs = V[t+1].reshape(-1,num_spds,num_yaws)[pos_idx,spd_idx,yaw_idx]

                        # Bellman backup
                        terminal = (reward < 0)[...,None]
                        Q[t,:,:,s,y] = reward[...,None] + (discount_factor * ~terminal * next_Vs.reshape(64,64,num_acts))

            # max over all actions
            V[t] = Q[t].max(dim=-1)[0]

        V = V.detach().numpy()

    end = time.time()
    print('total: {}'.format(end - start))

    return V


def main():
    dataset_paths = [
        # '/zfsauton/datasets/ArgoRL/brianyan/expert_data/',
        '/zfsauton/datasets/ArgoRL/brianyan/expert_data_ignorecars/',
        '/zfsauton/datasets/ArgoRL/brianyan/bad_data/'
        # '/home/brian/carla-rl/carla-rl/projects/affordance_maps/sample_data/'
    ]
    datasets = [SpatialDataset(path) for path in dataset_paths]
    dataset = torch.utils.data.ConcatDataset(datasets)

    model_weights = torch.load('ego_model.th')
    model = EgoModel()
    model.load_state_dict(model_weights)
    model.eval()

    model.share_memory()

    def label_worker(queue, model):
        try:
            while True:
                image_path, rewards, world_pts = queue.get(timeout=60)

                value_path = image_path.split('/')
                value_path[-2] = 'reward'
                value_path[-1] = value_path[-1][:-4] + '_value'
                value_path = '/'.join(value_path)

                V = solve_for_value_function_mp(rewards, world_pts, model)
                np.save(value_path, V)
                print('Saved to {}'.format(value_path))
        except queue.Empty as e:
            print('Queue empty, terminating')
            return
        except Exception as e:
            print(e)
            return

    NUM_PROCESSES = 30
    processes = []
    sample_queue = mp.Queue(maxsize=len(dataset))

    print('Spawning {} labelers'.format(NUM_PROCESSES))

    # spawn labeling workers
    for _ in range(NUM_PROCESSES):
        p = mp.Process(target=label_worker, args=(sample_queue, model))
        p.start()
        processes.append(p)

    print('Populating queue')

    # fill queue as needed
    for sample in tqdm(dataset):
        sample_queue.put(sample)

        while sample_queue.qsize() >= NUM_PROCESSES+10:
            time.sleep(.1)

    print('Finished populating queue. Waiting for queue to empty')

    # stall until queue is empty
    while sample_queue.qsize() > 0:
        time.sleep(.1)

    print('Killing labeler processes')

    # kill processes
    for p in processes:
        p.join()

    print('Done')


if __name__ == '__main__':
    main()
