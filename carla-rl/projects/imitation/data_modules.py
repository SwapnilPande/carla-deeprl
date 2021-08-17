""" PyTorch datasets for offline RL experiments """

import glob
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl

from common.utils import get_reward, preprocess_rgb, preprocess_topdown, \
    get_angle_to_next_node, get_obs, get_action, get_dir
# from .replay_buffer import Experience, ReplayBuffer, PERBuffer
from environment import CarlaEnv

"""
Offline dataset handling
"""

class OfflineCarlaDataset(Dataset):
    """ Offline dataset """

    def __init__(self, path, use_images=True, H=1, max_trajectories=-1):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        if max_trajectories != -1:
            trajectory_paths = trajectory_paths[:max_trajectories]

        self.path = path
        self.use_images = use_images
        self.H = H

        self.obs = []
        self.actions, self.rewards, self.terminals = [], [], []

        for trajectory_path in trajectory_paths:
            samples = []
            json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
            image_paths = sorted(glob.glob('{}/rgb/*.png'.format(trajectory_path)))
            traj_length = min(len(json_paths), len(image_paths))

            for i in range(traj_length):
                with open(json_paths[i]) as f:
                    sample = json.load(f)
                sample['image_path'] = image_paths[i]
                samples.append(sample)

            if traj_length <= H:
                continue

            for i in range(H-1, traj_length, H):
                # TODO: this is definitely the slow way
                image_paths = [samples[t]['image_path'] for t in range(i-H+1, i+1)]
                mlp_features = [samples[t]['obs'] for t in range(i-H+1, i+1)]

                obs = image_paths, mlp_features
                # reward = samples[i]['reward']
                # terminal = samples[i]['done']
                action = [samples[t]['action'] for t in range(i-H+1, i+1)]

                self.obs.append(obs)
                self.actions.append(action)
                # self.rewards.append(reward)
                # self.terminals.append(terminal)

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        image_paths, mlp_features = self.obs[idx]

        mlp_features = torch.FloatTensor(mlp_features).reshape(self.H,8)
        action = torch.FloatTensor(self.actions[idx])
        # reward = torch.FloatTensor([self.rewards[idx]])
        # terminal = torch.Tensor([self.terminals[idx]])

        # mlp_features[:,[1,2,7]] = 0.  # hide privileged information

        if self.use_images:
            image = torch.cat([preprocess_rgb(cv2.imread(path), image_size=(64,64))[None] for path in image_paths], dim=0)
            return (image, mlp_features), action
        else:
            return mlp_features.flatten(), action

    def __len__(self):
        return len(self.actions)

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]


class OfflineCarlaDataModule(pl.LightningDataModule):
    """ Datamodule for offline driving data """

    def __init__(self, cfg):
        super().__init__()
        self.paths = cfg.dataset_paths
        self.use_images = cfg.use_images
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.val_dataset_idx = cfg.val_dataset_idx
        self.max_train_trajectories = cfg.max_train_trajectories
        self.max_val_trajectories = cfg.max_val_trajectories
        self.H = cfg.horizon_length

        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        train_datasets = [d for i,d in enumerate(self.paths) if i not in self.val_dataset_idx]
        val_datasets = [d for i,d in enumerate(self.paths) if i in self.val_dataset_idx]

        train_datasets = [OfflineCarlaDataset(use_images=self.use_images, path=path, H=self.H, max_trajectories=self.max_train_trajectories) for path in train_datasets]
        val_datasets = [OfflineCarlaDataset(use_images=self.use_images, path=path, H=self.H, max_trajectories=self.max_val_trajectories) for path in val_datasets]

        self.train_data = torch.utils.data.ConcatDataset(train_datasets)
        self.val_data = torch.utils.data.ConcatDataset(val_datasets)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
