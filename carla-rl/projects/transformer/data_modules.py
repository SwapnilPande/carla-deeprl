""" PyTorch datasets for offline RL experiments """

import glob
from pathlib import Path
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl

from environment import CarlaEnv
from utils import preprocess_rgb, preprocess_topdown


class TransformerDataset(Dataset):

    def __init__(self, path, K=26):
        super().__init__()

        if 'stacked_states.npy' in os.listdir(path) and 'stacked_actions.npy' in os.listdir(path):
            self.obs = np.load('{}/stacked_states.npy'.format(path)).reshape(-1,K,8)
            self.actions = np.load('{}/stacked_actions.npy'.format(path))
        else:
            trajectory_paths = glob.glob('{}/*'.format(path))
            assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

            self.path = path
            self.K = K

            self.obs, self.actions, self.rewards = [], [], []
            self.timesteps = []

            for trajectory_path in trajectory_paths:
                samples = []
                json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
                traj_length = len(json_paths)

                for i in range(traj_length):
                    with open(json_paths[i]) as f:
                        sample = json.load(f)
                    samples.append(sample)

                if traj_length <= (self.K + 1):
                    continue

                for i in range(traj_length-K+1):
                    obs = [samples[i+t]['obs'] for t in range(K)]
                    # reward = [samples[i+t]['reward'] for t in range(K)]

                    # for t in range(K-2,-1,-1):
                    #     reward[t] += reward[t+1]

                    action = [samples[i+t]['action'] for t in range(K)]

                    self.obs.append(obs)
                    self.actions.append(action)
                    # self.rewards.append(reward)

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        obs = self.obs[idx]
        actions = self.actions[idx]

        obs = torch.FloatTensor(obs).squeeze(1)
        actions = torch.FloatTensor(actions)

        return obs, actions

    def __len__(self):
        return len(self.obs)


class TransformerDataModule(pl.LightningDataModule):

    def __init__(self, train_paths, val_paths, batch_size=1024):
        super().__init__()
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.batch_size = batch_size

        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        train_datasets = [TransformerDataset(path) for path in self.train_paths]
        self.train_data = torch.utils.data.ConcatDataset(train_datasets)
        val_datasets = [TransformerDataset(path) for path in self.val_paths]
        self.val_data = torch.utils.data.ConcatDataset(val_datasets)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=8)


class VQVAEDataset(Dataset):

    def __init__(self, path):
        super().__init__()

        if 'states.npy' in os.listdir(path) and 'actions.npy' in os.listdir(path):
            self.obs = np.load('{}/states.npy'.format(path))
            self.actions = np.load('{}/actions.npy'.format(path))
        else:
            trajectory_paths = glob.glob('{}/*'.format(path))
            assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

            self.obs, self.actions = [], []

            for trajectory_path in trajectory_paths:
                samples = []
                json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
                traj_length = len(json_paths)

                for i in range(traj_length):
                    with open(json_paths[i]) as f:
                        sample = json.load(f)
                    samples.append(sample)

                if traj_length <= (self.K + 1):
                    continue

                for i in range(traj_length-K+1):
                    obs = samples[i]['obs']
                    action = samples[i]['action']

                    self.obs.append(obs)
                    self.actions.append(action)

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        obs = self.obs[idx]
        actions = self.actions[idx]

        obs = torch.FloatTensor(obs).flatten()
        actions = torch.FloatTensor(actions).flatten()
        return obs, actions

    def __len__(self):
        return len(self.actions)


class VQVAEDataModule(pl.LightningDataModule):

    def __init__(self, train_paths, val_paths, batch_size=1024):
        super().__init__()
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.batch_size = batch_size

        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        train_datasets = [VQVAEDataset(path) for path in self.train_paths]
        self.train_data = torch.utils.data.ConcatDataset(train_datasets)
        val_datasets = [VQVAEDataset(path) for path in self.val_paths]
        self.val_data = torch.utils.data.ConcatDataset(val_datasets)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4)
