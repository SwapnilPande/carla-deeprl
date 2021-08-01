""" PyTorch datasets for offline RL experiments """

import glob
from pathlib import Path
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from environment import CarlaEnv
from utils import preprocess_rgb, preprocess_topdown


class TransformerDataset(Dataset):

    def __init__(self, path, K=26):
        super().__init__()

        if 'stacked_states.npy' in os.listdir(path) and 'stacked_actions.npy' in os.listdir(path):
            self.obs = np.load('{}/stacked_states.npy'.format(path)).reshape(-1,K,8)
            self.actions = np.load('{}/stacked_actions.npy'.format(path)).reshape(-1,K,2)
        else:
            trajectory_paths = glob.glob('{}/*'.format(path))
            # assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)
            if len(trajectory_paths) == 0:
                print('No trajectories found in {}'.format(path))

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

    def __init__(self, train_paths, val_paths=None, batch_size=1024):
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
        if self.val_paths is not None:
            val_datasets = [TransformerDataset(path) for path in self.val_paths]
            self.val_data = torch.utils.data.ConcatDataset(val_datasets)
        else:
            train_size = int(len(self.train_data) * .9)
            val_size = len(self.train_data) - train_size
            self.train_data, self.val_data = torch.utils.data.random_split(self.train_data, (train_size, val_size))

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
            if len(trajectory_paths) == 0:
                print('No trajectories found in {}'.format(path))

            self.obs, self.actions = [], []

            for trajectory_path in trajectory_paths:
                samples = []
                json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
                traj_length = len(json_paths)

                for i in range(traj_length):
                    with open(json_paths[i]) as f:
                        sample = json.load(f)
                    samples.append(sample)

                if traj_length < 26:
                    continue

                for i in range(traj_length):
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

    def __init__(self, train_paths, val_paths=None, batch_size=1024):
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
        if self.val_paths is not None:
            val_datasets = [VQVAEDataset(path) for path in self.val_paths]
            self.val_data = torch.utils.data.ConcatDataset(val_datasets)
        else:
            train_size = int(len(self.train_data) * .9)
            val_size = len(self.train_data) - train_size
            self.train_data, self.val_data = torch.utils.data.random_split(self.train_data, (train_size, val_size))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4)


class TokenizedDataset(Dataset):

    def __init__(self, obs, actions, weights):
        super().__init__()

        self.obs = obs
        self.actions = actions
        self.weights = weights

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        obs = self.obs[idx]
        actions = self.actions[idx]

        obs = torch.LongTensor(obs)
        actions = torch.LongTensor(actions).flatten()
        return obs, actions

    def __len__(self):
        return len(self.actions)

    def get_sampler(self):
        sampler = WeightedRandomSampler(self.weights, len(self.weights))
        return sampler


class TokenizedDataModule(pl.LightningDataModule):

    def __init__(self, path, batch_size=1024):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        obs = np.load('{}/stacked_state_tokens.npy'.format(self.path)).reshape(-1,26)
        actions = np.load('{}/stacked_action_tokens.npy'.format(self.path)).reshape(-1,26)
        weights = np.load('{}/action_weights.npy'.format(self.path))

        num_samples = len(obs)
        train_size = int(.9 * num_samples)
        val_size = num_samples - train_size

        idx = np.arange(num_samples)
        np.random.shuffle(idx)
        train_idx = idx[:train_size]
        val_idx = idx[train_size:]

        self.train_data = TokenizedDataset(obs[train_idx], actions[train_idx], weights[train_idx])
        self.val_data = TokenizedDataset(obs[val_idx], actions[val_idx], weights[val_idx])

    def train_dataloader(self):
        sampler = self.train_data.get_sampler()
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=sampler, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4)