""" PyTorch datasets for offline RL experiments """

import glob
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl

from environment import CarlaEnv
from utils import preprocess_rgb, preprocess_topdown


class TransformerDataset(Dataset):

    def __init__(self, path, K=25):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.path = path
        self.K = K

        self.obs, self.actions, self.rewards = [], [], []
        self.timesteps = []

        self.image_paths = []

        for trajectory_path in trajectory_paths:
            samples = []
            image_paths = sorted(glob.glob('{}/topdown/*.png'.format(trajectory_path)))
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
                reward = [samples[i+t]['reward'] for t in range(K)]

                for t in range(K-2,-1,-1):
                    reward[t] += reward[t+1]

                action = [samples[i+t]['action'] for t in range(K)]
                # timesteps = [i+t for t in range(K)]

                self.obs.append(obs)
                self.actions.append(action)
                self.rewards.append(reward)
                # self.timesteps.append(timesteps)

                image_path = [image_paths[i+t] for t in range(K)]
                self.image_paths.append(image_path)

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        obs = self.obs[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        # timesteps = self.timesteps[idx]

        obs = torch.FloatTensor(obs).squeeze(1)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)[...,None]
        # timesteps = torch.LongTensor(timesteps).reshape(-1)

        images = torch.stack([preprocess_rgb(cv2.imread(path)) for path in self.image_paths[idx]], dim=0)

        return obs, images, actions, rewards #, timesteps

    def __len__(self):
        return len(self.rewards)

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]


class TransformerDataModule(pl.LightningDataModule):

    def __init__(self, paths, batch_size=4):
        super().__init__()
        self.paths = paths
        self.batch_size = batch_size

        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        datasets = [TransformerDataset(path) for path in self.paths]
        self.dataset = torch.utils.data.ConcatDataset(datasets)
        train_size = int(len(self.dataset) * .9)
        val_size = len(self.dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, (train_size, val_size))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4)


if __name__ == '__main__':
    dataset = TransformerDataset('/home/brian/carla-rl/carla-rl/projects/affordance_maps/autopilot_data')