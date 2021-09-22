""" PyTorch datasets for offline RL experiments """

import glob
import json
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SymbolicDataset(Dataset):

    def __init__(self, path):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.path = path
        self.sample_paths = []

        for trajectory_path in tqdm(trajectory_paths):
            paths = sorted(glob.glob('{}/*.json'.format(trajectory_path)))
            traj_length = len(paths)

            for i in range(traj_length):
                self.sample_paths.append(paths[i])

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        with open(sample_path, 'r') as f:
            sample = json.load(f)

        # TODO: transform points to ego frame

        obs = sample['obs']
        next_obs = sample['next_obs']
        action = torch.FloatTensor(sample['action'])
        reward = torch.FloatTensor([sample['reward']])
        done = torch.tensor([sample['done']])

        return obs, action, reward, done, next_obs

    def __len__(self):
        return len(self.sample_paths)


def symbolic_collate_fn(data):
    data = list(zip(*data))
    for i, _ in enumerate(data):
        if isinstance(data[i][0], torch.Tensor):
            data[i] = torch.stack(data[i])
    return data


class MLPDataset(Dataset):

    def __init__(self, path):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.path = path
        self.sample_paths = []

        for trajectory_path in tqdm(trajectory_paths):
            paths = sorted(glob.glob('{}/*.json'.format(trajectory_path)))
            traj_length = len(paths)

            for i in range(traj_length):
                self.sample_paths.append(paths[i])

        print('Number of samples: {}'.format(len(self)))

    def _extract_obs(self, obs_dict):
        obs = torch.FloatTensor([
            obs_dict['next_orientation'],
            obs_dict['dist_to_trajectory'],
            obs_dict['ego_features']['speed'],
            obs_dict['light'],
            obs_dict['obstacle_dist'],
            obs_dict['obstacle_speed']
        ])
        return obs

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        with open(sample_path, 'r') as f:
            sample = json.load(f)

        # TODO: transform points to ego frame

        obs = self._extract_obs(sample['obs'])
        next_obs = self._extract_obs(sample['next_obs'])
        action = torch.FloatTensor(sample['action'])
        reward = torch.FloatTensor([sample['reward']])
        done = torch.tensor([sample['done']])

        return obs, action, reward, done, next_obs

    def __len__(self):
        return len(self.sample_paths)
