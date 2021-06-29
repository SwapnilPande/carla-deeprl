import glob
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
import carla

from common.utils import get_reward, preprocess_rgb, preprocess_topdown, \
    get_angle_to_next_node, get_obs, get_action, get_dir
from environment import CarlaEnv
from client_bounding_boxes import ClientSideBoundingBoxes
from utils import CALIBRATION


class SpatialDataset(Dataset):
    def __init__(self, path, use_images=True, T=5, val=False):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        if val:
            trajectory_paths = trajectory_paths[:5]
        else:
            trajectory_paths = trajectory_paths[5:]

        self.path = path
        self.use_images = use_images
        self.T = T

        self.images = []
        self.rewards = []
        self.values = []

        for trajectory_path in trajectory_paths:
            samples = []

            image_paths = sorted(glob.glob('{}/topdown/*.png'.format(trajectory_path)))
            reward_paths = sorted(glob.glob('{}/reward/*.png'.format(trajectory_path)))
            value_paths = sorted(glob.glob('{}/reward/*.npy'.format(trajectory_path)))

            traj_length = min(len(image_paths), len(reward_paths), len(value_paths))

            if traj_length < T:
                continue

            for i in range(traj_length-T+1):
                self.images.append([image_paths[i+t] for t in range(T)])
                self.rewards.append([reward_paths[i+t] for t in range(T)])
                self.values.append([value_paths[i+t] for t in range(T)])

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        image_paths = self.images[idx]
        reward_paths = self.rewards[idx]
        value_paths = self.values[idx]

        images = [preprocess_rgb(cv2.imread(image_path)) for image_path in image_paths]
        rewards = [(preprocess_rgb(cv2.imread(reward_path)) * 255)[0] for reward_path in reward_paths]
        values = [torch.from_numpy(np.load(value_path)) for value_path in value_paths]

        images = torch.stack(images)
        rewards = torch.stack(rewards)
        values = torch.stack(values)

        return images, rewards, values

    def __len__(self):
        return len(self.rewards)


class SpatialDataModule(pl.LightningDataModule):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        train_datasets = [SpatialDataset(path, val=False) for path in self.paths]
        self.train_data = torch.utils.data.ConcatDataset(train_datasets)
        val_datasets = [SpatialDataset(path, val=True) for path in self.paths]
        self.val_data = torch.utils.data.ConcatDataset(val_datasets)
        # train_size = int(len(self.dataset) * .95)
        # val_size = len(self.dataset) - train_size
        # self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, (train_size, val_size))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32, shuffle=False, num_workers=4)


class EgoDataset(Dataset):

    def __init__(self, path, T=5):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.path = path
        self.T = T

        self.ego_transforms = []
        self.camera_transforms = []

        self.speeds = []
        self.actions = []

        for trajectory_path in trajectory_paths:
            samples = []
            json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
            traj_length = len(json_paths)

            for i in range(traj_length):
                with open(json_paths[i], 'r') as f:
                    sample = json.load(f)
                samples.append(sample)

            # if traj_length < 50:
            #     continue

            for i in range((traj_length) -T+1):
                ego_transforms = [samples[(i+t)]['actor_tf'] for t in range(T)]
                speed = [samples[(i+t)]['speed'] for t in range(T)]
                action = [samples[(i+t)]['action'] for t in range(T)]

                self.ego_transforms.append(ego_transforms)
                self.speeds.append(speed)
                self.actions.append(action)

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        ego_tf = self.ego_transforms[idx]
        speed = self.speeds[idx]
        action = self.actions[idx]

        ego_tf = torch.FloatTensor(ego_tf)
        speed = torch.FloatTensor(speed)[:,None]
        action = torch.FloatTensor(action)

        return ego_tf, speed, action

    def __len__(self):
        return len(self.speeds)


class EgoDataModule(pl.LightningDataModule):
    def __init__(self, paths, val_path=None):
        super().__init__()
        self.paths = paths
        self.val_path = val_path
        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        datasets = [EgoDataset(path) for path in self.paths]
        self.train_data = torch.utils.data.ConcatDataset(datasets)
        if self.val_path:
            self.val_data = EgoDataset(self.val_path)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, shuffle=True, num_workers=4)

    # def val_dataloader(self):
    #     if self.val_path:
    #         return DataLoader(self.val_data, batch_size=32, shuffle=False, num_workers=4)


def list_to_transform(l):
    loc = carla.Location(x=l[0], y=l[1], z=l[2])
    rot = carla.Rotation(pitch=l[3], yaw=l[4], roll=l[5])
    tf = carla.Transform(location=loc, rotation=rot)
    return tf


if __name__ == '__main__':
    dataset = SpatialDataset('/zfsauton/datasets/ArgoRL/brianyan/expert_data')
    # dataset[100]