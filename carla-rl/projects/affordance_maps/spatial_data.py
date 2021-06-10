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


CALIBRATION = np.array([[32, 0, 32],
                        [0, 32, 32],
                        [0,  0,  1]])



class SpatialDataset(Dataset):

    def __init__(self, path, use_images=True, T=5):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.path = path
        self.use_images = use_images
        self.T = T

        self.image_paths = []
        self.rewards = []
        self.terminals = []
        self.ego_transforms = []
        self.camera_transforms = []

        for trajectory_path in trajectory_paths:
            samples = []
            json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
            image_paths = sorted(glob.glob('{}/topdown/*.png'.format(trajectory_path)))
            reward_paths = sorted(glob.glob('{}/reward/*.png'.format(trajectory_path)))
            traj_length = min(len(json_paths), len(image_paths), len(reward_paths))

            for i in range(traj_length):
                with open(json_paths[i], 'r') as f:
                    sample = json.load(f)
                samples.append(sample)

            if traj_length < T: # (self.frame_stack + 1):
                continue

            for i in range(traj_length-T+1):
                # rewards = [samples[i+t]['reward'] for t in range(T)]
                # reward_positions = np.array(samples[i]['reward_positions'])
                # reward_labels = np.array(samples[i]['reward_labels'])
                terminals = np.array(samples[i]['done'])
                # ego_transforms = np.array(samples[i]['actor_tf'])
                camera_transform = np.array(samples[i]['camera_tf'])

                self.image_paths.append(image_paths[i])
                # self.rewards.append((reward_positions, reward_labels))
                self.rewards.append(reward_paths[i])
                self.terminals.append(terminals)
                # self.ego_transforms.append(ego_transforms)
                self.camera_transforms.append(camera_transform)

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        reward_path = self.rewards[idx]

        terminals = self.terminals[idx]
        # ego_transforms = self.ego_transforms[idx]
        camera_transform = self.camera_transforms[idx]

        image = preprocess_rgb(cv2.imread(image_path))
        rewards = (preprocess_rgb(cv2.imread(reward_path)) * 255)[:1]
        terminals = torch.LongTensor(terminals)

        # rewards = np.zeros(reward_labels.shape)
        # rewards[reward_labels == 'empty'] = 0
        # rewards[reward_labels == 'offroad'] = -1
        # rewards[reward_labels == 'badlane'] = -1
        # rewards[reward_labels == 'vehicle'] = -1
        
        # binarize rewards
        # rewards = torch.FloatTensor(rewards)

        # camera_transform = list_to_transform(camera_transform)
        # reward_transforms = [carla.Transform(location=carla.Location(x=pos[0], y=pos[1])) for pos in reward_positions]
        # reward_camera_pts = ClientSideBoundingBoxes.get_bounding_boxes(reward_transforms, camera_transform, CALIBRATION)
        # reward_camera_pts = torch.LongTensor(reward_camera_pts)

        # project transforms into camera
        # ego_transforms = [list_to_transform(tf) for tf in ego_transforms]
        # camera_transform = list_to_transform(camera_transform)
        # ego_pts = ClientSideBoundingBoxes.get_bounding_boxes(ego_transforms, camera_transform, CALIBRATION)
        # ego_pts = torch.LongTensor(ego_pts)

        # import matplotlib.pyplot as plt
        # ego_pts = np.array(reward_camera_pts)
        # plt.imshow(image)
        # plt.scatter(reward_camera_pts[:,0], reward_camera_pts[:,1])
        # plt.show()

        # return image, reward_camera_pts, rewards, terminals
        return image, rewards

    def __len__(self):
        return len(self.rewards)


class SpatialDataModule(pl.LightningDataModule):
    def __init__(self, paths, val_path):
        super().__init__()
        self.paths = paths
        self.val_path = val_path
        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        datasets = [SpatialDataset(path) for path in self.paths]
        self.train_data = torch.utils.data.ConcatDataset(datasets)
        self.val_data = SpatialDataset(self.val_path)
        # self.dataset = OfflineCarlaDataset(use_images=self.use_images, path=self.path, frame_stack=self.frame_stack)
        # train_size = int(len(self.dataset) * .9)
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

            if traj_length < 25:
                continue

            for i in range(25, (traj_length)//5 -T+1):
                ego_transforms = [samples[(i+t)*5]['actor_tf'] for t in range(T)]
                speed = [samples[(i+t)*5]['speed'] for t in range(T)]
                action = [samples[(i+t)*5]['action'] for t in range(T)]

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
    def __init__(self, paths, val_path):
        super().__init__()
        self.paths = paths
        self.val_path = val_path
        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        datasets = [EgoDataset(path) for path in self.paths]
        self.train_data = torch.utils.data.ConcatDataset(datasets)
        self.val_data = EgoDataset(self.val_path)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32, shuffle=False, num_workers=4)


def list_to_transform(l):
    loc = carla.Location(x=l[0], y=l[1], z=l[2])
    rot = carla.Rotation(pitch=l[3], yaw=l[4], roll=l[5])
    tf = carla.Transform(location=loc, rotation=rot)
    return tf


if __name__ == '__main__':
    dataset = SpatialDataset('/media/brian/linux-data/reward_maps')
    dataset[100]