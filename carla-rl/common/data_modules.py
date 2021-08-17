""" PyTorch datasets for offline RL experiments """

import glob
from pathlib import Path
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl

from .utils import get_reward, preprocess_rgb, preprocess_topdown, \
    get_angle_to_next_node, get_obs, get_action, get_dir
from .replay_buffer import Experience, ReplayBuffer, PERBuffer
from environment import CarlaEnv

"""
Offline dataset handling
"""

class OfflineCarlaDataset(Dataset):
    """ Offline dataset """

    def __init__(self, path, use_images=True, frame_stack=1):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.path = path
        self.use_images = use_images
        self.frame_stack = frame_stack

        self.obs, self.next_obs = [], []
        self.actions, self.rewards, self.terminals = [], [], []

        for trajectory_path in trajectory_paths:
            samples = []
            json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
            image_paths = sorted(glob.glob('{}/topdown/*.png'.format(trajectory_path)))
            traj_length = min(len(json_paths), len(image_paths))

            for i in range(traj_length):
                with open(json_paths[i]) as f:
                    sample = json.load(f)
                sample['image_path'] = image_paths[i]
                samples.append(sample)

            if traj_length <= (self.frame_stack + 1):
                continue

            image_buffer = [samples[0]['image_path'] for _ in range(self.frame_stack)]
            next_image_buffer = [samples[0]['image_path'] for _ in range(self.frame_stack)]

            for i in range(1, traj_length):
                image_path = samples[i-1]['image_path']
                next_image_path = samples[i]['image_path'] # TODO: images don't work for this

                image_buffer.pop(0)
                image_buffer.append(image_path)
                next_image_buffer.pop(0)
                next_image_buffer.append(next_image_path)

                obs = image_buffer[:], samples[i]['obs']
                next_obs = next_image_buffer[:], samples[i]['next_obs']
                reward = samples[i]['reward']
                terminal = samples[i]['done']
                action = samples[i]['action']
                if action[1] <= 0:
                    action[1] = -.8

                if np.isnan(obs[1]).any():
                    print(trajectory_path)
                    break

                self.obs.append(obs)
                self.next_obs.append(next_obs)
                self.actions.append(action)
                self.rewards.append(reward)
                self.terminals.append(terminal)

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        image_paths, mlp_features = self.obs[idx]
        next_image_paths, next_mlp_features = self.next_obs[idx]

        mlp_features = torch.FloatTensor(mlp_features).clamp(-4, 4)
        next_mlp_features = torch.FloatTensor(next_mlp_features).clamp(-4, 4)

        action = torch.FloatTensor(self.actions[idx])
        reward = torch.FloatTensor([self.rewards[idx]])
        terminal = torch.Tensor([self.terminals[idx]])

        if self.use_images:
            image = torch.cat([preprocess_rgb(cv2.imread(path)) for path in image_paths], dim=0)
            next_image = torch.cat([preprocess_rgb(cv2.imread(path)) for path in next_image_paths], dim=0)
            return (image, mlp_features), action, reward, (next_image, next_mlp_features), terminal
        else:
            return mlp_features.flatten(), action, reward, next_mlp_features.flatten(), terminal

    def __len__(self):
        return len(self.rewards)

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
        self.frame_stack = cfg.frame_stack
        self.num_workers = cfg.num_workers
        self.train_val_split = cfg.train_val_split

        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        datasets = [OfflineCarlaDataset(use_images=self.use_images, path=path, frame_stack=self.frame_stack) for path in self.paths]
        self.dataset = torch.utils.data.ConcatDataset(datasets)
        # self.dataset = OfflineCarlaDataset(use_images=self.use_images, path=self.path, frame_stack=self.frame_stack)
        train_size = int(len(self.dataset) * self.train_val_split)
        val_size = len(self.dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, (train_size, val_size))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


"""
Online dataset handling
"""

class OnlineCarlaDataset(IterableDataset):
    """ Online dataset with environment interaction """

    def __init__(self, agent, env, cfg):
        self.agent = agent
        self.env = env
        self.replay_buffer = PERBuffer(int(cfg.buffer_size))
        self.cfg = cfg

    def populate(self, size, data_module=None):
        """ Populates replay buffer.

        If datamodule provided, sample from datamodule. Otherwise, collect interactions.
        """
        if data_module is None:
            num_steps = 0
            obs = self.env.reset()
            while num_steps < size:
                action = np.random.uniform(-1., 1., (2,))
                next_obs, reward, done, info = self.env.step(action)
                # reward = np.array([
                #     info['reward_dict']['dist_to_trajectory'],
                #     info['reward_dict']['speed'] + info['reward_dict']['acceleration'],
                #     info['reward_dict']['collision'],
                #     info['reward_dict']['light']
                # ])

                experience = Experience(obs, action, reward, next_obs, done)
                self.replay_buffer.append(experience)

                obs = next_obs if not done else self.env.reset()
                num_steps += 1
        else:
            dataloader = DataLoader(data_module.dataset, batch_size=1, shuffle=True)
            num_steps = 0
            while True:
                for batch in dataloader:
                    obs = batch[0].numpy().flatten()
                    action = batch[1].numpy().flatten()
                    reward = batch[2].numpy().flatten()
                    next_obs = batch[3].numpy().flatten()
                    done = batch[4].item()

                    experience = Experience(obs, action, reward, next_obs, done)
                    self.replay_buffer.append(experience)

                    num_steps += 1
                    if num_steps >= size:
                        return

    def generate_batch(self):
        num_steps = 0
        obs = self.env.reset()
        while True:
            # Interact with environment using agent
            action = self.agent.predict(obs)[0]
            next_obs, reward, done, _ = self.env.step(action)

            # Store new experience
            experience = Experience(obs, action, reward, next_obs, done)
            self.replay_buffer.append(experience)

            obs = next_obs if not done else self.env.reset()

            if num_steps % self.cfg.train_every_n_steps == 0:
                # Sample from replay buffer for training step
                batch, indices, weights = self.replay_buffer.sample(self.cfg.batch_size)
                # batch = self.replay_buffer.sample(self.cfg.batch_size)
                for idx, _ in enumerate(batch[0]):
                    yield (batch[0][idx], batch[1][idx], batch[2][idx], batch[3][idx], batch[4][idx]), indices[idx], weights[idx]

            num_steps += 1
            if num_steps >= self.cfg.epoch_size:
                break

    def __iter__(self):
        return self.generate_batch()


class OnlineCarlaDataModule(pl.LightningDataModule):
    """ Datamodule for online driving data """

    def __init__(self, agent, env, cfg):
        super().__init__()
        self.agent = agent
        self.env = env
        self.cfg = cfg
        self.dataset = OnlineCarlaDataset(self.agent, self.env, self.cfg)

    def setup(self, stage):
        pass

    def populate(self, size, data_module=None):
        self.dataset.populate(size=size, data_module=data_module)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size)


class RepLearningDataset(Dataset):
    def __init__(self, path, use_images=True, frame_stack=1):
        trajectory_paths = glob.glob('{}/*'.format(path))
        assert len(trajectory_paths) > 0, 'No trajectories found in {}'.format(path)

        self.path = path
        self.use_images = use_images
        self.frame_stack = frame_stack

        self.obs, self.rewards = [], []
        self.actions = []

        for trajectory_path in trajectory_paths:
            samples = []
            json_paths = sorted(glob.glob('{}/measurements/*.json'.format(trajectory_path)))
            rgb_paths = sorted(glob.glob('{}/rgb/*.png'.format(trajectory_path)))
            seg_paths = sorted(glob.glob('{}/rgb/*.png'.format(trajectory_path)))
            traj_length = min(len(json_paths), len(rgb_paths), len(seg_paths))

            for i in range(traj_length):
                with open(json_paths[i]) as f:
                    sample = json.load(f)
                sample['rgb_path'] = rgb_paths[i]
                sample['seg_path'] = seg_paths[i]
                samples.append(sample)

            if traj_length <= (self.frame_stack + 1):
                continue

            rgb_buffer = [samples[0]['rgb_path'] for _ in range(self.frame_stack)]
            seg_buffer = [samples[0]['seg_path'] for _ in range(self.frame_stack)]

            for i in range(1, traj_length):
                rgb_path = samples[i-1]['rgb_path']
                seg_path = samples[i-1]['seg_path']

                rgb_buffer.pop(0)
                rgb_buffer.append(rgb_path)
                seg_buffer.pop(0)
                seg_buffer.append(seg_path)

                obs = rgb_buffer[:], seg_buffer[:], samples[i]['obs']
                reward = samples[i]['reward']
                action = samples[i]['action']

                self.obs.append(obs)
                self.rewards.append(reward)
                self.actions.append(action)

        print('Number of samples: {}'.format(len(self)))

    def __getitem__(self, idx):
        rgb_paths, seg_paths, mlp_features = self.obs[idx]

        mlp_features = torch.FloatTensor(mlp_features).clamp(-4, 4)
        reward = torch.FloatTensor([self.rewards[idx]])
        action = torch.FloatTensor([self.actions[idx]])

        rgb = torch.cat([preprocess_rgb(cv2.imread(path)) for path in rgb_paths], dim=0)
        seg = torch.cat([preprocess_rgb(cv2.imread(path)) for path in seg_paths], dim=0)
        return rgb, seg, mlp_features, reward, action

    def __len__(self):
        return len(self.rewards)

    def sample(self):
        idx = np.random.randint(len(self))
        return self[idx]


class RepLearningDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.paths = cfg.dataset_paths
        self.batch_size = cfg.batch_size
        self.frame_stack = cfg.frame_stack
        self.num_workers = cfg.num_workers
        self.train_val_split = cfg.train_val_split

        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        datasets = [RepLearningDataset(path=path, frame_stack=self.frame_stack) for path in self.paths]
        self.dataset = torch.utils.data.ConcatDataset(datasets)
        train_size = int(len(self.dataset) * self.train_val_split)
        val_size = len(self.dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, (train_size, val_size))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
