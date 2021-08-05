import glob
import json

import click
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class ReactiveDataset(Dataset):
    def __init__(self, path):
        self.trajectory_paths = glob.glob("{}/*".format(path))
        assert len(self.trajectory_paths) > 0, "No trajectories found in {}".format(
            path
        )

        self.wide_rgb_paths = []
        self.wide_seg_paths = []
        self.narrow_rgb_paths = []
        self.narrow_seg_paths = []
        self.rewards = []
        self.values = []
        self.action_values = []
        self.measurements = []
        self.world_pts = []
        self.terminals = []
        self.cmds = []

        for trajectory_path in self.trajectory_paths:
            samples = []
            print(trajectory_path)
            json_paths = sorted(
                glob.glob("{}/measurements/*.json".format(trajectory_path))
            )
            wide_rgb_paths = sorted(
                glob.glob("{}/wide_rgb/*.png".format(trajectory_path))
            )
            wide_seg_paths = sorted(
                glob.glob("{}/wide_seg/*.png".format(trajectory_path))
            )
            narrow_rgb_paths = sorted(
                glob.glob("{}/narrow_rgb/*.png".format(trajectory_path))
            )
            narrow_seg_paths = sorted(
                glob.glob("{}/narrow_seg/*.png".format(trajectory_path))
            )
            reward_paths = sorted(glob.glob("{}/reward/*.npy".format(trajectory_path)))
            value_paths = sorted(glob.glob("{}/value/*.npy".format(trajectory_path)))
            action_value_paths = sorted(
                glob.glob("{}/action_value/*.npy".format(trajectory_path))
            )
            world_paths = sorted(glob.glob("{}/world/*.npy".format(trajectory_path)))
            traj_length = min(len(json_paths), len(wide_rgb_paths), len(reward_paths))

            for i in range(traj_length):
                with open(json_paths[i], "r") as f:
                    sample = json.load(f)
                    samples.append(sample)
                terminals = np.array(samples[i]["done"])

                self.wide_rgb_paths.append(wide_rgb_paths[i])
                self.wide_seg_paths.append(wide_seg_paths[i])
                self.narrow_rgb_paths.append(narrow_rgb_paths[i])
                self.narrow_seg_paths.append(narrow_seg_paths[i])
                self.rewards.append(reward_paths[i])
                self.values.append(value_paths[i])
                self.action_values.append(action_value_paths[i])
                self.world_pts.append(world_paths[i])
                self.measurements.append(json_paths[i])

                with open(json_paths[i], "r") as f:
                    measurement = json.load(f)
                    self.cmds.append(measurement["cmd_value"])
                self.terminals.append(terminals)

        print("Number of samples: {}".format(len(self)))

    def __getitem__(self, idx):
        # TODO(jhoang): Implement augmenter
        wide_rgb = torch.Tensor(cv2.imread(self.wide_rgb_paths[idx]))
        wide_seg = torch.Tensor(
            cv2.imread(self.wide_seg_paths[idx], cv2.IMREAD_GRAYSCALE)
        )
        narrow_rgb = torch.Tensor(cv2.imread(self.narrow_rgb_paths[idx]))
        narrow_seg = torch.Tensor(
            cv2.imread(self.narrow_seg_paths[idx], cv2.IMREAD_GRAYSCALE)
        )

        rewards = torch.FloatTensor(np.load(self.rewards[idx]))
        values = torch.FloatTensor(np.load(self.values[idx]))
        action_values = torch.FloatTensor(np.load(self.action_values[idx]))
        world_pts = torch.FloatTensor(np.load(self.world_pts[idx]))
        with open(self.measurements[idx], "r") as f:
            measurement = json.load(f)

        return (
            wide_rgb,
            wide_seg,
            narrow_rgb,
            narrow_seg,
            rewards,
            values,
            action_values,
            world_pts,
            measurement,
        )

    def __len__(self):
        return len(self.rewards)

    def create_weights(self):
        cmds = np.array(self.cmds)
        self.weights = np.ones_like(cmds)
        self.weights[cmds != 4] = 10

    def sampler(self):
        self.create_weights()
        return WeightedRandomSampler(self.weights, len(self.rewards))


class ReactiveDatasetModule(pl.LightningDataModule):
    def __init__(self, path, val_path=None):
        super().__init__()
        self.path = path
        self.val_path = val_path
        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        self.train_data = ReactiveDataset(self.path)
        if self.val_path:
            self.val_data = ReactiveDataset(self.val_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=32,
            num_workers=4,
            sampler=self.train_data.sampler(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=32, num_workers=4, sampler=self.val_data.sampler()
        )


@click.command()
@click.option("--dataset-path", type=str, required=True)
def test_load_dataset(dataset_path):
    dataset = ReactiveDataset(dataset_path)
    image_path, rewards, world_pts = dataset[0]


if __name__ == "__main__":
    test_load_dataset()
