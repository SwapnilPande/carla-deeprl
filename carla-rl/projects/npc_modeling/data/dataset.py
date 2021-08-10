from glob import glob
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pytorch_lightning as pl


class NPCDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        actor_paths = glob(f"{path}/*/[0-9]*")
        self.transformations = []
        for actor_path in actor_paths:
            transform_path = os.path.join(actor_path, "transforms.npy")
            self.transformations.append(np.load(transform_path))

    def __getitem__(self, index):
        return self.transformations[index]

    def __len__(self):
        return len(self.actor_paths)


class NPCDatasetModule(pl.LightningDataModule):
    def __init__(self, path, val_path=None):
        super().__init__()
        self.path = path
        self.val_path = val_path
        self.dataset = None
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        self.train_data = NPCDataset(self.path)
        if self.val_path:
            self.val_data = NPCDataset(self.val_path)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, num_workers=4, shuffle=True,)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32, num_workers=4, shuffle=True,)

