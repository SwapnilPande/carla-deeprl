import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from omegaconf import DictConfig, OmegaConf
import hydra

from spatial_data import SpatialDataset, SpatialDataModule


class RewardMapper(pl.LightningModule):
    # mostly copied from LBC architecture

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])
        # self.speed_encoder = nn.Sequential(
        #     nn.Linear(1, 128),
        #     nn.ReLU(),
        #     nn.Linear(128,128),
        # )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(512,256,3,2,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,3,2,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,16,3,2,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,3,2,1,1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8,2,1,1,0),
            nn.Softmax(dim=1)
        )
        self.model = nn.Sequential(
            self.backbone,
            self.upconv
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.upconv(x)
        return x

    def training_step(self, batch, batch_idx):
        image, ego_pts, rewards, terminals = batch
        batch_size = rewards.shape[0]
        reward_map = self.forward(image)
        pred = reward_map[torch.arange(batch_size)[:,None].repeat(1,5), :, ego_pts[torch.arange(batch_size),:,0], ego_pts[torch.arange(batch_size),:,1]]

        pred = pred.reshape(-1,2)
        terminals = terminals.flatten()

        loss = F.cross_entropy(pred, terminals)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, ego_pts, rewards, terminals = batch
        batch_size = rewards.shape[0]
        reward_map = self.forward(image)
        pred = reward_map[torch.arange(batch_size)[:,None].repeat(1,ego_pts.size(1)), :, ego_pts[torch.arange(batch_size),:,0], ego_pts[torch.arange(batch_size),:,1]]

        pred = pred.reshape(-1,2)
        terminals = terminals.flatten()
        loss = F.cross_entropy(pred, terminals)
        self.log('val/loss', loss)

        pred_labels = pred.argmax(dim=1)
        accuracy = (pred_labels == terminals).sum() / len(terminals)
        self.log('val/acc', accuracy)

        reward_map_viz = (reward_map.argmax(dim=1)[:16].unsqueeze(1)).type(torch.uint8) * 255
        reward_map_viz = torchvision.utils.make_grid(reward_map_viz)
        self.logger.experiment.add_image('binarized_predictions', reward_map_viz)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-3)


@hydra.main(config_path='conf', config_name='train.yaml')
def main(cfg):
    model = RewardMapper()
    dataset_paths = [
        '/media/brian/linux-data/reward_maps'
    ]
    dm = SpatialDataModule(dataset_paths)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []
    checkpoint_callback = ModelCheckpoint(period=1, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, dm)


if __name__ == '__main__':
    main()
