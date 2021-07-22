import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models.segmentation import fcn_resnet50
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from omegaconf import DictConfig, OmegaConf
import hydra

from spatial_data import SpatialDataModule


class StackedValueNetwork(pl.LightningModule):
    def __init__(self, T=5):
        super().__init__()
        self.T = T
        resnet = fcn_resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet.backbone.children())[:-1],
            nn.Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(256 * T,512,3,2,1,1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 20, kernel_size=(1,1), stride=(1,1), bias=False)
        )

    def forward(self, images):
        batch_size = len(images)
        images = images.reshape(batch_size * self.T, 3, 64, 64)
        features = self.backbone(images).reshape(batch_size, self.T * 256, 8, 8)
        out = self.upconv(features)
        return out

    def training_step(self, batch, batch_idx):
        images, rewards, values = batch
        batch_size = len(images)
        pred_values = self.forward(images)
        values = values[:,-1].reshape(batch_size,16,16,20).permute(0,3,1,2)
        loss = F.mse_loss(values, pred_values)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, rewards, values = batch
        batch_size = len(images)
        pred_values = self.forward(images)
        values = values[:,-1].reshape(batch_size,16,16,20).permute(0,3,1,2)
        loss = F.mse_loss(values, pred_values)
        self.log('val/loss', loss)

        pred_values = (-pred_values[0].clamp(-1,0) * 255).type(torch.uint8)
        pred_values = pred_values[:,None].repeat(1,3,1,1)
        pred_values = torchvision.utils.make_grid(pred_values)

        values = (-values[0].clamp(-1,0) * 255).type(torch.uint8)
        values = values[:,None].repeat(1,3,1,1)
        values = torchvision.utils.make_grid(values)

        self.logger.experiment.add_image('val/pred', pred_values)
        self.logger.experiment.add_image('val/gt', values)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class RecurrentValueNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = fcn_resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet.backbone.children())[:-1],
            nn.Conv2d(1024, 128, kernel_size=(1,1), stride=(1,1), bias=False)
        )
        self.rnn = nn.GRU(input_size=128*8*8, hidden_size=512, batch_first=True)
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(128,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 20, kernel_size=(1,1), stride=(1,1), bias=False)
        )

    def forward(self, images, hidden=None):
        batch_size, T = images.shape[0], images.shape[1]
        images = images.reshape(batch_size * T, 3, 64, 64)
        features = self.backbone(images).reshape(batch_size, T, 128*8*8)
        out, hidden = self.rnn(features, hidden)
        out = out.reshape(batch_size * T, 128, 2, 2)
        preds = self.upconv(out).reshape(batch_size,T,20,16,16)
        return preds, hidden

    def training_step(self, batch, batch_idx):
        images, rewards, values = batch
        batch_size, T = images.shape[0], images.shape[1]
        pred_values, hidden = self.forward(images)
        values = values.reshape(batch_size,T,16,16,20).permute(0,1,4,2,3)
        loss = F.mse_loss(values, pred_values)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, rewards, values = batch
        batch_size, T = images.shape[0], images.shape[1]
        pred_values, hidden = self.forward(images)
        values = values.reshape(batch_size,T,16,16,20).permute(0,1,4,2,3)
        loss = F.mse_loss(values, pred_values)
        self.log('val/loss', loss)

        pred_values = (-pred_values[0].clamp(-1,0) * 255).type(torch.uint8)
        pred_values = pred_values[:,:,None].repeat(1,1,3,1,1).reshape(-1,3,16,16)
        pred_values = torchvision.utils.make_grid(pred_values)

        values = (-values[0].clamp(-1,0) * 255).type(torch.uint8)
        values = values[:,:,None].repeat(1,1,3,1,1).reshape(-1,3,16,16)
        values = torchvision.utils.make_grid(values)

        self.logger.experiment.add_image('val/pred', pred_values)
        self.logger.experiment.add_image('val/gt', values)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class TransformerValueNetwork(pl.LightningModule):
    def __init__(self, T=5):
        super().__init__()
        self.T = T
        resnet = fcn_resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            *list(resnet.backbone.children())[:-1],
            nn.Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        )
        self.rnn = nn.GRU(input_size=256*8*8, hidden_size=2048, batch_first=True)
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(512,256,3,2,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 20, kernel_size=(1,1), stride=(1,1), bias=False)
        )

    def forward(self, images):
        batch_size = len(images)
        images = images.reshape(batch_size * self.T, 3, 64, 64)
        features = self.backbone(images).reshape(batch_size, self.T, 256*8*8)
        out, hn = self.rnn(features)
        out = out.reshape(batch_size * self.T, 512, 2, 2)
        preds = self.upconv(out).reshape(batch_size,self.T,20,16,16)
        return preds

    def training_step(self, batch, batch_idx):
        images, rewards, values = batch
        batch_size = len(images)
        pred_values = self.forward(images)
        values = values.reshape(batch_size,self.T,16,16,20).permute(0,1,4,2,3)
        loss = F.mse_loss(values, pred_values)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, rewards, values = batch
        batch_size = len(images)
        pred_values = self.forward(images)
        values = values.reshape(batch_size,self.T,16,16,20).permute(0,1,4,2,3)
        loss = F.mse_loss(values, pred_values)
        self.log('val/loss', loss)

        pred_values = (-pred_values[0].clamp(-1,0) * 255).type(torch.uint8)
        pred_values = pred_values[:,:,None].repeat(1,1,3,1,1).reshape(-1,3,16,16)
        pred_values = torchvision.utils.make_grid(pred_values)

        values = (-values[0].clamp(-1,0) * 255).type(torch.uint8)
        values = values[:,:,None].repeat(1,1,3,1,1).reshape(-1,3,16,16)
        values = torchvision.utils.make_grid(values)

        self.logger.experiment.add_image('val/pred', pred_values)
        self.logger.experiment.add_image('val/gt', values)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


@hydra.main(config_path='conf', config_name='train.yaml')
def main(cfg):
    if cfg.value_model == 'stacked':
        model = StackedValueNetwork()
    elif cfg.value_model == 'recurrent':
        model = RecurrentValueNetwork()
    elif cfg.value_model == 'transformer':
        raise NotImplementedError
    else:
        raise NotImplementedError

    dm = SpatialDataModule(['/zfsauton/datasets/ArgoRL/brianyan/expert_data/'])
    dm.setup(None)

    # Setting up logger and checkpoint/eval callbacks
    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    cfg.trainer.gpus = str(cfg.trainer.gpus) # str denotes gpu id, not quantity

    # if cfg.train_offline:
    trainer = pl.Trainer(**cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
        max_epochs=40)
    trainer.fit(model, dm)

    print('Done')

if __name__ == '__main__':
    main()
