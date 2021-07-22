"""
VQ-VAE implementation
Reference: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from six.moves import xrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
import hydra

from models.vqvae import VQVAE
from data_modules import TransformerDataModule, VQVAEDataModule


@hydra.main(config_path='', config_name='vae.yaml')
def main(cfg):
    model = VQVAE()

    dm = VQVAEDataModule(['/zfsauton/datasets/ArgoRL/brianyan/town01_expert_speed=0.5/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town01_expert_speed=0.75/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town01_expert_speed=1.0/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town01_noisy_speed=0.5/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town01_noisy_speed=0.75/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town01_noisy_speed=1.0/',
                        # '/zfsauton/datasets/ArgoRL/brianyan/town01_random/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town04_expert_speed=0.5/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town04_expert_speed=0.75/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town04_expert_speed=1.0/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town04_noisy_speed=0.5/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town04_noisy_speed=0.75/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town04_noisy_speed=1.0/',
                        # '/zfsauton/datasets/ArgoRL/brianyan/town04_random/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town06_expert_speed=0.5/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town06_expert_speed=0.75/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town06_expert_speed=1.0/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town06_noisy_speed=0.5/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town06_noisy_speed=0.75/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town06_noisy_speed=1.0/'],
                        # '/zfsauton/datasets/ArgoRL/brianyan/town06_random/'],
                       ['/zfsauton/datasets/ArgoRL/brianyan/town03_expert_speed=0.5/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town03_expert_speed=0.75/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town03_expert_speed=1.0/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town03_noisy_speed=0.5/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town03_noisy_speed=0.75/',
                        '/zfsauton/datasets/ArgoRL/brianyan/town03_noisy_speed=1.0/',
                        # '/zfsauton/datasets/ArgoRL/brianyan/town03_random/'])
                       ])
    dm.setup(None)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=1, save_top_k=-1)
    callbacks.append(checkpoint_callback)
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=[1],
        max_epochs=100
    )
    trainer.fit(model, dm)

    print('Done')


if __name__ == '__main__':
    main()