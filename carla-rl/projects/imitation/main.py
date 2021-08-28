import numpy as np 
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from models import RecurrentAttentionAgent
from data_modules import OfflineCarlaDataModule


""" Iterative attention and the glimpse VAE 

Exploring the inner workings of iterative attention approaches
How does our representation of the image evolve over time with additional glimpses?

1) Do additional glimpses help in recurrent architectures? 

2) Can we quantify the improvements? Is information gain correlated at all with downstream performance?

3) If we tighten the attentional bottleneck, how does that change the need for iteration?
Vertical versus horizontal depth in attention processing: making larger queries with minimal feedback versus making tiny queries with lots of feedback

"""


@hydra.main(config_path='conf', config_name='train.yaml')
def main(cfg):
    # For reproducibility
    # seed_everything(cfg.seed)

    # Loading agent and environment
    agent = RecurrentAttentionAgent(**cfg.agent)

    # Setting up logger and checkpoint/eval callbacks
    logger = TensorBoardLogger(save_dir=os.getcwd(), name='', version='')
    callbacks = []

    checkpoint_callback = ModelCheckpoint(period=cfg.checkpoint_freq, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    cfg.trainer.gpus = str(cfg.trainer.gpus) # str denotes gpu id, not quantity

    offline_data_module = OfflineCarlaDataModule(cfg.data_module)
    offline_data_module.setup(None)

    try:
        trainer = pl.Trainer(**cfg.trainer, 
            logger=logger,
            callbacks=callbacks,
            max_epochs=cfg.num_epochs)
        trainer.fit(agent, offline_data_module)
    finally:
        env.close()

    print('Done')