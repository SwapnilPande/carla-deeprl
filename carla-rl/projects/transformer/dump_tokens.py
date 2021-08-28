import argparse
import os
import time
import glob

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

# from leaderboard.utils.statistics_manager import StatisticsManager

from omegaconf import DictConfig, OmegaConf
import hydra

# from stable_baselines.common.vec_env import DummyVecEnv
# from agents.tf.ppo import PPO

from data_modules import TransformerDataModule
# from algorithms.bc import BC, ImageBC
# from algorithms.sac import SAC, ImageSAC
from models.decision_transformer import DecisionTransformer
from models.trajectory_transformer import TrajectoryTransformer
from models.ebm import EBMTransformer
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *
from utils import preprocess_rgb, preprocess_topdown
from models.action_binner import *


@hydra.main(config_path='../affordance_maps/conf', config_name='train.yaml')
def main(cfg):
    PATH = '/zfsauton/datasets/ArgoRL/brianyan/carla_dataset'
    agent = TrajectoryTransformer()

    paths = glob.glob('/zfsauton/datasets/ArgoRL/brianyan/carla_dataset/town01/*/')
    offline_data_module = TransformerDataModule(paths)
    offline_data_module.setup(None)

    state_token_counter = torch.zeros(256)
    action_token_counter = torch.zeros(256)

    state_tokens_all = []
    action_tokens_all = []

    from tqdm import tqdm
    for (states, actions) in tqdm(offline_data_module.train_dataloader()):
        state_tokens = agent.state_tokenizer(states)
        action_tokens = action_tokenizer(actions, expand_dim=True, pad_to_size=256).reshape(-1,256)

        state_tokens_all.append(state_tokens.argmax(dim=-1).detach().cpu())
        action_tokens_all.append(action_tokens.argmax(dim=-1).detach().cpu())

        state_token_counts = state_tokens.sum(dim=0).detach().cpu()
        action_token_counts = action_tokens.sum(dim=0).detach().cpu()

        state_token_counter += state_token_counts
        action_token_counter += action_token_counts

    state_token_counter = state_token_counter.numpy()
    action_token_counter = action_token_counter.numpy()

    np.save('{}/state_token_count'.format(PATH), state_token_counter)
    np.save('{}/action_token_count'.format(PATH), action_token_counter)

    state_tokens = np.concatenate([st.reshape(-1,26) for st in state_tokens_all], axis=0)
    action_tokens = np.concatenate([act.reshape(-1,26) for act in action_tokens_all], axis=0)

    np.save('{}/stacked_state_tokens'.format(PATH), state_tokens)
    np.save('{}/stacked_action_tokens'.format(PATH), action_tokens)

    # action_tokens = np.load('{}/stacked_action_tokens.npy'.format(PATH))

    # state_token_counter = np.load('{}/state_token_count.npy'.format(PATH))
    # action_token_counter = np.load('{}/action_token_count.npy'.format(PATH))

    # state_tokens = np.load('{}/stacked_state_tokens.npy'.format(PATH))
    # action_tokens = np.load('{}/stacked_action_tokens.npy'.format(PATH))

    # compute weights
    state_weights = state_token_counter.sum() / state_token_counter
    state_weights[np.isinf(state_weights)] = 0
    action_weights = action_token_counter.sum() / action_token_counter
    action_weights[np.isinf(action_weights)] = 0

    state_weights_all = state_weights[state_tokens].sum(axis=-1)
    action_weights_all = action_weights[action_tokens].sum(axis=-1)

    np.save('{}/state_weights'.format(PATH), state_weights_all)
    np.save('{}/action_weights'.format(PATH), action_weights_all)

    # only keep trajectories where actions change
    # action_weights_all = ((action_tokens != action_tokens[:,:1]).sum(axis=-1) > 5).astype(int)

    # np.save('{}/state_weights'.format(PATH), action_weights_all)

    print('Done')


if __name__ == '__main__':
    main()
