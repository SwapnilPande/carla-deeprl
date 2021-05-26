from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hydra

from .utils import to_np, soft_update_params
from agents.torch.models import make_conv_preprocessor


class DBC(pl.LightningModule):
    """ Uses DBC to learn representations for control using bisimulation metrics
    https://arxiv.org/abs/2006.10742
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.soft_target_tau = 1e-2

        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim)
        )
        self.dynamics_model = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim + 1)
        )
        self.target_dynamics_model = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.obs_dim + 1)
        )

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_dynamics_model.load_state_dict(self.dynamics_model.state_dict())

        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters())

    def training_step(self, batch, batch_idx, optimizer_idx):
        # train encoder + dynamics
        curr_z, action, reward, next_z, terminal = batch
        batch_size = curr_z.size(0)
        perm = np.random.permutation(batch_size)
        curr_z_2 = curr_z[perm]
        reward_2 = reward[perm]

        curr_z_and_action = torch.cat([curr_z, action], dim=1)
        pred_z_and_reward = self.dynamics_model(curr_z_and_action)
        pred_z, pred_reward = pred_z_and_reward[:,:-1], pred_z_and_reward[:,-1:]
        pred_z_2 = pred_z[perm]

        z_dist = F.smooth_l1_loss(curr_z, curr_z_2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward_2, reduction='none')
        t_dist = F.smooth_l1_loss(pred_z, pred_z_2, reduction='none')

        b_dist = r_dist + t_dist # bisimulation distance
        z_loss = F.mse_loss(b_dist, z_dist)
        self.log('encoder_loss', z_loss)

        pred_loss = F.mse_loss(pred_z, next_z.detach())
        reward_loss = F.mse_loss(pred_reward, reward)
        self.log('pred_loss', pred_loss)
        self.log('reward_loss', reward_loss)

        total_loss = z_loss + pred_loss + reward_loss
        self.log('total_dbc_loss', total_loss)

        self.encoder_optimizer.zero_grad()
        self.dynamics_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.dynamics_optimizer.step()

        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )

        for target_param, param in zip(self.target_dynamics_model.parameters(), self.dynamics_model.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_target_tau) + param.data * self.soft_target_tau
            )

    def configure_optimizers(self):
        return [self.dynamics_optimizer, self.encoder_optimizer]

    @property
    def automatic_optimization(self):
        return False