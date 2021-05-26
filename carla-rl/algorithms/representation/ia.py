from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import pytorch_lightning as pl
import hydra

from .utils import to_np, soft_update_params
from agents.torch.utils import COLOR
from agents.torch.models import make_conv_preprocessor


def create_resnet_basic_block(
    width_output_feature_map, height_output_feature_map, nb_channel_in, nb_channel_out):
    basic_block = nn.Sequential(
        nn.Upsample(size=(width_output_feature_map, height_output_feature_map), mode="nearest"),
        nn.Conv2d(
            nb_channel_in,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            nb_channel_out,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
    )
    return basic_block


class IA(pl.LightningModule):
    """ Trains encoder using implicit affordances approach with auxiliary tasks (e.g. semantic segmentation)
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()

        resnet = resnet18(pretrained=True)
        resnet.layer2[0].downsample[0].kernel_size = (2, 2)
        resnet.layer3[0].downsample[0].kernel_size = (2, 2)
        resnet.layer4[0].downsample[0].kernel_size = (2, 2)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2],
            nn.ReLU(),
            nn.Conv2d(512, 32, kernel_size=3, padding=1),
        )
        '''
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2,2), stride=(2,2), bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            create_resnet_basic_block(8, 8, 512, 256),
            create_resnet_basic_block(16, 16, 256, 128),
            create_resnet_basic_block(32, 32, 128, 64),
            create_resnet_basic_block(64, 64, 64, 16),
            nn.Conv2d(16, 5, kernel_size=(1,1), stride=(1,1), bias=False)
        )
        self.rgb_decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2,2), stride=(2,2), bias=False),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            create_resnet_basic_block(8, 8, 512, 256),
            create_resnet_basic_block(16, 16, 256, 128),
            create_resnet_basic_block(32, 32, 128, 64),
            create_resnet_basic_block(64, 64, 64, 16),
            nn.Conv2d(16, 3, kernel_size=(1,1), stride=(1,1), bias=False),
            nn.Sigmoid()
        )
        # self.state_decoder = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 9)
        # )
        '''
        self.action_decoder = nn.Sequential(
              nn.Flatten(),
              nn.Linear(128, 2)
        )

    def training_step(self, batch, batch_idx):
        rgb, seg, state, reward, action = batch
        # seg = F.one_hot((seg * 255).long()[:,0], 5).permute(0,3,1,2)
        batch_size = rgb.size(0)
        state = torch.cat([state.reshape(batch_size, -1), reward.reshape(batch_size, -1)], dim=1)

        '''
        encoding = self.encoder(rgb)
        pred_seg = self.seg_decoder(encoding)
        seg_loss = F.binary_cross_entropy_with_logits(pred_seg, seg.float())

        # pred_state = self.state_decoder(encoding)
        # state_loss = F.mse_loss(pred_state, state) * .1

        pred_rgb = self.rgb_decoder(encoding)
        rgb_loss = F.mse_loss(pred_rgb, rgb)

        self.log('train/seg_loss', seg_loss)
        # self.log('train/state_loss', state_loss)

        self.log('train/rgb_loss', rgb_loss)

        # return state_loss + seg_loss
        return seg_loss + rgb_loss
        '''

        encoding = self.encoder(rgb)
        pred_action = self.action_decoder(encoding)
        bc_loss = F.mse_loss(pred_action.reshape(-1,2), action.reshape(-1,2))
        self.log('train/bc_loss', bc_loss)
        return bc_loss

    def validation_step(self, batch, batch_idx):
        rgb, seg, state, reward, action = batch
        # seg = F.one_hot((seg * 255).long()[:,0], 5).permute(0,3,1,2)
        batch_size = rgb.size(0)
        state = torch.cat([state.reshape(batch_size, -1), reward.reshape(batch_size, -1)], dim=1)

        '''
        encoding = self.encoder(rgb)
        pred_seg = self.seg_decoder(encoding)
        seg_loss = F.binary_cross_entropy_with_logits(pred_seg, seg.float())

        # pred_state = self.state_decoder(encoding)
        # state_loss = F.mse_loss(pred_state, state) * .1

        pred_rgb = self.rgb_decoder(encoding)
        rgb_loss = F.mse_loss(pred_rgb, rgb)

        self.log('val/seg_loss', seg_loss)
        # self.log('val/state_loss', state_loss)

        self.log('val/rgb_loss', rgb_loss)

        seg_viz = COLOR[pred_seg.argmax(-3).cpu()][:4]
        rgb_viz = (pred_rgb.cpu()[:4].permute(0,2,3,1).numpy() * 255).astype(int)
        gt_viz = COLOR[seg.argmax(-3).cpu()][:4]
        # rgb_viz = (rgb * 255).cpu()[:4].permute(0,2,3,1).numpy()

        seg_viz = np.concatenate(seg_viz, axis=1)
        rgb_viz = np.concatenate(rgb_viz, axis=1)
        gt_viz = np.concatenate(gt_viz, axis=1)

        viz = np.concatenate([seg_viz, rgb_viz, gt_viz], axis=0)
        viz = torch.tensor(viz).permute(2,0,1)
        self.logger.experiment.add_image('val/predictions', viz, self.current_epoch)
        '''
        encoding = self.encoder(rgb)
        pred_action = self.action_decoder(encoding)
        bc_loss = F.mse_loss(pred_action.reshape(-1,2), action.reshape(-1,2))
        self.log('val/bc_loss', bc_loss)

    def configure_optimizers(self):
        return [optim.Adam(self.encoder.parameters())]
        # return [self.dynamics_optimizer, self.encoder_optimizer]

    # @property
    # def automatic_optimization(self):
    #     return False
