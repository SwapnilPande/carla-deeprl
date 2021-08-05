import torch
from torch import nn
import math
import torch.optim as optim
import numpy as np

import torch.nn.functional as F
import pytorch_lightning as pl

from projects.reactive_mbrl.algorithms.dynamic_programming import STEERINGS, THROTS
from projects.reactive_mbrl.data.reward_map import MAP_SIZE
from projects.reactive_mbrl.losses.action_loss import calculate_action_loss
from projects.reactive_mbrl.losses.segmentation_loss import calculate_segmentation_loss
from projects.reactive_mbrl.models.camera_model import CameraModel, action_logits


class CameraAgent(pl.LightningModule):
    def __init__(self, config, comet_logger=None):
        super().__init__()
        self.model = CameraModel(config.model)
        self.seg_weight = config.train["seg_weight"]
        self.comet_logger = comet_logger
        self.config = config

    # TODO(jhoang): Consider topdown image as well?
    def forward(self, wide_rgb, narr_rgb):
        return self.model(wide_rgb, narr_rgb)

    def training_step(self, batch, batch_idx):
        (
            wide_rgbs,
            wide_segs,
            narr_rgbs,
            narr_segs,
            _,
            _,
            action_values,
            _,
            measurement,
        ) = batch
        loss, comet_logs = self.compute_losses(
            preprocess_inputs(
                wide_rgbs, wide_segs, narr_rgbs, narr_segs, action_values
            ),
            measurement["cmd_value"],
            prefix="train",
        )
        self.comet_logger.log_metrics(comet_logs)

        return {"loss": loss}

    """
    def validation_step(self, batch, batch_idx):
        wide_rgbs, wide_segs, narr_rgbs, narr_segs, rewards, _ = batch
        loss, comet_logs = self.compute_losses(
            preprocess_inputs(wide_rgbs, wide_segs, narr_rgbs, narr_segs), prefix="val"
        )
        self.comet_logger.log_metrics(comet_logs)

        return {"val_loss": loss}
        """

    def compute_losses(self, inputs, cmd, prefix="train"):
        wide_rgbs, wide_segs, narr_rgbs, narr_segs, action_values = inputs
        action_logits, pred_wide_seg, pred_narr_seg = self.model(wide_rgbs, narr_rgbs)
        # 2 is the index of 0-yaw in [-1, -0.5, 0, 0.5 1] which points upward
        action_value = action_values[
            :, int(MAP_SIZE / 2), int(MAP_SIZE / 2), :, 2, :
        ].unsqueeze(1)
        action_loss = calculate_action_loss(action_value, action_logits)
        seg_loss1 = calculate_segmentation_loss(wide_segs, pred_wide_seg)
        seg_loss2 = calculate_segmentation_loss(narr_segs, pred_narr_seg)
        seg_loss = (seg_loss1 + seg_loss2) / 2

        loss = action_loss + self.seg_weight * seg_loss

        comet_logs = {
            f"{prefix}/seg_loss": seg_loss,
            f"{prefix}/action_loss": action_loss,
            f"{prefix}/loss": loss,
        }

        return loss, comet_logs

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def predict(self, wide_rgb, narr_rgb, target_speed_idx=3):
        steer_logits, throt_logits, brake_logits = self.model.predict(
            wide_rgb, narr_rgb
        )

        steer_probs = F.softmax(steer_logits)
        throt_probs = F.softmax(throt_logits)
        brake_probs = F.softmax(brake_logits)

        _, steer_indices = steer_probs[target_speed_idx].topk(3)
        steer = STEERINGS[steer_indices].mean()
        _, throt_indices = throt_probs[target_speed_idx].topk(3)
        throt = THROTS[throt_indices].mean()
        return np.array([steer, throt])


def preprocess_inputs(wide_rgbs, wide_segs, narr_rgbs, narr_segs, action_values):
    wide_rgbs = wide_rgbs.float().permute(0, 3, 1, 2)
    narr_rgbs = narr_rgbs.float().permute(0, 3, 1, 2)
    wide_segs = wide_segs.long()
    narr_segs = narr_segs.long()

    return wide_rgbs, wide_segs, narr_rgbs, narr_segs, action_values
