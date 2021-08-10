import torch
from torch import nn
import math
import torch.optim as optim
import numpy as np

import torch.nn.functional as F
import pytorch_lightning as pl


class SimpleModel(pl.LightningModule):
    def __init__(self, config, comet_logger=None):
        super().__init__()
        self.model = CameraModel(config.model)
        self.comet_logger = comet_logger
        self.config = config

    # TODO(jhoang): Consider topdown image as well?
    def forward(self, wide_rgb, narr_rgb):
        return self.model(wide_rgb, narr_rgb)

    def training_step(self, batch, batch_idx):
        transforms = batch
        loss, comet_logs = self.compute_losses(
            transforms,
            prefix="train",
        )
        self.comet_logger.log_metrics(comet_logs)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        transforms = batch
        loss, comet_logs = self.compute_losses(
            transforms,
            prefix="val",
        )
        self.comet_logger.log_metrics(comet_logs)

        return {"val_loss": loss}

    def compute_losses(self, transforms, prefix="train"):
        predicted = self.model(transforms)
        waypoint_loss = calculate_waypoint_loss(predicted, transforms)
        yaw_loss = calculate_yaw_loss(predicted, transforms)

        loss = waypoint_loss + yaw_loss

        comet_logs = {
            f"{prefix}/waypoint_loss": waypoint_loss,
            f"{prefix}/yaw_loss": yaw_loss,
            f"{prefix}/loss": loss,
        }

        return loss, comet_logs

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

