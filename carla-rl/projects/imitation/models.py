import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18
import pytorch_lightning as pl
import numpy as np

from common.utils import preprocess_rgb
from perceiver_pytorch import Perceiver


class ConvAgent(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])
        self.measurement_mlp = nn.Linear(8, 2048)
        self.action_mlp = nn.Sequential(
            nn.Linear(2048,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward(self, image, mlp_features):
        image = self.conv(image)
        image = image.view(image.shape[0], -1)
        mlp_features = mlp_features.reshape(-1,8)
        mlp_features = self.measurement_mlp(mlp_features)
        features = image + mlp_features
        action = self.action_mlp(features)
        return action

    def predict(self, image, mlp_features):
        image = preprocess_rgb(image).cuda()[None]
        mlp_features = torch.FloatTensor(mlp_features).cuda().reshape(1,-1)
        out = self.forward(image, mlp_features)
        action = out.detach().cpu().numpy()
        return np.clip(action, -1, 1)

    def training_step(self, batch, batch_idx):
        (image, mlp_features), action, reward, (next_image, next_mlp_features), terminal = batch
        pred_action = self.forward(image, mlp_features)
        loss = F.mse_loss(pred_action, action)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (image, mlp_features), action, reward, (next_image, next_mlp_features), terminal = batch
        pred_action = self.forward(image, mlp_features)
        loss = F.mse_loss(pred_action, action)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


class PerceiverAgent(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.perceiver = Perceiver(
            input_channels = 3,          # number of channels for each token of the input
            input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
            depth = 6,                   # depth of net
            num_latents = 128,           # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = 256,            # latent dimension
            cross_heads = 1,             # number of heads for cross attention. paper said 1
            latent_heads = 8,            # number of heads for latent self attention, 8
            cross_dim_head = 64,
            latent_dim_head = 64,
            num_classes = 512,          # output number of classes
            attn_dropout = 0.,
            ff_dropout = 0.,
            weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
            fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
            self_per_cross_attn = 2      # number of self attention blocks per cross attention
        )
        self.measurement_mlp = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128,512)
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
        
    def forward(self, image, mlp_features):
        image = image.permute(0,2,3,1)
        image = self.perceiver(image)
        image = image.view(image.shape[0], -1)
        mlp_features = mlp_features.reshape(-1,8)
        mlp_features = self.measurement_mlp(mlp_features)
        features = torch.cat([image, mlp_features], dim=1)
        action = self.action_mlp(features)
        return action

    def predict(self, image, mlp_features):
        image = preprocess_rgb(image).cuda()[None]
        mlp_features = torch.FloatTensor(mlp_features).cuda().reshape(1,-1)
        out = self.forward(image, mlp_features)
        action = out.detach().cpu().numpy()
        return np.clip(action, -1, 1)

    def training_step(self, batch, batch_idx):
        (image, mlp_features), action, reward, (next_image, next_mlp_features), terminal = batch
        pred_action = self.forward(image, mlp_features)
        loss = F.mse_loss(pred_action, action)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (image, mlp_features), action, reward, (next_image, next_mlp_features), terminal = batch
        pred_action = self.forward(image, mlp_features)
        loss = F.mse_loss(pred_action, action)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
