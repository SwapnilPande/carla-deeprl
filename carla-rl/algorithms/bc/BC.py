import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from agents.torch.models import make_conv_preprocessor


class BC(pl.LightningModule):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU(),
            nn.Linear(750, action_dim)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out

    def predict(self, x):
        x = torch.FloatTensor(x).cuda()
        out = self.forward(x)
        return out.detach().cpu().numpy().reshape(-1, self.action_dim)

    def training_step(self, batch, batch_idx):
        state, action, _, _, _ = batch
        pred = self(state)
        import ipdb; ipdb.set_trace()
        loss = F.mse_loss(pred, action)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state, action, _, _, _ = batch
        pred = self(state)
        loss = F.mse_loss(pred, action)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [{'params': self.mlp.parameters()}]
        return torch.optim.Adam(params, lr=1e-3)


class ImageBC(pl.LightningModule):
    def __init__(self, img_dim, mlp_dim, action_dim, conv_arch='vanilla', freeze_conv=False, frame_stack=1):
        super().__init__()
        self.img_dim = img_dim
        self.mlp_dim = mlp_dim
        self.action_dim = action_dim
        self.freeze_conv = freeze_conv
        self.frame_stack = frame_stack

        self.conv = make_conv_preprocessor(256, arch=conv_arch, frame_stack=frame_stack, freeze_conv=freeze_conv)
        self.mlp = nn.Sequential(
            nn.Linear(256 + mlp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        img, mlp_features = x
        conv_features = self.conv(img)
        conv_features = conv_features.view(img.size(0), -1)
        features = torch.cat([conv_features, mlp_features], dim=1)
        out = self.mlp(features)
        return out

    def predict(self, x):
        img, mlp_features = x
        img = torch.FloatTensor(img).permute(2,0,1).cuda()
        mlp_features = torch.FloatTensor(mlp_features).cuda()
        pred = self.forward((img[None], mlp_features[None]))
        return pred.detach().cpu().numpy().reshape(-1, self.action_dim)

    def training_step(self, batch, batch_idx):
        state, action, _, _, _ = batch
        pred = self(state)
        loss = F.mse_loss(pred, action)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        state, action, _, _, _ = batch
        pred = self(state)
        loss = F.mse_loss(pred, action)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = [{'params': self.mlp.parameters()}]
        if not self.freeze_conv:
            params.append({'params': self.conv.parameters()})
        return torch.optim.Adam(params, lr=1e-3)
