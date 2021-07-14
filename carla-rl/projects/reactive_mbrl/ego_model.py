import math
import torch
import torch.nn.functional as F
from torch import nn

import pytorch_lightning as pl


class EgoModelRails(nn.Module):
    def __init__(self, dt=1./4):
        super().__init__()

        self.dt = dt

        # Kinematic bicycle model
        self.front_wb = nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.rear_wb  = nn.Parameter(torch.tensor(1.),requires_grad=True)

        self.steer_gain  = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.brake_accel = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.throt_accel = nn.Sequential(
            nn.Linear(1,1,bias=False),
        )

    def forward(self, locs, yaws, spds, acts):
        steer = acts[...,0:1]
        throt = acts[...,1:2]
        brake = acts[...,2:3].byte()

        accel = torch.where(brake, self.brake_accel.expand(*brake.size()), self.throt_accel(throt))
        wheel = self.steer_gain * steer

        beta = torch.atan(self.rear_wb/(self.front_wb+self.rear_wb) * torch.tan(wheel))

        next_locs = locs + spds * torch.cat([torch.cos(yaws+beta), torch.sin(yaws+beta)],-1) * self.dt
        next_yaws = yaws + spds / self.rear_wb * torch.sin(beta) * self.dt
        next_spds = spds + accel * self.dt

        return next_locs, next_yaws, F.relu(next_spds)


class EgoModel(pl.LightningModule):
    def __init__(self, dt=1./10):
        super().__init__()

        self.dt = dt

        # Kinematic bicycle model
        self.front_wb = nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.rear_wb  = nn.Parameter(torch.tensor(1.),requires_grad=True)

        self.steer_gain  = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.brake_accel = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.throt_accel = nn.Sequential(
            nn.Linear(1,1,bias=False),
        )
        # self.throt_accel[0].weight = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    def forward(self, locs, yaws, spds, acts):
        steer = acts[...,0:1]
        throt = acts[...,1:2]
        brake = (throt < 0).byte() # acts[...,2:3].byte()

        accel = torch.where(brake, self.brake_accel.expand(*brake.size()), self.throt_accel(throt))
        wheel = self.steer_gain * steer

        beta = torch.atan(self.rear_wb/(self.front_wb+self.rear_wb) * torch.tan(wheel))

        next_locs = locs + spds * torch.cat([torch.cos(yaws+beta), torch.sin(yaws+beta)],-1) * self.dt
        next_yaws = yaws + spds / self.rear_wb * torch.sin(beta) * self.dt
        next_spds = spds + accel * self.dt

        return next_locs, next_yaws, F.relu(next_spds)

    def training_step(self, batch, batch_idx):
        ego_tf, spds, acts = batch
        locs, yaws = ego_tf[:,:,:2], ego_tf[:,:,4:5] * math.pi / 180

        pred_locs = []
        pred_yaws = []

        pred_loc = locs[:,0]
        pred_yaw = yaws[:,0]
        pred_spd = spds[:,0]
        for t in range(4):
            act = acts[:,t]

            pred_loc, pred_yaw, pred_spd = self.forward(pred_loc, pred_yaw, pred_spd, act)

            pred_locs.append(pred_loc)
            pred_yaws.append(pred_yaw)

        pred_locs = torch.stack(pred_locs, 1)
        pred_yaws = torch.stack(pred_yaws, 1)

        loc_loss = F.l1_loss(pred_locs, locs[:,1:])
        ori_loss = F.l1_loss(torch.cos(pred_yaws), torch.cos(yaws[:,1:])) + F.l1_loss(torch.sin(pred_yaws), torch.sin(yaws[:,1:]))

        loss = loc_loss + ori_loss

        self.log('train/loc_loss', loc_loss)
        self.log('train/ori_loss', ori_loss)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ego_tf, spds, acts = batch
        locs, yaws = ego_tf[:,:,:2], ego_tf[:,:,4:5] * math.pi / 180

        pred_locs = []
        pred_yaws = []

        pred_loc = locs[:,0]
        pred_yaw = yaws[:,0]
        pred_spd = spds[:,0]
        for t in range(4):
            act = acts[:,t]

            pred_loc, pred_yaw, pred_spd = self.forward(pred_loc, pred_yaw, pred_spd, act)

            pred_locs.append(pred_loc)
            pred_yaws.append(pred_yaw)

        pred_locs = torch.stack(pred_locs, 1)
        pred_yaws = torch.stack(pred_yaws, 1)

        loc_loss = F.l1_loss(pred_locs, locs[:,1:])
        ori_loss = F.l1_loss(torch.cos(pred_yaws), torch.cos(yaws[:,1:])) + F.l1_loss(torch.sin(pred_yaws), torch.sin(yaws[:,1:]))

        loss = loc_loss + ori_loss

        self.log('val/loc_loss', loc_loss)
        self.log('val/ori_loss', ori_loss)
        self.log('val/loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
