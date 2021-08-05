import torch
from torch import nn

from projects.reactive_mbrl.pretrained.resnet import resnet34, resnet18


class SegmentationHead(nn.Module):
    def __init__(self, input_channels, num_labels):
        super().__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 256, 3, 2, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, num_labels, 1, 1, 0),
        )

    def forward(self, x):
        return self.upconv(x)


class Normalize(nn.Module):
    """ ImageNet normalization """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False)

    def forward(self, x):
        return (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]


class CameraModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Configs
        self.num_cmds = int(config["num_cmds"])
        self.num_steers = config["num_steers"]
        self.num_throts = config["num_throts"]
        self.num_speeds = config["num_speeds"]
        self.num_labels = len(config["seg_channels"])

        self.backbone_wide = resnet34(pretrained=True)
        self.seg_head_wide = SegmentationHead(512, self.num_labels + 1)
        self.backbone_narr = resnet18(pretrained=True)
        self.seg_head_narr = SegmentationHead(512, self.num_labels + 1)
        self.bottleneck_narr = nn.Sequential(nn.Linear(512, 64), nn.ReLU(True),)

        # self.num_acts = self.num_cmds*self.num_speeds*(self.num_steers+self.num_throts+1)
        self.num_acts = (
            self.num_cmds * self.num_speeds * (self.num_steers + self.num_throts)
        )

        self.wide_seg_head = SegmentationHead(512, self.num_labels + 1)
        self.act_head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, self.num_acts),
        )

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, wide_rgb, narr_rgb):

        wide_embed = self.backbone_wide(self.normalize(wide_rgb / 255.0))

        wide_seg_output = self.seg_head_wide(wide_embed)
        narr_embed = self.backbone_narr(self.normalize(narr_rgb / 255.0))
        narr_seg_output = self.seg_head_narr(narr_embed)

        embed = torch.cat(
            [
                wide_embed.mean(dim=[2, 3]),
                self.bottleneck_narr(narr_embed.mean(dim=[2, 3])),
            ],
            dim=1,
        )

        act_output = self.act_head(embed).view(
            #    -1, self.num_cmds, self.num_speeds, self.num_steers + self.num_throts + 1
            -1,
            self.num_cmds,
            self.num_speeds,
            self.num_steers + self.num_throts,
        )
        act_output = action_logits(act_output, self.num_steers, self.num_throts)

        return act_output, wide_seg_output, narr_seg_output

    @torch.no_grad()
    def predict(self, wide_rgb, narr_rgb):
        cmd = 0

        wide_embed = self.backbone_wide(self.normalize(wide_rgb / 255.0))
        narr_embed = self.backbone_narr(self.normalize(narr_rgb / 255.0))
        embed = torch.cat(
            [
                wide_embed.mean(dim=[2, 3]),
                self.bottleneck_narr(narr_embed.mean(dim=[2, 3])),
            ],
            dim=1,
        )

        # Action logits
        act_output = self.act_head(embed).view(
            #    -1, self.num_cmds, self.num_speeds, self.num_steers + self.num_throts + 1
            -1,
            self.num_cmds,
            self.num_speeds,
            self.num_steers + self.num_throts,
        )
        act_output = action_logits(act_output, self.num_steers, self.num_throts)

        # Action logits
        steer_logits = act_output[0, cmd, :, : self.num_steers]
        throt_logits = act_output[
            0, cmd, :, self.num_steers : self.num_steers + self.num_throts
        ]
        brake_logits = act_output[0, cmd, :, -1]

        return steer_logits, throt_logits, brake_logits


def action_logits(raw_logits, num_steers, num_throts):

    steer_logits = raw_logits[..., :num_steers]
    throt_logits = raw_logits[..., num_steers : num_steers + num_throts]
    # brake_logits = raw_logits[..., -1:]

    steer_logits = steer_logits.repeat(1, 1, 1, num_throts)
    throt_logits = throt_logits.repeat_interleave(num_steers, -1)

    # act_logits = torch.cat([steer_logits + throt_logits, brake_logits], dim=-1)
    act_logits = torch.cat([steer_logits + throt_logits], dim=-1)

    return act_logits
