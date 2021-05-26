import torch
import torch.nn as nn
import torch.nn.functional as TF
from torchvision.models import resnet18, wide_resnet50_2, vgg16


class AttentionCNN(nn.Module):
    def __init__(self, output_size, frame_stack=1):
        super().__init__()
        network = vgg16(pretrained=True) # wide_resnet50_2(pretrained=True) # resnet18(pretrained=True)
        conv1 = network.features[0]
        conv1.in_channels = conv1.in_channels * frame_stack
        conv1.weight = nn.Parameter(torch.cat([conv1.weight for _ in range(frame_stack)], 1))

        children = nn.Sequential(*network.features.children())
        self.conv_block_1 = nn.Sequential(*[c for c in children[:16] if not isinstance(c, nn.MaxPool2d)])
        self.conv_block_2 = children[17:24]
        self.conv_block_3 = children[24:]
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1024, 512)
        )
        self.projection_layer = nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False)
        self.attention_layer_1 = nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False)
        self.attention_layer_2 = nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False)
        self.attention_layer_3 = nn.Conv2d(512, 1, kernel_size=1, padding=0, bias=False)
        self.fc = nn.Linear(512 * 4, output_size)

        # freeze conv layers
        for param in self.conv_block_1.parameters():
            param.requires_grad = False

        for param in self.conv_block_2.parameters():
            param.requires_grad = False

        for param in self.conv_block_3.parameters():
            param.requires_grad = False

        # for param in self.conv_block_4.parameters():
        #     param.requires_grad = False

    def forward(self, x, attention_maps=False):
        N, C, W, H = x.size()
        x1 = TF.max_pool2d(self.conv_block_1(x), 2, 2)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        g = self.conv_block_4(x3).reshape(N, 512, 1, 1)

        x1 = self.projection_layer(x1)

        a1 = self.attention_layer_1(x1 + g)
        a2 = self.attention_layer_2(x2 + g)
        a3 = self.attention_layer_3(x3 + g)

        a1 = TF.softmax(a1.view(-1), 0).view(N, 1, 64, 64)
        a2 = TF.softmax(a2.view(-1), 0).view(N, 1, 32, 32)
        a3 = TF.softmax(a3.view(-1), 0).view(N, 1, 16, 16)

        g1 = torch.mul(a1, x1).view(N, 512, -1).sum(dim=2)
        g2 = torch.mul(a2, x2).view(N, 512, -1).sum(dim=2)
        g3 = torch.mul(a3, x3).view(N, 512, -1).sum(dim=2)

        g = torch.cat([g1, g2, g3, g.reshape(N, 512)], 1)
        out = self.fc(g)

        if attention_maps:
            return out, (a1, a2, a3)
        else:
            return out


class VanillaCNN(nn.Module):
    def __init__(self, output_size, frame_stack=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3 * frame_stack, 16, kernel_size=5, stride=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=3),
            nn.BatchNorm2d(32),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def make_conv_preprocessor(output_size, arch='vanilla', frame_stack=1, freeze_conv=True):
    if arch == 'resnet':
        network = resnet18(pretrained=True)
        conv1 = network.conv1
        conv1.in_channels = conv1.in_channels * frame_stack
        conv1.weight = nn.Parameter(torch.cat([conv1.weight for _ in range(frame_stack)], 1))
        network.fc = nn.Linear(512, output_size)
    elif arch == 'vanilla':
        network = VanillaCNN(output_size, frame_stack=frame_stack)
    elif arch == 'attention_cnn':
        network = AttentionCNN(output_size, frame_stack=frame_stack)
    else:
        print('conv preprocessor with arch ({}) not found'.format(arch))
        raise NotImplementedError

    if freeze_conv:
        for param in network.parameters():
            param.requires_grad = False

    return network
