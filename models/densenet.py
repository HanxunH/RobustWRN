'''
https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
DenseNet in PyTorch
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class RobustDenseNet(nn.Module):
    '''
    DenseNet121:
        depth_configs=[6, 12, 24, 16]
        channel_configs=[64, 128, 256, 512]
        growth_rate=32

    DenseNet169:
        depth_configs=[6, 12, 32, 32]
        channel_configs=[64, 128, 256, 640]
        growth_rate=32

    DenseNet201:
        depth_configs=[6, 12, 48, 32]
        channel_configs=[64, 128, 256, 896]
        growth_rate=32

    DenseNet161:
        depth_configs=[6, 12, 36, 24]
        channel_configs=[96, 192, 384, 1056]
        growth_rate=48
    '''
    def __init__(self, depth_configs=[6, 12, 24, 16], channel_configs=[64, 128, 256, 512],
                 growth_rate=32, reduction=0.5, num_classes=10):

        super(RobustDenseNet, self).__init__()
        self.growth_rate = growth_rate
        block = Bottleneck

        self.conv1 = nn.Conv2d(3, channel_configs[0], kernel_size=3, padding=1, bias=False)
        stages = []

        for i, nblocks in enumerate(depth_configs):
            block_out = nblocks * growth_rate + channel_configs[i]
            if i < len(depth_configs) - 1:
                out_planes = channel_configs[i+1]
                stage = nn.Sequential(self._make_dense_layers(block, channel_configs[i], nblocks),
                                      Transition(block_out, out_planes))
            else:
                stage = nn.Sequential(self._make_dense_layers(block, channel_configs[i], nblocks))
            stages.append(stage)

        self.stages = nn.Sequential(*stages)
        self.bn = nn.BatchNorm2d(block_out)
        self.linear = nn.Linear(block_out, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stages(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
