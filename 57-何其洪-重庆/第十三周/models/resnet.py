# -*- coding: utf-8 -*-
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, places,
                 stride=1, is_down_sample=False, expansion=4):
        super().__init__()
        self.is_down_sample = is_down_sample
        self.expansion = expansion

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, places, 1, 1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(places, places, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(places, places * self.expansion, 1, 1, bias=False),
            nn.BatchNorm2d(places * self.expansion)
        )

        if is_down_sample:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, places * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.is_down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.expansion = 4

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1)  # 输出 64, 56, 56
        )
        self.layer1 = self.make_layer(64, 64, 3, 1)
        self.layer2 = self.make_layer(256, 128, 4, 2)
        self.layer3 = self.make_layer(512, 256, 6, 2)

    def make_layer(self, in_channels, places, block_num, stride=1):
        layers = [Bottleneck(in_channels, places, stride, True, self.expansion)]
        block_num -= 1
        for i in range(block_num):
            layers.append(Bottleneck(places * self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
