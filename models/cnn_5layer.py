#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN_5layer(nn.Module):
    def __init__(self,  in_channel=1, num_cls=10):
        super(CNN_5layer, self).__init__()

        # 采样频率较高的数据第一层的卷积核的核大小一定要大一点
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=25, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=15, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 256, kernel_size=5, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4))

        self.layer4 = nn.Sequential(
            nn.Linear(256 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout())

        self.layer5 = nn.Linear(256, num_cls)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        x = self.layer5(x)

        return x