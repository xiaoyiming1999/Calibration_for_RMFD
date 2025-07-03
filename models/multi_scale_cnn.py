#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch import nn


class Multi_scale_CNN(nn.Module):
    def __init__(self, output_dim):
        super(Multi_scale_CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=25, stride=2, padding=0, dilation=1),
            nn.ReLU(True),
            nn.BatchNorm1d(16)
        )

        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)

        self.Mod1_MS1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.BatchNorm1d(32)
        )

        self.Mod1_MS2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(True),
            nn.BatchNorm1d(32)
        )

        self.Mod1_MS3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(True),
            nn.BatchNorm1d(32)
        )

        self.Mod1_MS4 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.ReLU(True),
            nn.BatchNorm1d(32)
        )

        self.SE_1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )

        self.Mod2_MS1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.BatchNorm1d(64)
        )

        self.Mod2_MS2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(True),
            nn.BatchNorm1d(64)
        )

        self.Mod2_MS3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(True),
            nn.BatchNorm1d(64)
        )

        self.Mod2_MS4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.ReLU(True),
            nn.BatchNorm1d(64)
        )

        self.SE_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )

        self.Adaptivepool = nn.AdaptiveAvgPool1d(4)

        self.layer1 = nn.Sequential(
            nn.Linear(4 * 256, 512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.layer3 = nn.Linear(512, output_dim)

    def forward(self, Input):

        conv1_output = self.conv1(Input)
        pool1 = self.pool1(conv1_output)

        Mod1_MS1_output = self.Mod1_MS1(pool1)
        Mod1_MS2_output = self.Mod1_MS2(pool1)
        Mod1_MS3_output = self.Mod1_MS3(pool1)
        Mod1_MS4_output = self.Mod1_MS4(pool1)

        Mod1_Con_F = torch.cat((Mod1_MS1_output,  Mod1_MS2_output, Mod1_MS3_output, Mod1_MS4_output), 1)

        squeeze_tensor = Mod1_Con_F.mean(dim=2)
        excitation_tensor = self.SE_1(squeeze_tensor)
        Mod1_Con_F = torch.mul(Mod1_Con_F, excitation_tensor.view(Mod1_Con_F.shape[0], 128, 1))

        Mod2_MS1_output = self.Mod2_MS1(Mod1_Con_F)
        Mod2_MS2_output = self.Mod2_MS2(Mod1_Con_F)
        Mod2_MS3_output = self.Mod2_MS3(Mod1_Con_F)
        Mod2_MS4_output = self.Mod2_MS4(Mod1_Con_F)

        Mod2_Con_F = torch.cat((Mod2_MS1_output, Mod2_MS2_output, Mod2_MS3_output, Mod2_MS4_output), 1)

        squeeze_tensor = Mod2_Con_F.mean(dim=2)
        excitation_tensor = self.SE_2(squeeze_tensor)
        Mod2_Con_F = torch.mul(Mod2_Con_F, excitation_tensor.view(Mod2_Con_F.shape[0], 256, 1))
        Mod2_Con_F = self.Adaptivepool(Mod2_Con_F)

        x = Mod2_Con_F.view(Mod2_Con_F.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

if __name__ == '__main__':
    model = Multi_scale_CNN(output_dim=8)
    input = torch.randn(size=[1, 1, 1024])
    output = model(input)

