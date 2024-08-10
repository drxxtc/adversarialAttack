import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(1, 4, 3, stride=1, padding=0)
        self.conv2 = nn.Conv1d(4, 16, 3, stride=1, padding=0)
        self.conv3 = nn.Conv1d(16, 32, 3, stride=1, padding=0)
        self.adaptivepool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 2)

    def forward(self, x):

        out = self.maxpool(self.act(self.conv1(x)))
        out = self.maxpool(self.act(self.conv2(out)))
        out = self.maxpool(self.act(self.conv3(out)))

        out = self.adaptivepool(out)

        out = self.flatten(out)
        out = self.act(self.linear1(out))
        out = self.linear2(out)

        return out