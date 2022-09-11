import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn as nn
from torch.nn import functional as F


class simple_conv_net(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        self.label = 'base'
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class simple_classifier(nn.Module):
    def __init__(self, label='', input_size = 100):
        super().__init__()
        self.label = label
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class pixelcnn_classifier(nn.Module):
    def __init__(self, label, pxcnn, device):
        super().__init__()
        self.label = label
        self.pxcnn = pxcnn
        self.classifier = simple_conv_net(in_channels=100).to(device)

    def forward(self, x):
        x_trf = self.pxcnn(x)
        return self.classifier(x_trf)