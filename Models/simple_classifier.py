import torch.nn as nn
import torch.nn.functional as F
import torch

class simple_conv_net(nn.Module):
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