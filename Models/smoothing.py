import torch
import torch.nn as nn


class Smooth(nn.Module):
    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 device : torch.device,
                 num_samples : int,
                 num_classes : int):
        super().__init__()
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier.to(device)
        self.sigma = sigma
        self.device = device
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.label = self.label + f'_Smooth_{round(sigma, 4)}_MTrain_{num_samples}'

    def forward(self, x):
        output = torch.zeros((x.size(dim=0), self.num_classes)).to(self.device)
        for i in range(self.num_samples):
            noise = torch.randn_like(x).to(self.device) * self.sigma ** 2
            output += self.base_classifier(x + noise)
        output /= self.num_samples
        return output

    def parameters(self, recurse: bool = True):
        return self.base_classifier.parameters()




