import torch
from Models import Smooth
import torch.nn as nn


class GenClf(nn.Module):
    def __init__(self, gen_model, clf):
        super().__init__()
        self.gen_model = gen_model
        self.clf = clf

    def forward(self, x):
        recon = self.gen_model.generate(x.detach())
        return self.clf(recon)


