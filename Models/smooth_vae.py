import torch
from Models import Smooth
import torch.nn as nn


class GenClf(nn.Module):
    def __init__(self, gen_model, clf, get_recon):
        super().__init__()
        self.gen_model = gen_model
        self.clf = clf
        self.get_recon = get_recon

    def forward(self, x):
        recon = self.get_recon(self.gen_model, x)
        return self.clf(recon)


