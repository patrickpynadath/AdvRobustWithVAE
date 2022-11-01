import torch
from Models import Smooth
import torch.nn as nn


class GenClf(nn.Module):
    def __init__(self, gen_model, clf):
        super().__init__()
        self.gen_model = gen_model
        self.clf = clf
        self.label = gen_model.label + clf.label

    def forward(self, x):
        self.gen_model.eval()
        recon = self.gen_model.generate(x.detach())
        return self.clf(recon)


