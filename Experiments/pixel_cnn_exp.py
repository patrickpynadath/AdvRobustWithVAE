import matplotlib

from Models.vae import VAE
from Training.train_vae import train_vae
from Training.train_natural import NatTrainerSmoothVAE
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
from Models.smoothing import SmoothVAE_Latent, SmoothVAE_Sample
from Models.classifiers import simple_conv_net
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from sklearn.metrics import calinski_harabasz_score
import numpy as np
from Utils.utils import get_class_subsets

