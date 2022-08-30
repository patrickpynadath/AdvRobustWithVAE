from Models.vae import VAE
from Training.train_vae import train_vae
from Models.smoothing import Smooth, SmoothVAE_Latent, SmoothVAE_Sample
from Models.simple_conv import simple_conv_net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.optim as optim
import os

class Experiment:

    def __init__(self, batch_size, training_logdir, hyperparam_logdir, device):
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.training_logdir = training_logdir + f"/{date_str}/"
        self.hyperparam_logdir = hyperparam_logdir + f"/{date_str}/"
        self.batch_size = batch_size

        transform = transforms.Compose(
            [transforms.ToTensor()])
        root_dir = r''
        trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                               download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = len(classes)
        self.trainset = trainset
        self.trainloader = trainloader
        self.testset = testset
        self.testloader = testloader
        self.device = device
        pass

    def get_base_models(self, base_label, lr):
        model = simple_conv_net(label=base_label).to(self.device)
        optimizers = [optim.SGD(model.parameters(), lr=lr)]
        criterions = [nn.CrossEntropyLoss()]
        return [model], optimizers, criterions

    def get_smooth_models(self, base_label, lr, sigmas, m_train):
        models = []
        optimizers = []
        criterions = []

        # instantiating the models
        for i, sigma in enumerate(sigmas):
            label = base_label + f'smooth_{sigma}'
            base = simple_conv_net()
            smooth = Smooth(base, label, self.num_classes, sigma, self.device, m_train, self.batch_size)
            optimizer = optim.SGD(smooth.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            models.append(smooth)
            optimizers.append(optimizer)
            criterions.append(criterion)
        return models, optimizers, criterions

    def get_smoothVAE_models(self,
                             base_label,
                             lr,
                             sigmas,
                             m_train,
                             vae_epochs,
                             with_VAE_grad,
                             vae_sigma=1,
                             model_version ='latent',
                             trainer_version='combined'):
        models = []
        optimizers = []
        criterions = []

        # instantiating the models
        for i, sigma in enumerate(sigmas):
            # training the VAE
            vae = VAE(label='', image_size=32, channel_num=3, kernel_num=50, z_size=100, device=self.device, var = vae_sigma).to(
                self.device)
            train_vae(vae, self.trainloader, len(self.trainset), epochs=vae_epochs)

            # making the smoothVAE model label
            label = base_label + f'smooth_{sigma}_vaegrad_{with_VAE_grad}_latentvar{vae_sigma}_trainerversion{trainer_version}_modelversion{model_version}'

            base = simple_conv_net()
            if model_version == 'latent':
                smoothVAE = SmoothVAE_Latent(base, label, self.num_classes, sigma, vae, self.device, m_train, self.batch_size)
            elif model_version == 'sample':
                smoothVAE = SmoothVAE_Sample(base, label, self.num_classes, sigma, vae, self.device, m_train, self.batch_size)
            if with_VAE_grad:
                optimizer = optim.SGD(smoothVAE.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(smoothVAE.base_classifier.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            models.append(smoothVAE)
            optimizers.append(optimizer)
            criterions.append(criterion)

        return models, optimizers, criterions