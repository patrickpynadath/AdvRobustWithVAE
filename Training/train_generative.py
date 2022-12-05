from torch import optim
import torch
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Utils import timestamp, get_cifar_sets
import torch.nn.functional as F
import yaml
import random
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os


def normalize_imgs(imgs):
    tmp = torch.flatten(imgs, start_dim=2)
    for batch_idx in range(tmp.size(0)):
        for channel_idx in range(3):
            tmp[batch_idx, channel_idx, :] -= tmp[batch_idx, channel_idx, :].min()
            tmp[batch_idx, channel_idx, :] /= tmp[batch_idx, channel_idx, :].max()
    return tmp.view(imgs.size())


class GenerativeTrainer:
    def __init__(self,
                 device,
                 model : torch.nn.Module,
                 use_tensorboard,
                 logdir,
                 batch_size,
                 lr):

        self.use_tensorboard = use_tensorboard
        self.model = model
        self.model_name = model.label
        self.logdir = logdir
        self.batch_size = batch_size
        self.device = device
        train_set, test_set = get_cifar_sets()
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        self.lr = lr

    def training_step(self, batch):
        real_img, labels = batch
        real_img = real_img.to(self.device)

        results = self.model.forward(real_img)
        train_loss = self.model.loss_function(*results)
        return {'total loss' : train_loss}

    def training_loop(self, num_epochs):
        optimizer = optim.SGD(params=self.model.parameters(), lr=self.lr)

        writer = None
        if self.use_tensorboard:
            writer = SummaryWriter(log_dir=self.logdir + f'/{self.model_name}_{timestamp()}/')

        for epoch in range(num_epochs):
            datastream = tqdm(enumerate(self.train_loader), total=len(self.train_loader), position=0, leave=True)
            train_epoch_res = {}
            val_epoch_res = {}
            self.model.train()
            for batch_idx, batch in datastream:
                optimizer.zero_grad()
                step_res = self.training_step(batch)
                loss = step_res['total loss']
                loss.backward()
                optimizer.step()

                # adding the results together
                for key, value in step_res.items():
                    if key in train_epoch_res:
                        train_epoch_res[key].append(value.item())
                    else:
                        train_epoch_res[key] = [value.item()]

                # tqdm loading bar
                datastream.set_description((
                    'epoch: {epoch} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                ).format(
                    epoch=epoch,
                    trained=batch_idx * len(batch[0]),
                    total=len(self.train_loader.dataset),
                    progress=(100. * batch_idx / len(self.train_loader)),
                ))
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_loader):
                    val_res = self.training_step(batch)
                    for key, value in val_res.items():
                        if key in val_epoch_res:
                            val_epoch_res[key].append(value.item())
                        else:
                            val_epoch_res[key] = [value.item()]
            if writer:
                # logging training data
                for key, value in train_epoch_res.items():
                    writer.add_scalar(f"Training/{key}", np.mean(value), epoch)
                for key, value in val_epoch_res.items():
                    writer.add_scalar(f"Val/{key}", np.mean(value), epoch)

                sampled_imgs_train = self.sample_reconstructions(mode='train')
                writer.add_images("Generated/training_reconstruction", sampled_imgs_train, epoch)
                sampled_imgs_test = self.sample_reconstructions(mode='test')
                writer.add_images("Generated/test_reconstruction", sampled_imgs_test, epoch)

    def sample_reconstructions(self, mode):
        assert mode in ['train', 'test']
        if mode == 'train':
            batch, labels = next(iter(self.train_loader))
            sampled_imgs = self.model.generate(batch.to(self.device))
            return torch.clamp(sampled_imgs, 0, 1)
        elif mode == 'test':
            batch, labels = next(iter(self.test_loader))
            sampled_imgs = self.model.generate(batch.to(self.device))
            return torch.clamp(sampled_imgs, 0, 1)


class VAETrainer(GenerativeTrainer):

    def training_step(self, batch):
        real_img, labels = batch
        real_img = real_img.to(self.device)

        recon, orig, mu, logvar = self.model.forward(real_img)
        total_loss, recon_loss, kld_loss = self.model.loss_fn(recon, orig, mu, logvar)
        return {'total loss': total_loss,
                'recon loss': recon_loss,
                'kld loss': kld_loss}




class VQVAETrainer(GenerativeTrainer):

    def training_step(self, batch):
        real_img, labels = batch
        real_img = real_img.to(self.device)
        recon, perp = self.model.forward(real_img)
        total_loss = mse_loss = F.mse_loss(recon, real_img)
        return {'total loss': total_loss,
                'mse loss': mse_loss,
                'perplexity': perp}


class AETrainer(GenerativeTrainer):

    def training_step(self, batch):
        real_img, labels = batch
        real_img = real_img.to(self.device)
        recon = self.model.forward(real_img)
        total_loss = mse_loss = F.mse_loss(recon, real_img)
        return {'total loss': total_loss,
                'mse loss': mse_loss}

