from torch import optim
import torch
from torch.utils.data import DataLoader
from Models.VAE_Models import vae_models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Utils import timestamp, get_cifar_sets
import yaml
import random
import numpy as np


import os


class VAETrainer:
    def __init__(self,
                 device,
                 use_tensorboard,
                 trainloader,
                 testloader,
                 logdir,
                 batch_size,
                 config_file_name,
                 **kwargs):

        with open(f"Models/VAE_Models/configs/{config_file_name}.yaml", 'r') as file:
            try:
                self.config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        self.use_tensorboard = use_tensorboard
        self.params = self.config['exp_params']
        self.model = vae_models[self.config['model_params']['name']](**self.config['model_params']).to(device)
        self.model_name = self.config['model_params']['name']
        self.logdir = logdir
        self.batch_size = batch_size
        self.device = device
        self.train_loader = trainloader
        self.test_loader = testloader

    # IMPORTANT NOTE: does not yet support VAE's that have multiple optimizers
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_img, labels = batch
        real_img = real_img.to(self.device)
        labels = labels.to(self.device)

        results = self.model.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        real_img = real_img.to(self.device)
        labels = labels.to(self.device)

        results = self.model.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def training_loop(self, num_epochs):
        model = self.model
        model.train()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        scheduler = None
        use_lr_sched = False
        if self.params['scheduler_gamma'] is not None:
            use_lr_sched = True
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                         gamma=self.params['scheduler_gamma'])
        writer = None
        if self.use_tensorboard:
            writer = SummaryWriter(log_dir=self.logdir + f'/{self.model_name}_{timestamp()}/')

        for epoch in range(num_epochs):
            datastream = tqdm(enumerate(self.train_loader), total=len(self.train_loader), position=0, leave=True)
            train_epoch_res = {}
            val_epoch_res = {}
            for batch_idx, batch in datastream:

                step_res = self.training_step(batch, batch_idx, optimizer_idx=0)
                loss = step_res['loss']
                loss.backward()
                optimizer.step()

                # adding the results together
                for key, value in step_res.items():
                    if key in train_epoch_res:
                        train_epoch_res[key] += value
                    else:
                        train_epoch_res[key] = value

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
            if use_lr_sched:
                scheduler.step()
                # val step
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_loader):
                    val_res = self.validation_step(batch, batch_idx)
                    for key, value in val_res.items():
                        if key in val_epoch_res:
                            val_epoch_res[key] += value
                        else:
                            val_epoch_res[key] = value
            if writer:
                # logging training data
                for key, value in train_epoch_res.items():
                    writer.add_scalar(f"Training/{key}", value)
                for key, value in val_epoch_res.items():
                    writer.add_scalar(f"Val/{key}", value)

                sampled_imgs_train = self.sample_reconstructions(mode='train')
                writer.add_images("Generated/training_reconstruction", sampled_imgs_train)
                sampled_imgs_test = self.sample_reconstructions(mode='test')
                writer.add_images("Generated/test_reconstruction", sampled_imgs_test)
                sampled_imgs_rand = self.sample_reconstructions('generate')
                writer.add_images("Generated/random_samples", sampled_imgs_rand)

    def sample_reconstructions(self, mode):
        assert mode in ['train', 'test', 'generate']
        if mode == 'train':
            batch, labels = next(iter(self.train_loader))
            sampled_imgs = self.model.generate(batch.to(self.device))
            return sampled_imgs
        elif mode == 'test':
            batch, labels = next(iter(self.test_loader))
            sampled_imgs = self.model.generate(batch.to(self.device))
            return sampled_imgs
        elif mode == 'generate':
            return self.model.sample(num_samples=self.batch_size, current_device=self.device)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                             gamma=self.params['scheduler_gamma'])
                return optimizer, scheduler
        except:
            pass



