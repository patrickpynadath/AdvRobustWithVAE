from torch import optim
import torch
from torch.utils.data import DataLoader
from Models.VAE_Models import vae_models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Utils import timestamp, get_cifar_sets
import torch.nn.functional as F
import yaml
import random
import numpy as np


import os
class VQVAETrainer:
    def __init__(self,
                 device,
                 tensorboard,
                 logdir,
                 trainset,
                 testset,
                 batch_size=256,
                 num_hiddens=128,
                 num_res_hiddens=32,
                 num_res_layers=2,
                 embedding_dim=64,
                 num_embeddings=512,
                 commitment_cost=.25,
                 decay=.9,
                 lr=1e-3):
        VQVAE2 = vae_models['VQVAE2']
        self.model = VQVAE2(num_hiddens, num_res_layers, num_res_hiddens,
                            num_embeddings, embedding_dim, commitment_cost, decay).to(device)
        self.train_loader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
        self.test_loader = DataLoader(dataset=testset, batch_size=batch_size,shuffle=True)
        self.logdir = logdir
        self.tensorboard = tensorboard
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, amsgrad=False)
        self.training_var = np.var(trainset.data)

    def training_step(self, batch):
        inputs = batch[0]
        inputs = inputs.to(self.device)
        self.optimizer.zero_grad()

        vq_loss, reconstruction, perplexity = self.model(inputs)
        recon_error = F.mse_loss(reconstruction, inputs) / self.training_var
        loss = recon_error + vq_loss
        loss.backward()
        self.optimizer.step()
        return recon_error, perplexity

    def validation_step(self, batch):
        print(batch)
        with torch.no_grad():
            inputs = batch[0]
            print(inputs)
            vq_loss, reconstruction, perplexity = self.model(inputs)
            recon_error = F.mse_loss(reconstruction, inputs) / self.training_var
        return recon_error, perplexity

    def training_loop(self, num_epochs):
        writer = None
        if self.tensorboard:
            writer = SummaryWriter(log_dir=self.logdir + f'/vqvae_{timestamp()}/')
        for epoch in range(num_epochs):
            self.model.train()
            train_res = {'recon_error' : [], 'perplexity' : []}
            val_res = {'recon_error' : [], 'perplexity' : []}
            datastream = tqdm(enumerate(self.train_loader), total=len(self.train_loader), position=0, leave=True)
            for batch_idx, batch in datastream:
                recon_error, perplexity = self.training_step(batch)
                train_res['recon_error'].append(recon_error.item())
                train_res['perplexity'].append(perplexity.item())

            for batch_idx, batch in self.test_loader:
                recon_error, perplexity = self.validation_step(batch)
                val_res['recon_error'].append(recon_error.item())
                val_res['perplexity'].append(perplexity.item())

            if writer:
                # logging training data
                writer.add_scalar(f"Training/ReconLoss", sum(train_res['recon_error'])/len(train_res['recon_error']), epoch)
                writer.add_scalar(f"Training/Perplexity", sum(train_res['perplexity'])/len(train_res['perplexity']), epoch)
                writer.add_scalar(f"Val/ReconLoss", sum(val_res['recon_error']) / len(val_res['recon_error']),
                                  epoch)
                writer.add_scalar(f"Val/Perplexity", sum(val_res['perplexity']) / len(val_res['perplexity']), epoch)
                train_batch = next(iter(self.train_loader))
                _, train_recon, _ = self.model(train_batch)
                writer.add_images("Generated/training_reconstruction", train_recon, epoch)
                test_batch = next(iter(self.test_loader))
                _, test_recon, _ = self.model(test_batch)
                writer.add_images("Generated/test_reconstruction", test_recon, epoch)
        return


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
            self.model.train()
            for batch_idx, batch in datastream:
                optimizer.zero_grad()
                step_res = self.training_step(batch, batch_idx, optimizer_idx=0)
                loss = step_res['loss']
                loss.backward()
                optimizer.step()

                # adding the results together
                for key, value in step_res.items():
                    if key in train_epoch_res:
                        train_epoch_res[key] += value.item()
                    else:
                        train_epoch_res[key] = value.item()

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
                            val_epoch_res[key] += value.item()
                        else:
                            val_epoch_res[key] = value.item()
            if writer:
                # logging training data
                for key, value in train_epoch_res.items():
                    writer.add_scalar(f"Training/{key}", value, epoch)
                for key, value in val_epoch_res.items():
                    writer.add_scalar(f"Val/{key}", value, epoch)

                sampled_imgs_train = self.sample_reconstructions(mode='train')
                writer.add_images("Generated/training_reconstruction", sampled_imgs_train, epoch)
                sampled_imgs_test = self.sample_reconstructions(mode='test')
                writer.add_images("Generated/test_reconstruction", sampled_imgs_test, epoch)
                if self.model_name != 'VQVAE':
                    sampled_imgs_rand = self.sample_reconstructions('generate')
                    writer.add_images("Generated/random_samples", sampled_imgs_rand, epoch)

    def sample_reconstructions(self, mode):
        assert mode in ['train', 'test', 'generate']
        if mode == 'train':
            batch, labels = next(iter(self.train_loader))
            sampled_imgs = self.model.generate(batch.to(self.device), labels=labels)
            return sampled_imgs
        elif mode == 'test':
            batch, labels = next(iter(self.test_loader))
            sampled_imgs = self.model.generate(batch.to(self.device), labels=labels)
            return sampled_imgs
        elif mode == 'generate':
            return self.model.sample(num_samples=self.batch_size, current_device=self.device, labels=torch.randint(low=0, high=10, size=(self.batch_size,)))

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



