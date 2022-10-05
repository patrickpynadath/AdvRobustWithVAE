from torch import optim
import torch
from torch.utils.data import DataLoader, dataset
from Models.VAE_Models import vae_models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Utils import timestamp, get_cifar_sets
import torch.nn.functional as F
import yaml
import random
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms


import os
def get_trained_vq_vae(training_logdir, num_training_updates):
    root_dir = r'../'
    device = 'cuda'
    vq_vae = vae_models['VQVAE']
    batch_size = 256


    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 64
    num_embeddings = 512

    commitment_cost = 0.25

    decay = 0.99

    learning_rate = 1e-3

    training_data = datasets.CIFAR10(root=root_dir, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    validation_data = datasets.CIFAR10(root=root_dir, train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                       ]))
    data_variance = np.var(training_data.data / 255.0)
    training_loader = DataLoader(training_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)
    validation_loader = DataLoader(validation_data,
                                   batch_size=32,
                                   shuffle=True,
                                   pin_memory=True)
    model = vq_vae(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    sw = SummaryWriter(training_logdir + f'vq_vae_{timestamp()}')
    for i in range(num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))

            model.eval()

            (valid_originals, _) = next(iter(validation_loader))
            valid_originals = valid_originals.to(device)

            vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
            _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
            valid_reconstructions = model._decoder(valid_quantize)
            sw.add_images('Val/Reconstructions', valid_reconstructions, i)

            (train_originals, _) = next(iter(training_loader))
            train_originals = train_originals.to(device)
            _, train_reconstructions, _, _ = model._vq_vae(train_originals)
            sw.add_images('Train/Reconstructions', train_reconstructions, i)
    return model


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



