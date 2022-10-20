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
import matplotlib.pyplot as plt
from Models import ResVAE
import os


def normalize_imgs(imgs):
    tmp = torch.flatten(imgs, start_dim=2)
    for batch_idx in range(tmp.size(0)):
        for channel_idx in range(3):
            tmp[batch_idx, channel_idx, :] -= tmp[batch_idx, channel_idx, :].min()
            tmp[batch_idx, channel_idx, :] /= tmp[batch_idx, channel_idx, :].max()
    return tmp.view(imgs.size())


def get_trained_vq_vae(training_logdir, epochs, device):
    root_dir = r'../'
    device = device
    vq_vae = vae_models['VQVAE2']
    batch_size = 256

    num_hiddens = 256
    num_residual_hiddens = 64
    num_residual_layers = 10

    embedding_dim = 128
    num_embeddings = 1024

    commitment_cost = 0.25

    decay = 0.99

    learning_rate = .005

    training_data = datasets.CIFAR10(root=root_dir, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))

    validation_data = datasets.CIFAR10(root=root_dir, train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor()
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
                  commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    sw = SummaryWriter(training_logdir + f'vq_vae_{timestamp()}')
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(training_loader):
            (data, _) = batch
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

        print('%d epoch' % (epoch + 1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))

        model.eval()

        (valid_originals, _) = next(iter(validation_loader))
        valid_originals = valid_originals.to(device)

        vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
        _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
        valid_reconstructions = model._decoder(valid_quantize)
        sw.add_images('Val/Reconstructions', normalize_imgs(valid_reconstructions), epoch)



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

        # with open(f"Models/VAE_Models/configs/{config_file_name}.yaml", 'r') as file:
        #     try:
        #         self.config = yaml.safe_load(file)
        #     except yaml.YAMLError as exc:
        #         print(exc)
        self.use_tensorboard = use_tensorboard
        # self.params = self.config['exp_params']
        # self.model = vae_models[self.config['model_params']['name']](**self.config['model_params']).to(device)
        # self.model_name = self.config['model_params']['name']
        self.model = ResVAE(latent_dim=100, encoder_depth=110, encoder_block='BottleNeck').to(device)
        self.model_name = f'ResVAE_110_{timestamp()}'
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

        results = self.model.forward(real_img)
        train_loss = self.model.loss_function(*results)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        real_img = real_img.to(self.device)
        labels = labels.to(self.device)

        results = self.model.forward(real_img)
        val_loss = self.model.loss_function(*results)

        return val_loss

    def training_loop(self, num_epochs):
        optimizer = optim.SGD(params=self.model.parameters(), lr=.1)
        scheduler = None
        use_lr_sched = False

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
            sampled_imgs = self.model.generate(batch.to(self.device))
            return sampled_imgs
        elif mode == 'test':
            batch, labels = next(iter(self.test_loader))
            sampled_imgs = self.model.generate(batch.to(self.device))
            return sampled_imgs
        elif mode == 'generate':
            return self.model.sample(num_samples=self.batch_size, device=self.device)




