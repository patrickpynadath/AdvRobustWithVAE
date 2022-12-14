import torch
import torch.nn as nn
from Models.Generative.encoder import Encoder
from Models.Generative.decoder import Decoder
import torch.nn.functional as F
from Utils import timestamp


class VAE(nn.Module):
    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 latent_size,
                 device,
                 beta):
        super(VAE, self).__init__()
        self.res_layers = num_residual_layers
        self.latent_size = latent_size
        self.beta = beta
        self.device = device
        self.num_hiddens = num_hiddens
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._fc_mu = nn.Linear(num_hiddens * 8 * 8, latent_size)
        self._fc_var = nn.Linear(num_hiddens * 8 * 8, latent_size)

        self._fc_dec = nn.Linear(latent_size, num_hiddens * 8 * 8)
        self._decoder = Decoder(num_hiddens,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.label= f'VAE_{timestamp()}'
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.uniform_(module.weight.data, -.08, .08)
            if module.bias is not None:
                module.bias.data.zero_()

    def encode(self, x):
        enc = self._encoder(x)
        pre_latent = enc.flatten(start_dim=1)
        mu = self._fc_mu(pre_latent)
        logvar = self._fc_var(pre_latent)
        return [mu, logvar]

    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return eps * std + mu

    def decode(self, z):
        dec = self._fc_dec(z)
        pre_recon = dec.view(-1, self.num_hiddens, 8, 8)
        x_recon = self._decoder(pre_recon)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), x, mu, logvar

    def get_encoding(self, x):
        mu, logvar = self.encode(x)
        z = self.parameterize(mu, logvar)
        return z


    def generate(self, x):
        return self.forward(x)[0]

    def loss_fn(self, recon, x, mu, logvar):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        recon_loss = F.mse_loss(recon, x)
        total_loss = recon_loss + self.beta * kld_loss
        return total_loss, recon_loss, kld_loss


def get_latent_code_vae(vae: VAE, x):
    return vae.encode(x)[0]
