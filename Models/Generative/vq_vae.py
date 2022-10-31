import torch.nn as nn
from Models.Generative.encoder import Encoder
from Models.Generative.decoder import Decoder
from .vector_quantizer import NSVQ
from Utils import timestamp


class VQVAE_NSVQ(nn.Module):
    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 num_embeddings,
                 embedding_dim,
                 batch_size,
                 epochs,
                 device,
                 num_training_samples):
        super(VQVAE_NSVQ, self).__init__()
        self.res_layers = num_residual_layers
        self.latent_size = embedding_dim
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self._vq_vae = NSVQ(num_embeddings, embedding_dim, epochs, num_training_samples, batch_size, device)

        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.label = f'VQVAE_{timestamp()}'

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        quantized, perplexity = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return x_recon, perplexity

    def generate(self, x):
        return self.forward(x)[0]