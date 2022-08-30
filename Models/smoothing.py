import torch
from Models.vae import VAE
import torch.nn as nn


class Smooth(nn.Module):
    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 device : torch.device,
                 num_samples : int,
                 num_classes : int):
        super().__init__()
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier.to(device)
        self.sigma = sigma
        self.device = device
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.label = f'Smooth_{round(sigma, 4)}_MTrain_{num_samples}'

    def forward(self, x):
        output = torch.zeros((x.size(dim=0), self.num_classes)).to(self.device)
        for i in range(self.num_samples):
            noise = torch.randn_like(x).to(self.device) * self.sigma ** 2
            output += self.base_classifier(x + noise)
        output /= self.num_samples
        return output

    def parameters(self, recurse: bool = True):
        return self.base_classifier.parameters()

class SmoothVAE_Latent(Smooth):

    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 trained_VAE: VAE,
                 device,
                 num_samples,
                 num_classes,
                 loss_coef):
        super().__init__(base_classifier, sigma, device, num_samples, num_classes)
        # needs to be trained on the same set as the base classifier
        self.trained_VAE = trained_VAE
        self.loss_coef = loss_coef
        self.label = f'SmoothVAE_Latent_{sigma}_MTrain_{num_samples}_losscoef_{loss_coef}'


    def forward(self, x):
        # sample latent code z from q given x.
        encoded = self.trained_VAE.encoder(x)
        mean, logvar = self.trained_VAE.q(encoded)
        z = self.trained_VAE.z(mean, logvar)
        z_projected = self.trained_VAE.project(z).view(
            -1, self.trained_VAE.kernel_num,
            self.trained_VAE.feature_size,
            self.trained_VAE.feature_size,
        )
        output = torch.zeros((x.size(dim=0), self.num_classes)).to(self.device)
        for i in range(self.num_samples):
            noise = torch.randn_like(z_projected).to(self.device) * self.sigma ** 2
            reconstr = self.trained_VAE.decoder(z_projected + noise)
            output += self.base_classifier(reconstr)
        output /= self.num_samples
        return output


class SmoothVAE_Sample(Smooth):
    ABSTAIN = -1

    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 trained_VAE: VAE,
                 device,
                 num_samples,
                 num_classes,
                 loss_coef):
        super().__init__(base_classifier, sigma, device, num_samples, num_classes)
        # needs to be trained on the same set as the base classifier
        self.trained_VAE = trained_VAE
        self.loss_coef = loss_coef
        self.label = f'SmoothVAE_Sample_{sigma}_MTrain_{num_samples}_losscoef_{loss_coef}'

    def forward(self, x):
        # sample latent code z from q given x.
        reconstruction = torch.zeros_like(x).to(self.device)
        for i in range(self.num_samples):
            _, output = self.trained_VAE(x)
            reconstruction += output
        reconstruction /= self.num_samples
        return self.base_classifier(reconstruction)




