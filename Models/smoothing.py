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
        """

        :param base_classifier: base classifier to apply SmoothVAE procedure to
        :param sigma: smoothing value for randomized smoothing procedure
        :param trained_VAE: VAE model
        :param device: device to move model and tensors to
        :param num_samples: number of samples to use for randomized smoothing
        :param num_classes: total possible classes
        :param loss_coef: the value to use to weight the VAE component of the loss function
        """
        super().__init__(base_classifier, sigma, device, num_samples, num_classes)
        self.trained_VAE = trained_VAE
        self.loss_coef = loss_coef
        self.label = f'SmoothVAE_Latent_{sigma}_MTrain_{num_samples}_losscoef_{loss_coef}'


    def forward(self, x):
        """

        :param x: torch tensor of [batch_size x channel_num x width x height]
        :return: y, where y is [batch_size x num_classes], a probabiliy distribution over potential classes
        """
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

    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 trained_VAE: VAE,
                 device,
                 num_samples,
                 num_classes,
                 loss_coef):
        """
        :param base_classifier: base classifier to apply SmoothVAE procedure to
        :param sigma: smoothing value for randomized smoothing procedure
        :param trained_VAE: VAE model
        :param device: device to move model and tensors to
        :param num_samples: number of samples to use for randomized smoothing
        :param num_classes: total possible classes
        :param loss_coef: the value to use to weight the VAE component of the loss function
        """
        super().__init__(base_classifier, sigma, device, num_samples, num_classes)

        self.trained_VAE = trained_VAE
        self.loss_coef = loss_coef
        self.label = f'SmoothVAE_Sample_{sigma}_MTrain_{num_samples}_losscoef_{loss_coef}'

    def forward(self, x):
        """
        :param x: torch tensor of [batch_size x channel_num x width x height]
        :return: y, where y is [batch_size x num_classes], a probabiliy distribution over potential classes
        """
        reconstruction = torch.zeros_like(x).to(self.device)
        for i in range(self.num_samples):
            _, output = self.trained_VAE(x)
            reconstruction += output
        reconstruction /= self.num_samples
        return self.base_classifier(reconstruction)


class SmoothVAE_PreProcess(Smooth):
    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 trained_VAE: VAE,
                 device,
                 num_samples,
                 num_classes):
        """
        :param base_classifier: base classifier to apply SmoothVAE procedure to
        :param sigma: smoothing value for randomized smoothing procedure
        :param trained_VAE: VAE model
        :param device: device to move model and tensors to
        :param num_samples: number of samples to use for randomized smoothing
        :param num_classes: total possible classes
        :param loss_coef: the value to use to weight the VAE component of the loss function
        """
        super().__init__(base_classifier, sigma, device, num_samples, num_classes)

        self.trained_VAE = trained_VAE
        self.label = f'SmoothVAE_PreProcess_{sigma}_MTrain_{num_samples}'

    def forward(self, x):
        encoded = self.trained_VAE.encoder(x)
        z_mean, z_var = self.trained_VAE.q(encoded)
        print(z_mean.size())
        z_projected = self.trained_VAE.project(z_mean).view(
            -1, self.trained_VAE.kernel_num,
            self.trained_VAE.feature_size,
            self.trained_VAE.feature_size,
        )
        print(z_projected.size())
        return self.base_classifier(z_projected)




