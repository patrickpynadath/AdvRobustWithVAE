import torch
from Models import Smooth, Conv_VAE


class SmoothVAE_Latent(Smooth):

    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 trained_VAE : Conv_VAE,
                 device : str,
                 num_samples : int,
                 num_classes : int,
                 vae_param : bool):
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
        self.main_module = torch.nn.Sequential(self.trained_VAE, self.base_classifier)
        self.label = f'SmoothVAE_Latent_{sigma}_MTrain_{num_samples}_VAEBeta_{self.trained_VAE.beta}'
        self.vae_param = vae_param


    def forward(self, x):
        """

        :param x: torch tensor of [batch_size x channel_num x width x height]
        :return: y, where y is [batch_size x num_classes], a probabiliy distribution over potential classes
        """
        encoded = self.trained_VAE.encoder(x)
        mu, log_var = self.trained_VAE.q(encoded)
        z = self.trained_VAE.z(mu, log_var)
        noise_placeholder = torch.zeros_like(z).to(self.device)
        for i in range(self.num_samples):
            noise = torch.randn_like(z).to(self.device) * self.sigma ** 2
            noise_placeholder += noise
        noise_placeholder /= self.num_samples
        z_proj = self.trained_VAE.project_z(z + noise_placeholder)
        recon = self.trained_VAE.decoder(z_proj)
        output = self.base_classifier(recon)
        return output

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        if self.vae_param:
            return self.main_module.named_parameters(prefix, recurse)
        else:
            return self.base_classifier.named_parameters(prefix, recurse)

    def parameters(self, recurse: bool = True):
        if self.vae_param:
            return self.main_module.parameters(recurse=recurse)
        else:
            return self.base_classifier.parameters(recurse=recurse)


class SmoothVAE_Sample(Smooth):

    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 trained_VAE,
                 device,
                 num_samples,
                 num_classes,
                 vae_param = False):
        """
        :param base_classifier: base classifier to apply SmoothVAE procedure to
        :param sigma: smoothing value for randomized smoothing procedure
        :param trained_VAE: VAE model
        :param device: device to move model and tensors to
        :param num_samples: number of samples to use for randomized smoothing
        :param num_classes: total possible classes
        """
        super().__init__(base_classifier, sigma, device, num_samples, num_classes)
        self.trained_VAE = trained_VAE
        self.main_module = torch.nn.Sequential(self.trained_VAE, self.base_classifier)
        self.label = f'SmoothVAE_Sample_{sigma}_MTrain_{num_samples}_VAEBeta_{self.trained_VAE.beta}'
        self.vae_param = vae_param

    def parameters(self, recurse: bool = True):
        if self.vae_param:
            return self.main_module.parameters(recurse=recurse)
        else:
            return self.base_classifier.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        if self.vae_param:
            return self.main_module.named_parameters(prefix, recurse)
        else:
            return self.base_classifier.named_parameters(prefix, recurse)


    def forward(self, x):
        """
        :param x: torch tensor of [batch_size x channel_num x width x height]
        :return: y, where y is [batch_size x num_classes], a probabiliy distribution over potential classes
        """
        reconstruction = torch.zeros_like(x).to(self.device)
        for i in range(self.num_samples):
            output = self.trained_VAE(x)[0]
            reconstruction += output
        reconstruction /= self.num_samples
        return self.base_classifier(reconstruction)


class SmoothVAE_PreProcess(Smooth):
    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 trained_VAE,
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
        self.label = self.label + f'_SmoothVAE_PreProcess_{sigma}_MTrain_{num_samples}_VAEBeta_{self.trained_VAE.beta}'

    def forward(self, x):
        encoded = self.trained_VAE.encoder(x)
        z_mean, z_var = self.trained_VAE.q(encoded)
        z = self.trained_VAE.z(z_mean, z_var)
        return self.base_classifier(z)