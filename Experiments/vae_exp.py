from Models.vae import VAE
from Training.train_vae import train_vae
from Training.train_natural import NatTrainerSmoothVAE
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import tqdm
from Experiments.exp import Experiment
import os
import datetime

# TODO: make compatable with tensorboard imagedata
# purpose of this experiment is to provide empirical results  as to how VAE handles gaussian peturb
class PeturbExperiment(Experiment):
    def __init__(self, batch_size, log_dir, out_dir, device):
        super().__init__(batch_size, log_dir,
                         out_dir +"/VAE_EXP/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), device)


    def sample_noise(self, shape, var):
        return torch.randn(size = shape).to(self.device) * var

    def generate_VAE(self, kernel_num, z_size, vae_epochs=10, sigma_vae = 1):
        vae = VAE(label='vae', image_size=32, channel_num=3, kernel_num=kernel_num,
                  z_size=z_size, device=self.device, var = sigma_vae).to(self.device)
        train_vae(vae, self.trainloader, len(self.trainset), epochs=vae_epochs, batch_size=self.batch_size)
        return vae

    def get_latent_rep(self, vae: VAE, x):
        encoded = vae.encoder(x)
        z_mean, z_var = vae.q(encoded)
        z_projected = vae.project(z_mean).view(
            -1, vae.kernel_num,
            vae.feature_size,
            vae.feature_size,
        )
        return z_projected

    def get_reconstruction(self, vae : VAE, z):
        z = z.to(vae.device)
        return vae.decoder(z)

    # getting data on what results we would normally get from the VAE
    def normal_VAE_forward(self, vae : VAE, x):
        z = self.get_latent_rep(vae, x)
        reconstruction = self.get_reconstruction(vae, z)
        return z, reconstruction

    def sample_space_peturb(self, vae: VAE, x : torch.Tensor, var):
        peturb = self.sample_noise(x.size(), var)
        x += peturb
        z = self.get_latent_rep(vae, x)
        reconstruction = self.get_reconstruction(vae, z)
        return z, reconstruction

    def latent_space_peturb(self, vae : VAE, x, var):
        z = self.get_latent_rep(vae, x)
        peturb = self.sample_noise(z.size(), var)
        z += peturb
        reconstruction = self.get_reconstruction(vae, z)
        return z, reconstruction

    # helper function for getting norm data giving tensor
    def get_norm_data(self, raw_data : torch.Tensor, norm_type):
        if norm_type == 'l2':
            norm_data = [torch.norm(raw_data[j, :]).item() for j in range(len(raw_data))]
            return norm_data
        elif norm_type == 'linf' :
            norm_data = [torch.max(torch.abs(raw_data[j, :])).item() for j in range(len(raw_data))]
            return norm_data

    def generate_hist(self, data, file_name, plt_title, show=True):
        plt.title(plt_title)
        plt.hist(data)
        if show:
            plt.show()
        plt.savefig(file_name)
        return


    def norm_analysis(self, vae : VAE, noise_vars, train_set = True):
        if train_set:
            datastream = enumerate(self.trainloader)
        else:
            datastream = enumerate(self.testloader)

        # getting the baseline
        baseline = {'latent' : {'l2' : [], 'linf' : []}, 'sample' : {'l2' : [], 'linf' : []}}
        peturb_latent = [{'latent' : {'l2' : [], 'linf' : []}, 'sample' : {'l2' : [], 'linf' : []}} for _ in noise_vars]
        peturb_sample= [{'latent' : {'l2' : [], 'linf' : []}, 'sample' : {'l2' : [], 'linf' : []}} for _ in noise_vars]
        unpeturb = {'latent' : {'l2' : [], 'linf' : []}, 'sample' : {'l2' : [], 'linf' : []}}
        vae_sigma = vae.var

        for i, data in datastream:
            inputs = data[0].to(self.device)
            # getting baseline data
            z, recon = self.normal_VAE_forward(vae, inputs)
            for norm_type in ['l2', 'linf']:
                # unpeturb data
                unpeturb['latent'][norm_type] += self.get_norm_data(z, norm_type)
                unpeturb['sample'][norm_type] += self.get_norm_data(recon - inputs, norm_type)

                # baseline data
                z_ideal = self.sample_noise(z.size(), vae.var)
                baseline['latent'][norm_type] += self.get_norm_data(z_ideal, norm_type)
                baseline['sample'][norm_type] += self.get_norm_data(inputs, norm_type)
            for var_idx, var in enumerate(noise_vars):
                for norm_type in ['l2', 'linf']:
                # peturbing the original representation and putting it through the VAE
                    z, recon = self.sample_space_peturb(vae, inputs, var)
                    peturb_sample[var_idx]['latent'][norm_type] += self.get_norm_data(z, norm_type)
                    peturb_sample[var_idx]['sample'][norm_type] += self.get_norm_data(recon - inputs, norm_type)

                    z, recon = self.latent_space_peturb(vae, inputs, var)
                    peturb_latent[var_idx]['latent'][norm_type] += self.get_norm_data(z, norm_type)
                    peturb_latent[var_idx]['sample'][norm_type] += self.get_norm_data(recon - inputs, norm_type)

        for norm_type in ['l2', 'linf']:
            dataset_name = 'train' if train_set else 'test'
            # latent to sample petubr
            plt_title = f"Peturb latent rep norm analysis {norm_type} for {dataset_name}"
            file_name = f"peturb_latent_rep_{norm_type}"
            self.plt_norm_analysis(baseline, unpeturb, peturb_latent, noise_vars, norm_type, plt_title, file_name)

            plt_title = f"Peturb sample rep norm analysis {norm_type} for {dataset_name}"
            file_name = f"peturb_sample_rep_{norm_type}"
            self.plt_norm_analysis(baseline, unpeturb, peturb_sample, noise_vars, norm_type, plt_title, file_name)
        return

    def plt_norm_analysis(self, baseline, unpeturb, peturb, peturb_vars, norm_type, title, file_name):

        f, a = plt.subplots(len(peturb_vars) + 2, 2, figsize=(8, 4 * (len(peturb_vars) + 2)))
        f.suptitle(title + f'norm {norm_type}')

        for col_idx, rep_space in enumerate(['latent', 'sample']):
            ax = a[0, col_idx]
            plt_title = f'Baseline {rep_space} norm {norm_type}'
            ax.set_title(plt_title)
            ax.hist(baseline[rep_space][norm_type])

            ax = a[1, col_idx]
            plt_title = f'Unpeturb VAE {rep_space} norm {norm_type}'
            ax.set_title(plt_title)
            ax.hist(unpeturb[rep_space][norm_type])

            for var_idx, var in enumerate(peturb_vars):
                ax = a[var_idx + 2, col_idx]
                plt_title = f'Peturb with var = {var}'
                ax.set_title(plt_title)
                ax.hist(peturb[var_idx][rep_space][norm_type])

        f.savefig(self.out_dir + f"/{file_name}")
        plt.show()
        return

    # retrieving the VAE where the gradient updates for the classifier are allowed to affect the VAE paramaters
    def get_smoothVAE_vae(self, clf_epochs, vae_loss_coef, vae_epochs, lr, sigmas, m_train, smoothVAE_version):
        models, optimizers, criterions = self.get_smoothVAE_models(base_label='smoothVAE', lr=lr, sigmas=sigmas, m_train=m_train,
                                                                   vae_epochs=vae_epochs, with_VAE_grad=True, model_version=smoothVAE_version)
        smoothVAE_trainer = NatTrainerSmoothVAE(models, trainloader = self.trainloader, testloader = self.testloader, device = self.device,
                                                optimizers=optimizers, criterions = criterions, log_dir=self.log_dir,
                                                vae_loss_coef=vae_loss_coef, use_tensorboard=False)
        smoothVAE_trainer.training_loop(epochs=clf_epochs)

        vae_models = [m.trained_VAE for m in models]
        return vae_models


    def get_baseVAE_vae(self, lr, sigmas, m_train, vae_epochs, smoothVAE_version):
        models, optimizers, criterions = self.get_smoothVAE_models(base_label='smoothVAE', lr=lr, sigmas=sigmas,
                                                                   m_train=m_train,
                                                                   vae_epochs=vae_epochs, with_VAE_grad=True,
                                                                   model_version=smoothVAE_version)
        vae_models = [m.trained_VAE for m in models]
        return vae_models