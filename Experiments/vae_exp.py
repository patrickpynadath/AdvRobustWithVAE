from Models.vae import VAE
from Training.train_vae import train_vae
from Training.train_natural import NatTrainerSmoothVAE
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime
from Models.smoothing import SmoothVAE_Latent, SmoothVAE_Sample
from Models.simple_conv import simple_conv_net
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from sklearn.metrics import calinski_harabasz_score
import numpy as np
from Utils.utils import get_class_loaders

# purpose of this experiment is to provide empirical results  as to how VAE handles gaussian peturb
class PeturbExperiment:
    def __init__(self,
                 batch_size,
                 log_dir,
                 device):

        self.batch_size = batch_size

        transform = transforms.Compose(
            [transforms.ToTensor()])
        root_dir = r'*/'
        trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                               download=True, transform=transform)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = len(classes)
        self.trainset = trainset
        self.trainloader = trainloader
        self.testset = testset
        self.testloader = testloader
        self.device = device
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = log_dir + f"/{date_str}/"

    def sample_noise(self, shape, var):
        return torch.randn(size = shape).to(self.device) * var

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

        res_writer = SummaryWriter(log_dir= self.log_dir + vae.label)

        for norm_type in ['l2', 'linf']:
            dataset_name = 'train' if train_set else 'test'
            # latent to sample peturb
            plt_title = f"Peturb latent rep norm analysis {norm_type} for {dataset_name}"
            fig = self.plt_norm_analysis(baseline, unpeturb, peturb_latent, noise_vars, norm_type, plt_title)
            res_writer.add_figure(tag=f"peturb_latent_{norm_type}_{dataset_name}", figure=fig)
            fig.close()
            # sample to latent peturb
            plt_title = f"Peturb sample rep norm analysis {norm_type} for {dataset_name}"
            fig = self.plt_norm_analysis(baseline, unpeturb, peturb_sample, noise_vars, norm_type, plt_title)
            res_writer.add_figure(tag=f"peturb_sample_{norm_type}_{dataset_name}", figure=fig)
            fig.close()
        return

    def plt_norm_analysis(self, baseline, unpeturb, peturb, peturb_vars, norm_type, title):

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
        return f

    def get_trained_vanilla_vae(self, kernel_num, z_size, epochs):
        vae = VAE(image_size=32,
                  channel_num=3,
                  kernel_num=kernel_num,
                  z_size = z_size,
                  device = self.device)
        train_vae(vae, self.trainloader, epochs)
        return vae

    def get_trained_smoothVAE_vae(self,
                                  smoothVAE_version,
                                  kernel_num,
                                  z_size,
                                  epochs_VAE,
                                  epochs_CLF,
                                  smoothing_sigma,
                                  num_smoothing_samples,
                                  loss_coef,
                                  lr):
        vae = self.get_trained_vanilla_vae(kernel_num, z_size, epochs_VAE)
        base_clf =simple_conv_net().to(self.device)
        if smoothVAE_version == 'latent':
            smoothVAE = SmoothVAE_Latent(base_classifier=base_clf,
                                         sigma = smoothing_sigma,
                                         trained_VAE=vae,
                                         device=self.device,
                                         num_samples=num_smoothing_samples,
                                         num_classes = self.num_classes,
                                         loss_coef = loss_coef)
        elif smoothVAE_version == 'sample':
            smoothVAE = SmoothVAE_Sample(base_classifier=base_clf,
                                         sigma = smoothing_sigma,
                                         trained_VAE=vae,
                                         device=self.device,
                                         num_samples=num_smoothing_samples,
                                         num_classes = self.num_classes,
                                         loss_coef = loss_coef)

        trainer = NatTrainerSmoothVAE(model = smoothVAE,
                                      trainloader=self.trainloader,
                                      testloader=self.testloader,
                                      device = self.device,
                                      optimizer = SGD(smoothVAE.parameters(), lr=lr),
                                      criterion=CrossEntropyLoss,
                                      use_tensorboard=False, # not interested in the trained classifier -- only care about VAE
                                      log_dir = '')
        trainer.training_loop(epochs_CLF)
        return smoothVAE.trained_VAE

    def run_var_ratio(self, vae, dataset_name, num_samples_per_class):

        summary_writer = SummaryWriter(log_dir=self.log_dir + vae.label)
        if dataset_name == 'train' :
            dataset = self.trainloader.dataset
        elif dataset_name == 'test':
            dataset = self.testloader.dataset
        dataloader_dct = get_class_loaders(dataset, num_samples_per_class)
        to_concat_reps = []
        to_concat_labels = []
        for label in dataloader_dct.keys():
            loader = dataloader_dct[label]
            class_batch, class_labels = next(loader)
            to_concat_labels += [c.item() for c in class_labels]
            class_batch = class_batch.to(self.device)
            latent_reps = self.get_latent_rep(vae, class_batch)
            to_concat_reps.append(latent_reps.to('cpu').numpy())
        X = np.concatenate(to_concat_labels, axis=0)
        labels = to_concat_labels
        var_ratio = calinski_harabasz_score(X, labels)
        summary_writer.add_scalar(f"VarRatioCriteria/{dataset_name}", scalar_value=var_ratio)
        return