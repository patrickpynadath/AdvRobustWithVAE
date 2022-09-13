import torch
from Utils import get_cifar_sets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from Adversarial import PGD_L2
import torchattacks
from Models import VAE, simple_conv_net
from Training import train_vae, NatTrainer
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class ManifoldModelingExp:

    def __init__(self,
                 result_dir,
                 lr,
                 batch_size,
                 device):
        trainset, testset = get_cifar_sets()
        self.trainset = trainset
        self.testset = testset
        self.result_dir = result_dir
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def get_trained_vae(self):
        pass

    def get_trained_pixelnn(self):
        pass

    def get_adv_examples(self, trained_classifier, attack_eps, adversary_type, steps, num_attacks = 1000, dataset_name = 'train'):

        if dataset_name == 'train':
            dataset = self.trainset
        elif dataset_name == 'test':
            dataset = self.testset
        samples_idx = np.random.randint(low=0, high=len(dataset), size=num_attacks)
        original_im_dim = tuple([num_attacks]) + dataset[0][0].size()
        original_im = torch.zeros(original_im_dim)
        labels = torch.zeros(size=(num_attacks,))
        for i, idx in enumerate(samples_idx):
            original_im[i, :] = dataset[idx][0]
            labels[i] = dataset[idx][1]
        labels = labels.type(torch.LongTensor)
        original_im, labels = original_im.to(self.device), labels.to(self.device)
        attacks = torch.zeros_like(original_im).to(self.device)
        if adversary_type == 'l2':
            attacker = PGD_L2(trained_classifier, steps=steps, max_norm=attack_eps, device=self.device)
            attacks.add(attacker.attack(original_im, labels))
        elif adversary_type == 'linf':
            attacker = torchattacks.PGD(trained_classifier, attack_eps, steps)
            attacks.add(attacker(original_im, labels).to(self.device))
        return original_im, attacks

    def get_vae_loss(self, vae, inputs):
        reconstruction_losses = []
        kl_losses = []
        for input_idx in range(len(inputs)):
            current_input = inputs[input_idx]
            current_input = current_input[None]
            (mean, logvar), reconstruction = vae(current_input)
            reconstruction_loss = vae.reconstruction_loss(reconstruction, current_input)
            kl_loss = vae.kl_divergence_loss(mean, logvar)
            reconstruction_losses.append(reconstruction_loss)
            kl_losses.append(kl_loss)
        return {'reconstruction' : reconstruction_losses, 'KL' : kl_losses}


    def get_trained_vanilla_vae(self, kernel_num, z_size, epochs):
        vae = VAE(image_size=32,
                  channel_num=3,
                  kernel_num=kernel_num,
                  z_size = z_size,
                  device = self.device)
        vae = vae.to(self.device)
        train_vae(vae, DataLoader(self.trainset, self.batch_size), epochs)
        return vae

    def get_trained_clf(self, clf_lr, clf_epochs):
        clf = simple_conv_net()
        trainloader = DataLoader(self.trainset, batch_size=100)
        testloader = DataLoader(self.testset, batch_size=100)
        trainer = NatTrainer(clf, trainloader, testloader, self.device, SGD(clf.parameters(), clf_lr), CrossEntropyLoss(), log_dir=self.result_dir)
        trainer.training_loop(clf_epochs)
        return clf


    def create_hist_vae_loss(self, tag, dataset_name, natural_data, attacked_data):



        sw = SummaryWriter(log_dir=self.result_dir)
        # make the histograms
        # how do I return it?
        f, a = plt.subplots(2, 2)
        f.suptitle(f"VAE Loss Analysis for {dataset_name}")
        ax = a[0, 0]
        ax.set_title("Recon. Loss for Natural Data")
        ax.hist(natural_data['reconstruction'])

        ax = a[0, 1]
        ax.set_title(f"Recon. Loss for Attacked Data")
        ax.hist(attacked_data['reconstruction'])

        ax = a[1, 0]
        ax.set_title("KL Loss for Natural Data")
        ax.hist(natural_data['KL'])

        ax = a[1, 1]
        ax.set_title("KL Loss for Attacked Data")
        ax.hist(attacked_data['KL'])

        sw.add_figure(tag=tag, figure=f)
        plt.close(f)
        pass
