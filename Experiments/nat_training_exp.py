from Training.train_natural import NatTrainer, NatTrainerSmoothVAE
from Training.train_vae import train_vae
from Tests.classifier_test import ClassifierTest
from Models.simple_conv import simple_conv_net
from Models.smoothing import Smooth, SmoothVAE_Latent, SmoothVAE_Sample
from Models.vae import VAE

import datetime
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim




class Adv_Robustness_NaturalTraining:
    def __init__(self,
                 training_logdir, # directory where tensorboard logs for training will go, gets written by NatTrainer
                 hyperparam_logdir, # directory where hyperparam data will be written
                 lr,
                 batch_size,
                 device):
        """

        :param training_logdir: directory to put training metrics, gets passed to NatTrainer object
        :param hyperparam_logdir: directory to put data for different hyperparams, written directly in this class
        :param lr: learning rate for all classifiers trained
        :param batch_size: batch size for training
        :param device: device to send models and tensors to
        """


        # getting the training and test loaders
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.training_logdir = training_logdir + f"/{date_str}/"
        self.hyperparam_logdir = hyperparam_logdir + f"/{date_str}/"
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
        self.lr = lr


    def adv_rob_baseclf(self,
                        clf_epochs,
                        adv_type,
                        adv_norms,
                        adv_steps,
                        num_attacks):
        base_clf = simple_conv_net().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(base_clf.parameters(), lr=self.lr)
        trainer = NatTrainer(model = base_clf,
                             trainloader= self.trainloader,
                             testloader= self.testloader,
                             device = self.device,
                             optimizer = optimizer,
                             criterion = criterion,
                             log_dir = self.training_logdir,
                             use_tensorboard=True)
        trainer.training_loop(clf_epochs)

        adv_tester = ClassifierTest(model = base_clf,
                                    testloader = self.testloader,
                                    device = self.device,
                                    batch_size = self.batch_size)
        nat_acc = adv_tester.test_clean()
        adv_accs = adv_tester.test_adv(adversary_type=adv_type,
                                       attack_eps_values=adv_norms,
                                       steps = adv_steps,
                                       num_attacks=num_attacks)
        return nat_acc, adv_accs

    def adv_rob_smoothclf(self,
                          clf_epochs,
                          smoothing_sigma,
                          smoothing_num_samples,
                          adv_type,
                          adv_norms,
                          adv_steps,
                          num_attacks):

        base_clf = simple_conv_net()
        smooth_clf = Smooth(base_classifier=base_clf,
                            sigma = smoothing_sigma,
                            device = self.device,
                            num_samples= smoothing_num_samples)
        optimizer = optim.SGD(smooth_clf.parameters(), lr = self.lr)
        criterion = nn.CrossEntropyLoss()
        trainer = NatTrainer(model = smooth_clf,
                             trainloader = self.trainloader,
                             testloader= self.testloader,
                             device = self.device,
                             criterion = criterion,
                             optimizer = optimizer,
                             log_dir = self.training_logdir,
                             use_tensorboard=True)
        trainer.training_loop(clf_epochs)
        adv_tester = ClassifierTest(model = smooth_clf,
                                    testloader=self.testloader,
                                    device = self.device,
                                    batch_size= self.batch_size)
        nat_acc = adv_tester.test_clean()
        adv_accs = adv_tester.test_adv(adversary_type=adv_type,
                                       attack_eps_values=adv_norms,
                                       steps = adv_steps,
                                       num_attacks=num_attacks)
        return nat_acc, adv_accs

    # need to get VAE as well, so need to also pass those hyperparam in
    def adv_rob_smoothvae_clf(self,
                              clf_epochs,
                              smoothingVAE_sigma,
                              smoothing_num_samples,
                              smoothVAE_version,
                              vae_loss_coef,
                              vae_img_size,
                              vae_channel_num,
                              vae_kern_num,
                              vae_z_size,
                              vae_epochs,
                              with_vae_grad,
                              adv_type,
                              adv_norms,
                              adv_steps,
                              num_attacks):

        base_clf = simple_conv_net()
        vae = VAE(image_size=vae_img_size,
                  channel_num=vae_channel_num,
                  kernel_num=vae_kern_num,
                  z_size=vae_z_size,
                  device=self.device)
        if vae_epochs != 0:
            train_vae(model=vae,
                      data_loader=self.trainloader,
                      len_dataset= len(self.trainloader.dataset),
                      epochs=vae_epochs)
        if smoothVAE_version =='latent':
            smoothVAE_clf = SmoothVAE_Latent(base_classifier=base_clf,
                                             sigma = smoothingVAE_sigma,
                                             trained_VAE=vae,
                                             device=self.device,
                                             num_samples = smoothing_num_samples)
        elif smoothVAE_version == 'sample':
            smoothVAE_clf = SmoothVAE_Sample(base_classifier=base_clf,
                                             sigma=smoothingVAE_sigma,
                                             trained_VAE=vae,
                                             device=self.device,
                                             num_samples=smoothing_num_samples)

        if with_vae_grad:
            optimizer = optim.SGD(smoothVAE_clf.parameters(), lr=self.lr)
        else:
            optimizer = optim.SGD(smoothVAE_clf.base_classifier.parameters(), lr = self.lr)

        criterion = nn.CrossEntropyLoss()
        trainer = NatTrainerSmoothVAE(model = smoothVAE_clf,
                                      trainloader=self.trainloader,
                                      testloader=self.testloader,
                                      device=self.device,
                                      optimizer=optimizer,
                                      criterion = criterion,
                                      log_dir=self.training_logdir,
                                      vae_loss_coef=vae_loss_coef,
                                      use_tensorboard=True)
        trainer.training_loop(clf_epochs)
        adv_tester = ClassifierTest(model = smoothVAE_clf,
                                    testloader=self.testloader,
                                    device=self.device,
                                    batch_size=self.batch_size)
        nat_acc = adv_tester.test_clean()
        adv_accs = adv_tester.test_adv(adversary_type=adv_type,
                                       attack_eps_values=adv_norms,
                                       steps=adv_steps,
                                       num_attacks=num_attacks)
        return nat_acc, adv_accs

