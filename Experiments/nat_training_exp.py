from Training.train_natural import NatTrainer, NatTrainerSmoothVAE
from Training.train_vae import train_vae
from Tests.classifier_test import ClassifierTest
from Models import simple_conv_net, simple_classifier, \
    pixelcnn_classifier, resnet_cifar, resnet, Smooth, \
    SmoothVAE_Latent, SmoothVAE_Sample, SmoothVAE_PreProcess, vae_models, PixelCNN

from Training.train_pixelcnn import train_pixel_cnn

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
        label = 'resnet50'
        base_clf = resnet(depth=110, num_classes=10).to(self.device)
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
        trainer.training_loop(clf_epochs, label)

        adv_tester = ClassifierTest(model = base_clf,
                                    testloader = self.testloader,
                                    device = self.device,
                                    batch_size = self.batch_size)
        nat_acc = adv_tester.test_clean()
        adv_accs = adv_tester.test_adv(adversary_type=adv_type,
                                       attack_eps_values=adv_norms,
                                       steps = adv_steps,
                                       num_attacks=num_attacks)
        return nat_acc, adv_accs, label

    def adv_rob_smoothclf(self,
                          clf_epochs,
                          smoothing_sigma,
                          smoothing_num_samples,
                          adv_type,
                          adv_norms,
                          adv_steps,
                          num_attacks):
        """
        :param clf_epochs: epochs to train classifier with
        :param smoothing_sigma: smoothing value for randomized smoothing procedure
        :param smoothing_num_samples: number of samples to use for randomized smoothing procedure
        :param adv_type: 'l2' or 'linf'
        :param adv_norms: list of max norms for PGD attack
        :param adv_steps: number of steps to use for PGD attack
        :param num_attacks: number of adversarial examples to evaluate trained model against
        :return: natural accuracy of model, list of adversarial robustness, and label of model
        """
        label= f"resnet110_smooth_sigma_{round(smoothing_sigma, 3)}"
        base_clf = resnet(depth=110, num_classes=10).to(self.device)
        smooth_clf = Smooth(base_classifier=base_clf,
                            sigma = smoothing_sigma,
                            device = self.device,
                            num_samples= smoothing_num_samples,
                            num_classes=self.num_classes)
        optimizer = optim.SGD(smooth_clf.base_classifier.parameters(), lr = self.lr)
        criterion = nn.CrossEntropyLoss()
        trainer = NatTrainer(model = smooth_clf,
                             trainloader = self.trainloader,
                             testloader= self.testloader,
                             device = self.device,
                             criterion = criterion,
                             optimizer = optimizer,
                             log_dir = self.training_logdir,
                             use_tensorboard=True)
        trainer.training_loop(clf_epochs, label= f"resnet110_smooth_sigma_{round(smoothing_sigma, 3)}")
        adv_tester = ClassifierTest(model = smooth_clf,
                                    testloader=self.testloader,
                                    device = self.device,
                                    batch_size= self.batch_size)
        nat_acc = adv_tester.test_clean()
        adv_accs = adv_tester.test_adv(adversary_type=adv_type,
                                       attack_eps_values=adv_norms,
                                       steps = adv_steps,
                                       num_attacks=num_attacks)
        return nat_acc, adv_accs, label

    def adv_rob_smoothvae_preprocess(self,
                                      clf_epochs,
                                      smoothing_num_samples,
                                      vae_img_size,
                                      vae_channel_num,
                                      vae_kern_num,
                                      vae_z_size,
                                      vae_epochs,
                                      adv_type,
                                      adv_norms,
                                      adv_steps,
                                      num_attacks,
                                      vae_beta=1):
        base_clf = resnet(depth=110, num_classes=10).to(self.device)
        label = f"resnet110_smooth_sigma_VAE_PreProcess"
        VAE = vae_models['VampVAE']
        vae = VAE(in_channels=3, latent1_dim=10, latent2_dim=10, img_size=32)
        if vae_epochs != 0:
            train_vae(model=vae,
                      data_loader=self.trainloader,
                      epochs=vae_epochs)
        smoothVAE_clf = SmoothVAE_PreProcess(base_classifier=base_clf,
                                             sigma=0,
                                             trained_VAE=vae,
                                             device=self.device,
                                             num_samples=smoothing_num_samples,
                                             num_classes=self.num_classes)
        optimizer = optim.SGD(params=base_clf.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        trainer = NatTrainer(model=smoothVAE_clf,
                                      trainloader=self.trainloader,
                                      testloader=self.testloader,
                                      device=self.device,
                                      optimizer=optimizer,
                                      criterion=criterion,
                                      log_dir=self.training_logdir,
                                      use_tensorboard=True)
        trainer.training_loop(clf_epochs, label)
        adv_tester = ClassifierTest(model=smoothVAE_clf,
                                    testloader=self.testloader,
                                    device=self.device,
                                    batch_size=self.batch_size)
        nat_acc = adv_tester.test_clean()
        adv_accs = adv_tester.test_adv(adversary_type=adv_type,
                                       attack_eps_values=adv_norms,
                                       steps=adv_steps,
                                       num_attacks=num_attacks)
        return nat_acc, adv_accs, smoothVAE_clf.label

    # need to get VAE as well, so need to also pass those hyperparam in
    def adv_rob_smoothvae_clf(self,
                              clf_epochs,
                              smoothingVAE_sigma,
                              smoothing_num_samples,
                              smoothVAE_version,
                              vae_loss_coef,
                              vae_epochs,
                              adv_type,
                              adv_norms,
                              adv_steps,
                              num_attacks,
                              vae_beta=1):
        """
        :param clf_epochs: epochs to use for classifier
        :param smoothingVAE_sigma: smoothing value for SmoothVAE
        :param smoothing_num_samples: number of samples to use for smoothing
        :param smoothVAE_version: 'latent' or 'sample'
        :param vae_loss_coef: determines how much to weight VAE component of loss function
        :param vae_img_size: dimensions for VAE input
        :param vae_channel_num: number of channels for VAE input
        :param vae_kern_num: number of kernels to use for VAE input
        :param vae_z_size: dimension of latent space
        :param vae_epochs: epochs to train VAE model
        :param with_vae_grad: switch for allowing VAE param to be updated during backprop
        :param adv_type: 'l2' or 'linf'
        :param adv_norms: list of max norms to use for PGD attack
        :param adv_steps: steps for PGD attack
        :param num_attacks: how many attacks to evaluate model against
        :return: natural accuracy, list of adversarial robustness, and model label
        """

        base_clf = resnet(depth=110, num_classes=10).to(self.device)
        label = f"resnet110_smoothVAE_{smoothVAE_version}_sigma_{smoothingVAE_sigma}_VAE_beta_{vae_beta}"
        VAE = vae_models['VampVAE']
        vae = VAE(in_channels=3, latent1_dim=10, latent2_dim=10, img_size=32)
        if vae_epochs != 0:
            train_vae(model=vae,
                      data_loader=self.trainloader,
                      epochs=vae_epochs)
        if smoothVAE_version =='latent':
            smoothVAE_clf = SmoothVAE_Latent(base_classifier=base_clf,
                                             sigma = smoothingVAE_sigma,
                                             trained_VAE=vae,
                                             device=self.device,
                                             num_samples = smoothing_num_samples,
                                             num_classes=self.num_classes,
                                             loss_coef=vae_loss_coef)
        elif smoothVAE_version == 'sample':
            smoothVAE_clf = SmoothVAE_Sample(base_classifier=base_clf,
                                             sigma=smoothingVAE_sigma,
                                             trained_VAE=vae,
                                             device=self.device,
                                             num_samples=smoothing_num_samples,
                                             num_classes=self.num_classes,
                                             loss_coef=vae_loss_coef)


        optimizer = optim.SGD(smoothVAE_clf.base_classifier.parameters(), lr = self.lr)

        criterion = nn.CrossEntropyLoss()
        trainer = NatTrainerSmoothVAE(model = smoothVAE_clf,
                                      trainloader=self.trainloader,
                                      testloader=self.testloader,
                                      device=self.device,
                                      optimizer=optimizer,
                                      criterion = criterion,
                                      log_dir=self.training_logdir,
                                      use_tensorboard=True)
        trainer.training_loop(clf_epochs, label)
        adv_tester = ClassifierTest(model = smoothVAE_clf,
                                    testloader=self.testloader,
                                    device=self.device,
                                    batch_size=self.batch_size)
        nat_acc = adv_tester.test_clean()
        adv_accs = adv_tester.test_adv(adversary_type=adv_type,
                                       attack_eps_values=adv_norms,
                                       steps=adv_steps,
                                       num_attacks=num_attacks)
        return nat_acc, adv_accs, label

    def adv_rob_pixelcnn_clf(self, epochs, adv_type, adv_norms, adv_steps, num_attacks):
        pxcnn = PixelCNN().to(device=self.device)
        train_pixel_cnn(1, pxcnn, self.device, self.trainloader)
        clf = pixelcnn_classifier('pxl_cnn_clf_poc', pxcnn, self.device)
        clf_trainer = NatTrainer(model = clf,
                                 trainloader=self.trainloader,
                                 testloader=self.testloader,
                                 device=self.device,
                                 optimizer=optim.SGD(params=clf.classifier.parameters(), lr=self.lr),
                                 criterion=nn.CrossEntropyLoss(),
                                 log_dir=self.training_logdir,
                                 use_tensorboard=True)

        clf_trainer.training_loop(50)
        adv_tester = ClassifierTest(model=clf,
                                    testloader=self.testloader,
                                    device=self.device,
                                    batch_size=self.batch_size)
        nat_acc = adv_tester.test_clean()
        adv_accs = adv_tester.test_adv(adversary_type=adv_type,
                                       attack_eps_values=adv_norms,
                                       steps=adv_steps,
                                       num_attacks=num_attacks)
        return nat_acc, adv_accs, clf.label

