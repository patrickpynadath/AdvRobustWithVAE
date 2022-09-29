import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torchvision
from Adversarial import PGD_L2
from torchattacks import PGD
from Models import SmoothVAE_Sample, SmoothVAE_Latent, ResNet, Smooth
from Training import NatTrainer, VAETrainer


class BaseExp:
    def __init__(self,
                 training_logdir,
                 exp_logdir,
                 device,
                 train_set=None,
                 test_set=None):

        self.training_logdir = training_logdir
        self.exp_logdir = exp_logdir
        self.device = device
        transform = transforms.Compose(
            [transforms.ToTensor()])
        root_dir = r'*/'
        if not train_set:
            train_set = torchvision.datasets.CIFAR10(root=root_dir,
                                                     train=True,
                                                     download=True,
                                                     transform=transform)
        if not test_set:
            test_set = torchvision.datasets.CIFAR10(root=root_dir,
                                                    train=False,
                                                    download=True,
                                                    transform=transform)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.num_classes = len(classes)
        self.train_set = train_set
        self.test_set = test_set
        self.device = device

    def get_loaders(self, batch_size):
        train_loader = DataLoader(self.train_set, batch_size, shuffle=True)
        test_loader = DataLoader(self.test_set, batch_size, shuffle=False)
        return train_loader, test_loader

    def get_trained_vae(self,
                        batch_size,
                        epochs,
                        vae_model,
                        **kwargs):
        train_loader, test_loader = self.get_loaders(batch_size)
        vae_trainer = VAETrainer(self.device,
                                 True,
                                 train_loader,
                                 test_loader,
                                 self.training_logdir,
                                 batch_size,
                                 vae_model,
                                 **kwargs)
        vae_trainer.training_loop(epochs)
        return vae_trainer.model

    def get_trained_resnet(self,
                           net_depth,
                           block_name,
                           batch_size,
                           optimizer,
                           lr,
                           epochs,
                           use_step_lr,
                           lr_schedule_step,
                           lr_schedule_gamma,
                           ):
        resnet = ResNet(depth=net_depth,
                        num_classes=self.num_classes,
                        block_name=block_name)
        train_loader, test_loader = self.get_loaders(batch_size)
        clf_trainer = NatTrainer(model=resnet,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 device=self.device,
                                 optimizer=optimizer,
                                 lr=lr,
                                 log_dir=self.training_logdir,
                                 use_tensorboard=True,
                                 use_step_lr=use_step_lr,
                                 lr_schedule_step=lr_schedule_step,
                                 lr_schedule_gamma=lr_schedule_gamma)
        clf_trainer.training_loop(epochs)
        return resnet

    def get_trained_smooth_resnet(self,
                                  net_depth,
                                  block_name,
                                  m_train,
                                  batch_size,
                                  optimizer,
                                  lr,
                                  epochs,
                                  smoothing_sigma,
                                  use_step_lr,
                                  lr_schedule_step,
                                  lr_schedule_gamma,
                                  ):
        resnet = ResNet(depth=net_depth,
                        num_classes=self.num_classes,
                        block_name=block_name)
        smooth_resnet = Smooth(base_classifier=resnet,
                               sigma=smoothing_sigma,
                               device=self.device,
                               num_samples=m_train,
                               num_classes=self.num_classes)
        train_loader, test_loader = self.get_loaders(batch_size)
        clf_trainer = NatTrainer(model=smooth_resnet,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 device=self.device,
                                 optimizer=optimizer,
                                 lr=lr,
                                 log_dir=self.training_logdir,
                                 use_tensorboard=True,
                                 use_step_lr=use_step_lr,
                                 lr_schedule_step=lr_schedule_step,
                                 lr_schedule_gamma=lr_schedule_gamma)
        clf_trainer.training_loop(epochs)
        return smooth_resnet

    def get_trained_smooth_vae_resnet(self,
                                      net_depth,
                                      block_name,
                                      img_size,
                                      num_channel,
                                      vae_kern_num,
                                      m_train,
                                      batch_size_clf,
                                      batch_size_vae,
                                      vae_latent_size,
                                      vae_beta,
                                      optimizer,
                                      lr_clf,
                                      epochs_clf,
                                      epochs_vae,
                                      smoothing_sigma,
                                      smooth_vae_version,
                                      use_step_lr,
                                      lr_schedule_step,
                                      lr_schedule_gamma,
                                      use_vae_param=False):
        resnet = ResNet(depth=net_depth,
                        num_classes=self.num_classes,
                        block_name=block_name)
        trained_vae = self.get_trained_vae(batch_size=batch_size_vae,
                                           img_size=img_size,
                                           num_channel=num_channel,
                                           kernel_num=vae_kern_num,
                                           latent_size=vae_latent_size,
                                           vae_beta=vae_beta,
                                           epochs=epochs_vae)
        assert smooth_vae_version in ['sample', 'latent']
        if smooth_vae_version == 'sample':
            smooth_vae = SmoothVAE_Sample(base_classifier=resnet,
                                          sigma=smoothing_sigma,
                                          trained_VAE=trained_vae,
                                          device=self.device,
                                          num_samples=m_train,
                                          vae_param=use_vae_param,
                                          num_classes=self.num_classes)
        elif smooth_vae_version == 'latent':
            smooth_vae = SmoothVAE_Latent(base_classifier=resnet,
                                          sigma=smoothing_sigma,
                                          trained_VAE=trained_vae,
                                          device=self.device,
                                          num_samples=m_train,
                                          vae_param=use_vae_param,
                                          num_classes=self.num_classes)
        train_loader, test_loader = self.get_loaders(batch_size_clf)
        clf_trainer = NatTrainer(model=smooth_vae,
                                 train_loader=train_loader,
                                 test_loader=test_loader,
                                 device=self.device,
                                 optimizer=optimizer,
                                 lr=lr_clf,
                                 log_dir=self.training_logdir,
                                 use_tensorboard=True,
                                 use_step_lr=use_step_lr,
                                 lr_schedule_gamma=lr_schedule_gamma,
                                 lr_schedule_step=lr_schedule_step)
        clf_trainer.training_loop(epochs_clf)
        return smooth_vae

    def get_adv_examples(self,
                         trained_clf,
                         attack_eps,
                         adversary_type,
                         steps,
                         num_attacks=1000,
                         dataset_name='train'):
        dataset = None
        assert dataset_name in ['train', 'test']
        assert adversary_type in ['l2', 'linf']
        if dataset_name == 'train':
            dataset = self.train_set
        elif dataset_name == 'test':
            dataset = self.test_set
        samples_idx = torch.randint(low=0, high=len(dataset), size=(num_attacks,))
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
            attacker = PGD_L2(trained_clf, steps=steps, max_norm=attack_eps, device=self.device)
            attacks.add(attacker.attack(original_im, labels))
        elif adversary_type == 'linf':
            attacker = PGD(trained_clf, attack_eps, steps)
            tmp = attacker(original_im, labels)
            attacks += tmp
        return original_im, attacks, labels

    def eval_clf_clean(self,
                       model,
                       batch_size=100):
        num_correct = 0
        test_loader = DataLoader(dataset=self.test_set,
                                 batch_size=batch_size)
        model.eval()
        total = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            raw_res = model(inputs)
            base_pred = torch.argmax(raw_res, dim=1)
            num_correct += sum([1 if base_pred[i].item() == labels[i].item() else 0 for i in range(len(base_pred))])
            total += len(inputs)
        nat_acc = num_correct / total
        return nat_acc

    def eval_clf_adv(self,
                     model,
                     adversary_type,
                     attack_eps_value,
                     steps,
                     num_attacks,
                     dataset_name):
        original_samples, attacks, labels = self.get_adv_examples(trained_clf=model,
                                                                  attack_eps=attack_eps_value,
                                                                  steps=steps,
                                                                  num_attacks=num_attacks,
                                                                  dataset_name=dataset_name,
                                                                  adversary_type=adversary_type)
        base_pred = torch.argmax(model(attacks), dim=1)
        score = sum(
            [1 if base_pred[i].item() == labels[i].item() else 0 for i in range(num_attacks)]) / num_attacks
        return score
