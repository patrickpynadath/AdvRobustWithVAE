from Models import GenClf
from torchattacks import PGD, PGDL2
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from Utils import get_cifar_sets, timestamp
from Experiments import get_norm_constrained_noise
from Experiments.helper_functions import get_adv_examples
from tqdm import tqdm

class AdversarialTrainer:

    def __init__(self,
                 model,
                 attacker_type,
                 attack_eps,
                 device,
                 log_dir,
                 attacker_steps,
                 warmup_epochs = 50,
                 batch_size = 64,
                 use_tensorboard = False,
                 lr = .01):
        self.model = model
        self.attacker_type = attacker_type
        self.attack_eps = attack_eps
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.attacker_steps = attacker_steps
        self.optim = SGD(self.model.parameters(), lr=lr)
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.warmup_epochs = 50
        self.criterion = nn.CrossEntropyLoss()
        train_set, test_set = get_cifar_sets()
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def train_step(self, batch, eps):
        self.optim.zero_grad()
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        adv_images = get_adv_examples(clf = self.model,
                                      attack_eps=eps,
                                      adversary_type=self.attacker_type,
                                      steps=self.attacker_steps,
                                      nat_img=imgs,
                                      labels = labels)
        # getting the adversarial loss and adversarial score
        adv_outputs = self.model(adv_images)
        adv_pred = torch.argmax(adv_outputs, dim=1)
        adv_score = sum([1 if adv_pred[i].item() == labels[i].item() else 0 for i in range(len(adv_pred))])
        adv_loss = self.criterion(adv_outputs, labels)
        adv_loss.backward()
        self.optim.step()
        return {'adv loss' : adv_loss.item(),
                'adv score' : adv_score}

    def val_loop(self):
        # need to do for both adv and natural
        self.model.eval()
        adv_loss_total = 0
        nat_loss_total = 0

        adv_score_total = 0
        nat_score_total = 0

        total = 0
        for i, batch in enumerate(self.test_loader):
            imgs, labels = batch
            total += len(labels)
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            adv_imgs = get_adv_examples(self.model,
                                        self.attack_eps,
                                        self.attacker_type,
                                        self.attacker_steps,
                                        imgs,
                                        labels)
            adv_outputs = self.model(adv_imgs)
            adv_pred = torch.argmax(adv_outputs, dim=1)
            adv_loss = self.criterion(adv_outputs, labels)
            adv_loss_total += adv_loss.item()
            adv_score = sum([1 if adv_pred[i].item() == labels[i].item() else 0 for i in range(len(labels))])
            adv_score_total += adv_score

            nat_outputs = self.model(imgs)
            nat_pred = torch.argmax(nat_outputs, dim=1)
            nat_loss = self.criterion(nat_outputs, labels)
            nat_loss_total += nat_loss.item()
            nat_score = sum([1 if nat_pred[i].item() == labels[i].item() else 0 for i in range(len(labels))])
            nat_score_total += nat_score

        nat_acc = nat_score_total / total
        adv_acc = adv_score_total / total
        return {'adv loss' : adv_loss_total,
                'adv acc' : adv_acc,
                'nat loss' : nat_loss_total,
                'nat acc' : nat_acc}

    def training_loop(self, epochs):
        writer = SummaryWriter(log_dir=self.log_dir + f'/{self.model.label}_{timestamp()}/')
        for epoch in range(epochs):
            total_loss = 0
            num_correct = 0
            train_total = 0
            if epoch < self.warmup_epochs:
                eps = epoch / self.warmup_epochs * self.attack_eps
            else:
                eps = self.attack_eps
            with tqdm(enumerate(self.train_loader), total=len(self.train_loader)) as datastream:
                for i, data in datastream:
                    datastream.set_description(
                        f"Epoch {epoch + 1} / {epochs} | Iteration {i + 1} / {len(self.train_loader)}")
                    train_total += len(data[1])
                    train_step_data = self.train_step(data, eps)
                    total_loss += train_step_data['adv loss']
                    num_correct += train_step_data['adv score']
            val_step = self.val_loop()
            train_acc = num_correct / train_total
            train_loss = total_loss
            if self.use_tensorboard:
                writer.add_scalar('AdvTraining/Train/AdvAccuracy', train_acc, epoch)
                writer.add_scalar('AdvTraining/Train/AdvLoss', train_loss, epoch)
                writer.add_scalar('AdvTraining/Val/AdvAccuracy', val_step['adv acc'], epoch)
                writer.add_scalar('AdvTraining/Val/NatAccuracy', val_step['nat acc'], epoch)
                writer.add_scalar('AdvTraining/Val/AdvLoss', val_step['adv loss'], epoch)
                writer.add_scalar('AdvTraining/Val/NatLoss', val_step['nat loss'], epoch)
                writer.flush()
        return


class AdversarialTrainerEnsemble(AdversarialTrainer):
    # gen_loss_fn must take the generative model, the adversarial inputs, and the untouched inputs
    def __init__(self,
                 model,
                 attacker_type,
                 attack_eps,
                 device,
                 log_dir,
                 attacker_steps,
                 to_optimize,
                 gen_loss_fn,
                 batch_size = 64,
                 use_tensorboard = False,
                 lr = .01):
        super(AdversarialTrainerEnsemble, self).__init__(model,
                                                         attacker_type,
                                                         attack_eps,
                                                         device,
                                                         log_dir,
                                                         attacker_steps,
                                                         batch_size = batch_size,
                                                         use_tensorboard = use_tensorboard,
                                                         lr = lr)
        if to_optimize == 'gen':
            self.optim = SGD(self.model.gen_parameters(), lr=lr)
        elif to_optimize == 'clf':
            self.optim = SGD(self.model.clf_parameters(), lr=lr)
        elif to_optimize == 'ens':
            self.optim = SGD(self.model.parameters(), lr=lr)
        self.to_optimize = to_optimize
        self.gen_loss_fn = gen_loss_fn


        # can be clf, gen, or ensemble

    def train_step(self, batch):
        self.optim.zero_grad()
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        adv_images = get_adv_examples(clf=self.model,
                                      attack_eps=self.attack_eps,
                                      adversary_type=self.attacker_type,
                                      steps=self.attacker_steps,
                                      nat_img=imgs,
                                      labels=labels)
        # getting the adversarial loss and adversarial score
        adv_outputs = self.model(adv_images)
        adv_pred = torch.argmax(adv_outputs, dim=1)
        adv_score = sum([1 if adv_pred[i].item() == labels[i].item() else 0 for i in range(len(adv_pred))])
        adv_loss = self.criterion(adv_outputs, labels)
        if self.to_optimize == 'ensemble' or self.to_optimize == 'clf':
            adv_loss.backward()
            self.optim.step()
        elif self.to_optimize == 'gen':
            gen_loss = self.gen_loss_fn(self.model.gen_model, imgs, adv_images)
            gen_loss.backward()
            self.optim.step()
        return {'adv loss': adv_loss.item(),
                'adv score': adv_score}




