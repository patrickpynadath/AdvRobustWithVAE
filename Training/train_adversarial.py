import torch
import torch.nn as nn
from tqdm import tqdm
from Utils.utils import requires_grad_
from typing import List
from Adversarial.attacks import PGD_L2
from torchattacks import PGD
from torch.utils.tensorboard import SummaryWriter

# train the classifier, returning an array of the training loss and accuracy
class AdvTrainer:
    def __init__(self, models: List[nn.Module], trainloader, testloader, device, optimizers, criterions, eps, attack_steps, attack_type, log_dir, use_tensorboard=False):
        self.models = models
        self.num_models = len(models)
        self.optimizers = optimizers
        self.use_tensorboard = use_tensorboard
        self.criterions = criterions
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.eps = eps
        self.attack_steps = attack_steps
        self.attack_type = attack_type
        self.log_dir = log_dir


    def _train_step(self, data):
        inputs, labels = data[0], data[1]

        # noticed that code from attacks doesn't work without this cast
        labels = labels.type(torch.LongTensor)
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        step_data = []

        # getting the adversarial examples
        attacks = torch.zeros_like(inputs).to(self.device)
        for i, model in enumerate(self.models):


            # getting the correct optimizer
            optimizer = self.optimizers[i]
            criterion = self.criterions[i]
            # zero the parameter gradients
            optimizer.zero_grad()

            # not accidently attaching gradients to model params during this step
            requires_grad_(model, False)
            model.eval()
            if self.attack_type == 'l2':
                attacker = PGD_L2(model, steps=self.attack_steps, max_norm=self.eps, device=self.device)
                attacks.add(attacker.attack(inputs, labels))
            elif self.attack_type == 'linf':
                attacker = PGD(model, self.eps, steps=self.attack_steps)
                attacks.add(attacker(inputs, labels))

            model.train()
            requires_grad_(model, True)
            outputs_attack = model(attacks)
            outputs_clean = model(inputs)

            # calculating training accuracy
            pred_attack = torch.argmax(outputs_attack, dim=1)
            pred_clean = torch.argmax(outputs_clean, dim=1)

            num_correct_adv = sum([1 if pred_attack[i].item() == labels[i].item() else 0 for i in range(len(inputs))])
            num_correct_nat = sum([1 if pred_clean[i].item() == labels[i].item() else 0 for i in range(len(inputs))])

            # forward + backward + optimize
            loss = criterion(outputs_attack, labels)
            loss.backward()
            optimizer.step()

            # looking at gradient information
            parameters = model.parameters()
            grad_info = torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters])

            grad_mean = torch.mean(grad_info).item()
            model_data = {'grad mean' : grad_mean, 'clean score' : num_correct_nat, 'adv score' : num_correct_adv, 'loss' : loss.item()}
            step_data.append(model_data)
            attacks.zero_()

        return step_data

    def val_step(self):

        total_loss = [0 for _ in range(self.num_models)]
        adv_score = [0 for _ in range(self.num_models)]
        nat_score = [0 for _ in range(self.num_models)]
        val_total = 0
        step_data = []
        for i, data in enumerate(self.testloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            attacks = torch.zeros_like(inputs).to(self.device)
            val_total += len(inputs)

            for j, model in enumerate(self.models):
                criterion = self.criterions[j]
                requires_grad_(model, False)
                model.eval()

                if self.attack_type == 'l2':
                    attacker = PGD_L2(model, steps=self.attack_steps, max_norm=self.eps, device=self.device)
                    attacks.add(attacker.attack(inputs, labels))
                elif self.attack_type == 'linf':
                    attacker = PGD(model, self.eps, steps=self.attack_steps)
                    attacks.add(attacker(inputs, labels))

                # looking at val loss
                outputs_attack = model(attacks)
                outputs_clean = model(inputs)
                total_loss[j] += criterion(outputs_attack, labels).item()

                # looking at val accuracy
                pred_attack = torch.argmax(outputs_attack, dim=1)
                pred_clean = torch.argmax(outputs_clean, dim=1)
                adv_score[j] += sum(
                    [1 if pred_attack[k].item() == labels[k].item() else 0 for k in range(len(inputs))])
                nat_score[j] += sum([1 if pred_clean[i].item() == labels[i].item() else 0 for i in range(len(inputs))])


        for j in range(self.num_models):
            adv_acc = adv_score[j]/val_total
            clean_acc = nat_score[j]/val_total
            val_loss = total_loss[j]
            model_data = {'adv acc' : adv_acc, 'clean acc' : clean_acc, 'val loss' : val_loss}
            step_data.append(model_data)

        return step_data


    def training_loop(self, epochs):

        adv_train_loss_data = [[] for _ in range(self.num_models)]
        adv_train_acc_attack_data = [[] for _ in range(self.num_models)]
        adv_train_acc_clean_data = [[] for _ in range(self.num_models)]

        adv_val_loss_data = [[] for _ in range(self.num_models)]
        adv_val_acc_attack_data = [[] for _ in range(self.num_models)]
        adv_val_acc_clean_data = [[] for _ in range(self.num_models)]

        # instantiate the writers for each model here
        writers = []
        for model in self.models:
            log_dir = self.log_dir + f"/{model.label}"
            writer = SummaryWriter(log_dir = log_dir)
            writers.append(writer)

        for epoch in range(epochs):
            total_loss = [0 for _ in range(self.num_models)]
            num_correct_adv = [0 for _ in range(self.num_models)]
            num_correct_clean = [0 for _ in range(self.num_models)]
            grad_mean = [[] for _ in range(self.num_models)]
            train_total = 0
            with tqdm(enumerate(self.trainloader), total=len(self.trainloader)) as datastream:
                for i, data in datastream:
                    datastream.set_description(f"Epoch {epoch + 1} / {epochs} | Iteration {i + 1} / {len(self.trainloader)}")
                    step_data = self._train_step(data)
                    for j in range(self.num_models):

                        # need to adjust how it unpacks the values
                        grad_mean[j].append(step_data[j]['grad mean'])
                        num_correct_clean[j] += step_data[j]['clean score']
                        num_correct_adv[j] += step_data[j]['adv score']
                        total_loss[j] += step_data[j]['loss']
                    train_total += len(data[1])
            val_step_data = self.val_step()


            for j in range(self.num_models):
                train_adv_acc = num_correct_adv[j] / train_total
                train_clean_acc = num_correct_clean[j] / train_total
                train_loss = total_loss[j]

                val_adv_acc = val_step_data[j]['adv acc']
                val_clean_acc = val_step_data[j]['clean acc']
                val_loss = val_step_data[j]['val loss']

                if self.use_tensorboard:
                    writer = writers[j]
                    writer.add_scalar('AdvTraining/Loss/Train', train_loss, epoch)
                    writer.add_scalar('AdvTraining/Loss/Val', val_loss, epoch)
                    writer.add_scalar('AdvTraining/CleanAcc/Train', train_clean_acc, epoch)
                    writer.add_scalar('AdvTraining/AdvAcc/Train', train_adv_acc, epoch)
                    writer.add_scalar('AdvTraining/CleanAcc/Val', val_clean_acc, epoch)
                    writer.add_scalar('AdvTraining/AdvAcc/Val', val_adv_acc, epoch)
                    writer.add_scalar('AdvTraining/Grad/AvgGradNorm', sum(grad_mean[j]) / len(grad_mean[j]), epoch)
                    writer.flush()

                adv_train_acc_clean_data[j].append(train_clean_acc)
                adv_train_acc_attack_data[j].append(train_adv_acc)
                adv_train_loss_data[j].append(train_loss)
                adv_val_acc_clean_data[j].append(val_clean_acc)
                adv_val_acc_attack_data[j].append(val_adv_acc)
                adv_val_loss_data[j].append(val_loss)


        return adv_train_acc_attack_data, adv_train_acc_clean_data, adv_train_loss_data, adv_val_acc_attack_data, \
               adv_val_acc_clean_data, adv_val_loss_data

