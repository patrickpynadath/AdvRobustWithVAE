import torch
import torch.nn as nn
from tqdm import tqdm
from Utils.utils import requires_grad_
from typing import List
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Models.smoothing import SmoothVAE_Sample, SmoothVAE_Latent
from torch.optim import Optimizer

class NatTrainer:
    def __init__(self,
                 model : nn.Module,
                 trainloader : DataLoader,
                 testloader : DataLoader,
                 device : str,
                 optimizer : Optimizer,
                 criterion,
                 log_dir : str,
                 use_tensorboard=False):
        """
        :param model : maps from [batch x channel x height x width] to [batch x num_classes]
        :param trainloader : torch DataLoader for iterating through training batches
        :param testloader : torch DataLoader for iterating through testing batches
        :param device: indicates which device to store tensors on, either 'cpu' or 'cuda'
        :param optimizer: Optimizer object from torch.optim, updates model weights
        :param criterion: loss function from torch.nn
        :param log_dir: where to point the SummaryWriter for storing training logs
        :param use_tensorboard: switch for using tensorboard
        """
        self.model = model
        self.use_tensorboard = use_tensorboard
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.log_dir = log_dir
        self.optimizer = optimizer
        self.criterion = criterion

    def _train_step(self, data : torch.Tensor):
        """
        :param data: torch.Tensor of [batch_size x channel x height x width]
        :return: dictionary where key, value pairs are metrics to be stored for batch step
        """
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        requires_grad_(self.model, True)
        self.model.train()
        optimizer = self.optimizer
        criterion = self.criterion
        optimizer.zero_grad()
        outputs = self.model(inputs)
        pred = torch.argmax(outputs, dim=1)
        batch_score = sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(inputs))])

        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        parameters = self.model.parameters()
        norm_type = 2
        grad_info = torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters])
        grad_mean = torch.mean(grad_info).item()

        step_data = {'grad mean' : grad_mean, 'loss' : batch_loss.item(), 'score' : batch_score}
        return step_data


    def val_step(self):
        """

        :return: dictionary where key, value pairs are metrics for validation step to be stored
        """
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                total_samples += len(inputs)
                requires_grad_(self.model, False)
                self.model.eval()
                criterion = self.criterion

                # generate model outputs, get loss
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # get accuracy values
                pred = torch.argmax(outputs, dim=1)
                num_correct = sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(inputs))])

                # update statistics
                batch_loss = loss.item()
                total_loss += batch_loss
                total_correct += num_correct


        acc = total_correct/total_samples
        step_data = {'val acc' : acc, 'val loss': total_loss}

        return step_data

    def training_loop(self, epochs : int):
        """
        runs a training loop for given epochs
        :param epochs: number of epochs to train self.classifier
        :return:
        """
        writer = SummaryWriter(log_dir=self.log_dir + f'/{self.model.label}/')

        for epoch in range(epochs):  # loop over the dataset multiple times

            # initializing metrics to track
            total_loss = 0
            num_correct = 0
            train_total = 0

            with tqdm(enumerate(self.trainloader), total=len(self.trainloader)) as datastream:

                grad_mean = []
                for i, data in datastream:
                    datastream.set_description(
                        f"Epoch {epoch + 1} / {epochs} | Iteration {i + 1} / {len(self.trainloader)}")
                    train_total += len(data[1])
                    train_step_data = self._train_step(data)

                    total_loss += train_step_data['loss']
                    num_correct += train_step_data['score']
                    grad_mean.append(train_step_data['grad mean'])

            # setting up metrics for validation
            val_step_data = self.val_step()

            train_acc = num_correct/train_total
            train_loss = total_loss
            val_acc = val_step_data['val acc']
            val_loss = val_step_data['val loss']
            model_grad_mean = sum(grad_mean)/len(grad_mean)

            if self.use_tensorboard:
                writer.add_scalar('NatTraining/Accuracy/Train', train_acc, epoch)
                writer.add_scalar('NatTraining/Accuracy/Val', val_acc, epoch)
                writer.add_scalar('NatTraining/Loss/Train', train_loss, epoch)
                writer.add_scalar('NatTraining/Loss/Val', val_loss, epoch)
                writer.add_scalar('NatTraining/Grad/AvgGradNorm', model_grad_mean, epoch)
                writer.flush()
        return


class NatTrainerSmoothVAE(NatTrainer):
    def __init__(self,
                 model,
                 trainloader : DataLoader,
                 testloader : DataLoader,
                 device : str,
                 optimizer : Optimizer,
                 criterion,
                 log_dir : str,
                 vae_loss_coef : float,
                 use_tensorboard=False):
        """

        :param model: SmoothVAE model, can be SmoothVAE_sample or SmoothVAE_latent
        :param trainloader: Dataloader object for iterating through training batches
        :param testloader: Dataloader object for iterating through testing batches
        :param device: indicates what device to send all tensor data to
        :param optimizer: Optimizer for updating model weights
        :param criterion: Loss function
        :param log_dir: directory to store the training metrics, used by SummaryWriter
        :param vae_loss_coef: how much to weight loss from VAE objective
        :param use_tensorboard: switch for displaying/saving data via tensorboard
        """
        super().__init__(model, trainloader, testloader, device, optimizer, criterion, log_dir, use_tensorboard)
        self.vae_loss_coef = vae_loss_coef

    def _train_step(self, data):
        """
        Goes through a single parameter update given batch data
        :param data: tensor of dimensions [batch_size x num_channels x width x height]
        :return: dct of metrics to keep track of in main training loop
        """
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        step_data = []

        model = self.model

        requires_grad_(model, True)
        model.train()

        optimizer = self.optimizer
        criterion = self.criterion
        optimizer.zero_grad()
        outputs = model(inputs)
        pred = torch.argmax(outputs, dim=1)
        batch_score = sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(inputs))])

        # calculating classification loss
        classification_loss = criterion(outputs, labels)

        # calculating the loss from the VAE
        if model.loss_coef != 0:
            (mean, logvar), inputs_reconstructed = model.trained_VAE(inputs)
            reconstruction_loss = model.trained_VAE.reconstruction_loss(inputs_reconstructed, inputs)
            kl_divergence_loss = model.trained_VAE.kl_divergence_loss(mean, logvar)

            # combining for total loss
            batch_loss = classification_loss + (reconstruction_loss + kl_divergence_loss) * self.model.loss_coef
        else:
            batch_loss = classification_loss
        batch_loss.backward()
        optimizer.step()



        parameters = model.parameters()
        norm_type = 2
        grad_info = torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters])
        grad_mean = torch.mean(grad_info).item()

        model_data = {'grad mean' : grad_mean, 'loss' : batch_loss.item(), 'score' : batch_score}
        return model_data

    def val_step(self):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, data in enumerate(self.testloader):
                model = self.model
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                total_samples += len(inputs)
                requires_grad_(model, False)
                model.eval()
                criterion = self.criterion
                # generate model outputs, get loss
                outputs = model(inputs)
                classification_loss = criterion(outputs, labels)
                if model.loss_coef != 0:
                    (mean, logvar), inputs_reconstructed = model.trained_VAE(inputs)
                    reconstruction_loss = model.trained_VAE.reconstruction_loss(inputs_reconstructed, inputs)
                    kl_divergence_loss = model.trained_VAE.kl_divergence_loss(mean, logvar)
                    batch_loss = classification_loss + (kl_divergence_loss + reconstruction_loss) * self.model.loss_coef
                else:
                    batch_loss = classification_loss
                # get accuracy values
                pred = torch.argmax(outputs, dim=1)
                num_correct = sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(inputs))])

                # update statistics
                total_loss += batch_loss
                total_correct += num_correct
        acc = total_correct / total_samples
        step_data = {'val acc': acc, 'val loss': total_loss}
        return step_data
