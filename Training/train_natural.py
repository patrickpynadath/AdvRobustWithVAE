import torch
import torch.nn as nn
from tqdm import tqdm
from Utils.utils import requires_grad_
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from Utils import timestamp


class NatTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: str,
                 optimizer: str,
                 lr: float,
                 log_dir: str,
                 use_tensorboard=False,
                 use_step_lr=False,
                 lr_schedule_step=1,
                 lr_schedule_gamma=1,
                 smooth = False,
                 noise_sd = 0):
        """
        :param model : maps from [batch x channel x height x width] to [batch x num_classes]
        :param train_loader : torch DataLoader for iterating through training batches
        :param test_loader : torch DataLoader for iterating through testing batches
        :param device: indicates which device to store tensors on, either 'cpu' or 'cuda'
        :param optimizer: Optimizer object from torch.optim, updates model weights
        :param log_dir: where to point the SummaryWriter for storing training logs
        :param use_tensorboard: switch for using tensorboard
        :param use_step_lr:
        """
        self.model = model.to(device)
        self.use_tensorboard = use_tensorboard
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.log_dir = log_dir
        self.smooth = smooth
        self.noise_sd = noise_sd
        assert optimizer in ['sgd', 'adam']
        if optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(),
                                 lr=lr)
        elif optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                  lr=lr)
        self.use_step_lr = use_step_lr
        if use_step_lr:
            self.lr_scheduler = StepLR(self.optimizer,
                                       step_size=lr_schedule_step,
                                       gamma=lr_schedule_gamma)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, data: torch.Tensor):
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
        if self.smooth:
            inputs = inputs + torch.randn_like(inputs, device='cuda') * self.noise_sd
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

        step_data = {'grad mean': grad_mean, 'loss': batch_loss.item(), 'score': batch_score}
        return step_data

    def val_step(self):
        """

        :return: dictionary where key, value pairs are metrics for validation step to be stored
        """
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                total_samples += len(inputs)
                requires_grad_(self.model, False)
                self.model.eval()
                criterion = self.criterion

                # generate model outputs, get loss
                if self.smooth:
                    inputs = inputs + torch.randn_like(inputs, device=self.device) * self.noise_sd
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # get accuracy values
                pred = torch.argmax(outputs, dim=1)
                num_correct = sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(inputs))])

                # update statistics
                batch_loss = loss.item()
                total_loss += batch_loss
                total_correct += num_correct

        acc = total_correct / total_samples
        step_data = {'val acc': acc, 'val loss': total_loss}

        return step_data

    def training_loop(self, epochs: int):
        """
        runs a training loop for given epochs
        :param epochs: number of epochs to train self.classifier
        :return:
        """
        writer = SummaryWriter(log_dir=self.log_dir + f'/{self.model.label}_{timestamp()}/')

        for epoch in range(epochs):  # loop over the dataset multiple times

            # initializing metrics to track
            total_loss = 0
            num_correct = 0
            train_total = 0

            with tqdm(enumerate(self.train_loader), total=len(self.train_loader)) as datastream:

                grad_mean = []
                for i, data in datastream:
                    datastream.set_description(
                        f"Epoch {epoch + 1} / {epochs} | Iteration {i + 1} / {len(self.train_loader)}")
                    train_total += len(data[1])
                    train_step_data = self.train_step(data)

                    total_loss += train_step_data['loss']
                    num_correct += train_step_data['score']
                    grad_mean.append(train_step_data['grad mean'])
            self.lr_scheduler.step()
            # setting up metrics for validation
            val_step_data = self.val_step()

            train_acc = num_correct / train_total
            train_loss = total_loss
            val_acc = val_step_data['val acc']
            val_loss = val_step_data['val loss']
            model_grad_mean = sum(grad_mean) / len(grad_mean)

            if self.use_tensorboard:
                writer.add_scalar('NatTraining/Accuracy/Train', train_acc, epoch)
                writer.add_scalar('NatTraining/Accuracy/Val', val_acc, epoch)
                writer.add_scalar('NatTraining/Loss/Train', train_loss, epoch)
                writer.add_scalar('NatTraining/Loss/Val', val_loss, epoch)
                writer.add_scalar('NatTraining/Grad/AvgGradNorm', model_grad_mean, epoch)
                writer.flush()
        return


