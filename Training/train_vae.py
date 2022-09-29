from torch import optim
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from Models import Conv_VAE
from torch.utils.tensorboard import SummaryWriter
from Utils import timestamp, get_cifar_sets
from torchvision.utils import make_grid

# source: https://github.com/SashaMalysheva/Pytorch-VAE

class Vanilla_VAE_Trainer:
    def __init__(self,
                 model: Conv_VAE,
                 logdir: object,
                 batch_size: object = 32,
                 lr: object = 3e-04,
                 weight_decay: object = 1e-5,
                 device: object = 'cpu',
                 use_tensorboard: object = True,
                 trainloader: object = None,
                 testloader: object = None) -> object:
        self.use_tensorboard = use_tensorboard
        self.model = model
        self.logdir = logdir
        self.batch_size = batch_size
        self.lr = lr
        self.device= device
        self.weight_decay = weight_decay
        trainset, testset = get_cifar_sets()
        if not trainloader:
            self.trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
        if not testloader:
            self.testloader = DataLoader(testset, batch_size = batch_size, shuffle=False)


    def _train_step(self, data, optimizer):
        inputs, labels = data
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        optimizer.zero_grad()
        reconstruction = self.model(inputs)
        mean, logvar = self.model.get_mean_logvar(inputs)
        reconstruction_loss = self.model.reconstruction_loss(reconstruction, inputs)
        kl_divergence_loss = self.model.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss - self.model.beta * kl_divergence_loss

        # backprop gradients from the loss
        total_loss.backward()
        optimizer.step()
        return {'recon_loss' : reconstruction_loss.item(),
                'kl_loss' : kl_divergence_loss.item(),
                'total_loss' : total_loss.item()}

    def training_loop(self, num_epochs, num_sample_img):
        model = self.model
        model.train()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        writer = None
        if self.use_tensorboard:
            writer = SummaryWriter(log_dir=self.logdir + f'/{model.label}_{timestamp()}/')

        for epoch in range(num_epochs):
            datastream = tqdm(enumerate(self.trainloader), total=len(self.trainloader),  position=0, leave=True)
            for batch_idx, batch in datastream:
                step_res = self._train_step(batch, optimizer)
                recon_loss = step_res['recon_loss']
                kl_loss = step_res['kl_loss']
                total_loss = step_res['total_loss']

                # tqdm loading bar
                datastream.set_description((
                    'epoch: {epoch} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'loss => '
                    'total: {total_loss:.4f} | '
                    'recon: {recon:.4f} | '
                    'kl: {kl:.4f} | '
                ).format(
                    epoch=epoch,
                    trained=batch_idx * len(batch[0]),
                    total=len(self.trainloader.dataset),
                    progress=(100. * batch_idx / len(self.trainloader)),
                    total_loss=total_loss,
                    recon=recon_loss,
                    kl=kl_loss
                ))
                if writer:
                    writer.add_scalar("Loss/KL", kl_loss, epoch)
                    writer.add_scalar("Loss/reconstruction", recon_loss, epoch)
                    writer.add_scalar("Loss/total", total_loss, epoch)
                    sampled_imgs_train = self.sample_reconstructions(mode='train', num_img=num_sample_img)
                    writer.add_images("Generated/training_reconstruction", sampled_imgs_train)
                    sampled_imgs_test = self.sample_reconstructions(mode='test', num_img=num_sample_img)
                    writer.add_images("Generated/test_reconstruction", sampled_imgs_test)
                    sampled_imgs_rand = self.sample_reconstructions('generate', num_sample_img)
                    writer.add_images("Generated/random_samples", sampled_imgs_rand)

    def sample_reconstructions(self, mode, num_img=25):
        assert mode in ['train', 'test', 'generate']
        if mode == 'train':
            dataset = self.trainloader.dataset
            sampled_idx = torch.randint(low=0, high=len(dataset), size=(num_img,))
            sampled_imgs = self.model(dataset[sampled_idx].to(self.device))
            return sampled_imgs
        elif mode == 'test':
            dataset = self.testloader.dataset
            sampled_idx = torch.randint(low=0, high=len(dataset), size=(num_img,))
            sampled_imgs = self.model(dataset[sampled_idx].to(self.device))
            return sampled_imgs
        elif mode == 'generate':
            return self.model.sample(size=num_img)




