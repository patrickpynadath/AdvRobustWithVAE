from torch.utils.data import DataLoader
from Models import PixelCNN, discretized_mix_logistic_loss
from Training import train_pixel_cnn
from Experiments.base_exp import BaseExp
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def get_pxcnn_loss(px_cnn, inputs):
    losses = []
    for input_idx in range(len(inputs)):
        current_input = inputs[input_idx]
        current_input = current_input[None]
        output = px_cnn(current_input)
        loss = discretized_mix_logistic_loss(current_input, output)
        losses.append(loss.cpu().detach().item())
    return losses


def get_vae_loss(vae, inputs):
    reconstruction_losses = []
    kl_losses = []
    for input_idx in range(len(inputs)):
        current_input = inputs[input_idx]
        current_input = current_input[None]
        (mean, logvar), reconstruction = vae(current_input)
        reconstruction_loss = vae.reconstruction_loss(reconstruction, current_input)
        kl_loss = vae.kl_divergence_loss(mean, logvar)
        reconstruction_losses.append(reconstruction_loss.cpu().detach().item())
        kl_losses.append(kl_loss.cpu().detach().item())
    return {'reconstruction': reconstruction_losses, 'KL': kl_losses}


class ManifoldModelingExp(BaseExp):
    def __init__(self,
                 training_logdir,
                 exp_logdir,
                 device,
                 clf_train_set=None,
                 test_set=None):
        super().__init__(training_logdir,
                         f'{exp_logdir}/ManifoldExp',
                         device,
                         clf_train_set=clf_train_set,
                         test_set=test_set)

    def get_trained_pixel_cnn(self,
                              epochs,
                              batch_size=32):
        px_cnn = PixelCNN().to(self.device)
        train_loader = DataLoader(self.clf_train_set,
                                  batch_size=batch_size)
        train_pixel_cnn(epochs,
                        px_cnn,
                        self.device,
                        train_loader)
        return px_cnn

    def create_hist_vae_loss(self,
                             tag,
                             dataset_name,
                             natural_data,
                             attacked_data):
        sw_dir = self.exp_logdir + f"/{tag}"
        sw = SummaryWriter(log_dir=sw_dir)

        f, a = plt.subplots(2, 2, figsize=(8, 10))
        f.suptitle(f"VAE Loss Analysis for {dataset_name}")
        ax = a[0, 0]
        ax.set_title("Recon. Loss for Natural Data")
        ax.hist(natural_data['reconstruction'], bins=20)

        ax = a[0, 1]
        ax.set_title(f"Recon. Loss for Attacked Data")
        ax.hist(attacked_data['reconstruction'], bins=20)

        ax = a[1, 0]
        ax.set_title("KL Loss for Natural Data")
        ax.hist(natural_data['KL'], bins=20)

        ax = a[1, 1]
        ax.set_title("KL Loss for Attacked Data")
        ax.hist(attacked_data['KL'], bins=20)

        sw.add_figure(tag=tag, figure=f)
        plt.close(f)
        pass

    def create_hist_pxcnn_loss(self,
                               tag,
                               dataset_name,
                               natural_data,
                               attacked_data):
        sw_dir = self.exp_logdir + f"/{tag}"
        sw = SummaryWriter(log_dir=sw_dir)

        f, a = plt.subplots(1, 2, figsize=(8, 10))
        f.suptitle(f"Loss for PixelCNN on {dataset_name}")

        ax = a[0]
        ax.set_title("Loss for Natural Data")
        ax.hist(natural_data, bins=20)

        ax = a[1]
        ax.set_title("Loss for Attacked Data")
        ax.hist(attacked_data, bins=20)

        sw.add_figure(tag=tag, figure=f)
        plt.close(f)

    def single_exp_loop(self,
                        use_step_lr,
                        lr_schedule_step,
                        lr_schedule_gamma,
                        img_size=32,
                        num_channel=3,
                        kernel_num=32,
                        latent_size=100,
                        vae_beta=1,
                        vae_epochs=100,
                        pxcnn_epochs=100,
                        num_samples=1000,
                        net_depth=20,
                        clf_epochs=100,
                        clf_lr=.01,
                        clf_optimizer='adam',
                        clf_batch_size=100,
                        block_name='BasicBlock',
                        dataset_name='test',
                        adv_steps=8,
                        adv_type='l2',
                        adv_eps=2 / 255):
        vae = self.get_trained_vae(img_size, num_channel, kernel_num, latent_size, vae_beta, self.device, vae_epochs)
        px_cnn = self.get_trained_pixel_cnn(epochs=pxcnn_epochs)
        trained_resnet = self.get_trained_resnet(net_depth=net_depth,
                                                 block_name=block_name,
                                                 batch_size=clf_batch_size,
                                                 optimizer=clf_optimizer,
                                                 lr=clf_lr,
                                                 epochs=clf_epochs,
                                                 use_step_lr=use_step_lr,
                                                 lr_schedule_step=lr_schedule_step,
                                                 lr_schedule_gamma=lr_schedule_gamma)
        original_images, attacked_images, labels = self.get_adv_examples(trained_clf=trained_resnet,
                                                                         attack_eps=adv_eps,
                                                                         adversary_type=adv_type,
                                                                         steps=adv_steps,
                                                                         num_attacks=num_samples,
                                                                         dataset_name=dataset_name)
        vae_nat_data = get_vae_loss(vae, original_images)
        vae_adv_data = get_vae_loss(vae, attacked_images)
        pxcnn_loss_original = get_pxcnn_loss(px_cnn, original_images)
        pxcnn_loss_adv = get_pxcnn_loss(px_cnn, attacked_images)
        self.create_hist_vae_loss(tag=vae.label,
                                  dataset_name=dataset_name,
                                  natural_data=vae_nat_data,
                                  attacked_data=vae_adv_data)
        self.create_hist_pxcnn_loss(tag=px_cnn.label,
                                    dataset_name=dataset_name,
                                    natural_data=pxcnn_loss_original,
                                    attacked_data=pxcnn_loss_adv)
        return
