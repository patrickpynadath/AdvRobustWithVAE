from Utils import torch_to_numpy, timestamp
import torch
from torch.linalg import vector_norm
from Experiments import BaseExp
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt


def get_latent_rep(vae, x):
    z_mean, _ = vae.encode(x)
    return z_mean


def get_res_hist(title, res):
    f, a = plt.subplots(1, 2, figsize=(16, 8))
    f.suptitle(title)

    ax = a[0]
    ax.set_title("L2 Differences")
    ax.hist(res['l2'], bins=20)

    ax = a[1]
    ax.set_title("Linf Differences")
    ax.hist(res['l_inf'], bins=20)
    return f


def get_norm_comparison(diff: torch.Tensor):
    # flattening along every dimension except for batch
    diff = diff.flatten(start_dim=1)
    l_2 = torch_to_numpy(vector_norm(diff, ord=2, dim=1))
    l_inf = torch_to_numpy(vector_norm(diff, ord=float('inf'), dim=1))
    return {'l2': l_2, 'l_inf': l_inf}


class VaeAdvGaussianExp(BaseExp):
    def __init__(self, training_logdir, exp_logdir, device):
        super().__init__(training_logdir, exp_logdir, device)
        transform = transforms.Compose([transforms.ToTensor()])
        root_dir = r'*/'
        self.train_set = torchvision.datasets.CIFAR10(root=root_dir,
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(root=root_dir,
                                                train=False,
                                                download=True,
                                                transform=transform)

    def input_img_comparison(self,
                             trained_vae,
                             trained_clf,
                             dataset,
                             attacker_type,
                             num_samples,
                             attack_eps,
                             num_steps):
        original_samples, attacked_samples, labels = self.get_adv_examples(trained_clf=trained_clf,
                                                                           attack_eps=attack_eps,
                                                                           adversary_type=attacker_type,
                                                                           steps=num_steps,
                                                                           num_attacks=num_samples,
                                                                           dataset_name=dataset)

        norm_constrained_gaussian = self.get_norm_constrained_noise(original_samples, norm=attack_eps)
        res = {'gauss_comp': get_norm_comparison(norm_constrained_gaussian),
               'adv_comp': get_norm_comparison(attacked_samples - original_samples)}
        return res

    def latent_code_comparison(self,
                               trained_vae,
                               trained_clf,
                               dataset,
                               attacker_type,
                               num_samples,
                               attack_eps,
                               num_steps):
        original_samples, attacked_samples, labels = self.get_adv_examples(trained_clf=trained_clf,
                                                                           attack_eps=attack_eps,
                                                                           adversary_type=attacker_type,
                                                                           steps=num_steps,
                                                                           num_attacks=num_samples,
                                                                           dataset_name=dataset)

        norm_constrained_gaussian = self.get_norm_constrained_noise(original_samples, norm=attack_eps) + original_samples
        gaussian_codes = get_latent_rep(vae=trained_vae,
                                        x=norm_constrained_gaussian)
        adv_codes = get_latent_rep(vae=trained_vae,
                                   x=attacked_samples)
        original_codes = get_latent_rep(vae=trained_vae,
                                        x=original_samples)
        res = {'gauss_comp': get_norm_comparison(gaussian_codes - original_codes),
               'adv_comp': get_norm_comparison(adv_codes - original_codes)}
        return res

    def get_norm_constrained_noise(self, original_samples, norm, ord=float('inf')):
        orig_shape = original_samples.size()
        flatten_orig = torch.flatten(original_samples, start_dim=1)

        norm_constrained_gaussian = torch.zeros_like(flatten_orig).to(self.device)
        for i in range(orig_shape[0]):
            gaussian_noise = torch.randn_like(flatten_orig[i, :])
            norm_constrained_gaussian[i, :] = (gaussian_noise * norm / torch.linalg.vector_norm(gaussian_noise, ord=ord))
        return torch.reshape(norm_constrained_gaussian, orig_shape)

    def save_ex_reconstructions(self,
                                sw: SummaryWriter,
                                trained_vae,
                                trained_clf,
                                dataset,
                                attack_eps,
                                attacker_type,
                                num_steps=8,
                                num_attacks=16):
        original_samples, adv_samples, labels = self.get_adv_examples(trained_clf=trained_clf,
                                                                      attack_eps=attack_eps,
                                                                      adversary_type=attacker_type,
                                                                      steps=num_steps,
                                                                      num_attacks=num_attacks,
                                                                      dataset_name=dataset)
        gaussian_samples = self.get_norm_constrained_noise(original_samples, norm=attack_eps) + original_samples
        original_reconstructions = trained_vae.generate(original_samples)
        adv_reconstructions = trained_vae.generate(adv_samples)
        gaussian_reconstructions = trained_vae.generate(gaussian_samples)
        sw.add_images(f"InputImages/Original", original_samples)
        sw.add_images(f"InputImages/Adversarial", adv_samples)
        sw.add_images(f"InputImages/Noise", gaussian_samples)
        sw.add_images(f"Reconstructions/Original", original_reconstructions)
        sw.add_images(f"Reconstructions/Adversarial", adv_reconstructions)
        sw.add_images(f"Reconstructions/Noise", gaussian_reconstructions)
        return

    def reconstruction_comparison(self,
                                  trained_vae,
                                  trained_clf,
                                  dataset,
                                  attacker_type,
                                  attack_eps,
                                  num_steps=8,
                                  num_attacks=1000):
        original_samples, adv_samples, labels = self.get_adv_examples(trained_clf=trained_clf,
                                                                      attack_eps=attack_eps,
                                                                      adversary_type=attacker_type,
                                                                      steps=num_steps,
                                                                      num_attacks=num_attacks,
                                                                      dataset_name=dataset)
        gaussian_samples = self.get_norm_constrained_noise(original_samples, norm=attack_eps) + original_samples
        original_reconstructions = trained_vae.generate(original_samples)
        adv_reconstructions = trained_vae.generate(adv_samples)
        gaussian_reconstructions = trained_vae.generate(gaussian_samples)
        res = {'gauss_comp': get_norm_comparison(gaussian_reconstructions - original_reconstructions),
               'adv_comp': get_norm_comparison(adv_reconstructions - original_reconstructions)}
        return res


def single_exp_loop_vae_adv(training_logdir, exp_logdir, device):
    adv_norms = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
    adv_type = 'linf'
    clf_epochs = 150
    vae_epochs = 150
    num_attacks = 1000
    exp = VaeAdvGaussianExp(training_logdir=training_logdir,
                            exp_logdir=exp_logdir,
                            device=device)
    vae = exp.get_trained_vae(batch_size=64,
                              epochs=vae_epochs,
                              vae_model='vae',
                              latent_dim=100,
                              in_channels=3)
    clf = exp.get_trained_resnet(net_depth=110,
                                 block_name='BottleNeck',
                                 batch_size=64,
                                 optimizer='sgd',
                                 lr=.1,
                                 epochs=clf_epochs,
                                 use_step_lr=True,
                                 lr_schedule_step=50,
                                 lr_schedule_gamma=.1)
    for eps in adv_norms:
        for dataset_name in ['train', 'test']:
            sw = SummaryWriter(
                log_dir=exp_logdir + f"/{timestamp()}/{dataset_name}/adv_norm_{round(eps, 3)}")
            exp.save_ex_reconstructions(sw, vae, clf, dataset_name, eps, attacker_type=adv_type)
            input_img_comparison = exp.input_img_comparison(vae, clf, dataset_name, adv_type, num_attacks, eps, 8)

            f = get_res_hist("Noise Difference for Input Images", input_img_comparison['gauss_comp'])
            sw.add_figure("Differences/InputImg/Noise", f)
            plt.close(f)
            f = get_res_hist("Adv Difference for Input Images", input_img_comparison['adv_comp'])
            sw.add_figure("Differences/InputImg/Adv", f)
            plt.close(f)

            latent_comparison = exp.latent_code_comparison(vae, clf, dataset_name, adv_type, num_attacks, eps, 8)
            f = get_res_hist("Noise Difference for Latent Codes", latent_comparison['gauss_comp'])
            sw.add_figure("Differences/Latent/Noise", f)
            plt.close(f)
            f = get_res_hist("Adv Difference for Latent Codes", latent_comparison['adv_comp'])
            sw.add_figure("Differences/Latent/Adv", f)
            plt.close(f)

            reconstruction_comparison = exp.reconstruction_comparison(vae, clf, dataset_name, adv_type, eps, 8, num_attacks=num_attacks)
            f = get_res_hist("Noise Difference for Reconstructions", reconstruction_comparison['gauss_comp'])
            sw.add_figure("Differences/Reconstruction/Noise", f)
            plt.close(f)
            f = get_res_hist("Adv Difference for Reconstructions", reconstruction_comparison['adv_comp'])
            sw.add_figure("Differences/Reconstruction/Adv", f)
            plt.close(f)
