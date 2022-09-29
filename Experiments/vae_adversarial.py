from Utils import torch_to_numpy
from Models import Conv_VAE
import torch
from torch.linalg import vector_norm
from Experiments import BaseExp


def get_latent_rep(vae: Conv_VAE, x):
    encoded = vae.encoder(x)
    z_mean, _ = vae.q(encoded)
    return z_mean


def get_norm_comparison(diff: torch.Tensor):
    # flattening along every dimension except for batch
    diff = diff.flatten(start_dim=1)
    l_2 = torch_to_numpy(vector_norm(diff, ord=2, dim=1))
    l_inf = torch_to_numpy(vector_norm(diff, ord='inf', dim=1))
    return {'l2': l_2, 'l_inf': l_inf}


class VaeAdvGaussianExp(BaseExp):
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

        norm_constrained_gaussian = self.get_norm_constrained_noise(original_samples, norm=attack_eps)
        gaussian_codes = get_latent_rep(vae=trained_vae,
                                        x=norm_constrained_gaussian)
        adv_codes = get_latent_rep(vae=trained_vae,
                                   x=attacked_samples)
        original_codes = get_latent_rep(vae=trained_vae,
                                        x=original_samples)
        res = {'gauss_comp': get_norm_comparison(gaussian_codes - original_codes),
               'adv_comp': get_norm_comparison(adv_codes - original_codes)}
        return res

    def get_norm_constrained_noise(self, original_samples, norm):
        norm_constrained_gaussian = torch.zeros_like(original_samples).to(self.device)
        for i in range(original_samples.size(0)):
            gaussian_noise = torch.randn_like(original_samples[i, :])
            norm_constrained_gaussian[i, :] = gaussian_noise / torch.linalg.vector_norm(gaussian_noise) * norm
        return norm_constrained_gaussian

    def reconstruction_comparison(self,
                                  trained_vae,
                                  trained_clf,
                                  dataset,
                                  attacker_type,
                                  attack_eps,
                                  num_steps,
                                  num_attacks=1000):
        original_samples, adv_samples, labels = self.get_adv_examples(trained_clf=trained_clf,
                                                                      attack_eps=attack_eps,
                                                                      adversary_type=attacker_type,
                                                                      steps=num_steps,
                                                                      num_attacks=num_attacks,
                                                                      dataset_name=dataset)
        gaussian_samples = self.get_norm_constrained_noise(original_samples, norm=attack_eps)
        original_reconstructions = trained_vae(original_samples)
        adv_reconstructions = trained_vae(adv_samples)
        gaussian_reconstructions = trained_vae(gaussian_samples)
        res = {'gauss_comp': get_norm_comparison(gaussian_reconstructions - original_reconstructions),
               'adv_comp': get_norm_comparison(adv_reconstructions - original_reconstructions)}
        return res

    def single_exp_loop(self):
        pass
