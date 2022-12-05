import torch
from torch import nn as nn
from torch.linalg import vector_norm
from torchattacks import PGD, PGDL2

from Utils import AVOID_ZERO_DIV, torch_to_numpy


def get_img_l2_norm(imgs):
    return vector_norm(imgs, dim=(1, 2, 3))


def norm_imgs_l2(imgs):
    norms = get_img_l2_norm(imgs).clamp(min=AVOID_ZERO_DIV)
    normalized_x = torch.zeros_like(imgs, device=imgs.device)
    for i in range(imgs.size(0)):
        normalized_x[i, :] = imgs[i, :] / norms[i]
    return normalized_x


def get_adv_examples(clf,
                     attack_eps,
                     adversary_type,
                     steps,
                     nat_img,
                     labels):
    if adversary_type == 'linf':
        attacker = PGD(clf, eps=attack_eps, steps=steps)
        return attacker(nat_img, labels)
    elif adversary_type == 'l2':
        attacker = PGDL2(clf, eps=attack_eps, steps=steps, alpha = (2 * attack_eps) / steps)
        return attacker(nat_img, labels)


def get_norm_constrained_noise(original_samples, norm, adv_type, device):
    if adv_type == 'l2':
        ord_type = 2
    elif adv_type == 'linf':
        ord_type = float('inf')
    orig_shape = original_samples.size()
    flatten_orig = torch.flatten(original_samples, start_dim=1)

    norm_constrained_gaussian = torch.zeros_like(flatten_orig).to(device)
    for i in range(orig_shape[0]):
        gaussian_noise = torch.randn_like(flatten_orig[i, :])
        norm_constrained_gaussian[i, :] = (
                    gaussian_noise * norm / torch.linalg.vector_norm(gaussian_noise, ord=ord_type))
    return torch.reshape(norm_constrained_gaussian, orig_shape)


def get_norm_comparison(diff: torch.Tensor, batch=True):
    # flattening along every dimension except for batch
    if batch:
        dim = 1
        diff = torch.flatten(diff, start_dim=1)
    else:
        dim = 0
        diff = torch.flatten(diff, start_dim=0)
    l_2 = torch_to_numpy(vector_norm(diff, ord=2, dim=dim))
    l_inf = torch_to_numpy(vector_norm(diff, ord=float('inf'), dim=dim))
    return {'l2': l_2, 'linf': l_inf}


class NormConstrainedAttacker:
    """ projected gradient desscent, with random initialization within the ball """
    def __init__(self, eps, model, num_iter=10):
        # define default attack parameters here:
        self.param = {'eps': eps,
                      'num_iter': num_iter,
                      'loss_fn': nn.CrossEntropyLoss()}
        self.model = model
        # parse thru the dictionary and modify user-specific params

    def generate(self, x, y):
        eps = self.param['eps']
        num_iter = self.param['num_iter']
        loss_fn = self.param['loss_fn']

        r = get_img_l2_norm(x)
        delta = torch.rand_like(x, requires_grad=True)
        delta.data = 2 * delta.detach() - 1
        delta.data = eps * norm_imgs_l2(delta.detach())
        delta.data = r * norm_imgs_l2(x + delta) - x

        for t in range(num_iter):
            self.model.zero_grad()
            loss = loss_fn(self.model(x + delta), y)
            loss.backward()

            delta_grad = delta.grad.detach()
            delta.data = delta + eps * norm_imgs_l2(delta_grad)
            # delta.data = delta + eps * delta_grad
            delta.data = r * norm_imgs_l2(x + delta) - x
            delta.grad.zero_()
        return delta.detach()
