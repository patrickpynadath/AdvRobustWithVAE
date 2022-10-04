from abc import ABC
import random
import torch
from torch.utils.data import Dataset, DataLoader


def generate_sphere(num_samples, dim, radius):
    randn_unscaled = torch.randn(num_samples, dim)
    randn_scaled = randn_unscaled * radius / torch.linalg.vector_norm(randn_unscaled, ord=2, dim=1)
    return randn_scaled


class SphereDataSet(Dataset, ABC):
    def __init__(self,
                 num_samples,
                 dim,
                 small_r,
                 big_r,
                 off_manifold_ratio=0,
                 num_off_layers = 5):
        super().__init__()
        num_small_r = num_big_r = int(num_samples * (1 - off_manifold_ratio) / 2)
        small_r_samples = generate_sphere(num_small_r, dim, small_r)
        small_r_labels = [0 for i in range(num_small_r)]
        big_r_samples = generate_sphere(num_big_r, dim, big_r)
        big_r_labels = [1 for i in range(num_big_r)]
        cur_data = torch.cat((small_r_samples, big_r_samples), 0)
        cur_labels = small_r_labels + big_r_labels
        if off_manifold_ratio != 0:
            off_manifold_samples_per_shell = (off_manifold_ratio * num_samples) / num_off_layers
            rad_step_size = (big_r - small_r) / (num_off_layers + 1)
            midpoint = (big_r - small_r) / 2
            cur_rad = small_r
            for i in range(num_off_layers):
                cur_rad += rad_step_size
                samples = generate_sphere(off_manifold_samples_per_shell, dim, cur_rad)
                if cur_rad < midpoint:
                    labels = [0 for i in range(off_manifold_samples_per_shell)]
                else:
                    labels = [1 for i in range(off_manifold_samples_per_shell)]
                cur_labels += labels
                cur_data = torch.cat((cur_data, samples), 0)

        # randomly shuffling the dataset
        idx = [i for i in range(cur_data.size()[0])]
        random.shuffle(idx)
        self.samples = cur_data[idx, :]
        self.labels = torch.Tensor(cur_labels).reshape(-1, 1)[idx, :]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.samples[idx, :]
        labels = self.labels[idx, :]
        return {'samples' : sample,
                'labels' : labels}


def normalize(x):
    x_norm = torch.linalg.vector_norm(x)
    normalized_x = x / x_norm
    return normalized_x


def generate(model, r, x, y, eps, num_iter, loss_fn):
    delta = torch.rand_like(x, requires_grad=True)
    delta.data = 2 * delta.detach() - 1
    delta.data = eps * normalize(delta.detach())
    delta.data = r * normalize(x + delta) - x

    for t in range(num_iter):
        model.zero_grad()
        loss = loss_fn(model(x + delta), y)
        loss.backward()

        delta_grad = delta.grad.detach()
        delta.data = delta + eps * normalize(delta_grad)
        # delta.data = delta + eps * delta_grad
        delta.data = r * normalize(x + delta) - x
        delta.grad.zero_()
    return delta.detach()
