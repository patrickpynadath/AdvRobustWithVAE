import os
import os.path
import torch
import yaml
from torch import nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import datetime
from torch.linalg import vector_norm

# code borrowed from https://github.com/SashaMalysheva/Pytorch-VAE/blob/master/utils.py
# given a dictionary of accuracies from adv rob raw test, returns in the form of a pandas dataframe
from torchattacks import PGD, PGDL2

from Models import AE, VAE, VQVAE_NSVQ, ResNet

AVOID_ZERO_DIV = 1e-12

def adv_raw_accs_to_df(adv_accs, fname):
    return


def adv_raw_accs_to_graph(adv_accs, fname):
    return


def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch


def requires_grad_(model: torch.nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def get_cifar_sets():
    root_dir = r'../'
    # transformations from https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/datasets.py
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                            download=True, transform=train_transform)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                           download=True, transform=test_transform)

    return trainset, testset


# given a dataloader, returns a dictionary where the key represents a class label, and the value is a list of the sample idx
def get_label_idx(dataset):
    label_idx_dct = {}
    for i in range(len(dataset)):
        label = dataset[i][1]
        if label in label_idx_dct:
            label_idx_dct[label].append(i)
        else:
            label_idx_dct[label] = [i]
    return label_idx_dct


def get_class_subsets(dataset):
    label_dct = get_label_idx(dataset)
    dataloader_dct = {}
    for i in range(len(label_dct)):
        class_subset = torch.utils.data.Subset(dataset, label_dct[i])
        dataloader_dct[i] = class_subset
    return dataloader_dct


def accuracies_to_dct(nat_acc, adv_accs, attack_norms, attack_type):
    res = {'Nat Acc': nat_acc}
    for i, acc in enumerate(adv_accs):
        key = f'hparam/AdvAcc_{attack_type}_{round(attack_norms[i], 4)}'
        res[key] = acc
    return res


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def torch_to_numpy(x: torch.Tensor):
    return x.cpu().detach().numpy()


# get the norm for batched data
def get_img_l2_norm(imgs):
    return vector_norm(imgs, dim=(1, 2, 3))

# normalize image data
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


def load_models(device):
    model_dct = get_untrained_models(device)

    for key in ['ae', 'vae', 'vqvae']:
        gen_model = model_dct[key]
        gen_path = f'PretrainedModels/{key.upper()}_{gen_model.res_layers}Res{gen_model.latent_size}Lat'
        gen_model.load_state_dict(torch.load(gen_path))
        model_dct[key] = gen_model
        resnet_ensemble = model_dct[f"resnet_{key}"]
        ens_path = f'PretrainedModels/Resnet{resnet_ensemble.depth}_{key.upper()}_{gen_model.res_layers}Res{gen_model.latent_size}Lat'
        resnet_ensemble.load_state_dict(torch.load(ens_path))
        model_dct[f"resnet_{key}"] = resnet_ensemble

    resnet = model_dct['resnet']
    resnet_path = f"PretrainedModels/Resnet{resnet.depth}"
    resnet.load_state_dict(torch.load(resnet_path))
    model_dct['resnet'] = resnet

    resnet_smooth = model_dct['resnetSmooth']
    resnet_smooth_path = f'PretrainedModels/Resnet110_SmoothSigma_.25'
    resnet_smooth.load_state_dict(torch.load(resnet_smooth_path))
    model_dct['resnetSmooth'] = resnet_smooth

    return model_dct


def get_untrained_models(device):
    with open("./Params/model_params.yaml", "r") as stream:
        model_params = yaml.safe_load(stream)

    # retrieving model parameters
    num_hiddens = int(model_params['enc_dec_params']['num_hiddens'])
    num_residual_hiddens = int(model_params['enc_dec_params']['num_residual_hiddens'])
    num_residual_layers = int(model_params['enc_dec_params']['num_residual_layers'])
    latent_size = int(model_params['enc_dec_params']['latent_size'])

    vae_beta = float(model_params['vae_params']['beta'])

    vqvae_embedding_dim = int(model_params['vqvae_params']['embedding_dim'])
    vqvae_num_embedding = int(model_params['vqvae_params']['num_embedding'])
    batch_size = int(model_params['training_param_gen']['batch_size'])
    training_samples = int(model_params['training_param_gen']['training_samples'])
    num_epochs_gen = int(model_params['training_param_gen']['epochs'])

    resnet_depth = int(model_params['resnet_params']['depth'])
    resnet_classes = int(model_params['resnet_params']['num_classes'])
    resnet_block = model_params['resnet_params']['block_name']

    ae = AE(num_hiddens, num_residual_layers, num_residual_hiddens, latent_size)
    vae = VAE(num_hiddens, num_residual_layers, num_residual_hiddens, latent_size, device, vae_beta)
    vqvae = VQVAE_NSVQ(num_hiddens, num_residual_layers, num_residual_hiddens, vqvae_num_embedding, vqvae_embedding_dim,
                       batch_size, num_epochs_gen, device, training_samples)
    resnet = ResNet(resnet_depth, resnet_classes, resnet_block, device=device)

    resnet_smooth = ResNet(resnet_depth, resnet_classes, resnet_block, device=device)
    return {'ae': ae,
            'vae': vae,
            'vqvae': vqvae,
            'resnet': resnet,
            'resnetSmooth': resnet_smooth,
            'resnet_ae': ResNet(resnet_depth, resnet_classes, resnet_block, device=device),
            'resnet_vae': ResNet(resnet_depth, resnet_classes, resnet_block, device=device),
            'resnet_vqvae': ResNet(resnet_depth, resnet_classes, resnet_block, device=device),
            'training_param_gen': model_params['training_param_gen'],
            'training_param_clf': model_params['training_param_clf']}


def get_latent_code_ae(ae: AE, x):
    return ae.encode(x)


def get_latent_code_vae(vae: VAE, x):
    return vae.encode(x)[0]


def get_latent_code_vqvae(vqvae: VQVAE_NSVQ, x):
    return vqvae.encode(x)[0]


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
