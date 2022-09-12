import os
import os.path
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# code borrowed from https://github.com/SashaMalysheva/Pytorch-VAE/blob/master/utils.py


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

def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def get_cifar_sets():
    root_dir = r'../'
    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                           download=True, transform=transform)

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
    res = {'Nat Acc' : nat_acc}
    for i, acc in enumerate(adv_accs):
        key = f'hparam/AdvAcc_{attack_type}_{round(attack_norms[i], 4)}'
        res[key] = acc
    return res


