import torch
from Models import ResNet, vae_models, VQVAE_CLF, Smooth, VAE_CLF
from Utils import get_cifar_sets
from torch.utils.data import DataLoader
import copy
from torchattacks import PGD, PGDL2
from tqdm import tqdm
import pickle


def run_adv_robust():
    device = 'cuda'
    resnet_base = get_base_resnet(device)
    vqvae_clf = get_vqvae_clf(device)
    vae_clf = get_vae_clf(device)
    smoothing_sigmas = [.1, .25, .5]
    smooth_models = get_resnet_smooth(device, smoothing_sigmas, resnet_base)

    _, test_dataset = get_cifar_sets()
    test_loader = DataLoader(test_dataset, batch_size=32)
    total_samples = len(test_dataset)

    linf_eps = [1/255, 2/255, 5/255, 10/255]
    l2_eps = [.5, 1, 1.5, 2]

    models = {'Base-Resnet': resnet_base,
              'VQVAE-Resnet(ensemble)': vqvae_clf,
              'VQVAE-Resnet(clf)': vqvae_clf.base_classifier,
              'VAE-Resnet(ensemble)' : vae_clf,
              'VAE-Resnet(clf)' : vae_clf.base_classifier}
    for idx, model in enumerate(smooth_models):
        models[f'Smooth-Resnet-{smoothing_sigmas[idx]}'] = model

    total_res = {}
    for name in models.keys():
        total_res[name] = {'nat acc': 0,
                           'linf adv': [0 for _ in range(len(linf_eps))],
                           'l2 adv': [0 for _ in range(len(l2_eps))]}
    attackers = {'linf adv': PGD, 'l2 adv': PGDL2}
    norm_lists = {'linf adv': linf_eps, 'l2 adv': l2_eps}
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, batch in progress_bar:
        # natural accuracy
        for model_name in models.keys():
            data, labels = batch
            data = data.to(device)
            model = models[model_name]
            outputs = model(data)
            pred = torch.argmax(outputs, dim=1)
            num_correct = get_num_correct(pred, labels)
            total_res[model_name]['nat acc'] += num_correct/total_samples

            # adversaries
            for attacker_type in attackers.keys():
                norms = norm_lists[attacker_type]
                for i in range(len(norms)):

                    if model_name == 'VQVAE-Resnet(ensemble)':
                        attacker = attackers[attacker_type](vqvae_clf.base_classifier, eps = norms[i], steps = 40)
                    elif model_name == 'VAE-Resnet(ensemble)':
                        attacker = attackers[attacker_type](vae_clf.base_classifier, eps = norms[i], steps = 40)
                    else:
                        attacker = attackers[attacker_type](model, eps=norms[i], steps=40)
                    attacked_data = attacker(data, labels)
                    outputs = model(attacked_data)
                    pred = torch.argmax(outputs, dim=1)
                    total_res[model_name][attacker_type][i] += get_num_correct(pred, labels)/total_samples

    # saving the data
    with open(r'res/prelim_adv_res.pickle', 'wb') as output_file:
        pickle.dump(total_res, output_file)
    print("Finished with adv testing")
    return


def get_num_correct(pred, labels):
    return sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(labels))])


def get_base_resnet(device):
    base_resnet_clf = ResNet(depth=110, block_name='BottleNeck', num_classes=10).to(device)
    base_resnet_clf.load_state_dict(torch.load('saved_models/resnet_updated'))
    return base_resnet_clf


def get_vae_clf(device):
    VAE = vae_models['VanillaVAE']
    vae = VAE(in_channels=3, latent_dim=512).to(device)
    vae.load_state_dict(torch.load('saved_models/vae_vanilla_base'))
    resnet_base_clf = ResNet(depth=110, block_name='BottleNeck', num_classes=10).to(device)
    resnet_base_clf.load_state_dict(torch.load('saved_models/resnet_vae_ensemble'))
    return VAE_CLF(base_classifier=resnet_base_clf, vae=vae)


def get_vqvae_clf(device):
    base_resnet_clf = ResNet(depth=110, block_name='BottleNeck', num_classes=10).to(device)
    base_resnet_clf.load_state_dict(torch.load('saved_models/resnet_ensemble'))

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    VQVAE = vae_models['VQVAE2']
    vq_vae = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                   num_embeddings, embedding_dim,
                   commitment_cost).to(device)
    vq_vae.load_state_dict(torch.load('saved_models/vqvae_ensemble'))
    return VQVAE_CLF(base_resnet_clf, vq_vae)


# for right  now, the resnet is NOT trained with smoothing -- so for right now, each of the smooth models
# should be fine with just pointer access to the model
def get_resnet_smooth(device,
                      smoothing_sigmas,
                      resnet,
                      m_train = 10):
    models = []
    for sigma in smoothing_sigmas:
        clf = Smooth(resnet, sigma, device, m_train, 10)
        models.append(clf)
    return models
