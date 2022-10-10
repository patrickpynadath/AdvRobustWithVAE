import torch
from Models import ResNet, vae_models, VQVAE_CLF
from Utils import get_cifar_sets
from torch.utils.data import DataLoader
import copy
from torchattacks import PGD, PGDL2
from tqdm import tqdm
import pickle


def run_adv_robust():
    device = 'cuda'
    base_resnet_clf = ResNet(depth=110, block_name='BottleNeck', num_classes=10).to(device)
    base_resnet_clf.load_state_dict(torch.load('saved_models/resnet_updated'))

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
    vq_vae.load_state_dict(torch.load('saved_models/vq_vae'))
    resnet = ResNet(depth=110, block_name='BottleNeck', num_classes=10).to(device)
    vq_vae2 = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                   num_embeddings, embedding_dim,
                   commitment_cost).to(device)
    vq_vae_clf = VQVAE_CLF(resnet, vq_vae2)
    vq_vae_clf.load_state_dict(torch.load('saved_models/vqvae_resnet'))
    _, test_dataset = get_cifar_sets()
    test_loader = DataLoader(test_dataset, batch_size=64)
    total_samples = len(test_dataset)
    linf_eps = [1/255, 2/255, 5/255, 10/255]
    l2_eps = [.5, 1, 1.5, 2]
    models = {'base_clf' : base_resnet_clf, 'VQVAE-Resnet(ensemble)' : vq_vae_clf, 'VQVAE-Resnet(clf)' : vq_vae_clf.base_classifier}
    total_res = {}
    for name in models.keys():
        total_res[name] = {'nat acc': 0,
                           'linf adv': [0 for _ in range(len(linf_eps))],
                           'l2 adv': [0 for _ in range(len(l2_eps))]}
    attackers = {'linf adv': PGD, 'l2 adv' : PGDL2}
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for batch_idx, batch in progress_bar:
        data, labels = batch
        data = data.to(device)

        # natural accuracy
        for model_name in models.keys():
            model = models[model_name]
            outputs = model(data)
            pred = torch.argmax(outputs, dim=1)
            num_correct = get_num_correct(pred, labels)
            total_res[model_name]['nat acc'] += num_correct/total_samples

            # linf adversaries
            for attacker_type in attackers.keys():
                for i in range(len(linf_eps)):

                    if model_name == 'VQVAE-Resnet(ensemble)':
                        attacker = attackers[attacker_type](vq_vae_clf.base_classifier, eps = linf_eps[i], steps = 40)
                    else:
                        attacker = attackers[attacker_type](model, eps = linf_eps[i], steps = 40)

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
