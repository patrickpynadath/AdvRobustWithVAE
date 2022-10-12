import torch
from Models import ResNet, vae_models, VQVAE_CLF, Smooth, VAE_CLF
from Utils import get_cifar_sets
from torch.utils.data import DataLoader
import copy
from torchattacks_cust import PGD, PGDL2
from tqdm import tqdm
import pickle

device = 'cuda'


def test_resnet_base(test_loader, attackers, norm_lists, acc_dct):
    resnet_base = get_base_resnet(device)
    resnet_base.eval()
    total = 0
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    print("Testing Base-Resnet")
    for batch_idx, batch in progress_bar:
        data, labels = batch
        data = data.to(device)
        outputs = resnet_base(data)
        pred = torch.argmax(outputs, dim=1)
        num_correct = get_num_correct(pred, labels)
        acc_dct['Base-Resnet']['nat acc'] += num_correct
        total += len(data)
    acc_dct['Base-Resnet']['nat acc'] /= total
    print(f"Nat Acc: {acc_dct['Base-Resnet']['nat acc']}")
    for attacker_type in attackers.keys():
        print(attacker_type)
        norms = norm_lists[attacker_type]
        for i in range(len(norms)):
            torch.cuda.empty_cache()
            print(f"Eps: {round(norms[i], 5)}")
            attacker = attackers[attacker_type](resnet_base, eps=norms[i], steps=20)
            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
            total = 0
            for batch_idx, batch in progress_bar:
                data, labels = batch
                attacked_data = attacker(data, labels)
                outputs = resnet_base(attacked_data)
                pred = torch.argmax(outputs, dim=1)
                acc_dct['Base-Resnet'][attacker_type][i] += get_num_correct(pred, labels)
                total += len(data)
            acc_dct['Base-Resnet'][attacker_type][i] /= total
            print(f"Adv Acc: {acc_dct['Base-Resnet'][attacker_type][i]}")
    print(f"Done testing {'Base-Resnet'}")
    return


def test_vq_vae_ensemble(test_loader, attackers, norm_lists, acc_dct):
    vqvae_clf = get_vqvae_clf(device)
    vqvae_clf.eval()
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    total = 0
    print('Testing VQVAE-Resnet(ensemble)')
    for batch_idx, batch in progress_bar:
        data, labels = batch
        data = data.to(device)
        outputs = vqvae_clf(data)
        pred = torch.argmax(outputs, dim=1)
        num_correct = get_num_correct(pred, labels)
        acc_dct['VQVAE-Resnet(ensemble)']['nat acc'] += num_correct
        total += len(data)
    acc_dct['VQVAE-Resnet(ensemble)']['nat acc'] /= total
    print(f"Nat Acc: {acc_dct['VQVAE-Resnet(ensemble)']['nat acc']}")
    for attacker_type in attackers.keys():
        print(attacker_type)
        norms = norm_lists[attacker_type]
        for i in range(len(norms)):
            torch.cuda.empty_cache()
            print(f"Eps: {round(norms[i], 5)}")
            attacker = attackers[attacker_type](vqvae_clf.base_classifier, eps=norms[i], steps=20)
            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
            total = 0
            for batch_idx, batch in progress_bar:
                data, labels = batch
                attacked_data = attacker(data, labels)
                outputs = vqvae_clf(attacked_data)
                pred = torch.argmax(outputs, dim=1)
                acc_dct['VQVAE-Resnet(ensemble)'][attacker_type][i] += get_num_correct(pred, labels)
                total += len(data)
            acc_dct['VQVAE-Resnet(ensemble)'][attacker_type][i] /= total
            print(f"Adv Acc: {acc_dct['VQVAE-Resnet(ensemble)'][attacker_type][i]}")
    print(f"Done testing {'VQVAE-Resnet(ensemble)'}")
    return


def test_vq_vae_clf(test_loader, attackers, norm_lists, acc_dct):
    vqvae_clf = get_vqvae_clf(device)
    model = vqvae_clf.base_classifier
    vqvae_clf.vq_vae.to('cpu')
    model.eval()
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    total = 0
    print('Testing VQVAE-Resnet(clf)')
    for batch_idx, batch in progress_bar:
        data, labels = batch
        data = data.to(device)
        outputs = model(data)
        pred = torch.argmax(outputs, dim=1)
        num_correct = get_num_correct(pred, labels)
        acc_dct['VQVAE-Resnet(clf)']['nat acc'] += num_correct
        total += len(data)
    acc_dct['VQVAE-Resnet(clf)']['nat acc'] /= total
    print(f"Nat Acc: {acc_dct['VQVAE-Resnet(clf)']['nat acc']}")
    for attacker_type in attackers.keys():
        print(attacker_type)
        norms = norm_lists[attacker_type]
        for i in range(len(norms)):
            torch.cuda.empty_cache()
            print(f"Eps: {round(norms[i], 5)}")
            attacker = attackers[attacker_type](model, eps=norms[i], steps=20)
            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
            total = 0
            for batch_idx, batch in progress_bar:
                data, labels = batch
                attacked_data = attacker(data, labels)
                outputs = model(attacked_data)
                pred = torch.argmax(outputs, dim=1)
                acc_dct['VQVAE-Resnet(clf)'][attacker_type][i] += get_num_correct(pred, labels)
                total += len(data)
            acc_dct['VQVAE-Resnet(clf)'][attacker_type][i] /= total
            print(f"Adv Acc: {acc_dct['VQVAE-Resnet(clf)'][attacker_type][i]}")
    print(f"Done testing {'VQVAE-Resnet(clf)'}")
    return


def test_vae_ensemble(test_loader, attackers, norm_lists, acc_dct):
    vae_clf = get_vae_clf(device)
    vae_clf.eval()
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    total = 0
    print('Testing VAE-Resnet(ensemble)')
    for batch_idx, batch in progress_bar:
        data, labels = batch
        data = data.to(device)
        outputs = vae_clf(data)
        pred = torch.argmax(outputs, dim=1)
        num_correct = get_num_correct(pred, labels)
        acc_dct['VAE-Resnet(ensemble)']['nat acc'] += num_correct
        total += len(data)
    acc_dct['VAE-Resnet(ensemble)']['nat acc'] /= total
    print(f"Nat Acc: {acc_dct['VAE-Resnet(ensemble)']['nat acc']}")
    for attacker_type in attackers.keys():
        print(attacker_type)
        norms = norm_lists[attacker_type]
        for i in range(len(norms)):
            torch.cuda.empty_cache()
            print(f"Eps: {round(norms[i], 5)}")
            attacker = attackers[attacker_type](vae_clf.base_classifier, eps=norms[i], steps=20)
            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
            total = 0
            for batch_idx, batch in progress_bar:
                data, labels = batch
                attacked_data = attacker(data, labels)
                outputs = vae_clf(attacked_data)
                pred = torch.argmax(outputs, dim=1)
                acc_dct['VAE-Resnet(ensemble)'][attacker_type][i] += get_num_correct(pred, labels)
                total += len(data)
            acc_dct['VAE-Resnet(ensemble)'][attacker_type][i] /= total
            print(f"Adv Acc: {acc_dct['VAE-Resnet(ensemble)'][attacker_type][i]}")
    print(f"Done testing {'VAE-Resnet(ensemble)'}")
    return


def test_vae_clf(test_loader, attackers, norm_lists, acc_dct):
    vae_clf = get_vae_clf(device)
    model = vae_clf.base_classifier
    vae_clf.vae.to('cpu')
    model.eval()
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    total = 0
    print('Testing VAE-Resnet(clf)')
    for batch_idx, batch in progress_bar:
        data, labels = batch
        data = data.to(device)
        outputs = model(data)
        pred = torch.argmax(outputs, dim=1)
        num_correct = get_num_correct(pred, labels)
        acc_dct['VAE-Resnet(clf)']['nat acc'] += num_correct
        total += len(data)
    acc_dct['VAE-Resnet(clf)']['nat acc'] /= total
    print(f"Nat Acc: {acc_dct['VAE-Resnet(clf)']['nat acc']}")
    for attacker_type in attackers.keys():
        print(attacker_type)
        norms = norm_lists[attacker_type]
        for i in range(len(norms)):
            torch.cuda.empty_cache()
            print(f"Eps: {round(norms[i], 5)}")
            attacker = attackers[attacker_type](model, eps=norms[i], steps=20)
            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
            total = 0
            for batch_idx, batch in progress_bar:
                data, labels = batch
                attacked_data = attacker(data, labels)
                outputs = model(attacked_data)
                pred = torch.argmax(outputs, dim=1)
                acc_dct['VAE-Resnet(clf)'][attacker_type][i] += get_num_correct(pred, labels)
                total += len(data)
            acc_dct['VAE-Resnet(clf)'][attacker_type][i] /= total
            print(f"Adv Acc: {acc_dct['VAE-Resnet(clf)'][attacker_type][i]}")
    print(f"Done testing {'VAE-Resnet(clf)'}")
    return


def test_smooth_resnet(test_loader, attackers, norm_lists, acc_dct, sigma):
    model = get_resnet_smooth(device, sigma)
    model.eval()
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    total = 0
    print(f'Testing Smooth-Resnet-{sigma}')
    for batch_idx, batch in progress_bar:
        data, labels = batch
        data = data.to(device)
        outputs = model(data)
        pred = torch.argmax(outputs, dim=1)
        num_correct = get_num_correct(pred, labels)
        acc_dct[f'Smooth-Resnet-{sigma}']['nat acc'] += num_correct
        total += len(data)
        torch.cuda.empty_cache()
    acc_dct[f'Smooth-Resnet-{sigma}']['nat acc'] /= total
    print(f"Nat Acc: {acc_dct[f'Smooth-Resnet-{sigma}']['nat acc']}")
    for attacker_type in attackers.keys():
        print(attacker_type)
        norms = norm_lists[attacker_type]
        for i in range(len(norms)):
            torch.cuda.empty_cache()
            print(f"Eps: {round(norms[i], 5)}")
            attacker = attackers[attacker_type](model.base_classifier, eps=norms[i], steps=20)
            progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
            total = 0
            for batch_idx, batch in progress_bar:
                data, labels = batch
                attacked_data = attacker(data, labels)
                outputs = model(attacked_data)
                pred = torch.argmax(outputs, dim=1)
                acc_dct[f'Smooth-Resnet-{sigma}'][attacker_type][i] += get_num_correct(pred, labels)
                total += len(data)
                torch.cuda.empty_cache()
            acc_dct[f'Smooth-Resnet-{sigma}'][attacker_type][i] /= total
            print(f"Adv Acc: {acc_dct[f'Smooth-Resnet-{sigma}'][attacker_type][i]}")
    print(f"Done testing {f'Smooth-Resnet-{sigma}'}")
    return


def run_adv_robust():

    smoothing_sigmas = [.25]

    _, test_dataset = get_cifar_sets()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    linf_eps = [1/255, 2/255, 5/255, 10/255]
    l2_eps = [.5, 1, 1.5, 2]
    models = ['Base-Resnet', 'VQVAE-Resnet(ensemble)', 'VQVAE-Resnet(clf)', 'VAE-Resnet(ensemble)', 'VAE-Resnet(clf)']
    for sigma in smoothing_sigmas:
        models.append(f'Smooth-Resnet-{sigma}')

    total_res = {}
    for name in models:
        total_res[name] = {'nat acc': 0,
                           'linf adv': [0 for _ in range(len(linf_eps))],
                           'l2 adv': [0 for _ in range(len(l2_eps))]}
    attackers = {'linf adv': PGD, 'l2 adv': PGDL2}
    norm_lists = {'linf adv': linf_eps, 'l2 adv': l2_eps}
    for sigma in smoothing_sigmas:
        try:
            test_smooth_resnet(test_loader, attackers, norm_lists, total_res, sigma)
            torch.cuda.empty_cache()
        except Exception as ex:
            dump_checkpoint(total_res)
            print(ex)
    try:
        test_resnet_base(test_loader, attackers, norm_lists, total_res)
        torch.cuda.empty_cache()
    except Exception as ex:
        dump_checkpoint(total_res)
        print(ex)
    try:
        test_vae_ensemble(test_loader, attackers, norm_lists, total_res)
        torch.cuda.empty_cache()
    except Exception as ex:
        dump_checkpoint(total_res)
        print(ex)
    try:
        test_vae_clf(test_loader, attackers, norm_lists, total_res)
        torch.cuda.empty_cache()
    except Exception as ex:
        dump_checkpoint(total_res)
        print(ex)
    try:
        test_vq_vae_ensemble(test_loader, attackers, norm_lists, total_res)
        torch.cuda.empty_cache()
    except Exception as ex:
        dump_checkpoint(total_res)
        print(ex)
    try:
        test_vq_vae_clf(test_loader, attackers, norm_lists, total_res)
        torch.cuda.empty_cache()
    except Exception as ex:
        dump_checkpoint(total_res)
        print(ex)
    # saving the data
    with open(r'res/prelim_adv_res.pickle', 'wb') as output_file:
        pickle.dump(total_res, output_file)
    print("Finished with adv testing")
    return


def dump_checkpoint(total_res):
    with open(r'res/prelim_adv_res_checkpoint.pickle', 'wb') as output_file:
        pickle.dump(total_res, output_file)
    return


def get_num_correct(pred, labels):
    return sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(labels))])


def get_base_resnet(device):
    base_resnet_clf = ResNet(depth=110, block_name='BottleNeck', num_classes=10).to(device)
    base_resnet_clf.load_state_dict(torch.load('saved_models/resnet_base'))
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
                      sigma,
                      m_train = 10):
    resnet = get_base_resnet(device)
    clf = Smooth(resnet, sigma, device, m_train, 10)
    return clf

