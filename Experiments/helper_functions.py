import pickle
from fractions import Fraction
import torch
import yaml
import os
from .base_exp import BaseExp
from Models import GenClf, AE, VAE, VQVAE_NSVQ, ResNet

from Training import AETrainer, VAETrainer, VQVAETrainer, NatTrainer


def train_models(training_metrics_dir, device):
    model_dct = get_untrained_models(device)
    gen_lr = float(model_dct['training_param_gen']['lr'])
    gen_batch_size = int(model_dct['training_param_gen']['batch_size'])
    gen_epochs = int(model_dct['training_param_gen']['epochs'])
    ae = model_dct['ae']
    ae_trainer = AETrainer(device, ae, True, training_metrics_dir, gen_batch_size, gen_lr)
    ae_trainer.training_loop(gen_epochs)
    torch.save(ae.state_dict(), f'PretrainedModels/AE_{ae.res_layers}Res{ae.latent_size}lat')

    vae = model_dct['vae']
    vae_trainer = VAETrainer(device, vae, True, training_metrics_dir, gen_batch_size, gen_lr)
    vae_trainer.training_loop(gen_epochs)
    torch.save(vae.state_dict(), f'PretrainedModels/VAE_{vae.res_layers}Res{vae.latent_size}lat')

    vqvae = model_dct['vqvae']
    vqvae_trainer = VQVAETrainer(device, vqvae, True, training_metrics_dir, gen_batch_size, gen_lr)
    vqvae_trainer.training_loop(gen_epochs)
    torch.save(vqvae.state_dict(), f'PretrainedModels/VQVAE_{vqvae.res_layers}Res{vqvae.latent_size}lat')

    resnet = model_dct['resnet']
    use_lr_step = False
    lr_gamma_step = int(model_dct['training_param_clf']['step_lr'])
    if lr_gamma_step != 0:
        use_lr_step = True
    gamma = float(model_dct['training_param_clf']['gamma'])
    lr = float(model_dct['training_param_clf']['lr'])
    epochs_clf = int(model_dct['training_param_clf']['epochs'])
    optimizer = model_dct['training_param_clf']['optimizer']
    clf_batch_size = int(model_dct['training_param_clf']['batch_size'])
    trainer = NatTrainer(resnet, device, optimizer, lr, training_metrics_dir, True,
                         use_step_lr=use_lr_step,
                         lr_schedule_step=lr_gamma_step,
                         lr_schedule_gamma=gamma,
                         batch_size=clf_batch_size)
    trainer.training_loop(epochs_clf)
    torch.save(resnet.state_dict(), f"PretrainedModels/Resnet_{resnet.depth}")
    return


def load_models(device):
    model_dct = get_untrained_models(device)
    ae = model_dct['ae']
    ae_path = f'PretrainedModels/AE_{ae.res_layers}Res{ae.latent_size}Lat'
    ae.load_state_dict(torch.load(ae_path))
    model_dct['ae'] = ae

    vae = model_dct['vae']
    vae_path = f"PretrainedModels/VAE_{vae.res_layers}Res{vae.latent_size}Lat"
    vae.load_state_dict(torch.load(vae_path))
    model_dct['vae'] = vae

    vqvae = model_dct['vqvae']
    vqvae_path = f"PretrainedModels/VQVAE_{vqvae.res_layers}Res{vqvae.latent_size}Lat"
    vqvae.load_state_dict(torch.load(vqvae_path))
    model_dct['vqvae'] = vqvae

    resnet = model_dct['resnet']
    resnet_path = f"PretrainedModels/Resnet{resnet.depth}"
    resnet.load_state_dict(torch.load(resnet_path))
    model_dct['resnet'] = resnet

    resnet_smooth=model_dct['resnetSmooth']
    resnet_smooth_path = f'PretrainedModels/Resnet110_SmoothSigma_.25'
    resnet_smooth.load_state_dict(torch.load(resnet_smooth_path))
    model_dct['resnetSmooth'] = resnet_smooth
    return model_dct


def run_raw_adv_rob(device):
    with open("./Params/exp_params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    num_steps = int(params['adv_params']['num_steps'])
    l2_eps = [float(eps) for eps in params['adv_params']['l2_eps']]
    linf_eps = [float(Fraction(eps)) for eps in params['adv_params']['linf_eps']]

    l2_accs = {'ae' : [], 'vae' : [], 'vqvae' : [], 'resnetSmooth': []}
    linf_accs = {'ae' : [], 'vae' : [], 'vqvae' : [], 'resnetSmooth': []}
    model_dct = load_models(device)
    exp = BaseExp(device)
    resnet = model_dct['resnet']
    print("Eval base resnet")
    nat_acc = exp.eval_clf_clean(resnet)
    adv_accs_l2 = exp.eval_clf_adv_raw(resnet, 'l2', l2_eps, num_steps)
    adv_accs_linf = exp.eval_clf_adv_raw(resnet, 'linf', linf_eps, num_steps)
    resnet_l2 = [nat_acc] + adv_accs_l2
    resnet_linf = [nat_acc] + adv_accs_linf

    for key in linf_accs.keys():
        print(f"Eval {key}")
        clf = GenClf(model_dct[key], resnet)
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs = exp.eval_clf_adv_raw(clf, 'linf', linf_eps, num_steps)
        linf_accs[key] = [nat_acc] + adv_accs

    for key in l2_accs.keys():
        print(f"Eval {key}")
        clf = GenClf(model_dct[key], resnet)
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs = exp.eval_clf_adv_raw(clf, 'l2', l2_eps, num_steps)
        l2_accs[key] = [nat_acc] + adv_accs

    l2_accs['base'] = resnet_l2
    linf_accs['base'] = resnet_linf
    res = {'l2': l2_accs, 'linf': linf_accs, 'l2_eps': l2_eps, 'linf_eps': linf_eps}
    with open("adv_rob_res_raw.pickle", "wb") as stream:
        pickle.dump(res, stream)
    return


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
    return {'ae' : ae,
            'vae' : vae,
            'vqvae' : vqvae,
            'resnet' : resnet,
            'resnetSmooth' : resnet_smooth,
            'training_param_gen':model_params['training_param_gen'],
            'training_param_clf':model_params['training_param_clf']}
