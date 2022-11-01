import torch
import yaml
from torchattacks import PGD, PGDL2
from Utils import torch_to_numpy
from torch.linalg import vector_norm
from Models import AE, VAE, VQVAE_NSVQ, ResNet

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
        norm_constrained_gaussian[i, :] = (gaussian_noise * norm / torch.linalg.vector_norm(gaussian_noise, ord=ord_type))
    return torch.reshape(norm_constrained_gaussian, orig_shape)


def get_adv_examples(clf,
                     attack_eps,
                     adversary_type,
                     steps,
                     nat_img,
                     labels):
    if adversary_type == 'linf':
        attacker = PGD(clf, eps=attack_eps, steps=steps)
    elif adversary_type == 'l2':
        attacker = PGDL2(clf, eps=attack_eps, steps=steps)
    return attacker(nat_img, labels)


# expects batched data
def get_norm_comparison(diff: torch.Tensor):
    # flattening along every dimension except for batch
    diff = diff.flatten(start_dim=1)
    l_2 = torch_to_numpy(vector_norm(diff, ord=2, dim=1))
    l_inf = torch_to_numpy(vector_norm(diff, ord=float('inf'), dim=1))
    return {'l2': l_2, 'linf': l_inf}