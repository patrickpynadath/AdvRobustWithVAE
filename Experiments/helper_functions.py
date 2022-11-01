import torch
import yaml
from torch.linalg import vector_norm
from torchattacks import PGD, PGDL2

from Models import AE, VAE, VQVAE_NSVQ, ResNet, GenClf
from Training import AETrainer, VAETrainer, VQVAETrainer, NatTrainer
from Utils import torch_to_numpy

trainer_dct = {'ae': AETrainer, 'vae': VAETrainer, 'vqvae': VQVAETrainer}


def train_models(training_metrics_dir, device):
    model_dct = get_untrained_models(device)
    gen_lr = float(model_dct['training_param_gen']['lr'])
    gen_batch_size = int(model_dct['training_param_gen']['batch_size'])
    gen_epochs = int(model_dct['training_param_gen']['epochs'])
    use_lr_step = False
    lr_gamma_step = int(model_dct['training_param_clf']['step_lr'])
    if lr_gamma_step != 0:
        use_lr_step = True
    gamma = float(model_dct['training_param_clf']['gamma'])
    lr = float(model_dct['training_param_clf']['lr'])
    epochs_clf = int(model_dct['training_param_clf']['epochs'])
    optimizer = model_dct['training_param_clf']['optimizer']
    clf_batch_size = int(model_dct['training_param_clf']['batch_size'])

    for key in ['ae', 'vae', 'vqvae']:
        gen_model = model_dct[key]
        gen_trainer = trainer_dct[key](device, gen_model, True, training_metrics_dir, gen_batch_size, gen_lr)
        gen_trainer.training_loop(gen_epochs)
        gen_name = f"{key.upper()}_{gen_model.res_layers}Res{gen_model.latent_size}lat"
        torch.save(gen_model.state_dict(), f"PretrainedModels/{gen_name}")
        # training the ensemble classifier
        resnet_ensemble = model_dct[f'resnet_{key}']
        clf = GenClf(gen_model, resnet_ensemble)
        clf_trainer = NatTrainer(clf, device, optimizer, lr, training_metrics_dir, True,
                                 use_step_lr=use_lr_step,
                                 lr_schedule_step=lr_gamma_step,
                                 lr_schedule_gamma=gamma,
                                 batch_size=clf_batch_size)
        clf_trainer.training_loop(epochs_clf)
        clf_name = f"Resnet{resnet_ensemble.depth}_{gen_name}"
        torch.save(resnet_ensemble.state_dict(), f"PretrainedModels/{clf_name}")

    resnet = model_dct['resnet']

    trainer = NatTrainer(resnet, device, optimizer, lr, training_metrics_dir, True,
                         use_step_lr=use_lr_step,
                         lr_schedule_step=lr_gamma_step,
                         lr_schedule_gamma=gamma,
                         batch_size=clf_batch_size)
    trainer.training_loop(epochs_clf)
    torch.save(resnet.state_dict(), f"PretrainedModels/Resnet_{resnet.depth}")

    resnet_smooth = model_dct['resnetSmooth']

    trainer = NatTrainer(resnet_smooth, device, optimizer, lr, training_metrics_dir, True,
                         use_step_lr=use_lr_step,
                         lr_schedule_step=lr_gamma_step,
                         lr_schedule_gamma=gamma,
                         batch_size=clf_batch_size, smooth=True, noise_sd=.25)
    trainer.training_loop(epochs_clf)
    torch.save(resnet.state_dict(), f"PretrainedModels/Resnet_{resnet_smooth.depth}_SmoothSigma_.25")
    return


def load_models(device):
    model_dct = get_untrained_models(device)

    for key in ['ae', 'vae', 'vqvae']:
        gen_model = model_dct[key]
        gen_path = f'PretrainedModels/{key.upper()}_{gen_model.res_layers}Res{gen_model.latent_size}Lat'
        gen_model.load_state_dict(torch.load(gen_path))
        model_dct[key] = gen_model
        resnet_ensemble = model_dct[f"resnet_{key}"]
        ens_path = f'PretrainedModels/Resnet{resnet_ensemble.depth}_{key.upper()}_{gen_model.res_layers}Res{gen_model.latent_size}Lat '
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


def get_norm_comparison(diff: torch.Tensor):
    # flattening along every dimension except for batch
    diff = diff.flatten(start_dim=1)
    l_2 = torch_to_numpy(vector_norm(diff, ord=2, dim=1))
    l_inf = torch_to_numpy(vector_norm(diff, ord=float('inf'), dim=1))
    return {'l2': l_2, 'linf': l_inf}
