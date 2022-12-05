import torch
import yaml

from Models import AE, VAE, VQVAE_NSVQ, ResNet


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
    model_dct['resnetGaussian'] = resnet_smooth

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
            'resnetGaussian': resnet_smooth,
            'resnet_ae': ResNet(resnet_depth, resnet_classes, resnet_block, device=device),
            'resnet_vae': ResNet(resnet_depth, resnet_classes, resnet_block, device=device),
            'resnet_vqvae': ResNet(resnet_depth, resnet_classes, resnet_block, device=device),
            'training_param_gen': model_params['training_param_gen'],
            'training_param_clf': model_params['training_param_clf']}
