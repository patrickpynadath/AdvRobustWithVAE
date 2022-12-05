import torch

from Models import GenClf, get_untrained_models
from Training import AETrainer, VAETrainer, VQVAETrainer, NatTrainer

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
