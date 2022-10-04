from Experiments.base_exp import BaseExp
import torch


def train_save_necessary_models(training_logdir,
                                exp_logdir,
                                device,
                                batch_size_vae=64,
                                resnet_depth=110,
                                clf_epochs=150,
                                vae_epochs=150,
                                use_step_lr=True,
                                lr_schedule_step=50,
                                lr_schedule_gamma=.1):
    smoothing_sigmas = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255, 32 / 255, 64 / 255, 128 / 255]
    exp = BaseExp(training_logdir=training_logdir,
                  exp_logdir=exp_logdir,
                  device=device,
                  clf_train_set=None,
                  vae_train_set=None,
                  test_set=None)
    print("TRAINING VAE")
    vae = exp.get_trained_vae(batch_size=batch_size_vae,
                              epochs=vae_epochs,
                              vae_model='vae')
    path = "saved_models/vae_base"
    torch.save(vae.state_dict(), path)
    print("SAVED VAE")
    print("TRAINING RESNET")
    resnet_clf = exp.get_trained_resnet(net_depth=resnet_depth,
                                        block_name='BottleNeck',
                                        batch_size=256,
                                        optimizer='sgd',
                                        lr=.01,
                                        epochs=clf_epochs,
                                        use_step_lr=use_step_lr,
                                        lr_schedule_step=lr_schedule_step,
                                        lr_schedule_gamma=lr_schedule_gamma)
    path = "saved_models/resnet_base"
    torch.save(resnet_clf.state_dict(), path)
    print("SAVED RESNET")
    for smooth_vae_type in ['sample', 'latent']:
        for sigma in smoothing_sigmas:
            print(f"TRAINING SMOOTH VAE {smooth_vae_type} with SIGMA = {round(sigma, 5)}")
            path = f"saved_models/smoothVAE_{smooth_vae_type}_sigma_{round(sigma, 5)}"
            smoothvae = exp.get_trained_smooth_vae_resnet(net_depth=resnet_depth,
                                                          block_name='BottleNeck',
                                                          m_train=10,
                                                          batch_size_clf=256,
                                                          batch_size_vae=batch_size_vae,
                                                          optimizer='sgd',
                                                          smoothing_sigma=sigma,
                                                          smooth_vae_version=smooth_vae_type,
                                                          epochs_clf=clf_epochs,
                                                          trained_vae=vae)
            torch.save(smoothvae.state_dict(), path)
            print("SAVED MODEL")

