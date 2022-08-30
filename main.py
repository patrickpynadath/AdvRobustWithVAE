from Experiments.nat_training_exp import Adv_Robustness_NaturalTraining
from Experiments.adv_training_exp import AdvExperiment
from Experiments.vae_exp import PeturbExperiment
import os
from Experiments.vae_exp import PeturbExperiment
from torch.utils.tensorboard import SummaryWriter
from Utils.utils import accuracies_to_dct

# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'
DEVICE = 'cuda'

# global parameters that control smoothing process
SMOOTHING_SIGMAS = [5/255, 10/255, 15/255, 20/255, 25/255]
SMOOTHINGVAE_SIGMAS = [5/255, 10/255, 15/255, 20/255, 25/255]
M_TRAIN = 10

# global parameters for measuring adversarial robustness
TEST_EPS_linf = [1/255, 2/255, 3/255, 4/255, 5/255, 10/255, 15/255, 20/255, 25/255]
#TEST_EPS_l2 = []
TEST_ATTACK_STEPS = 8
NUM_TEST_ATTACKS = 1000

# global parameters for classifier
BATCH_SIZE_CLF = 50
LR = .01
CLF_EPOCHS = 50

# global parameters for VAE
VAE_EPOCHS = [1, 10, 20, 30]
KERNEL_NUM = [10, 20, 50, 100]
LATENT_SIZE= [10, 20, 50, 100]
BATCH_SIZE_VAE = 32
# VAE_LOSS_COEFS = [0, .5, 1, 2]
# PETURBATION_NORMS = [1/255, 2/255, 5/255, 10/255, 20/255]

VAE_LOSS_COEFS = [0, .5, 1, 1.5, 2]


def adv_rob_linf_loop():

    adv_exp = Adv_Robustness_NaturalTraining(training_logdir=TRAIN_METRICS_DIR,
                                             hyperparam_logdir=HYPERPARAM_DIR,
                                             lr=LR,
                                             batch_size=BATCH_SIZE_CLF,
                                             device=DEVICE)
    hparam_writer = SummaryWriter(log_dir=adv_exp.hyperparam_logdir)
    # adding the hyperparameters for grid search


    # getting results for the baseline model -- plain classifier
    nat_acc, adv_accs, label = adv_exp.adv_rob_baseclf(clf_epochs=CLF_EPOCHS,
                            adv_type='linf',
                            adv_norms=TEST_EPS_linf,
                            adv_steps=TEST_ATTACK_STEPS,
                            num_attacks=NUM_TEST_ATTACKS)
    param_dct = {'Model' : 'base_clf',
                 'SmoothingSigma' : 0,
                 'LossCoef' : 0,
                 'VAE_Epoch' : 0,
                 'KernelNum' : 0,
                 'LatentSize' : 0}
    metric_dct = accuracies_to_dct(nat_acc, adv_accs, TEST_EPS_linf, 'linf')
    run_name = adv_exp.hyperparam_logdir + f"/{label}"
    hparam_writer.add_hparams(param_dct, metric_dct, run_name=run_name)

    # getting results for RandSmooth models
    for smoothing_sigma in SMOOTHING_SIGMAS:
        param_dct = {'Model': 'Smooth',
                     'SmoothingSigma': round(smoothing_sigma, 4),
                     'LossCoef': 0,
                     'VAE_Epoch': 0,
                     'KernelNum': 0,
                     'LatentSize': 0}
        nat_acc, adv_accs, label = adv_exp.adv_rob_smoothclf(clf_epochs=CLF_EPOCHS,
                                                      smoothing_sigma=smoothing_sigma,
                                                      smoothing_num_samples=M_TRAIN,
                                                      adv_type='linf',
                                                      adv_norms=TEST_EPS_linf,
                                                      adv_steps=TEST_ATTACK_STEPS,
                                                      num_attacks=NUM_TEST_ATTACKS)
        metric_dct = accuracies_to_dct(nat_acc, adv_accs, TEST_EPS_linf, 'linf')
        run_name = adv_exp.hyperparam_logdir + f"/{label}"
        hparam_writer.add_hparams(param_dct, metric_dct, run_name=run_name)

    # getting results for SmoothVAE models
    for smoothing_sigma in SMOOTHINGVAE_SIGMAS:
        for num_vae_epochs in VAE_EPOCHS:
            for kernel_num in KERNEL_NUM:
                for latent_size in LATENT_SIZE:
                    for loss_coef in VAE_LOSS_COEFS:
                        for model_type in ['sample', 'latent']:
                            param_dct = {'Model': f'SmoothVAE_{model_type}',
                                         'SmoothingSigma': round(smoothing_sigma, 4),
                                         'LossCoef': loss_coef,
                                         'VAE_Epoch': num_vae_epochs,
                                         'KernelNum': kernel_num,
                                         'LatentSize': latent_size}
                            nat_acc, adv_accs, label = adv_exp.adv_rob_smoothvae_clf(clf_epochs=CLF_EPOCHS,
                                                                              smoothingVAE_sigma=smoothing_sigma,
                                                                              smoothing_num_samples=M_TRAIN,
                                                                              smoothVAE_version=model_type,
                                                                              vae_loss_coef=loss_coef,
                                                                              vae_img_size=32,
                                                                              vae_channel_num=3,
                                                                              vae_kern_num=kernel_num,
                                                                              vae_z_size=latent_size,
                                                                              vae_epochs=num_vae_epochs,
                                                                              with_vae_grad=True,
                                                                              adv_type='linf',
                                                                              adv_norms=TEST_EPS_linf,
                                                                              adv_steps=TEST_ATTACK_STEPS,
                                                                              num_attacks=NUM_TEST_ATTACKS)
                            metric_dct = accuracies_to_dct(nat_acc, adv_accs, TEST_EPS_linf, 'linf')
                            run_name = adv_exp.hyperparam_logdir + f"/{label}"
                            hparam_writer.add_hparams(param_dct, metric_dct, run_name=run_name)
    return


if __name__ == '__main__':
    adv_rob_linf_loop()


