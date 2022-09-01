from Experiments.nat_training_exp import Adv_Robustness_NaturalTraining
from Experiments.vae_exp import PeturbExperiment
import os
from Experiments.vae_exp import PeturbExperiment
from torch.utils.tensorboard import SummaryWriter
from Utils.utils import accuracies_to_dct

# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'
VAE_EXP_DIR = '../ExperimentLogging/VaeExpMetrics'
DEVICE = 'cuda'

# global parameters that control smoothing process
SMOOTHING_SIGMAS = [5/255, 10/255, 15/255, 20/255, 25/255]
SMOOTHINGVAE_SIGMAS = [5/255, 10/255, 15/255, 20/255, 25/255]
M_TRAIN = 10

# global parameters for measuring adversarial robustness
TEST_EPS_linf = [1/255, 2/255, 3/255, 4/255, 5/255, 10/255, 15/255, 20/255, 25/255]
TEST_EPS_l2 = []
TEST_ATTACK_STEPS = 8
NUM_TEST_ATTACKS = 1000

# global parameters for classifier
BATCH_SIZE_CLF = 50
LR = .01
CLF_EPOCHS = 50

# global parameters for VAE
VAE_EPOCHS = [50]
KERNEL_NUM = [50]
LATENT_SIZE= [10, 100, 200]
BATCH_SIZE_VAE = 32
# VAE_LOSS_COEFS = [0, .5, 1, 2]
# PETURBATION_NORMS = [1/255, 2/255, 5/255, 10/255, 20/255]

VAE_LOSS_COEFS = [1]

def run_adv_rob_baseclf(exp : Adv_Robustness_NaturalTraining, summary_writer : SummaryWriter, adv_type : str, test_eps):
    nat_acc, adv_accs, label = exp.adv_rob_baseclf(clf_epochs=CLF_EPOCHS,
                            adv_type=adv_type,
                            adv_norms=test_eps,
                            adv_steps=TEST_ATTACK_STEPS,
                            num_attacks=NUM_TEST_ATTACKS)
    param_dct = {'Model' : 'base_clf',
                 'SmoothingSigma' : 0,
                 'LossCoef' : 0,
                 'VAE_Epoch' : 0,
                 'KernelNum' : 0,
                 'LatentSize' : 0}
    metric_dct = accuracies_to_dct(nat_acc, adv_accs, test_eps, adv_type)
    run_name = exp.hyperparam_logdir + f"/{label}"
    summary_writer.add_hparams(param_dct, metric_dct, run_name=run_name)
    return

def run_adv_rob_smoothclf(exp : Adv_Robustness_NaturalTraining, summary_writer : SummaryWriter, adv_type : str, test_eps):
    for smoothing_sigma in SMOOTHING_SIGMAS:
        nat_acc, adv_accs, label = exp.adv_rob_smoothclf(clf_epochs= CLF_EPOCHS,
                                                         smoothing_sigma=smoothing_sigma,
                                                         smoothing_num_samples=M_TRAIN,
                                                         adv_type=adv_type,
                                                         adv_norms=test_eps,
                                                         adv_steps=TEST_ATTACK_STEPS,
                                                         num_attacks=NUM_TEST_ATTACKS)
        param_dct = {'Model': 'Smooth',
                     'SmoothingSigma': round(smoothing_sigma, 4),
                     'LossCoef': 0,
                     'VAE_Epoch': 0,
                     'KernelNum': 0,
                     'LatentSize': 0}
        metric_dct = accuracies_to_dct(nat_acc, adv_accs, test_eps, adv_type)
        run_name = exp.hyperparam_logdir + f"/{label}"
        summary_writer.add_hparams(param_dct, metric_dct, run_name=run_name)
    return

def run_adv_rob_smoothVAE(exp : Adv_Robustness_NaturalTraining, summary_writer : SummaryWriter, adv_type : str, test_eps):
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
                            nat_acc, adv_accs, label = exp.adv_rob_smoothvae_clf(clf_epochs=CLF_EPOCHS,
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
                                                                              adv_type=adv_type,
                                                                              adv_norms=test_eps,
                                                                              adv_steps=TEST_ATTACK_STEPS,
                                                                              num_attacks=NUM_TEST_ATTACKS)
                            metric_dct = accuracies_to_dct(nat_acc, adv_accs, test_eps, adv_type)
                            run_name = exp.hyperparam_logdir + f"/{label}"
                            summary_writer.add_hparams(param_dct, metric_dct, run_name=run_name)
    return

def run_peturn_exp():

    exp = PeturbExperiment(batch_size= BATCH_SIZE_VAE,
                           log_dir=VAE_EXP_DIR,
                           device=DEVICE)
    # first, examining the VAE trained without classifier support
    return



def adv_rob_loop(adv_type):
    if adv_type == 'linf':
        test_eps = TEST_EPS_linf
    elif adv_type == 'l2' :
        test_eps = TEST_EPS_l2
    adv_exp = Adv_Robustness_NaturalTraining(training_logdir=TRAIN_METRICS_DIR,
                                             hyperparam_logdir=HYPERPARAM_DIR,
                                             lr=LR,
                                             batch_size=BATCH_SIZE_CLF,
                                             device=DEVICE)
    hparam_writer = SummaryWriter(log_dir=adv_exp.hyperparam_logdir)
    # run_adv_rob_baseclf(exp = adv_exp,
    #                     summary_writer= hparam_writer,
    #                     adv_type=adv_type,
    #                     test_eps=test_eps)
    #
    # run_adv_rob_smoothclf(exp = adv_exp,
    #                       summary_writer=hparam_writer,
    #                       adv_type=adv_type,
    #                       test_eps = test_eps)

    run_adv_rob_smoothVAE(exp = adv_exp,
                          summary_writer=hparam_writer,
                          adv_type=adv_type,
                          test_eps=test_eps)
    return


def peturb_analysis_loop(kernel_num, latent_size):
    # need to first generate the VAE

    return



if __name__ == '__main__':
    adv_rob_loop(adv_type='linf')


