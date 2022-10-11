from Experiments import run_hyperparam_clf, sanity_check, BaseExp, single_exp_loop_vae_adv, run_adv_rob_exp
from Models import Configuration, AutoEncoder
import torch
from Experiments import train_save_necessary_models
from Training import get_trained_vq_vae
from Experiments import run_adv_robust
# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'
VAE_ADV_EXP = '../ExperimentLogging/AdversarialExpVAE/'
ADV_ROB_EXP = '../ExperimentLogging/AdvRobExp/'
if __name__ == '__main__':
    run_adv_robust()
    # exp = BaseExp(training_logdir=TRAIN_METRICS_DIR,
    #               exp_logdir=VAE_ADV_EXP,
    #               device='cuda')
    # vae_clf = exp.get_vae_resnet(net_depth=110, block_name='BottleNeck')
    # torch.save(vae_clf.base_classifier, 'saved_models/resnet_vae_ensemble')
    # vq_vae = get_trained_vq_vae(TRAIN_METRICS_DIR, 15000)
    # torch.save(vq_vae.state_dict(), 'saved_models/vq_vae')
    # run_adv_robust()
