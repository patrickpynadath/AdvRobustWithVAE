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

    exp = BaseExp(training_logdir=TRAIN_METRICS_DIR,
                  exp_logdir=VAE_ADV_EXP,
                  device='cpu')
    vae = exp.get_trained_vanilla_vae(batch_size=128, epochs=100, vae_model='a')
