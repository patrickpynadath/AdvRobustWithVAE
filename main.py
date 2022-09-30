from Experiments import run_hyperparam_clf, sanity_check, BaseExp, single_exp_loop_vae_adv, run_adv_rob_exp
from Models import Configuration, AutoEncoder
import torch
# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'
VAE_ADV_EXP = '../ExperimentLogging/AdversarialExpVAE/'
ADV_ROB_EXP = '../ExperimentLogging/AdvRobExp/'
if __name__ == '__main__':
    run_adv_rob_exp(training_logdir=TRAIN_METRICS_DIR,
                    exp_logdir=ADV_ROB_EXP,
                    device='cuda',
                    resnet_depth=110,
                    clf_epochs=150,
                    vae_epochs=150)
    single_exp_loop_vae_adv(training_logdir=TRAIN_METRICS_DIR,
                            exp_logdir=VAE_ADV_EXP,
                            device='cuda')

