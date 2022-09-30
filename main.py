from Experiments import run_hyperparam_clf, sanity_check, BaseExp, single_exp_loop
from Models import Configuration, AutoEncoder
import torch
# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'
VAE_ADV_EXP = '../ExperimentLogging/AdversarialExpVAE/'
if __name__ == '__main__':
    single_exp_loop(TRAIN_METRICS_DIR, VAE_ADV_EXP, 'cuda')


