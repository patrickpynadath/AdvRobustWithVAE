from Experiments import run_hyperparam_clf, sanity_check, BaseExp
from Models import Configuration, AutoEncoder
import torch
# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'

if __name__ == '__main__':
    exp = BaseExp(training_logdir=TRAIN_METRICS_DIR,
                  exp_logdir=HYPERPARAM_DIR,
                  device='cuda')
    for model_name in ['betatc_vae', 'lvae', 'iwae', 'logcosh_vae', 'vae', 'vq_vae']:
        exp.get_trained_vae(64, 150, model_name, latent_dim=100,in_channels=3)


