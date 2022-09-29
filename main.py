from Experiments import run_hyperparam_clf, sanity_check, BaseExp
from Models import Configuration, AutoEncoder
import torch
import os
# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'

if __name__ == '__main__':
    # exp = BaseExp(training_logdir=TRAIN_METRICS_DIR,
    #               exp_logdir=HYPERPARAM_DIR,
    #               device='cuda')
    # for model_name in ['vae', 'vq_vae']:
    #     exp.get_trained_vae(64, 150, model_name, latent_dim=100,in_channels=3)
    print(os.curdir)
    conf = Configuration()
    autoencoder = AutoEncoder('cpu', conf)
    autoencoder.load_state_dict(torch.load("/saved_models/model_ema_norm_he-et-al.pth"))

