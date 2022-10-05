from Experiments import run_hyperparam_clf, sanity_check, BaseExp, single_exp_loop_vae_adv, run_adv_rob_exp
from Models import Configuration, AutoEncoder
import torch
from Experiments import train_save_necessary_models
# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'
VAE_ADV_EXP = '../ExperimentLogging/AdversarialExpVAE/'
ADV_ROB_EXP = '../ExperimentLogging/AdvRobExp/'
if __name__ == '__main__':
    exp = BaseExp(training_logdir=TRAIN_METRICS_DIR,
                  exp_logdir=ADV_ROB_EXP,
                  device='cuda')
    vq_vae = exp.get_trained_vqvae(150)
    torch.save(vq_vae.state_dict(), 'saved_models/vq_vae')
