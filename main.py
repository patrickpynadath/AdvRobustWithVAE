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
                  device='cuda')
    sigmas = [.1, .25, .3, .4, .5, 1]
    for s in sigmas:
        print(f"Training smooth resnet with sigma {s}")
        smooth_resnet = exp.get_trained_smooth_resnet(net_depth=110,
                                                      block_name='BottleNeck',
                                                      m_train=10,
                                                      batch_size=128,
                                                      optimizer='sgd',
                                                      lr=.15,
                                                      epochs=150,
                                                      smoothing_sigma=s,
                                                      use_step_lr=True,
                                                      lr_schedule_step=50,
                                                      lr_schedule_gamma=.1)
        torch.save(smooth_resnet.state_dict(), f'saved_models/smooth_resnet_sigma_{s}')
        torch.cuda.empty_cache()
    run_adv_robust()
