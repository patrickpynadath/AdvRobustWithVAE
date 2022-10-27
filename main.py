from Experiments import run_hyperparam_clf, sanity_check, BaseExp, single_exp_loop_vae_adv, run_adv_rob_exp
import torch
from Experiments import train_save_necessary_models
from Experiments import run_adv_robust
# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'
VAE_ADV_EXP = '../ExperimentLogging/AdversarialExpVAE/'
ADV_ROB_EXP = '../ExperimentLogging/AdvRobExp/'
if __name__ == '__main__':

    #get_trained_vq_vae(training_logdir=TRAIN_METRICS_DIR, epochs=100, device='cpu')
    exp = BaseExp(training_logdir=TRAIN_METRICS_DIR,
                  exp_logdir=VAE_ADV_EXP,
                  device='cuda')
    resnet = exp.get_trained_resnet(net_depth=110, block_name='BottleNeck', batch_size=64, optimizer='sgd', lr=.1, epochs=150,
                                    use_step_lr=True, lr_schedule_step=50, lr_schedule_gamma=.1)
    torch.save(resnet.state_dict(), 'PretrainedModels/ResnetBase')
    #run_adv_robust()
