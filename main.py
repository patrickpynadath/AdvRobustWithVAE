from Experiments import run_hyperparam_clf, sanity_check, BaseExp

# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'

if __name__ == '__main__':
    exp = BaseExp(training_logdir=TRAIN_METRICS_DIR,
                  exp_logdir=HYPERPARAM_DIR,
                  device='cpu')
    exp.get_trained_vae(64, 3, 'vampvae', latent_dim=100,in_channels=3)