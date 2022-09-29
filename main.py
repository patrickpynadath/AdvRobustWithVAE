from Experiments import run_hyperparam_clf, run_hyperparam_vae

# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'

if __name__ == '__main__':
    run_hyperparam_clf()
    run_hyperparam_vae()