import argparse
from Experiments.helper_functions import run_raw_adv_rob, train_models, load_models

# things to run from commandline: training the models, running prelim adv accs test
# training the models - make the parameters for the models a text file -- will make my life so much easier

# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'
VAE_ADV_EXP = '../ExperimentLogging/AdversarialExpVAE/'
ADV_ROB_EXP = '../ExperimentLogging/AdvRobExp/'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Experiments!')
    parser.add_argument('pretrained', help='use pretrained models', type=bool, required=True)
    parser.add_argument('device', help='device to run experiment on', type=str, required=True)
    args = parser.parse_args()
    if args.pretrained:
        train_models()
    model_dct = load_models()
    run_raw_adv_rob(args.device)



