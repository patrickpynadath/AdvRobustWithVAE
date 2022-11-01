import argparse
from Experiments import run_raw_adv_rob, train_models, load_models
from Utils import graph_adv_rob_res

# TODO: - Add plotting to scripts so it outputs well formatted graphs
# TODO: - Debug -- make sure what works right now works
# TODO: - Add script for latent code comparison, include analysis on auto encoder and vq vae
# TODO: - Add test for looking at certified accuracies of models
# TODO: - Add a function for quickly editing the YAML config files -- probably make my life easier in the long run

# Directory for tensorboard logging -- useful to keep track of training progress for models
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
ROBUST_DIR = '../ExperimentLogging/AdvRobRes'
LATENT_DIR = '../ExperimentLogging/LatentSpaceRes'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Experiments!')
    parser.add_argument('pretrained', help='use pretrained models', type=bool)
    parser.add_argument('device', help='device to run experiment on', type=str)
    args = parser.parse_args()
    if not args.pretrained:
        train_models(TRAIN_METRICS_DIR, args.device)
    run_raw_adv_rob(args.device)
    graph_adv_rob_res()



