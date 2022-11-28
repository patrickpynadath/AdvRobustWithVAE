import argparse
from Experiments import run_raw_adv_rob, train_models, load_models, get_class_comparisons, get_total_res_peturbation
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
    parser.add_argument('exp_num',
                        help='experiment to run, 0 = all, 1 = raw adv rob, 2 = class comparisons for norm difference, 3 = peturbation exp',
                        type=int)
    args = parser.parse_args()
    if not args.pretrained:
        train_models(TRAIN_METRICS_DIR, args.device)
    device = args.device
    exp_num = args.exp_num
    if exp_num == 0:
        run_raw_adv_rob(args.device)
        get_class_comparisons(device=args.device)
        get_total_res_peturbation(device=args.device, steps=8, ensemble=True)
    elif exp_num == 1:
        run_raw_adv_rob(args.device)
    elif exp_num == 2:
        get_class_comparisons(device=args.device)
    elif exp_num == 3:
        get_total_res_peturbation(device=args.device, steps=8, ensemble=True)



