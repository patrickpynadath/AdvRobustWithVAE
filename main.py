from Experiments import run_hyperparam_clf, run_hyperparam_vae, sanity_check

# global experiment variables that stay constant for every experiment
# logistical parameters
TRAIN_METRICS_DIR = '../ExperimentLogging/TrainMetrics/'
HYPERPARAM_DIR = '../ExperimentLogging/HyperParamMetrics/'

if __name__ == '__main__':
    sanity_check()
    clf_params, clf_score = run_hyperparam_clf()
    #vae_params, vae_loss = run_hyperparam_vae()
    print(f"Best Params for Clf with score {round(clf_score, 3)}: \n")
    print(clf_params)
    # print(f"Best Params for VAE with loss {round(vae_loss, 5)} \n")
    # print(vae_params)
