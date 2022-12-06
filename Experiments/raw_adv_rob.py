import pickle
from fractions import Fraction
import yaml
from .base_exp import BaseExp
from Models import GenClf, load_models


def run_raw_adv_rob(device):
    with open("./Params/exp_params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    num_steps = int(params['adv_params']['num_steps'])
    l2_eps = [float(eps) for eps in params['adv_params']['l2_eps']]
    linf_eps = [float(Fraction(eps)) for eps in params['adv_params']['linf_eps']]

    l2_accs = {}
    linf_accs = {}
    model_dct = load_models(device)
    exp = BaseExp(device)
    print(f"Eval Smooth Resnet")
    clf = model_dct['resnetGaussian']
    nat_acc = exp.eval_smoothclf_nat_raw(clf, .25, 100, .1)
    adv_accs = exp.eval_smoothclf_adv_raw(clf, .25, 100, 'l2', .1, l2_eps, num_steps)
    l2_accs["resnetSmooth"] = [nat_acc] + adv_accs

    adv_accs = exp.eval_smoothclf_adv_raw(clf, .25, 100, 'linf', .1, linf_eps, num_steps)
    linf_accs['resnetSmooth'] = [nat_acc] + adv_accs

    for key in ['resnet', 'resnetGaussian']:
        print(f"Eval base {key}")
        clf = model_dct[key]
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs_l2 = exp.eval_clf_adv_raw(clf, 'l2', l2_eps, num_steps)
        adv_accs_linf = exp.eval_clf_adv_raw(clf, 'linf', linf_eps, num_steps)
        l2_accs[key] = [nat_acc] + adv_accs_l2
        linf_accs[key] = [nat_acc] + adv_accs_linf

    resnet = model_dct['resnet']

    for key in ['ae', 'vae', 'vqvae']:
        print(f"Eval {key}")
        clf = GenClf(model_dct[key], resnet)
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs = exp.eval_clf_adv_raw(clf, 'linf', linf_eps, num_steps)
        linf_accs[key] = [nat_acc] + adv_accs

        print(f"Eval ensemble {key}")
        clf = GenClf(model_dct[key], model_dct[f"resnet_{key}"])
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs = exp.eval_clf_adv_raw(clf, 'linf', linf_eps, num_steps)
        linf_accs[f"resnet_{key}"] = [nat_acc] + adv_accs

    for key in ['ae', 'vae', 'vqvae']:
        print(f"Eval {key}")
        clf = GenClf(model_dct[key], resnet)
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs = exp.eval_clf_adv_raw(clf, 'l2', l2_eps, num_steps)
        l2_accs[key] = [nat_acc] + adv_accs

        print(f"Eval ensemble {key}")
        clf = GenClf(model_dct[key], model_dct[f"resnet_{key}"])
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs = exp.eval_clf_adv_raw(clf, 'l2', l2_eps, num_steps)
        l2_accs[f"resnet_{key}"] = [nat_acc] + adv_accs



    res = {'l2': l2_accs, 'linf': linf_accs, 'l2_eps': l2_eps, 'linf_eps': linf_eps}
    with open("adv_rob_res_raw.pickle", "wb") as stream:
        pickle.dump(res, stream)
    return


