import pickle
from fractions import Fraction

import yaml

from .helper_functions import load_models
from .base_exp import BaseExp
from Models import GenClf


def run_raw_adv_rob(device):
    with open("./Params/exp_params.yaml", "r") as stream:
        params = yaml.safe_load(stream)
    num_steps = int(params['adv_params']['num_steps'])
    l2_eps = [float(eps) for eps in params['adv_params']['l2_eps']]
    linf_eps = [float(Fraction(eps)) for eps in params['adv_params']['linf_eps']]

    l2_accs = {'ae' : [], 'vae' : [], 'vqvae' : [], 'resnetSmooth': [], 'resnet' : []}
    linf_accs = {'ae' : [], 'vae' : [], 'vqvae' : [], 'resnetSmooth': [], 'resnet' : []}
    model_dct = load_models(device)
    exp = BaseExp(device)
    for key in ['resnet', 'resnetSmooth']:
        print(f"Eval base {key}")
        clf = model_dct[key]
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs_l2 = exp.eval_clf_adv_raw(clf, 'l2', l2_eps, num_steps)
        adv_accs_linf = exp.eval_clf_adv_raw(clf, 'linf', linf_eps, num_steps)
        l2_accs[key] = [nat_acc] + adv_accs_l2
        linf_accs[key] = [nat_acc] + adv_accs_linf

    resnet = model_dct['resnet']

    for key in linf_accs.keys():
        print(f"Eval {key}")
        clf = GenClf(model_dct[key], resnet)
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs = exp.eval_clf_adv_raw(clf, 'linf', linf_eps, num_steps)
        linf_accs[key] = [nat_acc] + adv_accs

    for key in l2_accs.keys():
        print(f"Eval {key}")
        clf = GenClf(model_dct[key], resnet)
        nat_acc = exp.eval_clf_clean(clf)
        adv_accs = exp.eval_clf_adv_raw(clf, 'l2', l2_eps, num_steps)
        l2_accs[key] = [nat_acc] + adv_accs

    res = {'l2': l2_accs, 'linf': linf_accs, 'l2_eps': l2_eps, 'linf_eps': linf_eps}
    with open("adv_rob_res_raw.pickle", "wb") as stream:
        pickle.dump(res, stream)
    return
