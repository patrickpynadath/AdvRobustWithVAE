import matplotlib.pyplot as plt
import pickle
import os


def graph_adv_rob_res_total():
    adv_res = pickle.load(open('adv_rob_res_raw.pickle', 'rb'))
    adv_types = ['l2', 'linf']
    for adv in adv_types:
        eps = adv_res[f'{adv}_eps']
        for key in ["resnet_vqvae", "resnet_vae", "resnet_ae", "resnetSmooth", "resnet"]:
            plt.plot([0] + eps, adv_res[adv][key], label=key)
        plt.title(f"Adv Accuracy Scores {adv} All Eps")
        plt.legend(loc='upper right')
        figure = plt.gcf()
        figure.savefig(f"Plots/adv_rob_{adv}_total")
    return


def graph_adv_rob_res_local():
    adv_res = pickle.load(open('adv_rob_res_raw.pickle', 'rb'))
    adv_types = ['l2', 'linf']
    for adv in adv_types:
        eps = adv_res[f'{adv}_eps']
        for key in ["resnet_vqvae", "resnet_vae", "resnet_ae", "resnetSmooth", "resnet"]:
            x_total = [0] + eps
            plt.plot(x_total[:7], adv_res[adv][key][:7], label=key)
        plt.title(f"Adv Accuracy Scores {adv} Local Eps")
        plt.legend(loc='upper right')
        figure = plt.gcf()
        figure.savefig(f"Plots/adv_rob_{adv}_local")
    return


def plt_orig_rep_norm_diff(class_comp):
    for class_idx in range(10):
        pass
    return


def plt_latent_rep_norm_diff(class_comp):
    return


def plt_recon_rep_norm_diff(class_comp):
    return


def plt_peturb_corr_analysis(peturb_res):
    return


def plt_peturb_recon_analysis(peturb_res):
    return




