import matplotlib.pyplot as plt
import pickle


def graph_adv_rob_res():
    adv_res = pickle.load(open('adv_rob_res_raw.pickle', 'rb'))
    adv_types = ['l2', 'linf']
    for adv in adv_types:
        eps = adv_res[f'{adv}_eps']
        for key in adv_res[adv].keys():
            plt.plot(eps + [0], adv_res[adv][key], label=key)
        plt.title(f"Adv Accuracy Scores {adv}")
        plt.legend(loc='upper right')
        figure = plt.gcf()
        figure.savefig(f"Plots/adv_rob_{adv}")
    return
