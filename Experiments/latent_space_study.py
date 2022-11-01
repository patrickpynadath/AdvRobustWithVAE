from Utils import get_cifar_sets, get_label_idx
from Experiments.helper_functions import get_latent_code_ae, get_latent_code_vae, get_latent_code_vqvae, \
    get_norm_comparison
from Models import AE, VAE, VQVAE_NSVQ
import torch
import random
import pickle

latent_code_fn = {'ae': get_latent_code_ae, 'vae': get_latent_code_vae, 'vqvae': get_latent_code_vqvae}


# getting baseline comparison: differences within the same class, differences within a different class
def get_random_sample_latent_diffs(class_idx,
                                   model_dct,
                                   idx_dct,
                                   device,
                                   dataset,
                                   num_samples=1000):
    gen_models = ['ae', 'vae', 'vqvae']
    norm_types = ['l2', 'linf']
    norm_data_same = {}
    norm_data_diff = {}
    for t in norm_types:
        dct_same = {}
        dct_diff = {}
        for m in gen_models:
            dct_same[m] = []
            dct_diff[m] = []
        norm_data_same[t] = dct_same
        norm_data_diff[t] = dct_diff

    out_classes = [i for i in range(10)]
    out_classes.remove(class_idx)
    for sample_idx in range(num_samples):
        [orig_sample, in_sample] = random.sample(idx_dct[class_idx], 2)
        # out sample
        out_class = random.sample(out_classes, 1)[0]
        out_sample = random.sample(idx_dct[out_class], 1)[0]
        to_encode = torch.stack([dataset[idx][0] for idx in [orig_sample, in_sample, out_sample]]).to(device)

        for m in gen_models:
            gen_model = model_dct[m]
            codes = latent_code_fn[m](gen_model, to_encode)
            differences_same_class = get_norm_comparison(codes[0, :] - codes[1, :])
            differences_diff_class = get_norm_comparison(codes[0, :] - codes[2, :])

            for t in norm_types:
                norm_data_same[t][m].append(differences_same_class[t])
                norm_data_diff[t][m].append(differences_diff_class[t])
    res = {'same_class' : norm_data_same, 'diff_class' : norm_data_diff}
    with open("RawExpData/latent_class_norms.pickle", 'wb') as stream:
        pickle.dump(res, stream)
    return res


def get_peturbation_comparison(gen_model_type, eps, adv_type):
    return

