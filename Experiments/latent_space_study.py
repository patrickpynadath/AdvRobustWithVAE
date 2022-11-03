from .helper_functions import get_latent_code_ae, get_latent_code_vae, get_latent_code_vqvae, \
    get_norm_comparison, get_norm_constrained_noise, get_adv_examples
from tqdm import tqdm
import pandas as pd
import torch
import random

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
            gen_model = model_dct[m].to(device)
            gen_model.eval()
            codes = latent_code_fn[m](gen_model, to_encode)
            differences_same_class = get_norm_comparison(codes[0, :] - codes[1, :], batch=False)
            differences_diff_class = get_norm_comparison(codes[0, :] - codes[2, :], batch=False)

            for t in norm_types:
                norm_data_same[t][m].append(differences_same_class[t])
                norm_data_diff[t][m].append(differences_diff_class[t])
    res = {'same_class' : norm_data_same, 'diff_class' : norm_data_diff}
    df = pd.DataFrame()
    for k1 in res.keys():
        for k2 in res[k1].keys():
            for k3 in res[k1][k2].keys():
                df[f"{k1}_{k2}_{k3}"] = res[k1][k2][k3]

    return df


# will only focus on the VAE -- only one seemingly demonstrating significant change in robustness properties
# retrieve codes and reconstructions for original inputs, noise inputs, and adversarial inputs
def get_generative_outputs(gen_model, get_latent_code, clf, natural_imgs, labels, adv_type, eps, steps, device):
    # original codes
    orig_codes = get_latent_code(gen_model, natural_imgs)
    # get noise codes
    noise = get_norm_constrained_noise(natural_imgs, eps, adv_type, device)
    noise_imgs = natural_imgs + noise
    noise_codes = get_latent_code(gen_model, noise_imgs)
    # get adv codes
    adv_imgs = get_adv_examples(clf,
                                eps,
                                adv_type,
                                steps,
                                natural_imgs,
                                labels)
    adv_codes = get_latent_code(gen_model, adv_imgs)
    adv_orig_diff = get_norm_comparison((adv_imgs - natural_imgs).detach().cpu())[adv_type]
    noise_orig_diff = get_norm_comparison((noise_imgs - natural_imgs).detach().cpu())[adv_type]
    adv_code_norms = get_norm_comparison(adv_codes.detach().cpu())[adv_type]
    noise_code_norms = get_norm_comparison(noise_codes.detach().cpu())[adv_type]
    nat_code_norms = get_norm_comparison(orig_codes.detach().cpu())[adv_type]
    # getting reconstructions
    orig_recon = gen_model.generate(natural_imgs)
    noise_recon = gen_model.generate(noise_imgs)
    adv_recon = gen_model.generate(adv_imgs)

    orig_recon_diff = get_norm_comparison((orig_recon - natural_imgs).detach().cpu())[adv_type]
    orig_advrecon_diff = get_norm_comparison((adv_recon - natural_imgs).detach().cpu())[adv_type]
    orig_noiserecon_diff = get_norm_comparison((noise_recon - natural_imgs).detach().cpu())[adv_type]
    code_res = {"orig": orig_codes.detach().cpu(), "noise": noise_codes.detach().cpu(), "adv": adv_codes.detach().cpu()}
    recon_res = {"orig": orig_recon.detach().cpu(), "noise": noise_recon.detach().cpu(),
                 "adv": adv_recon.detach().cpu()}
    return {"codes": code_res, "recon": recon_res, "adv_orig_diff":
        adv_orig_diff, "noise_orig_diff": noise_orig_diff, 'orig_recon_diff': orig_recon_diff,
            'orig_noiserecon_diff': orig_noiserecon_diff, 'orig_advrecon_diff': orig_advrecon_diff,
            'nat_code_norms':nat_code_norms, 'noise_code_norms':noise_code_norms,
            'adv_code_norms': adv_code_norms}


# given a dict of the codes and recon for the orig, noise, and adv inputs,
# returns dict containing diff of orig and noise vs orig and adv


def peturbation_analysis(data_loader,
                         gen_model,
                         latent_code_fn,
                         clf,
                         adv_type,
                         eps,
                         steps,
                         device):
    total_code_res = {"noise_diff": [], "adv_diff": [], "actual_code": []}
    total_recon_res = {"noise_diff": [], "adv_diff": []}
    total_res = {"codes": total_code_res, "recon": total_recon_res,
                 "adv_orig_diff": [], "noise_orig_diff": [], "orig_recon_diff": [],
                 "orig_advrecon_diff": [], "orig_noiserecon_diff": [],
                 "noise_code_norms": [], "nat_code_norms": [],
                 "adv_code_norms": []}
    gen_model.to(device)
    clf.to(device)
    pg_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, batch in pg_bar:
        data, labels = batch
        data = data.to(device)

        batch_peturb_analysis = get_generative_outputs(gen_model, latent_code_fn,
                                                       clf, data, labels, adv_type, eps, steps, device)
        total_res["adv_orig_diff"] += list(batch_peturb_analysis["adv_orig_diff"])
        total_res["noise_orig_diff"] += list(batch_peturb_analysis["noise_orig_diff"])
        total_res["orig_recon_diff"] += list(batch_peturb_analysis["orig_recon_diff"])
        total_res["orig_advrecon_diff"] += list(batch_peturb_analysis["orig_advrecon_diff"])
        total_res["orig_noiserecon_diff"] += list(batch_peturb_analysis["orig_noiserecon_diff"])
        total_res["adv_code_norms"] += list(batch_peturb_analysis["adv_code_norms"])
        total_res["nat_code_norms"] += list(batch_peturb_analysis["nat_code_norms"])
        total_res["noise_code_norms"] += list(batch_peturb_analysis["noise_code_norms"])

        for k1 in ["codes", "recon"]:
            orig = batch_peturb_analysis[k1]['orig']
            noise = batch_peturb_analysis[k1]['noise']
            adv = batch_peturb_analysis[k1]['adv']
            noise_norm_diffs = get_norm_comparison(noise - orig)[adv_type]
            adv_norm_diffs = get_norm_comparison(orig - adv)[adv_type]
            total_res[k1]["noise_diff"] += list(noise_norm_diffs)
            total_res[k1]["adv_diff"] += list(adv_norm_diffs)
            if k1 == 'codes':
                total_res[k1]["actual_code"] += list(adv)
            torch.cuda.empty_cache()

        data.detach().cpu()
    df = pd.DataFrame()
    df["noise_code_nat_code_diff"] = total_res["codes"]["noise_diff"]
    df["adv_code_nat_code_diff"] = total_res["codes"]["adv_diff"]
    df["noise_recon_nat_recon_diff"] = total_res["recon"]["noise_diff"]
    df["adv_recon_nat_recon_diff"] = total_res["recon"]["adv_diff"]
    df["adv_orig_diff"] = total_res["adv_orig_diff"]
    df["noise_orig_diff"] = total_res["noise_orig_diff"]
    df["orig_recon_diff"] = total_res["orig_recon_diff"]
    df["orig_noiserecon_diff"] = total_res["orig_noiserecon_diff"]
    df["orig_advrecon_diff"] = total_res["orig_advrecon_diff"]
    df["noise_code_norms"] = total_res["noise_code_norms"]
    df["nat_code_norms"] = total_res["nat_code_norms"]
    df["adv_code_norms"] = total_res["adv_code_norms"]

    return df


def get_total_res(exp, model_dct, device, steps=8):
    peturb_res_total = {}
    norms = {'l2': [.25, .5, 1, 1.5, 2, 4, 6, 10, 15], 'linf': [2 / 255, 5 / 255, 10 / 255]}
    for m in ['ae', 'vae', 'vqvae']:
        adv_type_res = {}
        for adv_type in norms.keys():
            eps_res = {}
            print(f"{adv_type}")
            for eps in norms[adv_type]:
                print(eps)
                eps_res[eps] = peturbation_analysis(exp.test_loader, model_dct[m],
                                                    latent_code_fn[m], model_dct[f"resnet_{m}"], adv_type, eps, steps,
                                                    device)
            adv_type_res[adv_type] = eps_res
        peturb_res_total[m] = adv_type_res
    return peturb_res_total