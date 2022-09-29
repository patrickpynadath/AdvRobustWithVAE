from Models import ResNet, Conv_VAE
from Utils import get_cifar_sets
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import optuna
import torch
from torch.utils.tensorboard import SummaryWriter
import os

os.chdir("../")
# A bit shoddy, but hardcoding the device to use -- keeping until I
# find a better way to pass the device to the objective
DEVICE = "cuda"


def objective_clf(trial: optuna.trial.Trial):
    # summary writer

    epochs = 100
    train_set, test_set = get_cifar_sets()
    train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

    block_name = trial.suggest_categorical("block_name", ["BasicBlock", "BottleNeck"])
    optimizer_name = trial.suggest_categorical("optimizer_name", ["adam", "sgd"])
    start_lr = trial.suggest_float("starting_lr", 1e-6, .1)
    lr_gamma = trial.suggest_float("lr_gamma", 1e-6, .9)
    lr_step_size = trial.suggest_int("lr_step", 1, 1000)
    criterion = torch.nn.CrossEntropyLoss()
    depth = 20
    resnet = ResNet(depth=depth,
                    num_classes=10,
                    block_name=block_name).to(DEVICE)
    hyperparam_tag = f"{optimizer_name}_startlr_{round(start_lr, 5)}_gamma_{round(lr_gamma, 5)}_step_{lr_step_size}"
    sw = SummaryWriter(log_dir=f"../ExperimentLogging/HyperParamExp/{hyperparam_tag}")
    if optimizer_name == 'adam':
        optimizer = Adam(resnet.parameters(),
                         lr=start_lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(resnet.parameters(),
                        lr=start_lr)
    lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    for epoch in range(epochs):
        resnet.train()

        # training
        total_correct = 0
        total = 0
        for batch_idx, batch in enumerate(train_loader):
            data, labels = batch
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = resnet(data)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()
            optimizer.step()
            pred = torch.argmax(outputs, dim=1)
            num_correct = sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(data))])
            total += len(data)
            total_correct += num_correct
        train_acc = total_correct / total
        sw.add_scalar("Train/Acc", train_acc, epoch)

        # validation
        with torch.no_grad():
            total_correct = 0
            total = 0
            for batch_idx, batch in enumerate(test_loader):
                data, labels = batch
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = resnet(data)
                pred = torch.argmax(outputs, dim=1)
                num_correct = sum([1 if pred[i].item() == labels[i].item() else 0 for i in range(len(data))])
                total += len(data)
                total_correct += num_correct
            val_acc = total_correct / total
            sw.add_scalar("Val/Acc", val_acc, epoch)
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return val_acc


def objective_vae(trial: optuna.trial.Trial):
    epochs = 100
    train_set, test_set = get_cifar_sets()
    train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
    beta = trial.suggest_float("vae_beta", .1, 1)
    kernel_num = trial.suggest_int("kern_num", 8, 64)
    latent_size = trial.suggest_int("latent_size", 50, 200)
    vae = Conv_VAE(image_size=32, channel_num=3, kernel_num=kernel_num, z_size=latent_size, device=DEVICE, beta=beta)
    sw = SummaryWriter(log_dir=f"../ExperimentLogging/HyperParamExp/{vae.label}")
    optimizer = SGD(vae.parameters(),
                    lr=3e-04,
                    weight_decay=1e-5)

    for epoch in range(epochs):
        print(epoch)
        vae.train()
        recon_loss_epoc_train = 0
        for batch_idx, batch in enumerate(train_loader):
            print(batch_idx)
            inputs, labels = batch
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            reconstruction = vae(inputs)
            mean, logvar = vae.get_mean_logvar(inputs)
            reconstruction_loss = vae.reconstruction_loss(reconstruction, inputs)
            kl_divergence_loss = vae.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss - vae.beta * kl_divergence_loss

            # backprop gradients from the loss
            recon_loss_epoc_train += reconstruction_loss.item()
            total_loss.backward()
            optimizer.step()
        sw.add_scalar("Train/ReconLoss", recon_loss_epoc_train, epoch)

        with torch.no_grad():
            recon_loss_epoc_val = 0
            for batch_idx, batch in enumerate(test_loader):
                inputs, labels = batch
                inputs = inputs.to(DEVICE)
                reconstruction = vae(inputs)
                reconstruction_loss = vae.reconstruction_loss(reconstruction, inputs)
                recon_loss_epoc_val += reconstruction_loss.item()
            sw.add_scalar("Val/ReconLoss", recon_loss_epoc_val, epoch)
            trial.report(recon_loss_epoc_val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return recon_loss_epoc_val


def run_hyperparam_vae():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_vae, n_trials=100)


def run_hyperparam_clf():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_clf, n_trials=100)


if __name__ == '__main__':
    run_hyperparam_vae()
