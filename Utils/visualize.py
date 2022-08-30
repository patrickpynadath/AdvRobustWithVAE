import matplotlib.pyplot as plt
import numpy as np

def create_training_val_results(train_acc_v_epoch, train_loss_v_epoch, val_acc_v_epoch, val_loss_v_epoch, grad_mean, grad_var, image_title, fname):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    num_epochs = len(train_acc_v_epoch)
    fig.suptitle(image_title)
    axs[0].set_title("Training Loss v Val Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Avg Loss per sample")
    axs[0].plot(np.linspace(1, num_epochs, num=num_epochs, endpoint=True), train_loss_v_epoch, c='blue', label="Train Loss")
    axs[0].plot(np.linspace(1, num_epochs, num=num_epochs, endpoint=True), val_loss_v_epoch, c='red', label="Val Loss")

    axs[1].set_title("Training Acc v Val Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].plot(np.linspace(1, num_epochs, num=num_epochs, endpoint=True), train_acc_v_epoch, c='blue', label="Train Acc")
    axs[1].plot(np.linspace(1, num_epochs, num=num_epochs, endpoint=True), val_acc_v_epoch, c='red', label="Val Acc")

    axs[2].set_title("Mean of Gradient Norm vs updates")
    axs[2].set_xlabel("Update")
    axs[2].set_ylabel("Gradient l2 norm")
    axs[2].plot(np.linspace(1, len(grad_mean), num=len(grad_mean), endpoint=True), grad_mean, c='blue', label = 'Mean')
    axs[2].plot(np.linspace(1, len(grad_mean), num=len(grad_mean), endpoint=True), grad_var, c='red', label='Var')
    fig.savefig(f"{fname}")
    return
