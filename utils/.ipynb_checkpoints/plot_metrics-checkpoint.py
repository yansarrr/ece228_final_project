import matplotlib.pyplot as plt
import os

def plot_training_metrics(train_rocauc, valid_rocauc, test_rocauc, loss_list, save_dir="graphs", prefix=""):
    """
    Plots ROC-AUC (train/val/test) and training loss vs. epoch.

    Parameters:
        train_rocauc (List[float]): ROC-AUC scores for training set per epoch.
        valid_rocauc (List[float]): ROC-AUC scores for validation set per epoch.
        test_rocauc (List[float]): ROC-AUC scores for test set per epoch.
        loss_list (List[float]): Training loss per epoch.
        save_dir (str): Directory to save the plots. Default is 'graphs'.
        prefix (str): Optional filename prefix for saved plots.
    """
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"The save directory '{save_dir}' does not exist.")

    epochs = list(range(1, len(loss_list) + 1))

    # --- Plot ROC-AUC ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_rocauc, label='Train ROC-AUC')
    plt.plot(epochs, valid_rocauc, label='Valid ROC-AUC')
    plt.plot(epochs, test_rocauc, label='Test ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}roc_auc_plot.png"))
    plt.close()

    # --- Plot Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_list, color='red', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}loss_plot.png"))
    plt.close()
