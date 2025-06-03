import csv
import os
from datetime import datetime

def save_training_metrics(train_rocauc, valid_rocauc, test_rocauc, loss_list, save_dir="results", prefix="run"):
    """
    Saves training metrics to a CSV file after training.

    Parameters:
        train_rocauc (List[float]): Training ROC-AUC per epoch.
        valid_rocauc (List[float]): Validation ROC-AUC per epoch.
        test_rocauc (List[float]): Test ROC-AUC per epoch.
        loss_list (List[float]): Training loss per epoch.
        save_dir (str): Directory to save the CSV file.
        prefix (str): Filename prefix for the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_metrics_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train ROC-AUC", "Valid ROC-AUC", "Test ROC-AUC", "Loss"])
        for epoch, (tr, va, te, loss) in enumerate(zip(train_rocauc, valid_rocauc, test_rocauc, loss_list), 1):
            writer.writerow([epoch, tr, va, te, loss])

    print(f"[✔] Metrics saved to {filepath}")
