import matplotlib.pyplot as plt
import numpy as np
import csv
import os

#importing system to get to config file
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "config"))
sys.path.append(str(project_root / "config"))
from config import Config

def plot_training_curves(train_losses, val_losses, val_ious, save_dir=Config.SAVE_DIR):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # mIoU curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ious, label="Val mIoU", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_precision_recall(results, class_names):
    precisions = list(results["class_precision"].values())
    recalls = list(results["class_recall"].values())
    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, precisions, width, label="Precision")
    plt.bar(x + width / 2, recalls, width, label="Recall")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Per-Class Precision and Recall")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm_norm, class_names):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm_norm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Proportion")

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            plt.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.show()


def log_metrics_to_csv(csv_path, log_entry):
    """Append metrics to CSV file, creating header if new."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)
