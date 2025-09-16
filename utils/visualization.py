# utils/visualization.py

import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# --- Custom categorical colormap ---
burn_cmap = ListedColormap(["green", "yellow", "orange", "red", "gray"])


def visualize_batch(data_loader, num_samples: int = 4):
    """
    Visualize samples from a dataloader with pre-fire, post-fire, and label maps.
    Includes a shared colorbar for burn severity.
    """
    inputs, labels = next(iter(data_loader))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(num_samples, inputs.size(0))):
        pre_fire = inputs[i, :3, :, :].permute(1, 2, 0).numpy()
        post_fire = inputs[i, 3:6, :, :].permute(1, 2, 0).numpy()
        label = labels[i].numpy()

        axes[i, 0].imshow(pre_fire)
        axes[i, 0].set_title("Pre-fire")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(post_fire)
        axes[i, 1].set_title("Post-fire")
        axes[i, 1].axis("off")

        im = axes[i, 2].imshow(label, cmap=burn_cmap, vmin=0, vmax=4)
        axes[i, 2].set_title("Burn Severity Labels")
        axes[i, 2].axis("off")

    plt.tight_layout()

    # Add a single colorbar for all samples
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_ticks(range(5))
    cbar.set_ticklabels(["Unburned", "Low", "Moderate", "High", "Non-processing/Cloud"])

    plt.show()


def visualize_batch_with_overlay(
    data_loader, num_samples: int = 4, alpha: float = 0.5, save_dir: str = None, prefix: str = "sample"
):
    """
    Visualize samples with pre-fire, post-fire, label maps, and overlay (labels over post-fire image).
    Optionally saves figures to disk.
    """
    class_names = ["Unburned", "Low", "Moderate", "High", "Non-processing/Cloud"]

    inputs, labels = next(iter(data_loader))

    from matplotlib import gridspec
    fig = plt.figure(figsize=(20, 4 * num_samples))
    gs = gridspec.GridSpec(num_samples, 5, width_ratios=[1, 1, 1, 1, 0.15])

    for i in range(min(num_samples, inputs.size(0))):
        pre_fire = inputs[i, :3, :, :].permute(1, 2, 0).numpy()
        post_fire = inputs[i, 3:6, :, :].permute(1, 2, 0).numpy()
        label = labels[i].numpy()

        # Pre-fire
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(pre_fire)
        ax1.set_title("Pre-fire")
        ax1.axis("off")

        # Post-fire
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(post_fire)
        ax2.set_title("Post-fire")
        ax2.axis("off")

        # Labels
        ax3 = fig.add_subplot(gs[i, 2])
        im = ax3.imshow(label, cmap=burn_cmap, vmin=0, vmax=4)
        ax3.set_title("Burn Severity Labels")
        ax3.axis("off")

        # Overlay
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.imshow(post_fire)
        ax4.imshow(label, cmap=burn_cmap, alpha=alpha, vmin=0, vmax=4)
        ax4.set_title("Overlay")
        ax4.axis("off")

        # Add colorbar once
        if i == 0:
            cbar_ax = fig.add_subplot(gs[i, 4])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_ticks(range(5))
            cbar.set_ticklabels(class_names)

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{prefix}_viz.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved visualization to: {save_path}")

    plt.show()

def show_sample_by_index(dataset, idx: int, alpha: float = 0.5):
    """
    Visualize a single sample from the dataset by its index.
    Includes overlay, filename, colorbar, and class distribution.
    """
    input_tensor, label_tensor = dataset[idx]
    sample_file = dataset.sample_files[idx]

    pre_fire = input_tensor[:3].permute(1, 2, 0).numpy()
    post_fire = input_tensor[3:6].permute(1, 2, 0).numpy()
    label = label_tensor.numpy()

    class_names = ['Unburned', 'Low', 'Moderate', 'High', 'Non-processing/Cloud']

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle(f"Sample {idx} — File: {sample_file}", fontsize=14, y=1.05)

    axes[0].imshow(pre_fire)
    axes[0].set_title("Pre-fire")
    axes[0].axis("off")

    axes[1].imshow(post_fire)
    axes[1].set_title("Post-fire")
    axes[1].axis("off")

    axes[2].imshow(post_fire)
    im = axes[2].imshow(label, cmap=burn_cmap, alpha=alpha, vmin=0, vmax=4)
    axes[2].set_title("Overlay (Burn Severity)")
    axes[2].axis("off")

    # Horizontal colorbar below title
    cbar_ax = fig.add_axes([0.25, 0.9, 0.5, 0.03])
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=burn_cmap, norm=plt.Normalize(vmin=0, vmax=4)),
        cax=cbar_ax,
        orientation="horizontal"
    )
    cbar.set_ticks(range(5))
    cbar.set_ticklabels(class_names)
    cbar.ax.tick_params(labelsize=9)

    # Class distribution
    unique_labels, counts = np.unique(label, return_counts=True)
    percentages = {class_names[l]: (c / label.size) * 100 for l, c in zip(unique_labels, counts)}

    dist_text = " | ".join([f"{cls}: {pct:.1f}%" for cls, pct in percentages.items()])
    fig.text(0.5, 0.02, f"Class Distribution → {dist_text}", ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    plt.show()

