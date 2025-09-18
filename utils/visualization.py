# utils/visualization.py

import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as F

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

    

    def visualize_model_predictions(model, data_loader, device, num_samples=4, alpha=0.5, save_dir=None, prefix="predictions"):
        """
        Visualize model predictions against ground truth labels with input images.
        Shows: Pre-fire, Post-fire, Ground Truth Labels, Model Predictions, and Difference Map
        """
        model.eval()
        class_names = ['Unburned', 'Low', 'Moderate', 'High', 'Non-processing/Cloud']
        
        # Get a batch of data
        inputs, labels = next(iter(data_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
        
        # Move back to CPU for visualization
        inputs = inputs.cpu()
        labels = labels.cpu()
        predictions = predictions.cpu()
        
        from matplotlib import gridspec
        fig = plt.figure(figsize=(25, 5 * num_samples))
        gs = gridspec.GridSpec(num_samples, 6, width_ratios=[1, 1, 1, 1, 1, 0.15])
        
        for i in range(min(num_samples, inputs.size(0))):
            pre_fire = inputs[i, :3, :, :].permute(1, 2, 0).numpy()
            post_fire = inputs[i, 3:6, :, :].permute(1, 2, 0).numpy()
            label = labels[i].numpy()
            prediction = predictions[i].numpy()
            
            # Create difference map (correct=0, incorrect=1)
            difference = (label != prediction).astype(np.uint8)
            
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
            
            # Ground Truth Labels
            ax3 = fig.add_subplot(gs[i, 2])
            im = ax3.imshow(label, cmap=burn_cmap, vmin=0, vmax=4)
            ax3.set_title("Ground Truth")
            ax3.axis("off")
            
            # Model Predictions
            ax4 = fig.add_subplot(gs[i, 3])
            ax4.imshow(prediction, cmap=burn_cmap, vmin=0, vmax=4)
            ax4.set_title("Model Prediction")
            ax4.axis("off")
            
            # Difference Map (Red = incorrect, transparent = correct)
            ax5 = fig.add_subplot(gs[i, 4])
            ax5.imshow(post_fire)
            ax5.imshow(difference, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
            ax5.set_title("Errors (Red)")
            ax5.axis("off")
            
            # Calculate accuracy for this sample
            accuracy = np.mean(label == prediction) * 100
            ax5.text(0.02, 0.98, f'Acc: {accuracy:.1f}%', transform=ax5.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=10)
            
            # Add colorbar once
            if i == 0:
                cbar_ax = fig.add_subplot(gs[i, 5])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_ticks(range(5))
                cbar.set_ticklabels(class_names)
                cbar.ax.tick_params(labelsize=9)
        
        plt.tight_layout()
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{prefix}_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved prediction visualization to: {save_path}")
        
        plt.show()


    def visualize_single_prediction(model, dataset, device, idx, alpha=0.5, save_dir=None):
        """
        Visualize a single sample prediction with detailed analysis.
        """
        model.eval()
        class_names = ['Unburned', 'Low', 'Moderate', 'High', 'Non-processing/Cloud']
        
        # Get single sample
        input_tensor, label_tensor = dataset[idx]
        sample_file = dataset.sample_files[idx]
        
        # Add batch dimension and move to device
        inputs = input_tensor.unsqueeze(0).to(device)
        labels = label_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
        
        # Move back to CPU
        inputs = inputs.cpu()
        labels = labels.cpu()
        prediction = prediction.cpu()
        probabilities = probabilities.cpu()
        
        # Extract data
        pre_fire = inputs[0, :3, :, :].permute(1, 2, 0).numpy()
        post_fire = inputs[0, 3:6, :, :].permute(1, 2, 0).numpy()
        label = labels[0].numpy()
        pred = prediction[0].numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Sample {idx} — {sample_file}", fontsize=14)
        
        # Top row: Input images and ground truth
        axes[0, 0].imshow(pre_fire)
        axes[0, 0].set_title("Pre-fire")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(post_fire)
        axes[0, 1].set_title("Post-fire")
        axes[0, 1].axis("off")
        
        im1 = axes[0, 2].imshow(label, cmap=burn_cmap, vmin=0, vmax=4)
        axes[0, 2].set_title("Ground Truth")
        axes[0, 2].axis("off")
        
        # Bottom row: Prediction, overlay, and confidence
        im2 = axes[1, 0].imshow(pred, cmap=burn_cmap, vmin=0, vmax=4)
        axes[1, 0].set_title("Model Prediction")
        axes[1, 0].axis("off")
        
        # Overlay with errors highlighted
        difference = (label != pred)
        axes[1, 1].imshow(post_fire)
        axes[1, 1].imshow(pred, cmap=burn_cmap, alpha=alpha, vmin=0, vmax=4)
        axes[1, 1].imshow(difference, cmap='Reds', alpha=0.7)
        axes[1, 1].set_title("Prediction Overlay + Errors")
        axes[1, 1].axis("off")
        
        # Confidence map (max probability across classes)
        max_conf = torch.max(probabilities[0], dim=0)[0].numpy()
        im3 = axes[1, 2].imshow(max_conf, cmap='viridis', vmin=0, vmax=1)
        axes[1, 2].set_title("Prediction Confidence")
        axes[1, 2].axis("off")
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # Add main colorbar
        plt.colorbar(im1, ax=axes[:, :2].ravel().tolist(), fraction=0.046, pad=0.04, 
                    ticks=range(5), label='Burn Severity Classes')
        
        # Calculate and display metrics
        accuracy = np.mean(label == pred) * 100
        unique_gt, counts_gt = np.unique(label, return_counts=True)
        unique_pred, counts_pred = np.unique(pred, return_counts=True)
        
        gt_dist = {class_names[l]: (c / label.size) * 100 for l, c in zip(unique_gt, counts_gt)}
        pred_dist = {class_names[l]: (c / pred.size) * 100 for l, c in zip(unique_pred, counts_pred)}
        
        # Add text summary
        text_summary = f"Pixel Accuracy: {accuracy:.1f}%\n"
        text_summary += f"Mean Confidence: {max_conf.mean():.3f}\n\n"
        text_summary += "Ground Truth Distribution:\n"
        for cls, pct in gt_dist.items():
            text_summary += f"  {cls}: {pct:.1f}%\n"
        
        fig.text(0.02, 0.02, text_summary, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"single_prediction_{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved single prediction to: {save_path}")
        
        plt.show()
        
        return {
            'accuracy': accuracy,
            'ground_truth_dist': gt_dist,
            'prediction_dist': pred_dist,
            'mean_confidence': max_conf.mean()
        }

