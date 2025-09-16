import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class BurnSeverityMetrics:
    """Comprehensive metrics for burn severity mapping evaluation"""
    
    def __init__(self, num_classes: int = 5, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [
            'Unburned', 'Low', 'Moderate', 'High', 'Non-processing/Cloud'
        ]
        self.reset()
        
    def reset(self):
        """Reset all accumulated metrics"""
        self.total_correct = 0
        self.total_pixels = 0
        self.class_correct = torch.zeros(self.num_classes)
        self.class_total = torch.zeros(self.num_classes)
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with a new batch"""
        if predictions.dim() == 4:  # (B, num_classes, H, W)
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions

        pred_flat = pred_classes.flatten().cpu().numpy()
        target_flat = targets.flatten().cpu().numpy()

        # Update confusion matrix
        cm_batch = confusion_matrix(target_flat, pred_flat, labels=range(self.num_classes))
        self.cm += cm_batch

        # Accuracy
        self.total_correct += (pred_flat == target_flat).sum()
        self.total_pixels += target_flat.size

        # Per-class stats
        for class_idx in range(self.num_classes):
            tp = cm_batch[class_idx, class_idx]
            fn = cm_batch[class_idx, :].sum() - tp
            fp = cm_batch[:, class_idx].sum() - tp

            self.intersection[class_idx] += tp
            self.union[class_idx] += tp + fp + fn
            self.class_correct[class_idx] += tp
            self.class_total[class_idx] += cm_batch[class_idx, :].sum()

    def compute_confusion_matrix(self, normalize: Optional[str] = None) -> np.ndarray:
        """Return confusion matrix (optionally normalized)"""
        cm = self.cm.astype(float)
        if normalize == 'true':
            cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
        elif normalize == 'pred':
            cm = cm / (cm.sum(axis=0, keepdims=True) + 1e-8)
        elif normalize == 'all':
            cm = cm / cm.sum()
        return cm

    def compute_iou(self) -> Dict[str, float]:
        iou_per_class = []
        for i in range(self.num_classes):
            if self.union[i] > 0:
                iou = self.intersection[i] / self.union[i]
            else:
                iou = 0.0
            iou_per_class.append(float(iou))
        return {
            'mean_iou': float(np.mean(iou_per_class)),
            'class_iou': {self.class_names[i]: iou_per_class[i] for i in range(self.num_classes)}
        }

    def compute_dice(self) -> Dict[str, float]:
        dice_per_class = []
        for i in range(self.num_classes):
            tp = self.intersection[i]
            target = self.class_total[i]
            pred = (self.union[i] - self.class_total[i] + tp)
            if (target + pred) > 0:
                dice = (2.0 * tp) / (target + pred + 1e-8)
            else:
                dice = 0.0
            dice_per_class.append(float(dice))
        return {
            'mean_dice': float(np.mean(dice_per_class)),
            'class_dice': {self.class_names[i]: dice_per_class[i] for i in range(self.num_classes)}
        }

    def compute_precision_recall(self) -> Dict[str, Dict[str, float]]:
        """Compute precision and recall per class from confusion matrix"""
        precision_per_class = {}
        recall_per_class = {}

        for i, name in enumerate(self.class_names):
            tp = self.cm[i, i]
            fp = self.cm[:, i].sum() - tp
            fn = self.cm[i, :].sum() - tp

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            precision_per_class[name] = precision
            recall_per_class[name] = recall

        return {
            "class_precision": precision_per_class,
            "class_recall": recall_per_class
        }

    def compute_overall_accuracy(self) -> float:
        return 0.0 if self.total_pixels == 0 else self.total_correct / self.total_pixels

    def compute_all_metrics(self) -> Dict[str, Any]:
        results = {
            'overall_accuracy': self.compute_overall_accuracy()
        }
        results.update(self.compute_iou())
        results.update(self.compute_dice())
        results.update(self.compute_precision_recall())
        results['confusion_matrix'] = self.compute_confusion_matrix()
        results['confusion_matrix_normalized'] = self.compute_confusion_matrix(normalize='true')
        return results

## Quick test of evaluation metrics with random dummy data
"""def test_metrics():
    print("Testing Burn Severity Metrics...")
    print("=" * 50)

    num_classes = 5
    h, w = 32, 32
    batch_size = 2

    # Simulate predictions (logits)
    predictions = torch.randn(batch_size, num_classes, h, w)

    # Simulate targets with a few classes
    targets = torch.zeros(batch_size, h, w, dtype=torch.long)
    targets[:, :16, :16] = 0  # Unburned
    targets[:, 16:24, 16:24] = 1  # Low
    targets[:, 24:28, 24:28] = 2  # Moderate
    targets[:, 28:30, 28:30] = 3  # High
    targets[:, 30:, 30:] = 4  # Cloud/No-data

    # Initialize metrics
    metrics = BurnSeverityMetrics(num_classes=num_classes)

    # Update with batch
    metrics.update(predictions, targets)

    # Compute results
    results = metrics.compute_all_metrics()

    # Print summary
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Dice: {results['mean_dice']:.4f}")
    print("\nPer-class IoU:")
    for cls, val in results['class_iou'].items():
        print(f"  {cls}: {val:.4f}")
    print("\nPer-class Dice:")
    for cls, val in results['class_dice'].items():
        print(f"  {cls}: {val:.4f}")
    print("\nPer-class Precision:")
    for cls, val in results['class_precision'].items():
        print(f"  {cls}: {val:.4f}")
    print("\nPer-class Recall:")
    for cls, val in results['class_recall'].items():
        print(f"  {cls}: {val:.4f}")

    # Plot confusion matrix
    cm_norm = metrics.compute_confusion_matrix(normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=metrics.class_names,
        yticklabels=metrics.class_names,
        cbar_kws={'label': 'Proportion'}
    )
    plt.title("Confusion Matrix (Normalized)", fontsize=12)
    plt.xlabel("Predicted", fontsize=10)
    plt.ylabel("True", fontsize=10)
    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.yticks(fontsize=8, rotation=0)
    plt.tight_layout()
    plt.show()

    # Precision & recall bar chart
    precisions = list(results['class_precision'].values())
    recalls = list(results['class_recall'].values())
    x = np.arange(len(metrics.class_names))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, precisions, width, label='Precision')
    plt.bar(x + width/2, recalls, width, label='Recall')
    plt.xticks(x, metrics.class_names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Score")
    plt.title("Per-Class Precision and Recall")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_metrics()"""
