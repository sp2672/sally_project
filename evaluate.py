import torch
import sys
from pathlib import Path
import argparse

# Add project subdirectories to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "preprocessing"))
sys.path.append(str(project_root / "model_architecture"))
sys.path.append(str(project_root / "metrics"))
sys.path.append(str(project_root / "utils"))
sys.path.append(str(project_root / "config"))

from dataloader import create_data_loaders
from evaluation_metrics import BurnSeverityMetrics
from train_utils import plot_precision_recall, plot_confusion_matrix

# Import models
from unet import UNet
from resUnet import ResUNet
from attentionUnet import AttentionUNet

from config.config import Config


def get_model(name: str, num_classes: int):
    """Factory to choose model by name."""
    if name.lower() == "unet":
        return UNet(n_channels=Config.IN_CHANNELS, n_classes=num_classes)
    elif name.lower() == "resunet":
        return ResUNet(n_channels=Config.IN_CHANNELS, n_classes=num_classes)
    elif name.lower() == "attentionunet":
        return AttentionUNet(n_channels=Config.IN_CHANNELS, n_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")


def load_trained_model(model_path: str, model_name: str, device: torch.device):
    """Load a trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    # Create model architecture
    model = get_model(model_name, num_classes=Config.NUM_CLASSES)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"Successfully loaded {model_name} model")
    return model


def create_subset_loader(test_loader, max_samples: int):
    """Create a subset of the test loader for quick testing"""
    subset_data = []
    sample_count = 0
    
    for inputs, labels in test_loader:
        subset_data.append((inputs, labels))
        sample_count += inputs.size(0)  # batch size
        
        if sample_count >= max_samples:
            break
    
    return subset_data


def evaluate_model(model, test_loader, device, model_name: str, max_samples=None):
    """Evaluate model on test set (or subset)"""
    print(f"\nEvaluating {model_name} on test set...")
    
    # Create subset if requested
    if max_samples:
        print(f"Using subset of {max_samples} samples for quick testing")
        test_data = create_subset_loader(test_loader, max_samples)
        print(f"Test batches: {len(test_data)}")
    else:
        test_data = test_loader
        print(f"Test batches: {len(test_loader)}")
    
    # Initialize metrics
    metrics = BurnSeverityMetrics(num_classes=Config.NUM_CLASSES)
    
    # Evaluation loop
    with torch.no_grad():
        if max_samples:
            # Subset evaluation
            for batch_idx, (inputs, labels) in enumerate(test_data):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                metrics.update(outputs, labels)
                print(f"Progress: {batch_idx + 1}/{len(test_data)} batches")
        else:
            # Full evaluation
            for batch_idx, (inputs, labels) in enumerate(test_data):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                metrics.update(outputs, labels)
                
                # Progress indicator
                if (batch_idx + 1) % max(1, len(test_data) // 10) == 0:
                    print(f"Progress: {batch_idx + 1}/{len(test_data)} batches")
    
    # Compute final metrics
    results = metrics.compute_all_metrics()
    return results, metrics


def print_detailed_results(results: dict, model_name: str):
    """Print comprehensive evaluation results"""
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS - {model_name.upper()}")
    print(f"{'='*60}")
    
    # Overall metrics
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Mean IoU (mIoU):  {results['mean_iou']:.4f}")
    print(f"Mean Dice:        {results['mean_dice']:.4f}")
    
    # Per-class metrics
    print(f"\nPER-CLASS RESULTS:")
    print(f"{'Class':<20} {'IoU':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 50)
    
    class_names = ['Unburned', 'Low', 'Moderate', 'High', 'Non-processing/Cloud']
    for i, class_name in enumerate(class_names):
        iou = results['class_iou'][class_name]
        precision = results['class_precision'][class_name]
        recall = results['class_recall'][class_name]
        print(f"{class_name:<20} {iou:<8.4f} {precision:<10.4f} {recall:<8.4f}")


def save_results_to_file(results: dict, model_name: str, save_path: Path):
    """Save results to text file"""
    save_path.mkdir(parents=True, exist_ok=True)
    results_file = save_path / f"{model_name}_test_results.txt"
    
    with open(results_file, 'w') as f:
        f.write(f"TEST RESULTS - {model_name.upper()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {results['overall_accuracy']:.4f}\n")
        f.write(f"Mean IoU (mIoU):  {results['mean_iou']:.4f}\n")
        f.write(f"Mean Dice:        {results['mean_dice']:.4f}\n\n")
        
        f.write("PER-CLASS RESULTS:\n")
        f.write(f"{'Class':<20} {'IoU':<8} {'Precision':<10} {'Recall':<8}\n")
        f.write("-" * 50 + "\n")
        
        class_names = ['Unburned', 'Low', 'Moderate', 'High', 'Non-processing/Cloud']
        for class_name in class_names:
            iou = results['class_iou'][class_name]
            precision = results['class_precision'][class_name]
            recall = results['class_recall'][class_name]
            f.write(f"{class_name:<20} {iou:<8.4f} {precision:<10.4f} {recall:<8.4f}\n")
    
    print(f"Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on test set')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained model (.pth file)')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['unet', 'resunet', 'attentionunet'],
                        help='Model architecture name')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test (for quick testing)')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save evaluation plots')
    parser.add_argument('--save_results', action='store_true', 
                        help='Save results to file')
    
    args = parser.parse_args()
    
    # Setup
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Create test data loader
    print("Loading test data...")
    _, _, test_loader = create_data_loaders(
        dataset_path=Config.DATASET_PATH,
        batch_size=Config.BATCH_SIZE,
    )
    
    if len(test_loader) == 0:
        print("No test data available! Using validation data instead...")
        _, test_loader, _ = create_data_loaders(
            dataset_path=Config.DATASET_PATH,
            batch_size=Config.BATCH_SIZE,
        )
    
    # Load trained model
    model = load_trained_model(args.model_path, args.model_name, device)
    
    # Evaluate model (with optional subset)
    results, metrics = evaluate_model(model, test_loader, device, args.model_name, args.max_samples)
    
    # Print results
    print_detailed_results(results, args.model_name)
    
    # Save results if requested
    if args.save_results:
        save_results_to_file(results, args.model_name, Path("evaluation_results"))
    
    # Generate and save plots if requested
    if args.save_plots:
        print("\nGenerating evaluation plots...")
        plot_precision_recall(results, metrics.class_names)
        plot_confusion_matrix(results["confusion_matrix_normalized"], metrics.class_names)
        print("Plots displayed. Close plot windows to continue.")
    
    print(f"\nEvaluation of {args.model_name} completed successfully!")


if __name__ == "__main__":
    main()