import torch
import torch.optim as optim
import sys
from pathlib import Path

# Add project subdirectories to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "preprocessing"))
sys.path.append(str(project_root / "model_architecture"))
sys.path.append(str(project_root / "metrics"))
sys.path.append(str(project_root / "utils"))
sys.path.append(str(project_root / "config"))
sys.path.append(str(project_root / "train"))

from dataloader import create_data_loaders
from trainer import Trainer
from train_utils import plot_training_curves
from evaluation_metrics import BurnSeverityMetrics
from loss_functions import LossFunctionFactory

# Import models
from unet import UNet
from resUnet import ResUNet
from attentionUnet import AttentionUNet

# Config + seed
from config import Config
from seed_utils import set_seed


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


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Device
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Data - only need train and validation for training
    train_loader, val_loader, _ = create_data_loaders(
        dataset_path=Config.DATASET_PATH,
        batch_size=Config.BATCH_SIZE,
    )

    # Model
    model = get_model(Config.MODEL_NAME, num_classes=Config.NUM_CLASSES).to(device)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    criterion = LossFunctionFactory.create_loss_from_config(Config)

    # Optimizer
    if Config.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LR,
            weight_decay=Config.WEIGHT_DECAY
        )
    elif Config.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=Config.LR,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unsupported OPTIMIZER: {Config.OPTIMIZER}")

    # Trainer
    trainer = Trainer(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        save_dir=Config.SAVE_DIR,
        early_stop_patience=Config.EARLY_STOP_PATIENCE
    )
    metrics_fn = BurnSeverityMetrics(num_classes=Config.NUM_CLASSES)

    print(f"\nStarting training: {Config.MODEL_NAME} with {Config.LOSS_TYPE} loss")
    
    # Train model
    train_losses, val_losses, val_ious = trainer.fit(
        num_epochs=Config.NUM_EPOCHS,
        metrics_fn=metrics_fn,
        csv_path=str(Config.TRAINING_LOG),
        config=Config
    )

    print(f"\nTraining completed!")
    print(f"Best model saved at: {Config.SAVE_DIR / Config.BEST_MODEL_NAME}")
    print(f"Training log saved at: {Config.TRAINING_LOG}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_ious)

    print(f"\nTo evaluate the trained model, run:")
    print(f"python evaluate.py --model_path {Config.SAVE_DIR / Config.BEST_MODEL_NAME} --model_name {Config.MODEL_NAME} --save_plots --save_results")


if __name__ == "__main__":
    main()