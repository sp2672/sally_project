import torch
import torch.optim as optim
import sys
from pathlib import Path

# Add project subdirectories to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "preprocessing"))
sys.path.append(str(project_root / "model_architecture"))
sys.path.append(str(project_root / "metrics"))
sys.path.append(str(project_root / "utils"))
sys.path.append(str(project_root / "config"))
sys.path.append(str(project_root / "train"))

from preprocessing.dataloader import create_data_loaders
from train.trainer import Trainer
from metrics.evaluation_metrics import BurnSeverityMetrics
from metrics.loss_functions import LossFunctionFactory
from model_architecture.unet import UNet
from utils.seed_utils import set_seed
from config.config import Config 


def test_enhanced_trainer():
    """Test the enhanced trainer with minimal real data"""
    
    print("Testing Enhanced Trainer")
    print("=" * 50)
    
    # Print configuration
    Config.print_config()
    
    # Validate config
    try:
        Config.validate()
        print("Configuration validated successfully")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    # Set seed
    set_seed(Config.SEED)
    
    # Device
    device = Config.DEVICE
    print(f"\nUsing device: {device}")
    
    # Create data loaders with very limited samples
    print("\nCreating data loaders with limited samples...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_path=Config.DATASET_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # No multiprocessing for test
        max_samples=4   # Only 4 training samples total
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    print("\nCreating model...")
    model = UNet(n_channels=Config.IN_CHANNELS, n_classes=Config.NUM_CLASSES).to(device)
    
    # Loss
    print("Creating loss function...")
    criterion = LossFunctionFactory.create_loss_from_config(Config)
    
    # Optimizer
    print("Creating optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Enhanced Trainer
    print("\nInitializing enhanced trainer...")
    trainer = Trainer(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        save_dir=Config.SAVE_DIR,
        early_stop_patience=Config.EARLY_STOP_PATIENCE
    )
    
    # Metrics
    metrics_fn = BurnSeverityMetrics(num_classes=Config.NUM_CLASSES)
    
    print(f"\nStarting training test...")
    print("This will test all enhanced trainer features:")
    print("- Progress bars")
    print("- System monitoring")
    print("- Enhanced logging")
    print("- Error handling")
    print("- Configuration validation")
    
    # Run training
    try:
        train_losses, val_losses, val_ious = trainer.fit(
            num_epochs=Config.NUM_EPOCHS,
            metrics_fn=metrics_fn,
            csv_path=Config.TRAINING_LOG,
            config=Config
        )
        
        print(f"\nTraining test completed successfully!")
        print(f"Training log saved to: {Config.TRAINING_LOG}")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        print(f"Final val loss: {val_losses[-1]:.4f}")
        print(f"Final val mIoU: {val_ious[-1]:.4f}")
        
        # Check if log file was created
        if Path(Config.TRAINING_LOG).exists():
            print(f"CSV log file created successfully")
            
            # Show last few lines of CSV
            with open(Config.TRAINING_LOG, 'r') as f:
                lines = f.readlines()
                print("\nLast few lines of training log:")
                for line in lines[-3:]:
                    print(line.strip())
        
        print(f"\nEnhanced trainer test PASSED!")
        
    except Exception as e:
        print(f"\nTraining test FAILED with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_trainer()