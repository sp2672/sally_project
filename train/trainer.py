import torch
import os
import sys
from pathlib import Path

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))
sys.path.append(str(Path(__file__).parent.parent / "config"))

from train_utils import log_metrics_to_csv
from config.config import Config


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device,
                 save_dir=Config.SAVE_DIR, early_stop_patience=Config.EARLY_STOP_PATIENCE):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.early_stop_patience = early_stop_patience

        os.makedirs(save_dir, exist_ok=True)

        # Tracking
        self.best_miou = 0.0
        self.epochs_no_improve = 0
        self.ckpt_path = os.path.join(save_dir, Config.BEST_MODEL_NAME)

        # Logs
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return running_loss / len(self.train_loader)

    def validate_one_epoch(self, metrics_fn):
        self.model.eval()
        running_loss = 0.0
        metrics_fn.reset()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                metrics_fn.update(outputs, labels)
                
                # Memory cleanup
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_loss = running_loss / len(self.val_loader)
        results = metrics_fn.compute_all_metrics()
        return val_loss, results

    def fit(self, num_epochs, metrics_fn, csv_path="training_log.csv", config=None):
        # Convert csv_path to string if it's a Path object
        csv_path = str(csv_path)
        
        # Log hyperparameters if CSV doesn't exist
        if config is not None and not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["# Hyperparameters"])
                for key, value in vars(config).items():
                    if not key.startswith("__"):  # skip built-ins
                        writer.writerow([key, value])
                writer.writerow([])  # blank line before logs
                writer.writerow([
                    "epoch", "train_loss", "val_loss",
                    "accuracy", "mean_iou", "mean_dice"
                ])

        for epoch in range(num_epochs):
            try:
                train_loss = self.train_one_epoch()
                val_loss, val_results = self.validate_one_epoch(metrics_fn)

                # Store logs
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.val_ious.append(val_results["mean_iou"])

                # Save metrics to CSV
                log_entry = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": val_results["overall_accuracy"],
                    "mean_iou": val_results["mean_iou"],
                    "mean_dice": val_results["mean_dice"],
                }
                
                # Add per-class metrics
                for cls, val in val_results["class_precision"].items():
                    log_entry[f"precision_{cls.replace(' ', '_')}"] = val
                for cls, val in val_results["class_recall"].items():
                    log_entry[f"recall_{cls.replace(' ', '_')}"] = val

                log_metrics_to_csv(csv_path, log_entry)

                print(f"[Epoch {epoch+1}] "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val mIoU: {val_results['mean_iou']:.4f}")

                # Save best model (state_dict only)
                if val_results["mean_iou"] > self.best_miou:
                    self.best_miou = val_results["mean_iou"]
                    torch.save(self.model.state_dict(), self.ckpt_path)
                    print(f"Saved new best model at {self.ckpt_path} "
                          f"(mIoU={self.best_miou:.4f})")
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.early_stop_patience:
                        print("Early stopping triggered")
                        break
                        
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                break

        return self.train_losses, self.val_losses, self.val_ious