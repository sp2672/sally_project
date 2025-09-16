import torch
import os
import sys
import time
import psutil
from pathlib import Path
from tqdm import tqdm

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
        
        # Monitoring
        self.epoch_times = []
        self.memory_usage = []

    def _log_system_info(self):
        """Log system resource usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_cached = torch.cuda.memory_reserved() / (1024**3)   # GB
            gpu_util = f"GPU Memory: {gpu_memory:.1f}GB allocated, {gpu_cached:.1f}GB cached"
        else:
            gpu_util = "CPU only"
            
        ram_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        return {
            'gpu_memory_allocated': gpu_memory if torch.cuda.is_available() else 0,
            'gpu_memory_cached': gpu_cached if torch.cuda.is_available() else 0,
            'ram_usage_percent': ram_usage,
            'cpu_usage_percent': cpu_usage
        }

    def train_one_epoch(self):
        """Train for one epoch with progress tracking"""
        self.model.train()
        running_loss = 0.0
        
        # Progress bar for training
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
            
            # Memory cleanup every 10 batches
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return running_loss / len(self.train_loader)

    def validate_one_epoch(self, metrics_fn):
        """Validate for one epoch with progress tracking"""
        self.model.eval()
        running_loss = 0.0
        metrics_fn.reset()

        # Progress bar for validation
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                metrics_fn.update(outputs, labels)
                
                # Update progress bar
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'Val Loss': f'{avg_loss:.4f}'})
                
                # Memory cleanup
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        val_loss = running_loss / len(self.val_loader)
        results = metrics_fn.compute_all_metrics()
        return val_loss, results

    def fit(self, num_epochs, metrics_fn, csv_path="training_log.csv", config=None):
        """Enhanced training loop with comprehensive monitoring"""
        # Convert csv_path to string if it's a Path object
        csv_path = str(csv_path)
        
        # Validate config
        if config is not None:
            try:
                config.validate()
            except ValueError as e:
                print(f"Configuration error: {e}")
                return None, None, None
        
        # Log hyperparameters if CSV doesn't exist
        if config is not None and not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["# Hyperparameters"])
                for key, value in vars(config).items():
                    if not key.startswith("__"):
                        writer.writerow([key, value])
                writer.writerow([])
                writer.writerow([
                    "epoch", "train_loss", "val_loss", "accuracy", "mean_iou", "mean_dice",
                    "epoch_time", "gpu_memory_allocated", "gpu_memory_cached", "ram_usage_percent"
                ])

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            try:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                
                # Training
                train_loss = self.train_one_epoch()
                
                # Validation
                val_loss, val_results = self.validate_one_epoch(metrics_fn)
                
                # Timing
                epoch_time = time.time() - epoch_start
                self.epoch_times.append(epoch_time)
                
                # System monitoring
                system_info = self._log_system_info()
                self.memory_usage.append(system_info)

                # Store logs
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.val_ious.append(val_results["mean_iou"])

                # Enhanced logging
                log_entry = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "accuracy": val_results["overall_accuracy"],
                    "mean_iou": val_results["mean_iou"],
                    "mean_dice": val_results["mean_dice"],
                    "epoch_time": epoch_time,
                    **system_info
                }
                
                # Add per-class metrics
                for cls, val in val_results["class_precision"].items():
                    log_entry[f"precision_{cls.replace(' ', '_')}"] = val
                for cls, val in val_results["class_recall"].items():
                    log_entry[f"recall_{cls.replace(' ', '_')}"] = val

                log_metrics_to_csv(csv_path, log_entry)

                # Enhanced console output
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val mIoU: {val_results['mean_iou']:.4f} | "
                      f"Time: {epoch_time:.1f}s")
                
                if torch.cuda.is_available():
                    print(f"GPU Memory: {system_info['gpu_memory_allocated']:.1f}GB allocated")

                # Save best model
                if val_results["mean_iou"] > self.best_miou:
                    self.best_miou = val_results["mean_iou"]
                    torch.save(self.model.state_dict(), self.ckpt_path)
                    print(f">>> New best model saved! mIoU: {self.best_miou:.4f}")
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.early_stop_patience:
                        print(f"Early stopping triggered after {self.early_stop_patience} epochs without improvement")
                        break
                        
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Training summary
        total_time = time.time() - start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        print(f"\nTraining completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average epoch time: {avg_epoch_time:.1f}s")
        print(f"Best mIoU: {self.best_miou:.4f}")

        return self.train_losses, self.val_losses, self.val_ious