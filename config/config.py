import torch
from pathlib import Path
from datetime import datetime


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset ---
    DATASET_PATH = str(Path.home() / "Desktop" / "Landsat-BSA")
    NUM_CLASSES = 5
    IN_CHANNELS = 6

    # --- Training ---
    LOSS_TYPE = "ce"   # options: "ce", "focal", "ohem", "combined" ------ REMEMBER TO CHANGE ALWAYS
    FOCAL_ALPHA = [0.1, 1.0, 2.0, 3.0, 2.0]  #only for focal
    FOCAL_GAMMA = 2.0                        #only for focal
 
    OHEM_THRESHOLD = 0.7   # only OHEM
    OHEM_MIN_KEPT = 10000  # only OHEM

    MODEL_NAME = "unet"        # options: "unet", "resunet", "attentionunet" ------ REMEMBER TO CHANGE ALWAYS
    BATCH_SIZE = 16            # change ------ REMEMBER TO CHANGE ALWAYS
    NUM_EPOCHS = 50             # change ------ REMEMBER TO CHANGE ALWAYS
    LR = 0.001                  #change ------ REMEMBER TO CHANGE ALWAYS
    WEIGHT_DECAY = 1e-5     #--
    OPTIMIZER = "adam"         # options: "adam", "sgd" ------ REMEMBER TO CHANGE ALWAYS
    MOMENTUM = 0.9             # only used if OPTIMIZER="sgd"

    # --- Early stopping ---
    EARLY_STOP_PATIENCE = 7  # change early stopping

    # --- Saving ---
    SAVE_DIR = Path("logs") / "checkpoints"         # saving training log
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    BEST_MODEL_NAME = "baseline_unet_model.pth"            #change model name ------ REMEMBER TO CHANGE ALWAYS
    BEST_MODEL_FULL = "Baseline_unet_model_full.pth"  #change full model ------ REMEMBER TO CHANGE ALWAYS

    # --- Logging with timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TRAINING_LOG = SAVE_DIR / f"trainlog_{MODEL_NAME}_bs{BATCH_SIZE}_lr{LR}_loss{LOSS_TYPE}_{timestamp}.csv"

    @classmethod
    def get_class_weights(cls):
        """Return class weights for loss function"""
        return None
        
    @classmethod
    def validate(cls):
        """Validate configuration parameters"""
        errors = []
        
        # Check required paths
        if not Path(cls.DATASET_PATH).exists():
            errors.append(f"Dataset path does not exist: {cls.DATASET_PATH}")
            
        # Check numeric parameters
        if cls.BATCH_SIZE <= 0:
            errors.append(f"BATCH_SIZE must be positive, got {cls.BATCH_SIZE}")
            
        if cls.NUM_EPOCHS <= 0:
            errors.append(f"NUM_EPOCHS must be positive, got {cls.NUM_EPOCHS}")
            
        if cls.LR <= 0:
            errors.append(f"Learning rate must be positive, got {cls.LR}")
            
        # Check valid options
        valid_models = ["unet", "resunet", "attentionunet"]
        if cls.MODEL_NAME.lower() not in valid_models:
            errors.append(f"MODEL_NAME must be one of {valid_models}, got {cls.MODEL_NAME}")
            
        valid_losses = ["ce", "focal", "ohem", "combined"]
        if cls.LOSS_TYPE.lower() not in valid_losses:
            errors.append(f"LOSS_TYPE must be one of {valid_losses}, got {cls.LOSS_TYPE}")
            
        valid_optimizers = ["adam", "sgd"]
        if cls.OPTIMIZER.lower() not in valid_optimizers:
            errors.append(f"OPTIMIZER must be one of {valid_optimizers}, got {cls.OPTIMIZER}")
            
        # Warnings for potentially problematic values
        if cls.BATCH_SIZE > 32:
            warnings.warn(f"Large batch size {cls.BATCH_SIZE} may cause memory issues")
            
        if cls.LR > 1e-2:
            warnings.warn(f"High learning rate {cls.LR} may cause training instability")
            
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(errors))
            
        return True
        
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("Configuration:")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Loss: {cls.LOSS_TYPE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning rate: {cls.LR}")
        print(f"Optimizer: {cls.OPTIMIZER}")
        print(f"Dataset: {cls.DATASET_PATH}")
        print("=" * 40)