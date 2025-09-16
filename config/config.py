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
    LOSS_TYPE = "focal"   # options: "ce", "focal", "ohem", "combined"
    FOCAL_ALPHA = [0.1, 1.0, 2.0, 3.0, 2.0]
    FOCAL_GAMMA = 2.0

    OHEM_THRESHOLD = 0.7
    OHEM_MIN_KEPT = 10000

    MODEL_NAME = "unet"        # options: "unet", "resunet", "attentionunet"
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = "adam"         # options: "adam", "sgd"
    MOMENTUM = 0.9             # only used if OPTIMIZER="sgd"

    # --- Early stopping ---
    EARLY_STOP_PATIENCE = 7

    # --- Saving ---
    SAVE_DIR = Path("logs") / "checkpoints"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    BEST_MODEL_NAME = "best_model.pth"
    BEST_MODEL_FULL = "best_model_full.pth"

    # --- Logging with timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TRAINING_LOG = SAVE_DIR / f"trainlog_{MODEL_NAME}_bs{BATCH_SIZE}_lr{LR}_loss{LOSS_TYPE}_{timestamp}.csv"

    @classmethod
    def get_class_weights(cls):
        """Return class weights for loss function"""
        # Option 1: No class weights
        return None
        
        # Option 2: Use predefined weights (uncomment to use)
        # return torch.tensor([0.1, 1.0, 2.0, 3.0, 2.0], dtype=torch.float32)