from pathlib import Path
import os
import sys

class PathConfig:
    """Central configuration for all project paths"""

    # Base directories
    PROJECT_ROOT = Path(__file__).parent.parent

    # Always point dataset to Desktop
    DESKTOP = Path.home() / "Desktop"
    LANDSAT_BSA_PATH = DESKTOP / "Landsat-BSA"

    # Data splits
    TRAIN_DATA_PATH = LANDSAT_BSA_PATH / "train"
    VAL_DATA_PATH = LANDSAT_BSA_PATH / "val"
    TEST_DATA_PATH = LANDSAT_BSA_PATH / "test"

    # Model and training paths (inside project)
    MODELS_DIR = PROJECT_ROOT / "model_architecture"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"

    # Output directories
    SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Results subdirectories
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    COMPARISONS_DIR = RESULTS_DIR / "comparisons"

    # Logging subdirectories
    TENSORBOARD_DIR = LOGS_DIR / "tensorboard"
    EXPERIMENTS_DIR = LOGS_DIR / "experiments"
    CHECKPOINTS_DIR = LOGS_DIR / "checkpoints"

    # Model-specific saved model directories
    UNET_MODELS_DIR = SAVED_MODELS_DIR / "Unet"
    ATTENTION_MODELS_DIR = SAVED_MODELS_DIR / "Attention"
    RESUNET_MODELS_DIR = SAVED_MODELS_DIR / "ResUnet"

    # Other useful directories
    EVALUATION_DIR = PROJECT_ROOT / "evaluation"
    TEST_DIR = PROJECT_ROOT / "test"
    TRAIN_DIR = PROJECT_ROOT / "train"
    METRICS_DIR = PROJECT_ROOT / "metrics"
    UTILS_DIR = PROJECT_ROOT / "utils"

    PAPER_DIR = PROJECT_ROOT / "paper"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

    @classmethod
    def create_directories(cls):
        """Create project output directories"""
        directories = [
            cls.SAVED_MODELS_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR,
            cls.FIGURES_DIR,
            cls.TABLES_DIR,
            cls.COMPARISONS_DIR,
            cls.TENSORBOARD_DIR,
            cls.EXPERIMENTS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.UNET_MODELS_DIR,
            cls.ATTENTION_MODELS_DIR,
            cls.RESUNET_MODELS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {len(directories)} directories")

    @classmethod
    def verify_dataset_paths(cls, strict: bool = True):
        """Verify dataset exists on Desktop.
        If strict=True, exit immediately on missing dataset.
        """
        required_paths = [
            cls.LANDSAT_BSA_PATH,
            cls.TRAIN_DATA_PATH,
            cls.VAL_DATA_PATH,
            cls.TEST_DATA_PATH,
        ]
        missing = [p for p in required_paths if not p.exists()]
        if missing:
            print("❌ Missing dataset paths:")
            for p in missing:
                print(f"  {p}")
            if strict:
                sys.exit("🚨 Dataset not found on Desktop! Please copy Landsat-BSA to ~/Desktop.")
            return False
        print("✅ All dataset paths verified successfully")
        return True


# Default instance
default_paths = PathConfig()

def get_paths():
    return default_paths
