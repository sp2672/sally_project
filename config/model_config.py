from dataclasses import dataclass
from typing import Dict, Any
import sys
from pathlib import Path

# Add model_architecture to path
sys.path.append(str(Path(__file__).parent.parent / "model_architecture"))

# Import your model architectures
try:
    from unet import UNet
    from attentionUnet import AttentionUNet
    from resUnet import ResUNet
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    print("Make sure model_architecture folder contains the model files")


@dataclass
class ModelConfig:
    """Configuration for individual model architectures"""
    model_name: str
    model_class: Any
    n_channels: int = 6
    n_classes: int = 5
    bilinear: bool = False
    description: str = ""


class ModelConfigs:
    """Central configuration for all model architectures"""
    
    # Model-specific configurations
    UNET_CONFIG = ModelConfig(
        model_name="UNet",
        model_class=UNet if 'UNet' in globals() else None,
        n_channels=6,
        n_classes=5,
        bilinear=False,
        description="Baseline U-Net architecture with skip connections"
    )
    
    ATTENTION_UNET_CONFIG = ModelConfig(
        model_name="AttentionUNet", 
        model_class=AttentionUNet if 'AttentionUNet' in globals() else None,
        n_channels=6,
        n_classes=5,
        bilinear=False,
        description="U-Net with attention gates in skip connections"
    )
    
    RESUNET_CONFIG = ModelConfig(
        model_name="ResUNet",
        model_class=ResUNet if 'ResUNet' in globals() else None,
        n_channels=6,
        n_classes=5,
        bilinear=False,
        description="U-Net with residual connections in conv blocks"
    )
    
    # Model registry
    MODELS = {
        "unet": UNET_CONFIG,
        "attention_unet": ATTENTION_UNET_CONFIG,
        "resunet": RESUNET_CONFIG
    }
    
    # Aliases for convenience
    MODEL_ALIASES = {
        "baseline": "unet",
        "attention": "attention_unet",
        "residual": "resunet",
        "res": "resunet"
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        # Handle aliases
        if model_name.lower() in cls.MODEL_ALIASES:
            model_name = cls.MODEL_ALIASES[model_name.lower()]
        
        if model_name.lower() not in cls.MODELS:
            available_models = list(cls.MODELS.keys()) + list(cls.MODEL_ALIASES.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        return cls.MODELS[model_name.lower()]
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> Any:
        """Create a model instance"""
        config = cls.get_model_config(model_name)
        
        if config.model_class is None:
            raise ValueError(f"Model class not available for {model_name}. Check imports.")
        
        # Override config parameters with kwargs
        model_params = {
            "n_channels": kwargs.get("n_channels", config.n_channels),
            "n_classes": kwargs.get("n_classes", config.n_classes),
            "bilinear": kwargs.get("bilinear", config.bilinear)
        }
        
        return config.model_class(**model_params)
    
    @classmethod
    def get_all_models(cls) -> Dict[str, ModelConfig]:
        """Get all available model configurations"""
        return cls.MODELS.copy()
    
    @classmethod
    def list_models(cls):
        """Print all available models with descriptions"""
        print("Available Models:")
        print("=" * 50)
        
        for model_key, config in cls.MODELS.items():
            print(f"Key: '{model_key}'")
            print(f"Name: {config.model_name}")
            print(f"Description: {config.description}")
            print(f"Channels: {config.n_channels} -> {config.n_classes}")
            print("-" * 30)
        
        print("Aliases:")
        for alias, target in cls.MODEL_ALIASES.items():
            print(f"  '{alias}' -> '{target}'")
    
    @classmethod
    def compare_models(cls):
        """Compare model architectures"""
        print("Model Architecture Comparison:")
        print("=" * 80)
        
        for model_key, config in cls.MODELS.items():
            if config.model_class is None:
                continue
                
            try:
                model = config.model_class(
                    n_channels=config.n_channels,
                    n_classes=config.n_classes,
                    bilinear=config.bilinear
                )
                
                info = model.get_model_info()
                print(f"{info['model_name']}:")
                print(f"  Parameters: {info['total_parameters']:,}")
                print(f"  Size: {info['model_size_mb']:.1f} MB")
                print(f"  Input: {info['input_channels']} channels")
                print(f"  Output: {info['output_classes']} classes")
                print("-" * 40)
                
            except Exception as e:
                print(f"Error creating {config.model_name}: {e}")


# Model-specific hyperparameters (if models need different settings)
MODEL_SPECIFIC_PARAMS = {
    "unet": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5
    },
    "attention_unet": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5
    },
    "resunet": {
        "learning_rate": 8e-5,  # Slightly lower for residual networks
        "weight_decay": 1e-5
    }
}


def get_model_config(model_name: str) -> ModelConfig:
    """Convenience function to get model configuration"""
    return ModelConfigs.get_model_config(model_name)


def create_model(model_name: str, **kwargs):
    """Convenience function to create model"""
    return ModelConfigs.create_model(model_name, **kwargs)


if __name__ == "__main__":
    # Test the model configurations
    print("Testing Model Configurations:")
    print("=" * 50)
    
    # List all models
    ModelConfigs.list_models()
    print()
    
    # Compare model architectures
    ModelConfigs.compare_models()
    
    # Test model creation
    print("Testing Model Creation:")
    print("-" * 30)
    
    for model_name in ["unet", "attention_unet", "resunet"]:
        try:
            model = create_model(model_name)
            print(f"✓ {model_name}: Created successfully")
        except Exception as e:
            print(f"✗ {model_name}: Error - {e}")