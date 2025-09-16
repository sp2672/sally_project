import torch
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

def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...")
    
    try:
        from dataloader import create_data_loaders
        print("✓ dataloader import OK")
    except ImportError as e:
        print(f"✗ dataloader import failed: {e}")
        return False
    
    try:
        from trainer import Trainer
        print("✓ trainer import OK")
    except ImportError as e:
        print(f"✗ trainer import failed: {e}")
        return False
        
    try:
        from evaluation_metrics import BurnSeverityMetrics
        print("✓ metrics import OK")
    except ImportError as e:
        print(f"✗ metrics import failed: {e}")
        return False
        
    try:
        from loss_functions import LossFunctionFactory
        print("✓ loss functions import OK")
    except ImportError as e:
        print(f"✗ loss functions import failed: {e}")
        return False
        
    try:
        from unet import UNet
        from resUnet import ResUNet
        from attentionUnet import AttentionUNet
        print("✓ model imports OK")
    except ImportError as e:
        print(f"✗ model imports failed: {e}")
        return False
        
    try:
        from config.config import Config
        print("✓ config import OK")
    except ImportError as e:
        print(f"✗ config import failed: {e}")
        return False
        
    try:
        from seed_utils import set_seed
        print("✓ seed_utils import OK")
    except ImportError as e:
        print(f"✗ seed_utils import failed: {e}")
        return False
        
    return True

def test_config():
    """Test config has required methods"""
    print("\nTesting config...")
    
    try:
        from config import Config
        
        # Test get_class_weights method exists
        weights = Config.get_class_weights()
        print(f"✓ get_class_weights() works, returns: {weights}")
        
        # Test required attributes
        required_attrs = ['DEVICE', 'BATCH_SIZE', 'NUM_EPOCHS', 'LR', 'MODEL_NAME', 'LOSS_TYPE']
        for attr in required_attrs:
            if hasattr(Config, attr):
                print(f"✓ {attr}: {getattr(Config, attr)}")
            else:
                print(f"✗ Missing {attr}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_model_creation():
    """Test model creation works"""
    print("\nTesting model creation...")
    
    try:
        from unet import UNet
        model = UNet(n_channels=6, n_classes=5)
        print(f"✓ UNet created, parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_loss_function():
    """Test loss function creation"""
    print("\nTesting loss function...")
    
    try:
        from loss_functions import LossFunctionFactory
        from config import Config
        
        criterion = LossFunctionFactory.create_loss_from_config(Config)
        print(f"✓ Loss function created: {type(criterion).__name__}")
        return True
    except Exception as e:
        print(f"✗ Loss function creation failed: {e}")
        return False

def main():
    print("Quick Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_model_creation,
        test_loss_function
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    if all(results):
        print("ALL TESTS PASSED! Ready for training.")
    else:
        print("Some tests failed. Fix issues before training.")
        
    return all(results)

if __name__ == "__main__":
    main()