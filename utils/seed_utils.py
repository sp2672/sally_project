import torch
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """
    Ensure full reproducibility across torch, numpy, and random.
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA determinism
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to: {seed}")