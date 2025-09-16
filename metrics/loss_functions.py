import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "config"))


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Loss for semantic segmentation
    
    Focuses training on hard examples by selecting the pixels with highest loss
    and ignoring easy examples that contribute little to learning.
    """
    
    def __init__(self, threshold: float = 0.7, min_kept: int = 10000, weight: Optional[torch.Tensor] = None):
        """
        Args:
            threshold: Loss threshold for selecting hard examples (applied on per-pixel CE loss).
            min_kept: Minimum number of pixels to keep for training
            weight: Class weights tensor
        """
        super(OHEMLoss, self).__init__()
        self.threshold = threshold
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (batch_size, n_classes, H, W)
            targets: Ground truth labels (batch_size, H, W)
        """
        batch_size = predictions.size(0)
        
        # Calculate pixel-wise loss
        pixel_losses = self.criterion(predictions, targets)  # (batch_size, H, W)
        
        # Flatten for processing
        pixel_losses_flat = pixel_losses.view(batch_size, -1)  # (batch_size, H*W)
        
        total_loss = 0.0
        
        for b in range(batch_size):
            losses_b = pixel_losses_flat[b]
            
            # Keep at least min_kept pixels
            n_min = min(self.min_kept, losses_b.numel())
            
            # Sort losses in descending order and select top-k
            sorted_losses, _ = torch.sort(losses_b, descending=True)
            
            # Use threshold-based selection or top-k selection
            if self.threshold > 0:
                # Keep pixels above threshold or at least n_min pixels
                threshold_mask = sorted_losses > self.threshold
                n_keep = max(threshold_mask.sum().item(), n_min)
            else:
                # Just keep top n_min pixels
                n_keep = n_min
            
            # Average loss of selected hard examples
            hard_losses = sorted_losses[:n_keep]
            total_loss += hard_losses.mean()
        
        return total_loss / batch_size


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focuses training on hard examples by down-weighting easy examples
    """
    
    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Class weights
            gamma: Focusing parameter
            reduction: Loss reduction method
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (batch_size, n_classes, H, W)
            targets: Ground truth labels (batch_size, H, W)
        """
        # Move alpha to same device as predictions
        if self.alpha is not None and self.alpha.device != predictions.device:
            self.alpha = self.alpha.to(predictions.device)
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Calculate p_t
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t if alpha is provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.flatten()).view_as(targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for class imbalance
    """
    
    def __init__(self, weight: Optional[torch.Tensor] = None):
        """
        Args:
            weight: Class weights tensor
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions, targets)


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions
    """
    
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        """
        Args:
            losses: List of loss functions
            weights: Weights for each loss function
        """
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(predictions, targets)
        return total_loss


class LossFunctionFactory:
    """Factory for creating loss functions"""
    
    @staticmethod
    def create_loss(loss_type: str, **kwargs) -> nn.Module:
        """
        Create loss function based on type
        
        Args:
            loss_type: Type of loss function
            **kwargs: Additional arguments for loss function
            
        Returns:
            Loss function instance
        """
        loss_type = loss_type.lower()
        
        if loss_type == "ce" or loss_type == "crossentropy":
            weight = kwargs.get('weight', None)
            return WeightedCrossEntropyLoss(weight=weight)
        
        elif loss_type == "focal":
            alpha = kwargs.get('alpha', None)
            gamma = kwargs.get('gamma', 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma)
        
        elif loss_type == "ohem":
            threshold = kwargs.get('threshold', 0.7)
            min_kept = kwargs.get('min_kept', 10000)
            weight = kwargs.get('weight', None)
            return OHEMLoss(threshold=threshold, min_kept=min_kept, weight=weight)
        
        elif loss_type == "combined":
            losses = kwargs.get('losses', [])
            weights = kwargs.get('weights', [])
            return CombinedLoss(losses=losses, weights=weights)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. "
                           f"Available types: 'ce', 'focal', 'ohem', 'combined'")
    
    @staticmethod
    def create_loss_from_config(config) -> nn.Module:
        """
        Create loss function from training configuration
        
        Args:
            config: Training configuration object
            
        Returns:
            Configured loss function
        """
        # Import here to avoid circular imports
        try:
            from config.config import Config
        except ImportError:
            # Fallback if the import structure is different
            Config = config
        
        loss_type = config.LOSS_TYPE.lower()
        
        if loss_type == "ce":
            weight = config.get_class_weights()
            return WeightedCrossEntropyLoss(weight=weight)
        
        elif loss_type == "focal":
            alpha = config.FOCAL_ALPHA
            gamma = config.FOCAL_GAMMA
            return FocalLoss(alpha=alpha, gamma=gamma)
        
        elif loss_type == "ohem":
            threshold = config.OHEM_THRESHOLD
            weight = config.get_class_weights()
            return OHEMLoss(threshold=threshold, weight=weight)
        
        else:
            raise ValueError(f"Unsupported loss type in config: {loss_type}")