"""
Loss functions for semantic segmentation
Includes CrossEntropy, Dice, Focal, and combined losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation
    """
    
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        """
        Initialize Dice Loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss
        
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            
        Returns:
            Dice loss value
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()   # (B, C, H, W)
        
        # Handle ignore index
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).float().unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Calculate Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - Dice as loss
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -1,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Class weights (C,)
            gamma: Focusing parameter (higher = more focus on hard examples)
            ignore_index: Index to ignore
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss
        
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            
        Returns:
            Focal loss value
        """
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        
        # Flatten
        num_classes = logits.shape[1]
        log_probs = log_probs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        targets = targets.view(-1)
        
        # Get probabilities for ground truth class
        probs = torch.exp(log_probs)
        class_mask = F.one_hot(targets, num_classes=num_classes).float()
        probs_gt = (probs * class_mask).sum(dim=1)
        
        # Calculate focal weight
        focal_weight = (1 - probs_gt) ** self.gamma
        
        # Calculate cross entropy
        ce_loss = F.nll_loss(log_probs, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of multiple losses
    """
    
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        focal_gamma: float = 2.0,
        ignore_index: int = -1
    ):
        """
        Initialize combined loss
        
        Args:
            num_classes: Number of classes
            class_weights: Class weights for CE loss
            ce_weight: Weight for CrossEntropy loss
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            focal_gamma: Gamma parameter for Focal loss
            ignore_index: Index to ignore
        """
        super().__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # CrossEntropy Loss
        if ce_weight > 0:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
        
        # Dice Loss
        if dice_weight > 0:
            self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        # Focal Loss
        if focal_weight > 0:
            self.focal_loss = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma,
                ignore_index=ignore_index
            )
        
        print(f"Combined Loss initialized:")
        print(f"  CE weight:    {ce_weight}")
        print(f"  Dice weight:  {dice_weight}")
        print(f"  Focal weight: {focal_weight}")
        if class_weights is not None:
            print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss
        
        Args:
            logits: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        if self.ce_weight > 0:
            total_loss += self.ce_weight * self.ce_loss(logits, targets)
        
        if self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(logits, targets)
        
        if self.focal_weight > 0:
            total_loss += self.focal_weight * self.focal_loss(logits, targets)
        
        return total_loss


def create_loss_function(
    loss_type: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: 'ce', 'dice', 'focal', or 'combined'
        num_classes: Number of classes
        class_weights: Optional class weights
        **kwargs: Additional loss-specific arguments
        
    Returns:
        Loss function instance
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'ce' or loss_type == 'crossentropy':
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=kwargs.get('ignore_index', -1))
    
    elif loss_type == 'dice':
        return DiceLoss(ignore_index=kwargs.get('ignore_index', -1))
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=class_weights,
            gamma=kwargs.get('gamma', 2.0),
            ignore_index=kwargs.get('ignore_index', -1)
        )
    
    elif loss_type == 'combined':
        return CombinedLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            ce_weight=kwargs.get('ce_weight', 1.0),
            dice_weight=kwargs.get('dice_weight', 1.0),
            focal_weight=kwargs.get('focal_weight', 0.0),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            ignore_index=kwargs.get('ignore_index', -1)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...\n")
    
    batch_size = 4
    num_classes = 7
    height = width = 512
    
    logits = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test CE Loss
    print("=" * 60)
    print("CrossEntropy Loss")
    ce_loss = create_loss_function('ce', num_classes)
    loss_val = ce_loss(logits, targets)
    print(f"Loss value: {loss_val.item():.4f}")
    
    # Test Dice Loss
    print("\n" + "=" * 60)
    print("Dice Loss")
    dice_loss = create_loss_function('dice', num_classes)
    loss_val = dice_loss(logits, targets)
    print(f"Loss value: {loss_val.item():.4f}")
    
    # Test Focal Loss
    print("\n" + "=" * 60)
    print("Focal Loss")
    focal_loss = create_loss_function('focal', num_classes, gamma=2.0)
    loss_val = focal_loss(logits, targets)
    print(f"Loss value: {loss_val.item():.4f}")
    
    # Test Combined Loss
    print("\n" + "=" * 60)
    print("Combined Loss")
    class_weights = torch.ones(num_classes)
    combined_loss = create_loss_function(
        'combined',
        num_classes,
        class_weights=class_weights,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.5
    )
    loss_val = combined_loss(logits, targets)
    print(f"Loss value: {loss_val.item():.4f}")
