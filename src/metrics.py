"""
Evaluation metrics for semantic segmentation
Includes IoU, Dice, Precision, Recall, F1-score with per-class and mean calculations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class SegmentationMetrics:
    """Calculate segmentation metrics"""
    
    def __init__(self, num_classes: int, ignore_index: int = -1, class_names: Optional[Dict[int, str]] = None):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in calculations
            class_names: Optional mapping of class indices to names
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or {i: f'Class_{i}' for i in range(num_classes)}
        
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch
        
        Args:
            predictions: Model predictions (B, H, W) or (B, C, H, W)
            targets: Ground truth labels (B, H, W)
        """
        # Convert logits to class predictions if needed
        if predictions.ndim == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)
        
        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        # Create mask for valid pixels
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            predictions = predictions[mask]
            targets = targets[mask]
        
        # Update confusion matrix
        cm = confusion_matrix(
            targets,
            predictions,
            labels=np.arange(self.num_classes)
        )
        self.confusion_matrix += cm
        self.total_samples += len(targets)
    
    def compute_iou(self) -> Tuple[np.ndarray, float]:
        """
        Compute Intersection over Union (IoU / Jaccard Index)
        
        Returns:
            Tuple of (per-class IoU, mean IoU)
        """
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        
        # Avoid division by zero
        iou = np.zeros(self.num_classes, dtype=np.float32)
        for i in range(self.num_classes):
            if union[i] > 0:
                iou[i] = intersection[i] / union[i]
        
        # Mean IoU (excluding classes not in ground truth)
        present_classes = ground_truth_set > 0
        mean_iou = iou[present_classes].mean() if present_classes.any() else 0.0
        
        return iou, mean_iou
    
    def compute_dice(self) -> Tuple[np.ndarray, float]:
        """
        Compute Dice coefficient (F1 score for segmentation)
        
        Returns:
            Tuple of (per-class Dice, mean Dice)
        """
        # Dice = 2*TP / (2*TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        ground_truth_set = self.confusion_matrix.sum(axis=1)
        predicted_set = self.confusion_matrix.sum(axis=0)
        
        dice = np.zeros(self.num_classes, dtype=np.float32)
        for i in range(self.num_classes):
            denominator = ground_truth_set[i] + predicted_set[i]
            if denominator > 0:
                dice[i] = 2 * intersection[i] / denominator
        
        # Mean Dice
        present_classes = ground_truth_set > 0
        mean_dice = dice[present_classes].mean() if present_classes.any() else 0.0
        
        return dice, mean_dice
    
    def compute_precision_recall_f1(self) -> Dict[str, np.ndarray]:
        """
        Compute precision, recall, and F1-score per class
        
        Returns:
            Dictionary with per-class and macro-averaged metrics
        """
        # TP, FP, FN for each class
        tp = np.diag(self.confusion_matrix).astype(np.float32)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Calculate metrics
        precision = np.zeros(self.num_classes, dtype=np.float32)
        recall = np.zeros(self.num_classes, dtype=np.float32)
        f1 = np.zeros(self.num_classes, dtype=np.float32)
        
        for i in range(self.num_classes):
            if tp[i] + fp[i] > 0:
                precision[i] = tp[i] / (tp[i] + fp[i])
            if tp[i] + fn[i] > 0:
                recall[i] = tp[i] / (tp[i] + fn[i])
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': precision.mean(),
            'macro_recall': recall.mean(),
            'macro_f1': f1.mean()
        }
    
    def compute_pixel_accuracy(self) -> float:
        """
        Compute overall pixel accuracy
        
        Returns:
            Pixel accuracy (0-1)
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0.0
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return self.confusion_matrix.copy()
    
    def compute_all_metrics(self) -> Dict[str, any]:
        """
        Compute all metrics at once
        
        Returns:
            Dictionary with all metrics
        """
        iou_per_class, mean_iou = self.compute_iou()
        dice_per_class, mean_dice = self.compute_dice()
        pr_f1 = self.compute_precision_recall_f1()
        pixel_acc = self.compute_pixel_accuracy()
        
        metrics = {
            # Per-class metrics
            'iou_per_class': iou_per_class,
            'dice_per_class': dice_per_class,
            'precision_per_class': pr_f1['precision'],
            'recall_per_class': pr_f1['recall'],
            'f1_per_class': pr_f1['f1'],
            
            # Mean metrics
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'macro_precision': pr_f1['macro_precision'],
            'macro_recall': pr_f1['macro_recall'],
            'macro_f1': pr_f1['macro_f1'],
            
            # Overall metrics
            'pixel_accuracy': pixel_acc,
            'confusion_matrix': self.get_confusion_matrix(),
            
            # Sample info
            'total_samples': self.total_samples
        }
        
        return metrics
    
    def compute_flood_metrics(self, flood_classes: List[int] = [2, 3, 4, 5]) -> Dict[str, float]:
        """
        Compute metrics specifically for flood-related classes
        
        Args:
            flood_classes: List of class indices considered as flood
            
        Returns:
            Dictionary with flood-specific metrics
        """
        iou_per_class, _ = self.compute_iou()
        dice_per_class, _ = self.compute_dice()
        pr_f1 = self.compute_precision_recall_f1()
        
        # Get metrics for flood classes only
        flood_iou = [iou_per_class[i] for i in flood_classes if i < len(iou_per_class)]
        flood_dice = [dice_per_class[i] for i in flood_classes if i < len(dice_per_class)]
        flood_precision = [pr_f1['precision'][i] for i in flood_classes if i < len(pr_f1['precision'])]
        flood_recall = [pr_f1['recall'][i] for i in flood_classes if i < len(pr_f1['recall'])]
        flood_f1 = [pr_f1['f1'][i] for i in flood_classes if i < len(pr_f1['f1'])]
        
        # Filter out NaN values
        flood_iou = [x for x in flood_iou if not np.isnan(x)]
        flood_dice = [x for x in flood_dice if not np.isnan(x)]
        flood_precision = [x for x in flood_precision if not np.isnan(x)]
        flood_recall = [x for x in flood_recall if not np.isnan(x)]
        flood_f1 = [x for x in flood_f1 if not np.isnan(x)]
        
        return {
            'flood_mean_iou': np.mean(flood_iou) if flood_iou else 0.0,
            'flood_mean_dice': np.mean(flood_dice) if flood_dice else 0.0,
            'flood_mean_precision': np.mean(flood_precision) if flood_precision else 0.0,
            'flood_mean_recall': np.mean(flood_recall) if flood_recall else 0.0,
            'flood_mean_f1': np.mean(flood_f1) if flood_f1 else 0.0,
            'flood_iou_per_class': flood_iou
        }
    
    def format_metrics(self, metrics: Optional[Dict] = None) -> str:
        """
        Format metrics as readable string
        
        Args:
            metrics: Metrics dictionary (if None, compute from current state)
            
        Returns:
            Formatted string
        """
        if metrics is None:
            metrics = self.compute_all_metrics()
        
        lines = []
        lines.append("=" * 70)
        lines.append("SEGMENTATION METRICS")
        lines.append("=" * 70)
        
        # Overall metrics
        lines.append(f"\nOverall Metrics:")
        lines.append(f"  Pixel Accuracy:  {metrics['pixel_accuracy']:.4f}")
        lines.append(f"  Mean IoU:        {metrics['mean_iou']:.4f}")
        lines.append(f"  Mean Dice:       {metrics['mean_dice']:.4f}")
        lines.append(f"  Macro F1:        {metrics['macro_f1']:.4f}")
        
        # Per-class metrics
        lines.append(f"\nPer-Class Metrics:")
        lines.append(f"{'Class':<20} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        lines.append("-" * 70)
        
        for i in range(self.num_classes):
            class_name = self.class_names.get(i, f'Class_{i}')
            lines.append(
                f"{class_name:<20} "
                f"{metrics['iou_per_class'][i]:>8.4f} "
                f"{metrics['dice_per_class'][i]:>8.4f} "
                f"{metrics['precision_per_class'][i]:>8.4f} "
                f"{metrics['recall_per_class'][i]:>8.4f} "
                f"{metrics['f1_per_class'][i]:>8.4f}"
            )
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


class MetricsTracker:
    """Track metrics across training epochs"""
    
    def __init__(self, num_classes: int, class_names: Optional[Dict[int, str]] = None):
        """
        Initialize metrics tracker
        
        Args:
            num_classes: Number of classes
            class_names: Optional class name mapping
        """
        self.num_classes = num_classes
        self.class_names = class_names
        
        self.train_metrics = SegmentationMetrics(num_classes, class_names=class_names)
        self.val_metrics = SegmentationMetrics(num_classes, class_names=class_names)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def update_train(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update training metrics"""
        self.train_metrics.update(predictions, targets)
    
    def update_val(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update validation metrics"""
        self.val_metrics.update(predictions, targets)
    
    def reset_train(self):
        """Reset training metrics"""
        self.train_metrics.reset()
    
    def reset_val(self):
        """Reset validation metrics"""
        self.val_metrics.reset()
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        print_summary: bool = True
    ):
        """
        Log metrics for completed epoch
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Current learning rate
            print_summary: If True, print epoch summary (default: True)
        """
        # Compute metrics
        train_metrics = self.train_metrics.compute_all_metrics()
        val_metrics = self.val_metrics.compute_all_metrics()
        
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_iou'].append(train_metrics['mean_iou'])
        self.history['val_iou'].append(val_metrics['mean_iou'])
        self.history['train_dice'].append(train_metrics['mean_dice'])
        self.history['val_dice'].append(val_metrics['mean_dice'])
        self.history['train_f1'].append(train_metrics['macro_f1'])
        self.history['val_f1'].append(val_metrics['macro_f1'])
        self.history['learning_rate'].append(learning_rate)
        
        # Print summary if requested
        if print_summary:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train IoU:  {train_metrics['mean_iou']:.4f} | Val IoU:  {val_metrics['mean_iou']:.4f}")
            print(f"  Train Dice: {train_metrics['mean_dice']:.4f} | Val Dice: {val_metrics['mean_dice']:.4f}")
            print(f"  Train F1:   {train_metrics['macro_f1']:.4f} | Val F1:   {val_metrics['macro_f1']:.4f}")
            print(f"  LR: {learning_rate:.6f}")
        
        return train_metrics, val_metrics
    
    def get_best_epoch(self, metric: str = 'val_iou') -> Tuple[int, float]:
        """
        Get best epoch based on metric
        
        Args:
            metric: Metric name from history
            
        Returns:
            Tuple of (epoch, best_value)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return -1, 0.0
        
        values = self.history[metric]
        if 'loss' in metric:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return best_idx + 1, values[best_idx]


if __name__ == "__main__":
    # Test metrics
    from config import CLASS_NAMES
    
    print("Testing SegmentationMetrics...")
    
    num_classes = 7
    metrics = SegmentationMetrics(num_classes, class_names=CLASS_NAMES)
    
    # Create dummy predictions and targets
    batch_size = 4
    height = width = 512
    
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Update metrics
    metrics.update(predictions, targets)
    
    # Compute and print metrics
    print(metrics.format_metrics())
    
    # Test MetricsTracker
    print("\n" + "="*70)
    print("Testing MetricsTracker...")
    
    tracker = MetricsTracker(num_classes, class_names=CLASS_NAMES)
    tracker.update_train(predictions, targets)
    tracker.update_val(predictions, targets)
    
    train_metrics, val_metrics = tracker.log_epoch(
        epoch=1,
        train_loss=0.5,
        val_loss=0.6,
        learning_rate=0.001
    )
    
    best_epoch, best_val = tracker.get_best_epoch('val_iou')
    print(f"\nBest epoch: {best_epoch} with val_iou: {best_val:.4f}")
