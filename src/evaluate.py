"""
Evaluation script for trained flood detection models.
Evaluates models on test set and generates comprehensive reports.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent))

import config
from dataset import create_dataloaders
from models import create_model
from metrics import SegmentationMetrics, format_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate flood detection models')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet++', 'deeplabv3+', 'segformer'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save prediction masks')
    
    return parser.parse_args()


def evaluate_model(model, test_loader, device, save_predictions=False, output_dir=None):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use
        save_predictions: Whether to save prediction masks
        output_dir: Directory to save predictions
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes=config.NUM_CLASSES)
    
    all_predictions = []
    all_targets = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(predictions, targets)
            
            # Store for visualization
            if save_predictions:
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
    # Compute final metrics
    results = {
        'iou': metrics.compute_iou(),
        'dice': metrics.compute_dice(),
        'precision_recall_f1': metrics.compute_precision_recall_f1(),
        'pixel_accuracy': metrics.compute_pixel_accuracy(),
        'confusion_matrix': metrics.confusion_matrix.tolist()
    }
    
    # Save predictions if requested
    if save_predictions and output_dir:
        pred_dir = Path(output_dir) / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        np.save(pred_dir / 'predictions.npy', all_predictions)
        np.save(pred_dir / 'targets.npy', all_targets)
        print(f"Predictions saved to: {pred_dir}")
    
    return results, metrics


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def plot_metrics_comparison(results, class_names, output_path):
    """Plot per-class metrics comparison."""
    iou_per_class = results['iou']['per_class']
    dice_per_class = results['dice']['per_class']
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, iou_per_class, width, label='IoU', alpha=0.8)
    rects2 = ax.bar(x + width/2, dice_per_class, width, label='Dice', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics comparison saved to: {output_path}")


def create_evaluation_report(results, output_path):
    """Create text evaluation report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FLOOD DETECTION MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean IoU:           {results['iou']['mean']:.4f}\n")
        f.write(f"Mean Dice:          {results['dice']['mean']:.4f}\n")
        f.write(f"Pixel Accuracy:     {results['pixel_accuracy']:.4f}\n")
        f.write(f"Macro Precision:    {results['precision_recall_f1']['precision']:.4f}\n")
        f.write(f"Macro Recall:       {results['precision_recall_f1']['recall']:.4f}\n")
        f.write(f"Macro F1:           {results['precision_recall_f1']['f1']:.4f}\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<30} {'IoU':>10} {'Dice':>10}\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(config.CLASS_NAMES):
            iou = results['iou']['per_class'][i]
            dice = results['dice']['per_class'][i]
            f.write(f"{class_name:<30} {iou:>10.4f} {dice:>10.4f}\n")
        
        f.write("\n")
    
    print(f"Evaluation report saved to: {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}\n")
    
    # Create test dataloader
    print("Creating test dataloader...")
    _, _, test_loader = create_dataloaders(
        train_dir=config.PROCESSED_TRAIN_DIR,
        val_dir=config.PROCESSED_VAL_DIR,
        test_dir=config.PROCESSED_TEST_DIR,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS
    )
    print(f"Test batches: {len(test_loader)}\n")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(
        args.model,
        num_classes=config.NUM_CLASSES,
        in_channels=6,
        **config.MODEL_CONFIGS.get(args.model, {})
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}\n")
    
    # Evaluate
    results, metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=args.device,
        save_predictions=args.save_predictions,
        output_dir=output_dir
    )
    
    # Save results as JSON
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print formatted metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80 + "\n")
    print(format_metrics(metrics, config.CLASS_NAMES))
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Confusion matrix
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, config.CLASS_NAMES, output_dir / 'confusion_matrix.png')
    
    # Metrics comparison
    plot_metrics_comparison(results, config.CLASS_NAMES, output_dir / 'metrics_comparison.png')
    
    # Text report
    create_evaluation_report(results, output_dir / 'evaluation_report.txt')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
