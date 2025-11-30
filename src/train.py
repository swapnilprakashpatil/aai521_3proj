"""
Main training script for flood detection segmentation models.
Trains multiple models (U-Net++, DeepLabV3+, SegFormer) with comprehensive metrics tracking.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

import config
from dataset import create_dataloaders
from models import create_model
from losses import create_loss_function
from metrics import MetricsTracker
from trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train flood detection segmentation models')
    
    # Model selection
    parser.add_argument('--models', nargs='+', 
                        choices=['unet++', 'deeplabv3+', 'segformer', 'all'],
                        default=['all'],
                        help='Models to train (default: all)')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help=f'Number of epochs (default: {config.NUM_EPOCHS})')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help=f'Learning rate (default: {config.LEARNING_RATE})')
    parser.add_argument('--weight-decay', type=float, default=config.WEIGHT_DECAY,
                        help=f'Weight decay (default: {config.WEIGHT_DECAY})')
    
    # Loss configuration
    parser.add_argument('--loss', type=str, 
                        choices=['ce', 'dice', 'focal', 'combined'],
                        default=config.LOSS_TYPE,
                        help=f'Loss function (default: {config.LOSS_TYPE})')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str,
                        choices=['none', 'step', 'cosine', 'plateau'],
                        default=config.SCHEDULER_TYPE if config.USE_SCHEDULER else 'none',
                        help='Learning rate scheduler')
    
    # Mixed precision
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    
    # Early stopping
    parser.add_argument('--patience', type=int, 
                        default=config.EARLY_STOPPING_PATIENCE,
                        help=f'Early stopping patience (default: {config.EARLY_STOPPING_PATIENCE})')
    
    # Data
    parser.add_argument('--num-workers', type=int, default=config.NUM_WORKERS,
                        help=f'Number of data loader workers (default: {config.NUM_WORKERS})')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def create_scheduler(optimizer, scheduler_type, num_epochs):
    """Create learning rate scheduler."""
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, 
                     gamma=config.SCHEDULER_GAMMA)
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs,
                                eta_min=config.SCHEDULER_MIN_LR)
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                patience=5, verbose=True)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_model(model_name, args, train_loader, val_loader, class_weights, output_dir):
    """
    Train a single model.
    
    Args:
        model_name: Name of the model ('unet++', 'deeplabv3+', 'segformer')
        args: Command line arguments
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Tensor of class weights for loss function
        output_dir: Directory to save checkpoints and logs
    
    Returns:
        dict: Training history
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*80}\n")
    
    # Create model
    print(f"Creating {model_name} model...")
    model = create_model(
        model_name, 
        num_classes=config.NUM_CLASSES,
        in_channels=6,
        **config.MODEL_CONFIGS.get(model_name, {})
    )
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = create_loss_function(
        loss_type=args.loss,
        num_classes=config.NUM_CLASSES,
        class_weights=class_weights,
        device=args.device,
        ce_weight=config.LOSS_CE_WEIGHT,
        dice_weight=config.LOSS_DICE_WEIGHT,
        focal_weight=config.LOSS_FOCAL_WEIGHT,
        focal_gamma=config.FOCAL_GAMMA
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker(
        class_names=config.CLASS_NAMES,
        num_classes=config.NUM_CLASSES
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics_tracker=metrics_tracker,
        device=args.device,
        checkpoint_dir=output_dir / 'checkpoints',
        use_amp=not args.no_amp,
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        early_stopping_patience=args.patience
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Device: {args.device}")
    print(f"Mixed Precision: {not args.no_amp}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Loss Function: {args.loss}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Early Stopping Patience: {args.patience}\n")
    
    history = trainer.train(
        num_epochs=args.epochs,
        start_epoch=start_epoch
    )
    
    # Save final training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    # Print best metrics
    best_epoch = metrics_tracker.get_best_epoch('val_iou')
    if best_epoch is not None:
        print(f"\nBest epoch: {best_epoch}")
        print("Best validation metrics:")
        for key, value in metrics_tracker.history['val'][best_epoch].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"  {key}: {value:.4f}")
    
    return history


def main():
    """Main training function."""
    args = parse_args()
    
    # Determine which models to train
    if 'all' in args.models:
        models_to_train = ['unet++', 'deeplabv3+', 'segformer']
    else:
        models_to_train = args.models
    
    print(f"\nTraining models: {', '.join(models_to_train)}")
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_root = Path(config.OUTPUTS_DIR) / 'training' / timestamp
    else:
        output_root = Path(args.output_dir)
    
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_root}")
    
    # Save training configuration
    config_dict = {
        'models': models_to_train,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'loss': args.loss,
        'scheduler': args.scheduler,
        'use_amp': not args.no_amp,
        'patience': args.patience,
        'device': args.device,
        'num_workers': args.num_workers,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = output_root / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to: {config_path}\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dir=config.PROCESSED_TRAIN_DIR,
        val_dir=config.PROCESSED_VAL_DIR,
        test_dir=config.PROCESSED_TEST_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get class weights from training dataset
    class_weights = train_loader.dataset.get_class_weights()
    class_weights = class_weights.to(args.device)
    print(f"\nClass weights: {class_weights}")
    
    # Train each model
    all_histories = {}
    for model_name in models_to_train:
        model_output_dir = output_root / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            history = train_model(
                model_name=model_name,
                args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                output_dir=model_output_dir
            )
            all_histories[model_name] = history
            
        except Exception as e:
            print(f"\nError training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comparison summary
    if len(all_histories) > 1:
        comparison_path = output_root / 'model_comparison.json'
        comparison = {}
        
        for model_name, history in all_histories.items():
            # Get best validation metrics
            if history and 'val_metrics' in history:
                best_idx = max(range(len(history['val_metrics'])),
                             key=lambda i: history['val_metrics'][i].get('mean_iou', 0))
                comparison[model_name] = {
                    'best_epoch': best_idx + 1,
                    'best_metrics': history['val_metrics'][best_idx]
                }
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        for model_name, data in comparison.items():
            print(f"{model_name.upper()}:")
            print(f"  Best Epoch: {data['best_epoch']}")
            print(f"  Mean IoU: {data['best_metrics'].get('mean_iou', 0):.4f}")
            print(f"  Mean Dice: {data['best_metrics'].get('mean_dice', 0):.4f}")
            print(f"  Mean F1: {data['best_metrics'].get('mean_f1', 0):.4f}")
            print()
        
        print(f"Comparison saved to: {comparison_path}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
