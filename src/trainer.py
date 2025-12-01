"""
Training engine with automatic mixed precision, early stopping, and checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Optional, Callable
import time
from tqdm import tqdm
import json
import psutil
import os

from metrics import MetricsTracker
from losses import create_loss_function


def get_gpu_memory_usage():
    """
    Get current GPU memory usage.
    
    Returns:
        tuple: (allocated_gb, reserved_gb, total_gb)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return allocated, reserved, total
    return 0.0, 0.0, 0.0


def get_ram_usage():
    """
    Get current RAM usage.
    
    Returns:
        tuple: (used_gb, total_gb, percent)
    """
    process = psutil.Process(os.getpid())
    ram_info = psutil.virtual_memory()
    process_ram = process.memory_info().rss / 1e9  # Process RAM in GB
    total_ram = ram_info.total / 1e9
    used_ram = ram_info.used / 1e9
    percent = ram_info.percent
    return used_ram, total_ram, percent


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = 'min'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/IoU
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if should stop
        
        Args:
            value: Current metric value
            
        Returns:
            True if should stop
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Training engine for semantic segmentation models
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: nn.Module,
        num_classes: int,
        device: torch.device,
        checkpoint_dir: Path,
        experiment_name: str,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,
        early_stopping_patience: int = 15,
        gradient_accumulation_steps: int = 1,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            num_classes: Number of classes
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
            use_amp: Use automatic mixed precision
            gradient_clip_val: Gradient clipping value
            early_stopping_patience: Patience for early stopping
            gradient_accumulation_steps: Number of steps to accumulate gradients
            class_names: Optional class name mapping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(device)
        self.num_classes = num_classes
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Setup metrics tracking
        self.metrics_tracker = MetricsTracker(num_classes, class_names=class_names)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.0001,
            mode='max'  # Monitor val_iou (maximize)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_iou = 0.0
        self.best_epoch = 0
        
        print(f"\nTrainer initialized:")
        print(f"  Experiment: {experiment_name}")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {use_amp}")
        print(f"  Gradient clipping: {gradient_clip_val}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss
        """
        self.model.train()
        self.metrics_tracker.reset_train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar for training batches
        pbar = tqdm(self.train_loader, desc="  Training", leave=False, ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Optimizer step every N batches
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update scheduler per step for OneCycleLR
                    if self.scheduler is not None:
                        from torch.optim.lr_scheduler import OneCycleLR
                        if isinstance(self.scheduler, OneCycleLR):
                            self.scheduler.step()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Optimizer step every N batches
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip_val
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update scheduler per step for OneCycleLR
                    if self.scheduler is not None:
                        from torch.optim.lr_scheduler import OneCycleLR
                        if isinstance(self.scheduler, OneCycleLR):
                            self.scheduler.step()
            
            # Update metrics (use unscaled loss for tracking)
            total_loss += loss.item() * self.gradient_accumulation_steps
            self.metrics_tracker.update_train(outputs.detach(), masks)
            
            # Get resource usage
            gpu_alloc, gpu_reserved, gpu_total = get_gpu_memory_usage()
            ram_used, ram_total, ram_percent = get_ram_usage()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'avg': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate_epoch(self) -> float:
        """
        Validate for one epoch
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        self.metrics_tracker.reset_val()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Progress bar for validation batches
        pbar = tqdm(self.val_loader, desc="  Validation", leave=False, ncols=100)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
            
            # Update metrics
            total_loss += loss.item()
            self.metrics_tracker.update_val(outputs, masks)
            
            # Get resource usage
            gpu_alloc, gpu_reserved, gpu_total = get_gpu_memory_usage()
            ram_used, ram_total, ram_percent = get_ram_usage()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False, filename: str = 'checkpoint.pth', silent: bool = False):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
            filename: Checkpoint filename
            silent: If True, suppress print messages
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_iou': self.best_val_iou,
            'metrics_history': self.metrics_tracker.history
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            if not silent:
                print(f"  ✓ Best model saved (Val IoU: {self.best_val_iou:.4f})")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_iou = checkpoint['best_val_iou']
        self.metrics_tracker.history = checkpoint.get('metrics_history', self.metrics_tracker.history)
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best val IoU: {self.best_val_iou:.4f}")
    
    def train(self, num_epochs: int, resume_from: Optional[Path] = None):
        """
        Train model for specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Optional checkpoint path to resume from
        """
        # Resume from checkpoint if provided
        if resume_from and resume_from.exists():
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 1
        
        print(f"\n{'='*70}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*70}\n")
        
        training_start = time.time()
        
        # Track all epoch metrics for final summary
        epoch_summaries = []
        
        # Simple loop without overall progress bar (show per-epoch progress instead)
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch}/{num_epochs}]")
            print(f"{'='*70}")
            
            # Get resource usage at start of epoch
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics (with per-epoch print)
            train_metrics, val_metrics = self.metrics_tracker.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                print_summary=True  # Show individual epoch summaries
            )
            
            # Update learning rate (OneCycleLR steps per batch, others step per epoch)
            if self.scheduler:
                from torch.optim.lr_scheduler import OneCycleLR
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mean_iou'])
                elif not isinstance(self.scheduler, OneCycleLR):
                    # OneCycleLR already stepped in train_epoch
                    self.scheduler.step()
            
            # Check if best model
            is_best = val_metrics['mean_iou'] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics['mean_iou']
                self.best_epoch = epoch
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best, filename=f'checkpoint_epoch_{epoch}.pth', silent=False)
            
            epoch_time = time.time() - epoch_start
            
            # Get resource usage at end of epoch
            gpu_alloc, gpu_reserved, gpu_total = get_gpu_memory_usage()
            ram_used, ram_total, ram_percent = get_ram_usage()
            cpu_percent_end = psutil.cpu_percent(interval=0.1)
            
            # Print resource usage
            resource_info = f"  Resources: "
            if torch.cuda.is_available():
                resource_info += f"GPU: {gpu_alloc:.1f}/{gpu_total:.0f}GB | "
            resource_info += f"RAM: {ram_percent:.0f}% | CPU: {cpu_percent_end:.0f}%"
            print(resource_info)
            print(f"  Epoch time: {epoch_time:.2f}s")
            print()
            
            # Store epoch summary
            epoch_summaries.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_iou': train_metrics['mean_iou'],
                'val_iou': val_metrics['mean_iou'],
                'train_dice': train_metrics['mean_dice'],
                'val_dice': val_metrics['mean_dice'],
                'lr': current_lr,
                'time': epoch_time,
                'is_best': is_best,
                'gpu_alloc': gpu_alloc,
                'gpu_total': gpu_total,
                'ram_percent': ram_percent,
                'cpu_percent': cpu_percent_end
            })
            
            # Check early stopping
            if self.early_stopping(val_metrics['mean_iou']):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best Val IoU: {self.best_val_iou:.4f} at epoch {self.best_epoch}")
                break
        
        # Training complete
        total_time = time.time() - training_start
        
        # Get final resource usage
        final_gpu_alloc, final_gpu_reserved, final_gpu_total = get_gpu_memory_usage()
        final_ram_used, final_ram_total, final_ram_percent = get_ram_usage()
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Total epochs: {len(epoch_summaries)}")
        print(f"Total time: {total_time / 60:.2f} minutes ({total_time / 3600:.2f} hours)")
        print(f"Average time per epoch: {total_time / len(epoch_summaries):.2f}s")
        print(f"\nResource Usage:")
        if torch.cuda.is_available():
            print(f"  Peak GPU: {max([s['gpu_alloc'] for s in epoch_summaries]):.2f}/{final_gpu_total:.2f} GB")
            print(f"  Final GPU: {final_gpu_alloc:.2f}/{final_gpu_total:.2f} GB")
        print(f"  Peak RAM: {max([s['ram_percent'] for s in epoch_summaries]):.1f}%")
        print(f"  Final RAM: {final_ram_percent:.1f}%")
        print(f"\nBest Val IoU: {self.best_val_iou:.4f} at epoch {self.best_epoch}")
        print(f"Best model saved to: {self.checkpoint_dir / 'best_model.pth'}")
        
        # Print detailed epoch-by-epoch summary
        print(f"\n{'='*100}")
        print(f"EPOCH-BY-EPOCH SUMMARY")
        print(f"{'='*100}")
        
        header = f"{'Epoch':<6} {'TrLoss':<8} {'VaLoss':<8} {'TrIoU':<7} {'VaIoU':<7} {'TrDice':<7} {'VaDice':<7} {'LR':<10} {'Time':<7}"
        if torch.cuda.is_available():
            header += f" {'GPU(GB)':<9}"
        header += f" {'RAM%':<6} {'CPU%':<6} {'Best':<5}"
        print(header)
        print(f"{'-'*110}")
        
        for summary in epoch_summaries:
            best_marker = '*' if summary['is_best'] else ''
            line = (f"{summary['epoch']:<6} "
                   f"{summary['train_loss']:<8.4f} "
                   f"{summary['val_loss']:<8.4f} "
                   f"{summary['train_iou']:<7.4f} "
                   f"{summary['val_iou']:<7.4f} "
                   f"{summary['train_dice']:<7.4f} "
                   f"{summary['val_dice']:<7.4f} "
                   f"{summary['lr']:<10.6f} "
                   f"{summary['time']:<7.1f}")
            
            if torch.cuda.is_available():
                line += f" {summary['gpu_alloc']:<9.2f}"
            line += f" {summary['ram_percent']:<6.1f} {summary['cpu_percent']:<6.1f} {best_marker:<5}"
            print(line)
        
        print(f"{'='*100}\n")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists
            history = {k: [float(v) if not isinstance(v, list) else v for v in vals] 
                      for k, vals in self.metrics_tracker.history.items()}
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")
        
        return self.metrics_tracker.history


if __name__ == "__main__":
    print("Trainer module - use in training script")
