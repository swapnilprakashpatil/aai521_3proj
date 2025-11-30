"""
Experiment tracking utilities for model training.
Supports TensorBoard logging and experiment comparison.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Logger for tracking experiments with TensorBoard."""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        """
        Initialize experiment logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.full_log_dir = self.log_dir / experiment_name
        self.full_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.full_log_dir))
        
        # Store hyperparameters
        self.hparams = {}
        
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.hparams = hparams
        
        # Convert to flat dict for TensorBoard
        flat_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                flat_hparams[key] = value
            else:
                flat_hparams[key] = str(value)
        
        # Save to file
        hparams_path = self.full_log_dir / 'hyperparameters.json'
        with open(hparams_path, 'w') as f:
            json.dump(hparams, f, indent=2)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """
        Log metrics dictionary.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{prefix}{key}', value, step)
            elif isinstance(value, dict):
                # Handle nested dicts (e.g., per-class metrics)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self.writer.add_scalar(f'{prefix}{key}/{sub_key}', sub_value, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int):
        """Log image."""
        self.writer.add_image(tag, image, step, dataformats='HWC')
    
    def log_figure(self, tag: str, figure: plt.Figure, step: int):
        """Log matplotlib figure."""
        self.writer.add_figure(tag, figure, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        self.writer.add_text(tag, text, step)
    
    def close(self):
        """Close writer."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ExperimentComparator:
    """Compare multiple experiments."""
    
    def __init__(self, experiments_dir: Path):
        """
        Initialize comparator.
        
        Args:
            experiments_dir: Directory containing experiment logs
        """
        self.experiments_dir = Path(experiments_dir)
    
    def load_experiment_history(self, experiment_name: str) -> Optional[Dict]:
        """Load training history from experiment."""
        history_path = self.experiments_dir / experiment_name / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        return None
    
    def compare_experiments(self, experiment_names: list) -> Dict:
        """
        Compare multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for exp_name in experiment_names:
            history = self.load_experiment_history(exp_name)
            if history:
                # Get best metrics
                if 'val_metrics' in history and len(history['val_metrics']) > 0:
                    best_idx = max(range(len(history['val_metrics'])),
                                 key=lambda i: history['val_metrics'][i].get('mean_iou', 0))
                    
                    comparison[exp_name] = {
                        'best_epoch': best_idx + 1,
                        'best_val_metrics': history['val_metrics'][best_idx],
                        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
                        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None
                    }
        
        return comparison
    
    def plot_comparison(self, experiment_names: list, metric: str = 'mean_iou',
                       save_path: Optional[Path] = None):
        """
        Plot comparison of experiments.
        
        Args:
            experiment_names: List of experiment names
            metric: Metric to plot
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for exp_name in experiment_names:
            history = self.load_experiment_history(exp_name)
            if history and 'val_metrics' in history:
                epochs = range(1, len(history['val_metrics']) + 1)
                metric_values = [m.get(metric, 0) for m in history['val_metrics']]
                
                ax1.plot(epochs, metric_values, label=exp_name, marker='o', markersize=3)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f'Validation {metric.replace("_", " ").title()} Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training loss
        for exp_name in experiment_names:
            history = self.load_experiment_history(exp_name)
            if history and 'train_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                ax2.plot(epochs, history['train_loss'], label=exp_name, marker='o', markersize=3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_comparison_table(self, experiment_names: list) -> str:
        """
        Create markdown table comparing experiments.
        
        Args:
            experiment_names: List of experiment names
            
        Returns:
            Markdown formatted table
        """
        comparison = self.compare_experiments(experiment_names)
        
        # Create table header
        table = "| Experiment | Best Epoch | Mean IoU | Mean Dice | Mean F1 | Final Train Loss | Final Val Loss |\n"
        table += "|------------|------------|----------|-----------|---------|------------------|----------------|\n"
        
        # Add rows
        for exp_name, data in comparison.items():
            best_metrics = data['best_val_metrics']
            table += f"| {exp_name} | {data['best_epoch']} | "
            table += f"{best_metrics.get('mean_iou', 0):.4f} | "
            table += f"{best_metrics.get('mean_dice', 0):.4f} | "
            table += f"{best_metrics.get('mean_f1', 0):.4f} | "
            table += f"{data['final_train_loss']:.4f} | "
            table += f"{data['final_val_loss']:.4f} |\n"
        
        return table
