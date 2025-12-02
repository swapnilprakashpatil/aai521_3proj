"""
Visualization utilities for model training and validation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ValidationVisualizer:
    """Visualizer for light validation results with comprehensive analysis."""
    
    def __init__(self):
        """Initialize the visualizer with professional color scheme."""
        # Define professional color palette
        self.COLOR_PALETTE = {
            'primary': '#1e3a8a',      # Deep blue
            'secondary': '#7c3aed',    # Purple
            'success': '#10b981',      # Green
            'warning': '#f59e0b',      # Amber
            'danger': '#ef4444',       # Red
            'info': '#06b6d4',         # Cyan
            'dark': '#1f2937',         # Dark gray
            'light': '#f3f4f6'         # Light gray
        }
        
        # Model-specific colors (vibrant gradient)
        self.MODEL_COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4']
    
    def plot_validation_overview(self, validation_results: Dict, all_models: List[str]):
        """
        Plot training speed and success rate overview.
        
        Args:
            validation_results: Dictionary with validation results per model
            all_models: List of all model names
        """
        # Extract data
        model_names = []
        training_times = []
        statuses = []
        
        for model_name in all_models:
            if model_name in validation_results:
                model_names.append(model_name.upper())
                training_times.append(validation_results[model_name]['time'])
                statuses.append(validation_results[model_name]['status'])
        
        passed_count = sum(1 for s in statuses if s == 'passed')
        failed_count = sum(1 for s in statuses if s == 'failed')
        
        # Create figure
        fig = plt.figure(figsize=(18, 7))
        fig.patch.set_facecolor('white')
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Training Speed Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        colors_bars = [self.COLOR_PALETTE['success'] if s == 'passed' else self.COLOR_PALETTE['danger'] 
                       for s in statuses]
        bars = ax1.barh(model_names, training_times, color=colors_bars, alpha=0.85, 
                       edgecolor='white', linewidth=2)
        
        # Add hatching for failed models
        for i, bar in enumerate(bars):
            bar.set_hatch('//' if statuses[i] == 'failed' else '')
        
        ax1.set_xlabel('Validation Time (seconds)', fontsize=13, fontweight='bold', 
                      color=self.COLOR_PALETTE['dark'])
        ax1.set_ylabel('Model Architecture', fontsize=13, fontweight='bold', 
                      color=self.COLOR_PALETTE['dark'])
        ax1.set_title('Training Speed Comparison', fontsize=15, fontweight='bold', 
                     color=self.COLOR_PALETTE['primary'], pad=15)
        ax1.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)
        
        # Add time labels
        for i, (bar, time, status) in enumerate(zip(bars, training_times, statuses)):
            width = bar.get_width()
            label_color = self.COLOR_PALETTE['dark'] if status == 'passed' else self.COLOR_PALETTE['danger']
            ax1.text(width + max(training_times)*0.02, bar.get_y() + bar.get_height()/2, 
                    f'{time:.1f}s', ha='left', va='center', fontsize=11, 
                    fontweight='bold', color=label_color)
        
        # Add legend
        legend_elements = [
            Patch(facecolor=self.COLOR_PALETTE['success'], alpha=0.85, edgecolor='white', 
                 linewidth=2, label='Passed'),
            Patch(facecolor=self.COLOR_PALETTE['danger'], alpha=0.85, edgecolor='white', 
                 linewidth=2, label='Failed')
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.95)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot 2: Success Rate Donut Chart
        ax2 = fig.add_subplot(gs[0, 1])
        sizes = [passed_count, failed_count] if failed_count > 0 else [passed_count]
        labels_pie = ['Passed', 'Failed'] if failed_count > 0 else ['Passed']
        colors_pie = [self.COLOR_PALETTE['success'], self.COLOR_PALETTE['danger']] if failed_count > 0 else [self.COLOR_PALETTE['success']]
        explode = (0.08, 0.08) if failed_count > 0 else (0.08,)
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=None, colors=colors_pie, autopct='%1.1f%%',
                                             explode=explode, startangle=90, 
                                             textprops={'fontsize': 14, 'fontweight': 'bold'},
                                             wedgeprops={'edgecolor': 'white', 'linewidth': 3})
        
        # Draw circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=0)
        ax2.add_artist(centre_circle)
        
        # Add center text
        ax2.text(0, 0.1, f'{passed_count}/{len(all_models)}', ha='center', va='center',
                fontsize=32, fontweight='bold', color=self.COLOR_PALETTE['primary'])
        ax2.text(0, -0.2, 'Validated', ha='center', va='center',
                fontsize=13, fontweight='bold', color=self.COLOR_PALETTE['dark'])
        
        # Color the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(15)
            autotext.set_fontweight('bold')
        
        # Add custom legend
        legend_labels = [f'{labels_pie[i]}: {sizes[i]} model{"s" if sizes[i] != 1 else ""}' 
                         for i in range(len(sizes))]
        ax2.legend(wedges, legend_labels, loc='upper left', fontsize=11, framealpha=0.95)
        ax2.set_title('Validation Success Rate', fontsize=15, fontweight='bold', 
                     color=self.COLOR_PALETTE['primary'], pad=15)
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_analysis(self, validation_results: Dict, all_models: List[str], train_loader_len: int):
        """
        Plot learning progress and convergence analysis for passed models.
        
        Args:
            validation_results: Dictionary with validation results per model
            all_models: List of all model names
            train_loader_len: Length of training data loader for time estimation
        """
        passed_models = [(m, validation_results[m]['time']) for m in all_models 
                        if validation_results[m]['status'] == 'passed']
        
        if len(passed_models) == 0:
            print("No models passed validation. Skipping learning analysis.")
            return
        
        passed_models_sorted = sorted(passed_models, key=lambda x: x[1])
        
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor('white')
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        models_sorted = [m[0].upper() for m in passed_models_sorted]
        times_sorted = [m[1] for m in passed_models_sorted]
        colors_gradient = self.MODEL_COLORS[:len(models_sorted)]
        
        # Plot 1: Training Speed Ranking
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(range(len(models_sorted)), times_sorted, color=colors_gradient, 
                      alpha=0.85, edgecolor='white', linewidth=2)
        
        ax1.set_xlabel('Model (Fastest → Slowest)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Validation Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Training Speed Ranking', fontsize=14, fontweight='bold', 
                     color=self.COLOR_PALETTE['primary'], pad=12)
        ax1.set_xticks(range(len(models_sorted)))
        ax1.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=10)
        ax1.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)
        
        # Add value labels with rank
        for i, (bar, time) in enumerate(zip(bars, times_sorted)):
            height = bar.get_height()
            rank_label = f'#{i+1}'
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(times_sorted)*0.02,
                    f'{rank_label}\n{time:.1f}s', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color=self.COLOR_PALETTE['dark'])
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Plot 2: Speed Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        bp = ax2.boxplot([times_sorted], vert=True, patch_artist=True, widths=0.6,
                         boxprops=dict(facecolor=self.COLOR_PALETTE['info'], alpha=0.7, linewidth=2),
                         medianprops=dict(color=self.COLOR_PALETTE['danger'], linewidth=3),
                         whiskerprops=dict(linewidth=2, color=self.COLOR_PALETTE['dark']),
                         capprops=dict(linewidth=2, color=self.COLOR_PALETTE['dark']),
                         flierprops=dict(marker='o', markerfacecolor=self.COLOR_PALETTE['warning'], 
                                        markersize=8, linestyle='none', alpha=0.8))
        
        # Overlay individual points
        y_points = times_sorted
        x_points = np.random.normal(1, 0.04, size=len(y_points))
        ax2.scatter(x_points, y_points, alpha=0.6, s=100, c=colors_gradient, 
                   edgecolors='white', linewidth=2, zorder=3)
        
        ax2.set_ylabel('Validation Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_title('Speed Distribution Analysis', fontsize=14, fontweight='bold',
                     color=self.COLOR_PALETTE['primary'], pad=12)
        ax2.set_xticklabels(['All Models'])
        ax2.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax2.set_axisbelow(True)
        
        # Add statistics text
        stats_text = (f'Mean: {np.mean(times_sorted):.1f}s\n'
                     f'Median: {np.median(times_sorted):.1f}s\n'
                     f'Range: {np.ptp(times_sorted):.1f}s')
        ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=self.COLOR_PALETTE['light'], 
                         alpha=0.9, edgecolor=self.COLOR_PALETTE['primary'], linewidth=2),
                fontweight='bold')
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Plot 3: Estimated Full Training Time
        ax3 = fig.add_subplot(gs[1, 0])
        full_training_factor = (20 / 3) * (train_loader_len / 50)
        estimated_times = [t * full_training_factor / 3600 for t in times_sorted]  # hours
        
        bars3 = ax3.barh(models_sorted, estimated_times, color=colors_gradient, 
                        alpha=0.85, edgecolor='white', linewidth=2)
        
        ax3.set_xlabel('Estimated Training Time (hours)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax3.set_title('Projected Full Training Duration (20 epochs)', fontsize=14, 
                     fontweight='bold', color=self.COLOR_PALETTE['primary'], pad=12)
        ax3.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.8)
        ax3.set_axisbelow(True)
        
        # Add time labels
        for bar, time_h in zip(bars3, estimated_times):
            width = bar.get_width()
            ax3.text(width + max(estimated_times)*0.02, bar.get_y() + bar.get_height()/2,
                    f'{time_h:.1f}h', ha='left', va='center', fontsize=11, 
                    fontweight='bold', color=self.COLOR_PALETTE['dark'])
        
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Plot 4: Model Efficiency Score
        ax4 = fig.add_subplot(gs[1, 1])
        max_time = max(times_sorted)
        efficiency_scores = [(max_time - t) / max_time * 100 for t in times_sorted]
        scatter_sizes = [100 + score * 5 for score in efficiency_scores]
        
        scatter = ax4.scatter(range(len(models_sorted)), efficiency_scores, 
                             s=scatter_sizes, c=colors_gradient, alpha=0.7,
                             edgecolors='white', linewidth=2.5, zorder=3)
        
        ax4.plot(range(len(models_sorted)), efficiency_scores, 
                color=self.COLOR_PALETTE['primary'], alpha=0.3, linewidth=2, 
                linestyle='--', zorder=1)
        
        ax4.axhline(y=50, color=self.COLOR_PALETTE['warning'], linestyle='--', 
                   linewidth=2, alpha=0.5, label='Average Efficiency')
        
        ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Efficiency Score (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Model Training Efficiency', fontsize=14, fontweight='bold',
                     color=self.COLOR_PALETTE['primary'], pad=12)
        ax4.set_xticks(range(len(models_sorted)))
        ax4.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=10)
        ax4.set_ylim(-5, 105)
        ax4.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        ax4.set_axisbelow(True)
        
        # Add efficiency labels
        for i, (score, model) in enumerate(zip(efficiency_scores, models_sorted)):
            ax4.annotate(f'{score:.0f}%', (i, score), 
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9, fontweight='bold',
                        color=self.COLOR_PALETTE['dark'])
        
        ax4.legend(loc='lower left', fontsize=10, framealpha=0.95)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_performers(self, validation_results: Dict, all_models: List[str]):
        """
        Plot top performing models in podium style.
        
        Args:
            validation_results: Dictionary with validation results per model
            all_models: List of all model names
        """
        passed_models = [(m, validation_results[m]['time']) for m in all_models 
                        if validation_results[m]['status'] == 'passed']
        
        if len(passed_models) < 3:
            print(f"Need at least 3 passed models for podium. Only {len(passed_models)} available.")
            return
        
        passed_models_sorted = sorted(passed_models, key=lambda x: x[1])
        top_3 = passed_models_sorted[:3]
        
        fig = plt.figure(figsize=(18, 6))
        fig.patch.set_facecolor('white')
        
        top_names = [m[0].upper() for m in top_3]
        top_times = [m[1] for m in top_3]
        top_colors = self.MODEL_COLORS[:3]
        
        # Podium order (2nd, 1st, 3rd for visual appeal)
        podium_order = [1, 0, 2]
        podium_heights = [top_times[i] for i in podium_order]
        podium_labels = [top_names[i] for i in podium_order]
        podium_colors = [top_colors[i] for i in podium_order]
        positions = [0, 1, 2]
        medals = ['2nd', '1st', '3rd']
        
        ax = fig.add_subplot(111)
        bars = ax.bar(positions, podium_heights, color=podium_colors, 
                     alpha=0.85, edgecolor='white', linewidth=3, width=0.6)
        
        # Add podium effect
        for bar in bars:
            bar.set_hatch('...')
        
        ax.set_ylabel('Validation Time (seconds)', fontsize=13, fontweight='bold')
        ax.set_title('Top Performing Models - Speed Champions', fontsize=16, 
                     fontweight='bold', color=self.COLOR_PALETTE['primary'], pad=20)
        ax.set_xticks(positions)
        ax.set_xticklabels(podium_labels, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Add time and medal labels
        for i, (bar, time, medal) in enumerate(zip(bars, podium_heights, medals)):
            height = bar.get_height()
            # Medal emoji
            ax.text(bar.get_x() + bar.get_width()/2., height + max(podium_heights)*0.05,
                    medal, ha='center', va='bottom', fontsize=14, fontweight='bold')
            # Time
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{time:.1f}s', ha='center', va='center', fontsize=13, 
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        # Add recommendation box
        fastest_model = top_names[podium_order.index(0)]
        recommendation = (
            f"RECOMMENDATION:\n\n"
            f"- Fastest: {fastest_model} ({top_times[0]:.1f}s)\n"
            f"- Best for rapid iteration and experimentation\n"
            f"- {(top_times[-1]/top_times[0]):.1f}x faster than slowest model"
        )
        
        ax.text(0.98, 0.97, recommendation, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=1', facecolor=self.COLOR_PALETTE['success'], 
                         alpha=0.2, edgecolor=self.COLOR_PALETTE['success'], linewidth=3),
                fontweight='bold', family='monospace')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def print_validation_statistics(self, validation_results: Dict, all_models: List[str], train_loader_len: int):
        """
        Print detailed validation statistics.
        
        Args:
            validation_results: Dictionary with validation results per model
            all_models: List of all model names
            train_loader_len: Length of training data loader
        """
        passed_models = [(m, validation_results[m]['time']) for m in all_models 
                        if validation_results[m]['status'] == 'passed']
        failed_models = [m for m in all_models if validation_results[m]['status'] == 'failed']
        
        passed_count = len(passed_models)
        failed_count = len(failed_models)
        
        print("\n" + "="*90)
        print("VALIDATION PERFORMANCE ANALYSIS")
        print("="*90)
        
        if passed_count > 0:
            passed_models_sorted = sorted(passed_models, key=lambda x: x[1])
            passed_times = [m[1] for m in passed_models]
            avg_time = np.mean(passed_times)
            min_time = np.min(passed_times)
            max_time = np.max(passed_times)
            std_time = np.std(passed_times)
            
            fastest_model = passed_models_sorted[0][0]
            slowest_model = passed_models_sorted[-1][0]
            
            print(f"\nPassed Models Statistics ({passed_count} models):")
            print(f"   • Average time:    {avg_time:.1f}s")
            print(f"   • Fastest:         {fastest_model.upper()} ({min_time:.1f}s)")
            print(f"   • Slowest:         {slowest_model.upper()} ({max_time:.1f}s)")
            print(f"   • Time range:      {min_time:.1f}s - {max_time:.1f}s")
            print(f"   • Std deviation:   ±{std_time:.1f}s")
            print(f"   • Speed variation: {((max_time - min_time) / min_time * 100):.1f}% difference")
            
            # Estimated full training time
            full_training_factor = (20 / 3) * (train_loader_len / 50)
            estimated_total_time = avg_time * full_training_factor * passed_count / 3600
            
            print(f"\nEstimated Full Training Time (20 epochs, full dataset):")
            print(f"   • Per model (avg):     ~{avg_time * full_training_factor / 3600:.1f} hours")
            print(f"   • All {passed_count} models:        ~{estimated_total_time:.1f} hours")
            print(f"   • Sequential training: ~{estimated_total_time:.1f} hours")
            print(f"   • Parallel (3 pairs):  ~{estimated_total_time / 3:.1f} hours")
            
            # Efficiency insights
            print(f"\nEfficiency Insights:")
            speedup = max_time / min_time
            print(f"   • {fastest_model.upper()} is {speedup:.1f}x faster than {slowest_model.upper()}")
            print(f"   • Time saved by choosing fastest: {max_time - min_time:.1f}s per validation")
            
            # Top recommendations
            print(f"\nTop Recommendations:")
            for i, (model, time) in enumerate(passed_models_sorted[:3], 1):
                efficiency = (max_time - time) / max_time * 100
                print(f"   Rank {i}: {model.upper()} - {time:.1f}s (Efficiency: {efficiency:.0f}%)")
        
        if failed_count > 0:
            print(f"\n\nFailed Models ({failed_count}):")
            for model_name in failed_models:
                print(f"\n   • {model_name.upper()}:")
                error_msg = validation_results[model_name]['error']
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                print(f"     Error: {error_msg}")
        
        print("\n" + "="*90)


class TrainingVisualizer:
    """Visualizer for training metrics and model comparison."""
    
    @staticmethod
    def plot_training_history(history: Dict, model_name: str, save_path: Optional[Path] = None):
        """
        Plot training history for a single model.
        
        Args:
            history: Training history dictionary
            model_name: Name of the model
            save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # IoU plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(epochs, history['train_iou'], 'b-', label='Train IoU', linewidth=2)
        ax2.plot(epochs, history['val_iou'], 'r-', label='Val IoU', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('IoU', fontsize=11)
        ax2.set_title('Mean IoU', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Dice plot
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
        ax3.plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Dice', fontsize=11)
        ax3.set_title('Mean Dice Coefficient', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} Training Metrics', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print best metrics
        best_epoch = max(range(len(history['val_iou'])), key=lambda i: history['val_iou'][i])
        print(f"\n{'='*80}")
        print(f"{model_name} - Best Validation Metrics (Epoch {best_epoch + 1})")
        print(f"{'='*80}")
        print(f"Mean IoU: {history['val_iou'][best_epoch]:.4f}")
        print(f"Mean Dice: {history['val_dice'][best_epoch]:.4f}")
        print(f"Mean F1: {history['val_f1'][best_epoch]:.4f}")
        print(f"{'='*80}\n")
    
    @staticmethod
    def compare_models(histories: List[Dict], model_names: List[str], save_path: Optional[Path] = None):
        """
        Compare multiple models.
        
        Args:
            histories: List of training history dictionaries
            model_names: List of model names
            save_path: Optional path to save the figure
        """
        # Filter out None histories
        valid_data = [(h, n, c) for h, n, c in zip(histories, model_names, 
                      ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']) 
                      if h is not None]
        
        if not valid_data:
            print("No valid training histories to compare.")
            return
        
        valid_histories, valid_names, valid_colors = zip(*valid_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Validation Loss
        for history, name, color in zip(valid_histories, valid_names, valid_colors):
            epochs = range(1, len(history['val_loss']) + 1)
            axes[0, 0].plot(epochs, history['val_loss'], label=name, linewidth=2, color=color)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Validation IoU
        for history, name, color in zip(valid_histories, valid_names, valid_colors):
            epochs = range(1, len(history['val_iou']) + 1)
            axes[0, 1].plot(epochs, history['val_iou'], label=name, linewidth=2, color=color)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('IoU', fontsize=12)
        axes[0, 1].set_title('Validation Mean IoU Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Validation Dice
        for history, name, color in zip(valid_histories, valid_names, valid_colors):
            epochs = range(1, len(history['val_dice']) + 1)
            axes[1, 0].plot(epochs, history['val_dice'], label=name, linewidth=2, color=color)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Dice', fontsize=12)
        axes[1, 0].set_title('Validation Mean Dice Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Best Metrics Bar Chart
        x = np.arange(len(valid_names))
        width = 0.25
        
        iou_values = []
        dice_values = []
        f1_values = []
        
        for history in valid_histories:
            best_epoch = max(range(len(history['val_iou'])), key=lambda i: history['val_iou'][i])
            iou_values.append(history['val_iou'][best_epoch])
            dice_values.append(history['val_dice'][best_epoch])
            f1_values.append(history['val_f1'][best_epoch])
        
        axes[1, 1].bar(x - width, iou_values, width, label='IoU', alpha=0.8)
        axes[1, 1].bar(x, dice_values, width, label='Dice', alpha=0.8)
        axes[1, 1].bar(x + width, f1_values, width, label='F1', alpha=0.8)
        
        axes[1, 1].set_xlabel('Model', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_title('Best Validation Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(valid_names, rotation=45, ha='right')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print comparison table
        print(f"\n{'='*100}")
        print("MODEL COMPARISON - BEST VALIDATION METRICS")
        print(f"{'='*100}")
        print(f"{'Model':<20} {'Best Epoch':<12} {'Mean IoU':<12} {'Mean Dice':<12} {'Mean F1':<12}")
        print(f"{'-'*100}")
        
        for name, history in zip(valid_names, valid_histories):
            best_epoch = max(range(len(history['val_iou'])), key=lambda i: history['val_iou'][i])
            
            print(f"{name:<20} {best_epoch+1:<12} {history['val_iou'][best_epoch]:<12.4f} "
                  f"{history['val_dice'][best_epoch]:<12.4f} {history['val_f1'][best_epoch]:<12.4f}")
        
        print(f"{'='*100}\n")
