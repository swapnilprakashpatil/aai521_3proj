"""
Visualization utilities for preprocessing, training, and validation analysis.
Combines all visualization functions in a single module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# PREPROCESSING VISUALIZATION FUNCTIONS
# ============================================================================

def plot_flood_statistics(germany_stats: Dict, louisiana_stats: Dict) -> plt.Figure:
    """
    Plot flood statistics comparison between regions
    
    Args:
        germany_stats: Statistics dictionary for Germany region
        louisiana_stats: Statistics dictionary for Louisiana-East region
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    regions = ['Germany', 'Louisiana-East']
    flooded = [germany_stats['flooded_count'], louisiana_stats['flooded_count']]
    non_flooded = [germany_stats['non_flooded_count'], louisiana_stats['non_flooded_count']]
    
    x = np.arange(len(regions))
    width = 0.35
    
    # Plot 1: Counts
    axes[0].bar(x - width/2, flooded, width, label='Flooded', color='#e74c3c')
    axes[0].bar(x + width/2, non_flooded, width, label='Non-flooded', color='#2ecc71')
    axes[0].set_xlabel('Region')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Flooded vs Non-flooded Road Segments')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regions)
    axes[0].legend()
    
    # Plot 2: Percentages
    flooded_pct = [germany_stats['flooded_pct'], louisiana_stats['flooded_pct']]
    non_flooded_pct = [germany_stats['non_flooded_pct'], louisiana_stats['non_flooded_pct']]
    
    axes[1].bar(x, flooded_pct, width, label='Flooded %', color='#e74c3c')
    axes[1].bar(x, non_flooded_pct, width, bottom=flooded_pct, label='Non-flooded %', color='#2ecc71')
    axes[1].set_xlabel('Region')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Class Distribution')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regions)
    axes[1].legend()
    
    plt.tight_layout()
    return fig


def plot_sample_tile(tile_data: Dict, title: str = "Sample Tile") -> plt.Figure:
    """
    Visualize a sample tile with pre/post images and mask
    
    Args:
        tile_data: Dictionary containing 'pre', 'post', and 'mask' arrays
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(tile_data['pre'])
    axes[0].set_title('Pre-Event')
    axes[0].axis('off')
    
    axes[1].imshow(tile_data['post'])
    axes[1].set_title('Post-Event')
    axes[1].axis('off')
    
    axes[2].imshow(tile_data['mask'], cmap='tab10', vmin=0, vmax=6)
    axes[2].set_title('Flood Mask')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_preprocessing_comparison(raw_image: np.ndarray, 
                                  processed_image: np.ndarray,
                                  title: str = "Preprocessing Comparison") -> plt.Figure:
    """
    Compare raw and processed images side by side
    
    Args:
        raw_image: Original image
        processed_image: Preprocessed image
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(raw_image)
    axes[0].set_title('Raw Image')
    axes[0].axis('off')
    
    axes[1].imshow(processed_image)
    axes[1].set_title('Processed Image')
    axes[1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_quality_metrics(raw_metrics: Dict, processed_metrics: Dict) -> plt.Figure:
    """
    Plot quality metrics comparison between raw and processed images
    
    Args:
        raw_metrics: Quality metrics for raw image
        processed_metrics: Quality metrics for processed image
        
    Returns:
        Matplotlib figure object
    """
    metrics = list(raw_metrics.keys())
    raw_values = list(raw_metrics.values())
    processed_values = list(processed_metrics.values())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, raw_values, width, label='Raw', color='#3498db')
    ax.bar(x + width/2, processed_values, width, label='Processed', color='#e74c3c')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title('Image Quality Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_patch_grid(patches: List[Dict], 
                   n_samples: int = 6,
                   show_mask: bool = True) -> plt.Figure:
    """
    Display a grid of sample patches
    
    Args:
        patches: List of patch dictionaries with 'image' and 'mask' keys
        n_samples: Number of samples to display
        show_mask: Whether to show masks alongside images
        
    Returns:
        Matplotlib figure object
    """
    n_samples = min(n_samples, len(patches))
    rows = 2 if show_mask else 1
    
    fig, axes = plt.subplots(rows, n_samples, figsize=(n_samples*3, rows*3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        patch = patches[i]
        
        # Show image (pre-event or combined)
        if patch['image'].shape[2] == 6:
            # Combined pre+post
            axes[0, i].imshow(patch['image'][:, :, :3])
        else:
            axes[0, i].imshow(patch['image'])
        axes[0, i].set_title(f"Patch {i+1}")
        axes[0, i].axis('off')
        
        # Show mask if requested
        if show_mask:
            axes[1, i].imshow(patch['mask'], cmap='tab10', vmin=0, vmax=6)
            axes[1, i].set_title(f"Mask {i+1}")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_class_distribution(patches: List[Dict], 
                           class_names: Dict[int, str],
                           class_colors: Dict[int, List[int]]) -> plt.Figure:
    """
    Plot class distribution from patches
    
    Args:
        patches: List of patch dictionaries
        class_names: Mapping of class ID to name
        class_colors: Mapping of class ID to RGB color
        
    Returns:
        Matplotlib figure object
    """
    class_totals = {i: 0 for i in range(7)}
    
    for patch in patches:
        for cls, count in patch.get('class_distribution', {}).items():
            class_totals[int(cls)] += count
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    classes = list(class_totals.keys())
    counts = list(class_totals.values())
    labels = [f"{class_names.get(c, f'Class {c}')}\n{counts[i]:,}" 
             for i, c in enumerate(classes)]
    
    # Convert colors to normalized format
    colors = []
    for cls in classes:
        rgb = class_colors.get(cls, [128, 128, 128])
        r, g, b = [val/255.0 for val in rgb]
        colors.append((r, g, b))
    
    # Pie chart
    axes[0].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Class Distribution (Pixel Count)')
    
    # Bar chart
    axes[1].bar(classes, counts, color=colors)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Pixel Count')
    axes[1].set_title('Class Distribution (Bar Chart)')
    axes[1].set_xticks(classes)
    axes[1].set_xticklabels([class_names.get(c, f'C{c}') for c in classes], 
                            rotation=45, ha='right')
    
    for i, (cls, count) in enumerate(zip(classes, counts)):
        axes[1].text(cls, count, f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_augmentation_samples(image: np.ndarray, 
                              mask: np.ndarray,
                              augmentation_fn,
                              n_samples: int = 6) -> plt.Figure:
    """
    Show augmentation examples
    
    Args:
        image: Input image
        mask: Input mask
        augmentation_fn: Augmentation function to apply
        n_samples: Number of augmented samples to show
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*3, 6))
    
    for i in range(n_samples):
        augmented = augmentation_fn(image=image, mask=mask)
        
        axes[0, i].imshow(augmented['image'])
        axes[0, i].set_title(f'Aug {i+1} - Image')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(augmented['mask'], cmap='tab10', vmin=0, vmax=5)
        axes[1, i].set_title(f'Aug {i+1} - Mask')
        axes[1, i].axis('off')
    
    plt.suptitle('Data Augmentation Examples')
    plt.tight_layout()
    return fig


def plot_image_comparison_grid(comparison_samples: List[Dict],
                               title: str = "Quality Comparison") -> plt.Figure:
    """
    Plot comparison grid showing raw, degraded, and enhanced images
    
    Args:
        comparison_samples: List of comparison dictionaries with 'raw', 'degraded', 'enhanced'
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    n_samples = len(comparison_samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples*4))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(comparison_samples):
        axes[i, 0].imshow(sample['raw'])
        axes[i, 0].set_title('Raw')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(sample['degraded'])
        axes[i, 1].set_title('Degraded')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(sample['enhanced'])
        axes[i, 2].set_title('Enhanced')
        axes[i, 2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_change_detection(pre_image: np.ndarray,
                          post_image: np.ndarray,
                          diff_image: np.ndarray) -> plt.Figure:
    """
    Visualize change detection between pre and post images
    
    Args:
        pre_image: Pre-event image
        post_image: Post-event image
        diff_image: Difference/change map
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(pre_image)
    axes[0].set_title('Pre-Event')
    axes[0].axis('off')
    
    axes[1].imshow(post_image)
    axes[1].set_title('Post-Event')
    axes[1].axis('off')
    
    axes[2].imshow(diff_image, cmap='RdYlGn_r')
    axes[2].set_title('Change Map')
    axes[2].axis('off')
    
    plt.suptitle('Change Detection')
    plt.tight_layout()
    return fig


def plot_tile_overview(sample_data: Dict, class_names: Dict[int, str]) -> plt.Figure:
    """
    Visualize original tile with pre/post images, mask, and difference
    
    Args:
        sample_data: Dictionary with 'pre_image', 'post_image', 'mask' keys
        class_names: Mapping of class IDs to names
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Pre-event
    axes[0, 0].imshow(sample_data['pre_image'])
    axes[0, 0].set_title('Pre-Event Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Post-event
    axes[0, 1].imshow(sample_data['post_image'])
    axes[0, 1].set_title('Post-Event Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Mask
    axes[1, 0].imshow(sample_data['mask'], cmap='tab10')
    axes[1, 0].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference
    diff = np.abs(sample_data['post_image'] - sample_data['pre_image'])
    axes[1, 1].imshow(diff)
    axes[1, 1].set_title('Temporal Difference (|Post - Pre|)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig


def plot_clahe_comparison(sample_data: Dict, pre_enhanced: np.ndarray, 
                         post_enhanced: np.ndarray) -> plt.Figure:
    """
    Compare original vs CLAHE enhanced images with histograms
    
    Args:
        sample_data: Dictionary with 'pre_image' and 'post_image'
        pre_enhanced: CLAHE enhanced pre-event image
        post_enhanced: CLAHE enhanced post-event image
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Pre-event comparison
    axes[0, 0].imshow(sample_data['pre_image'])
    axes[0, 0].set_title('Pre-Event Original', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pre_enhanced)
    axes[0, 1].set_title('Pre-Event Enhanced (CLAHE)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Histogram comparison for pre-event
    for i in range(3):
        axes[0, 2].hist(sample_data['pre_image'][:, :, i].flatten(), bins=50, 
                       alpha=0.5, label=f'Ch{i} Orig')
        axes[0, 2].hist(pre_enhanced[:, :, i].flatten(), bins=50, 
                       alpha=0.5, label=f'Ch{i} Enh', linestyle='--')
    axes[0, 2].set_title('Pre-Event Histogram', fontsize=12)
    axes[0, 2].set_xlabel('Intensity')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.3)
    
    # Post-event comparison
    axes[1, 0].imshow(sample_data['post_image'])
    axes[1, 0].set_title('Post-Event Original', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(post_enhanced)
    axes[1, 1].set_title('Post-Event Enhanced (CLAHE)', fontsize=12)
    axes[1, 1].axis('off')
    
    # Histogram comparison for post-event
    for i in range(3):
        axes[1, 2].hist(sample_data['post_image'][:, :, i].flatten(), bins=50, 
                       alpha=0.5, label=f'Ch{i} Orig')
        axes[1, 2].hist(post_enhanced[:, :, i].flatten(), bins=50, 
                       alpha=0.5, label=f'Ch{i} Enh', linestyle='--')
    axes[1, 2].set_title('Post-Event Histogram', fontsize=12)
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_advanced_preprocessing(sample_data: Dict, pre_degraded: np.ndarray,
                               pre_enhanced: np.ndarray, pre_cloud_removed: np.ndarray,
                               pre_cloud_mask: np.ndarray, post_degraded: np.ndarray,
                               post_enhanced: np.ndarray, post_cloud_removed: np.ndarray,
                               post_cloud_mask: np.ndarray, pre_metrics: Dict,
                               post_metrics: Dict) -> plt.Figure:
    """
    Comprehensive visualization of advanced preprocessing (cloud removal + deblurring)
    
    Args:
        sample_data: Original data with 'pre_image' and 'post_image'
        pre_degraded: Synthetically degraded pre-event image
        pre_enhanced: Enhanced pre-event image
        pre_cloud_removed: Pre-event after cloud removal
        pre_cloud_mask: Detected cloud mask for pre-event
        post_degraded: Synthetically degraded post-event image
        post_enhanced: Enhanced post-event image
        post_cloud_removed: Post-event after cloud removal
        post_cloud_mask: Detected cloud mask for post-event
        pre_metrics: Dictionary with 'orig', 'degraded', 'enhanced' metrics for pre
        post_metrics: Dictionary with 'orig', 'degraded', 'enhanced' metrics for post
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(5, 3, figsize=(18, 30))
    
    # Column headers (Row 0)
    for ax in axes[0, :]:
        ax.axis('off')
    
    axes[0, 0].text(0.5, 0.5, 'ORIGINAL\n(Clean)', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    axes[0, 1].text(0.5, 0.5, 'DEGRADED\n(Clouds + Blur)', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    axes[0, 2].text(0.5, 0.5, 'ENHANCED\n(Preprocessed)', ha='center', va='center', 
                    fontsize=16, fontweight='bold', color='darkgreen',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # PRE-EVENT IMAGE (Row 1)
    axes[1, 0].imshow(sample_data['pre_image'])
    axes[1, 0].set_title(f'Pre-Event Original\nSharpness: {pre_metrics["orig"]["sharpness"]:.1f} | Contrast: {pre_metrics["orig"]["contrast"]:.3f}', 
                         fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pre_degraded)
    axes[1, 1].set_title(f'Pre-Event Degraded\nSharpness: {pre_metrics["degraded"]["sharpness"]:.1f} | Contrast: {pre_metrics["degraded"]["contrast"]:.3f}', 
                         fontsize=11, color='darkred', fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pre_enhanced)
    sharpness_imp = ((pre_metrics["enhanced"]["sharpness"]/pre_metrics["degraded"]["sharpness"]-1)*100)
    contrast_imp = ((pre_metrics["enhanced"]["contrast"]/pre_metrics["degraded"]["contrast"]-1)*100)
    axes[1, 2].set_title(f'Pre-Event Enhanced\nSharpness: {pre_metrics["enhanced"]["sharpness"]:.1f} (+{sharpness_imp:.0f}%) | Contrast: {pre_metrics["enhanced"]["contrast"]:.3f} (+{contrast_imp:.0f}%)', 
                         fontsize=11, color='darkgreen', fontweight='bold')
    axes[1, 2].axis('off')
    
    # PRE-EVENT DETAILS (Row 2)
    pre_cloud_cov = np.mean(pre_cloud_mask) * 100
    axes[2, 0].imshow(pre_cloud_mask, cmap='Reds', vmin=0, vmax=1)
    axes[2, 0].set_title(f'Pre-Event Cloud Mask\n{pre_cloud_cov:.1f}% coverage detected', 
                         fontsize=11, color='red')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(pre_cloud_removed)
    axes[2, 1].set_title('Pre-Event Cloud Removed\n(Before deblurring)', fontsize=11)
    axes[2, 1].axis('off')
    
    diff_pre = np.abs(pre_degraded - pre_enhanced)
    axes[2, 2].imshow(diff_pre)
    axes[2, 2].set_title('Pre-Event Changes\n(Difference Map)', fontsize=11)
    axes[2, 2].axis('off')
    
    # POST-EVENT IMAGE (Row 3)
    axes[3, 0].imshow(sample_data['post_image'])
    axes[3, 0].set_title(f'Post-Event Original\nSharpness: {post_metrics["orig"]["sharpness"]:.1f} | Contrast: {post_metrics["orig"]["contrast"]:.3f}', 
                         fontsize=11, fontweight='bold')
    axes[3, 0].axis('off')
    
    axes[3, 1].imshow(post_degraded)
    axes[3, 1].set_title(f'Post-Event Degraded\nSharpness: {post_metrics["degraded"]["sharpness"]:.1f} | Contrast: {post_metrics["degraded"]["contrast"]:.3f}', 
                         fontsize=11, color='darkred', fontweight='bold')
    axes[3, 1].axis('off')
    
    axes[3, 2].imshow(post_enhanced)
    post_sharpness_imp = ((post_metrics["enhanced"]["sharpness"]/post_metrics["degraded"]["sharpness"]-1)*100)
    post_contrast_imp = ((post_metrics["enhanced"]["contrast"]/post_metrics["degraded"]["contrast"]-1)*100)
    axes[3, 2].set_title(f'Post-Event Enhanced\nSharpness: {post_metrics["enhanced"]["sharpness"]:.1f} (+{post_sharpness_imp:.0f}%) | Contrast: {post_metrics["enhanced"]["contrast"]:.3f} (+{post_contrast_imp:.0f}%)', 
                         fontsize=11, color='darkgreen', fontweight='bold')
    axes[3, 2].axis('off')
    
    # POST-EVENT DETAILS (Row 4)
    post_cloud_cov = np.mean(post_cloud_mask) * 100
    axes[4, 0].imshow(post_cloud_mask, cmap='Reds', vmin=0, vmax=1)
    axes[4, 0].set_title(f'Post-Event Cloud Mask\n{post_cloud_cov:.1f}% coverage detected', 
                         fontsize=11, color='red')
    axes[4, 0].axis('off')
    
    axes[4, 1].imshow(post_cloud_removed)
    axes[4, 1].set_title('Post-Event Cloud Removed\n(Before deblurring)', fontsize=11)
    axes[4, 1].axis('off')
    
    diff_post = np.abs(post_degraded - post_enhanced)
    axes[4, 2].imshow(diff_post)
    axes[4, 2].set_title('Post-Event Changes\n(Difference Map)', fontsize=11)
    axes[4, 2].axis('off')
    
    plt.suptitle('Advanced Preprocessing: Cloud Removal + Deblurring\nOriginal → Synthetic Degradation → Enhanced (Pre & Post Event)', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_patch_samples(patches: List[Dict], n_samples: int = 8, 
                      patch_size: int = 512) -> plt.Figure:
    """
    Visualize sample patches with pre/post/mask/difference
    
    Args:
        patches: List of patch dictionaries
        n_samples: Number of samples to display
        patch_size: Size of patches
        
    Returns:
        Matplotlib figure object
    """
    n_samples = min(n_samples, len(patches))
    sample_patches = np.random.choice(patches, n_samples, replace=False)
    
    fig, axes = plt.subplots(4, n_samples, figsize=(n_samples*3, 12))
    
    for i, patch in enumerate(sample_patches):
        # Pre-event (first 3 channels)
        pre_patch = patch['image'][:, :, :3]
        axes[0, i].imshow(pre_patch)
        axes[0, i].set_title(f"Patch {i}\nPre-Event", fontsize=10)
        axes[0, i].axis('off')
        
        # Post-event (last 3 channels)
        post_patch = patch['image'][:, :, 3:6]
        axes[1, i].imshow(post_patch)
        axes[1, i].set_title('Post-Event', fontsize=10)
        axes[1, i].axis('off')
        
        # Mask
        axes[2, i].imshow(patch['mask'], cmap='tab10', vmin=0, vmax=5)
        axes[2, i].set_title('Mask', fontsize=10)
        axes[2, i].axis('off')
        
        # Difference
        diff_patch = np.abs(post_patch - pre_patch)
        axes[3, i].imshow(diff_patch)
        flood_status = 'FLOOD' if patch['is_flood_positive'] else 'OK'
        axes[3, i].set_title(f"Difference\n{flood_status}", fontsize=10)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_processed_sample(sample_img: np.ndarray, sample_mask: np.ndarray) -> plt.Figure:
    """
    Visualize a processed data sample with pre/post/mask
    
    Args:
        sample_img: Processed image array (6 channels)
        sample_mask: Corresponding mask
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sample_img[:, :, :3])  # Pre-event
    axes[0].set_title('Pre-Event (Processed)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(sample_img[:, :, 3:6])  # Post-event
    axes[1].set_title('Post-Event (Processed)', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(sample_mask, cmap='tab10')
    axes[2].set_title('Mask', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_raw_vs_processed_comparison(comparison_samples: List[Dict],
                                     calculate_metrics_fn) -> plt.Figure:
    """
    Compare raw vs processed full-resolution images
    
    Args:
        comparison_samples: List of dicts with 'region', 'tile', 'raw_pre', 'raw_post',
                           'processed_pre', 'processed_post' paths
        calculate_metrics_fn: Function to calculate quality metrics
        
    Returns:
        Matplotlib figure object
    """
    n_samples = len(comparison_samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5*n_samples))
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(comparison_samples):
        # Display images
        axes[idx, 0].imshow(sample['raw_pre'])
        axes[idx, 0].set_title(f"{sample['region']}\n{sample['tile']}\nRaw PRE-event", 
                               fontsize=11, fontweight='bold')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(sample['processed_pre'])
        axes[idx, 1].set_title(f"Processed PRE-event\n(CLAHE + Cloud Removal + Deblur)", 
                               fontsize=11, fontweight='bold', color='darkgreen')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(sample['raw_post'])
        axes[idx, 2].set_title(f"Raw POST-event", 
                               fontsize=11, fontweight='bold')
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(sample['processed_post'])
        axes[idx, 3].set_title(f"Processed POST-event\n(CLAHE + Cloud Removal + Deblur)", 
                               fontsize=11, fontweight='bold', color='darkgreen')
        axes[idx, 3].axis('off')
    
    plt.suptitle('Raw vs Processed Full-Resolution Images Comparison\n(Germany & Louisiana-East Datasets)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


# ============================================================================
# EDA (EXPLORATORY DATA ANALYSIS) VISUALIZATION FUNCTIONS
# ============================================================================

def plot_csv_metadata_analysis(csv_analysis: Dict) -> plt.Figure:
    """
    Plot comprehensive CSV metadata analysis (2x2 grid)
    
    Args:
        csv_analysis: Dictionary containing CSV analysis data per dataset
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CSV Metadata Analysis - Dataset Characteristics', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Number of annotations per dataset
    ax1 = axes[0, 0]
    datasets = []
    annotation_counts = []
    for dataset_name, data in csv_analysis.items():
        if 'mapping' in data:
            datasets.append(dataset_name)
            annotation_counts.append(data['mapping']['rows'])
    
    if datasets:
        bars1 = ax1.bar(datasets, annotation_counts, color='#06D6A0', edgecolor='black')
        ax1.set_title('Number of Annotations per Dataset', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=10)
        ax1.set_xlabel('Dataset', fontsize=10)
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        for i, (bar, count) in enumerate(zip(bars1, annotation_counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(annotation_counts)*0.02,
                    f'{count:,}', ha='center', fontweight='bold', fontsize=10)
    
    # Plot 2: Flood status distribution
    ax2 = axes[0, 1]
    flood_data_available = False
    for dataset_name, data in csv_analysis.items():
        if 'reference' in data:
            df_ref = data['reference']['dataframe']
            if 'flooded' in df_ref.columns:
                flooded_counts = df_ref['flooded'].value_counts()
                labels = ['Flooded' if k == 1 else 'Not Flooded' for k in flooded_counts.index]
                colors = ['#EF476F', '#118AB2']
                wedges, texts, autotexts = ax2.pie(flooded_counts, labels=labels, autopct='%1.1f%%',
                                                    colors=colors, startangle=90)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(11)
                ax2.set_title(f'Flood Status Distribution - {dataset_name}', 
                             fontsize=12, fontweight='bold')
                flood_data_available = True
                break
    
    if not flood_data_available:
        ax2.text(0.5, 0.5, 'No flood status data available', 
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Flood Status Distribution', fontsize=12, fontweight='bold')
    
    # Plot 3: Road segment length distribution
    ax3 = axes[1, 0]
    length_data_available = False
    for dataset_name, data in csv_analysis.items():
        if 'reference' in data:
            df_ref = data['reference']['dataframe']
            if 'length_m' in df_ref.columns:
                valid_lengths = df_ref['length_m'].dropna()
                if len(valid_lengths) > 0:
                    ax3.hist(valid_lengths, bins=50, color='#118AB2', edgecolor='black', alpha=0.7)
                    ax3.set_title(f'Road Segment Length Distribution - {dataset_name}', 
                                 fontsize=12, fontweight='bold')
                    ax3.set_xlabel('Length (meters)', fontsize=10)
                    ax3.set_ylabel('Frequency', fontsize=10)
                    ax3.grid(axis='y', alpha=0.3)
                    stats_text = f'Mean: {valid_lengths.mean():.1f}m\nMedian: {valid_lengths.median():.1f}m\nStd: {valid_lengths.std():.1f}m'
                    ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes,
                            fontsize=9, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    length_data_available = True
                    break
    
    if not length_data_available:
        ax3.text(0.5, 0.5, 'No road length data available', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Road Segment Length Distribution', fontsize=12, fontweight='bold')
    
    # Plot 4: Post-event image availability
    ax4 = axes[1, 1]
    datasets_with_post2 = []
    post2_percentages = []
    for dataset_name, data in csv_analysis.items():
        if 'mapping' in data:
            df_map = data['mapping']['dataframe']
            if 'post-event image 2' in df_map.columns:
                post2_count = df_map['post-event image 2'].notna().sum()
                post2_pct = (post2_count / len(df_map)) * 100
                datasets_with_post2.append(dataset_name)
                post2_percentages.append(post2_pct)
    
    if datasets_with_post2:
        bars4 = ax4.barh(datasets_with_post2, post2_percentages, color='#073B4C')
        ax4.set_title('Availability of 2nd Post-Event Images', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Percentage of Entries (%)', fontsize=10)
        ax4.set_ylabel('Dataset', fontsize=10)
        ax4.set_xlim(0, 100)
        ax4.grid(axis='x', alpha=0.3)
        for i, (bar, pct) in enumerate(zip(bars4, post2_percentages)):
            ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%', va='center', fontweight='bold', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No secondary post-event images', 
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Availability of 2nd Post-Event Images', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_pixel_distributions(datasets: List[Tuple[str, Path]], 
                            num_distributions: int = 2) -> Tuple[plt.Figure, Dict]:
    """
    Plot pixel value distributions for PRE and POST event images
    
    Args:
        datasets: List of (name, path) tuples for datasets
        num_distributions: Number of datasets to analyze
        
    Returns:
        Tuple of (figure, statistics dictionary)
    """
    fig, axes = plt.subplots(num_distributions, 2, figsize=(16, 6 * num_distributions))
    fig.suptitle('Pixel Value Distributions - PRE vs POST Event Images', 
                 fontsize=16, fontweight='bold')
    
    if num_distributions == 1:
        axes = axes.reshape(1, -1)
    
    pixel_statistics = {}
    
    for col_idx, (dataset_name, dataset_path) in enumerate(datasets[:num_distributions]):
        if dataset_path.exists():
            from PIL import Image
            
            pre_dir = dataset_path / "PRE-event"
            post_dir = dataset_path / "POST-event"
            pre_files = sorted(list(pre_dir.glob("*.png")) + list(pre_dir.glob("*.tif")))
            post_files = sorted(list(post_dir.glob("*.png")) + list(post_dir.glob("*.tif")))
            
            if len(pre_files) > 0 and len(post_files) > 0:
                sample_idx = min(5, len(pre_files) - 1)
                pre_img = np.array(Image.open(pre_files[sample_idx]))
                post_img = np.array(Image.open(post_files[sample_idx]))
                
                # Normalize if needed
                if pre_img.dtype == np.uint16:
                    pre_img = (pre_img / 65535.0 * 255).astype(np.uint8)
                if post_img.dtype == np.uint16:
                    post_img = (post_img / 65535.0 * 255).astype(np.uint8)
                
                # Flatten for histogram
                if len(pre_img.shape) == 3:
                    pre_flat = pre_img.reshape(-1, pre_img.shape[2])
                    post_flat = post_img.reshape(-1, post_img.shape[2])
                else:
                    pre_flat = pre_img.reshape(-1, 1)
                    post_flat = post_img.reshape(-1, 1)
                
                colors = ['red', 'green', 'blue'] if pre_flat.shape[1] == 3 else ['gray']
                
                for channel in range(min(3, pre_flat.shape[1])):
                    axes[col_idx, 0].hist(pre_flat[:, channel], bins=50, alpha=0.6, 
                                         color=colors[channel], label=f'{colors[channel].capitalize()}',
                                         density=True)
                    axes[col_idx, 1].hist(post_flat[:, channel], bins=50, alpha=0.6, 
                                         color=colors[channel], label=f'{colors[channel].capitalize()}',
                                         density=True)
                
                axes[col_idx, 0].set_title(f'{dataset_name}\nPRE-event Pixel Distribution', fontweight='bold')
                axes[col_idx, 0].set_xlabel('Pixel Value')
                axes[col_idx, 0].set_ylabel('Density')
                axes[col_idx, 0].legend()
                axes[col_idx, 0].grid(alpha=0.3)
                
                axes[col_idx, 1].set_title(f'{dataset_name}\nPOST-event Pixel Distribution', fontweight='bold')
                axes[col_idx, 1].set_xlabel('Pixel Value')
                axes[col_idx, 1].set_ylabel('Density')
                axes[col_idx, 1].legend()
                axes[col_idx, 1].grid(alpha=0.3)
                
                pixel_statistics[dataset_name] = {
                    'pre_mean': pre_img.mean(axis=(0,1)) if len(pre_img.shape) == 3 else pre_img.mean(),
                    'pre_std': pre_img.std(axis=(0,1)) if len(pre_img.shape) == 3 else pre_img.std(),
                    'post_mean': post_img.mean(axis=(0,1)) if len(post_img.shape) == 3 else post_img.mean(),
                    'post_std': post_img.std(axis=(0,1)) if len(post_img.shape) == 3 else post_img.std(),
                    'pre_percentiles': np.percentile(pre_img, [5, 25, 50, 75, 95]),
                    'post_percentiles': np.percentile(post_img, [5, 25, 50, 75, 95])
                }
    
    plt.tight_layout()
    return fig, pixel_statistics


def plot_temporal_change_detection(datasets: List[Tuple[str, Path]]) -> Tuple[plt.Figure, Dict]:
    """
    Comprehensive temporal change detection analysis (3x5 grid per dataset row)
    
    Args:
        datasets: List of (name, path) tuples for datasets
        
    Returns:
        Tuple of (figure, change metrics dictionary)
    """
    from PIL import Image
    import cv2
    
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle('Temporal Change Detection - Multiple Analysis Methods', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    change_metrics = {}
    
    for row_base, (dataset_name, dataset_path) in enumerate(datasets[:2]):  # Max 2 datasets
        if dataset_path.exists():
            pre_dir = dataset_path / "PRE-event"
            post_dir = dataset_path / "POST-event"
            
            pre_files = sorted(list(pre_dir.glob("*.png")) + list(pre_dir.glob("*.tif")))
            post_files = sorted(list(post_dir.glob("*.png")) + list(post_dir.glob("*.tif")))
            
            if len(pre_files) > 0 and len(post_files) > 0:
                # Find image pair with most changes
                max_samples_to_try = min(10, len(pre_files))
                best_sample_idx = 0
                max_diff = 0
                
                for test_idx in range(max_samples_to_try):
                    try:
                        test_pre = np.array(Image.open(pre_files[test_idx])).astype(float)
                        test_post = np.array(Image.open(post_files[test_idx])).astype(float)
                        if test_pre.shape == test_post.shape:
                            diff_sum = np.abs(test_pre - test_post).sum()
                            if diff_sum > max_diff:
                                max_diff = diff_sum
                                best_sample_idx = test_idx
                    except Exception:
                        continue
                
                sample_idx = best_sample_idx
                pre_img = np.array(Image.open(pre_files[sample_idx])).astype(float)
                post_img = np.array(Image.open(post_files[sample_idx])).astype(float)
                
                # Normalize to 0-255
                if pre_img.max() > 255:
                    pre_img = pre_img / pre_img.max() * 255
                if post_img.max() > 255:
                    post_img = post_img / post_img.max() * 255
                
                row = row_base
                
                # 1. PRE image
                axes[row, 0].imshow(pre_img.astype(np.uint8))
                axes[row, 0].set_title(f'{dataset_name}\nPRE-event', fontweight='bold', fontsize=11)
                axes[row, 0].axis('off')
                
                # 2. POST image
                axes[row, 1].imshow(post_img.astype(np.uint8))
                axes[row, 1].set_title('POST-event', fontweight='bold', fontsize=11)
                axes[row, 1].axis('off')
                
                # 3. Absolute difference
                abs_diff = np.abs(post_img - pre_img)
                if len(abs_diff.shape) == 3:
                    abs_diff_gray = abs_diff.mean(axis=2)
                else:
                    abs_diff_gray = abs_diff
                
                axes[row, 2].imshow(abs_diff_gray, cmap='hot')
                axes[row, 2].set_title('Absolute Difference\n|POST - PRE|', fontweight='bold', fontsize=11)
                axes[row, 2].axis('off')
                
                # 4. Thresholded change mask
                threshold = abs_diff_gray.mean() + 2 * abs_diff_gray.std()
                change_mask = (abs_diff_gray > threshold).astype(np.uint8) * 255
                axes[row, 3].imshow(change_mask, cmap='binary')
                axes[row, 3].set_title(f'Change Mask\n(threshold={threshold:.1f})', 
                                      fontweight='bold', fontsize=11)
                axes[row, 3].axis('off')
                
                # 5. Change overlay
                overlay = pre_img.copy().astype(np.uint8)
                if len(overlay.shape) == 3:
                    overlay[change_mask > 0, 0] = 255  # Red channel
                axes[row, 4].imshow(overlay)
                axes[row, 4].set_title('Change Overlay\n(Red = Changed)', fontweight='bold', fontsize=11)
                axes[row, 4].axis('off')
                
                # Calculate metrics
                mean_abs_diff = abs_diff_gray.mean()
                std_abs_diff = abs_diff_gray.std()
                max_change = abs_diff_gray.max()
                changed_pixels = (change_mask > 0).sum()
                total_pixels = change_mask.size
                change_pct = 100.0 * changed_pixels / total_pixels
                
                change_metrics[dataset_name] = {
                    'mean_abs_diff': mean_abs_diff,
                    'std_abs_diff': std_abs_diff,
                    'max_change': max_change,
                    'change_percentage': change_pct,
                    'changed_pixels': int(changed_pixels),
                    'total_pixels': int(total_pixels),
                    'threshold_used': threshold,
                    'sample_index': sample_idx,
                    'total_diff': max_diff
                }
    
    # Add summary in third row
    for i in range(5):
        axes[2, i].axis('off')
    
    summary_text = "CHANGE DETECTION METRICS\n\n"
    if change_metrics:
        for name, metrics in change_metrics.items():
            summary_text += f"{name}:\n"
            summary_text += f"  Sample: #{metrics.get('sample_index', 'N/A')}\n"
            summary_text += f"  Mean Δ: {metrics['mean_abs_diff']:.2f}\n"
            summary_text += f"  Std Δ:  {metrics['std_abs_diff']:.2f}\n"
            summary_text += f"  Max Δ:  {metrics['max_change']:.2f}\n"
            summary_text += f"  Threshold: {metrics.get('threshold_used', 0):.1f}\n"
            summary_text += f"  Changed: {metrics['change_percentage']:.2f}%\n"
            summary_text += f"  Pixels: {metrics['changed_pixels']:,} / {metrics['total_pixels']:,}\n\n"
    else:
        summary_text += "No change metrics computed."
    
    axes[2, 2].text(0.5, 0.5, summary_text, ha='center', va='center', 
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    return fig, change_metrics


def plot_edge_detection_analysis(datasets: List[Tuple[str, Path]]) -> plt.Figure:
    """
    Edge detection analysis for flood boundaries and structure identification (2x4 grid)
    
    Args:
        datasets: List of (name, path) tuples for datasets
        
    Returns:
        Matplotlib figure object
    """
    from PIL import Image
    import cv2
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Edge Detection Analysis - Identifying Flood Boundaries & Structures', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for row_idx, (dataset_name, dataset_path) in enumerate(datasets[:2]):
        if dataset_path.exists():
            post_dir = dataset_path / "POST-event"
            post_files = sorted(list(post_dir.glob("*.png")) + list(post_dir.glob("*.tif")))
            
            if len(post_files) > 0:
                sample_idx = min(5, len(post_files) - 1)
                img = np.array(Image.open(post_files[sample_idx]))
                
                # Convert to RGB if grayscale
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # Normalize
                if img.dtype == np.uint16:
                    img = (img / 65535.0 * 255).astype(np.uint8)
                elif img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # 1. Original
                axes[row_idx, 0].imshow(img)
                axes[row_idx, 0].set_title(f'{dataset_name}\nOriginal POST-event', 
                                          fontweight='bold', fontsize=10)
                axes[row_idx, 0].axis('off')
                
                # 2. Canny edge detection
                edges_canny = cv2.Canny(gray, 50, 150)
                axes[row_idx, 1].imshow(edges_canny, cmap='gray')
                axes[row_idx, 1].set_title('Canny Edge Detection', fontweight='bold', fontsize=10)
                axes[row_idx, 1].axis('off')
                
                # 3. Sobel edge detection
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                axes[row_idx, 2].imshow(sobel, cmap='gray')
                axes[row_idx, 2].set_title('Sobel Edge Detection', fontweight='bold', fontsize=10)
                axes[row_idx, 2].axis('off')
                
                # 4. Laplacian
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                axes[row_idx, 3].imshow(np.abs(laplacian), cmap='gray')
                axes[row_idx, 3].set_title('Laplacian Edge Detection', fontweight='bold', fontsize=10)
                axes[row_idx, 3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_geospatial_distribution(geospatial_data: Dict) -> plt.Figure:
    """
    Plot geospatial distribution of flood annotations
    
    Args:
        geospatial_data: Dictionary containing coordinates and geometries per dataset
        
    Returns:
        Matplotlib figure object
    """
    if not geospatial_data:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No geospatial data available', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    num_datasets = len(geospatial_data)
    fig, axes = plt.subplots(1, num_datasets, figsize=(12 * num_datasets, 10))
    
    if num_datasets == 1:
        axes = [axes]
    
    fig.suptitle('Geospatial Distribution of Flood Annotations', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for idx, (dataset_name, data) in enumerate(geospatial_data.items()):
        coords = data['coordinates']
        
        # Create 2D histogram for density
        h = axes[idx].hexbin(coords[:, 0], coords[:, 1], gridsize=50, cmap='YlOrRd', 
                            mincnt=1, edgecolors='black', linewidths=0.2)
        
        axes[idx].set_xlabel('X Coordinate (pixels)', fontweight='bold', fontsize=12)
        axes[idx].set_ylabel('Y Coordinate (pixels)', fontweight='bold', fontsize=12)
        axes[idx].set_title(f'{dataset_name}\nSpatial Density of Annotations\n'
                           f'({len(coords):,} points from {len(data["geometries"])} features)',
                           fontweight='bold', fontsize=13)
        
        plt.colorbar(h, ax=axes[idx], label='Point Density')
        axes[idx].grid(alpha=0.3, linestyle='--')
        
        # Mark center
        center = data['center']
        axes[idx].plot(center['x'], center['y'], 'b*', markersize=20, 
                      label=f'Center ({center["x"]:.0f}, {center["y"]:.0f})', 
                      markeredgecolor='white', markeredgewidth=2)
        axes[idx].legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_annotation_samples_grid(datasets: List[Tuple[str, Path]], 
                                 grid_size: Tuple[int, int] = (3, 6)) -> plt.Figure:
    """
    Display grid of annotation sample images (3x6 default)
    
    Args:
        datasets: List of (name, path) tuples for datasets
        grid_size: Tuple of (rows, cols) for grid layout
        
    Returns:
        Matplotlib figure object
    """
    from PIL import Image
    import cv2
    
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(24, 12))
    fig.suptitle('Sample Annotations from All Datasets', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    sample_idx = 0
    
    for dataset_name, dataset_path in datasets:
        if dataset_path.exists():
            ann_dir = dataset_path / "annotations"
            if ann_dir.exists():
                ann_files = sorted(list(ann_dir.glob("*.png")) + list(ann_dir.glob("*.tif")))
                
                # Display up to 6 samples from this dataset
                for i in range(min(6, len(ann_files))):
                    if sample_idx >= rows * cols:
                        break
                    
                    row = sample_idx // cols
                    col = sample_idx % cols
                    
                    try:
                        ann_img = np.array(Image.open(ann_files[i]))
                        
                        if ann_img.max() > 1:
                            ann_display = ann_img
                        else:
                            ann_display = ann_img * 255
                        
                        axes[row, col].imshow(ann_display, cmap='tab10')
                        axes[row, col].set_title(f'{dataset_name}\n{ann_files[i].stem[:20]}', 
                                                fontsize=9, fontweight='bold')
                        axes[row, col].axis('off')
                        sample_idx += 1
                    except Exception as e:
                        axes[row, col].text(0.5, 0.5, f'Error loading\n{ann_files[i].name}',
                                          ha='center', va='center', fontsize=8)
                        axes[row, col].axis('off')
                        sample_idx += 1
    
    # Hide remaining empty subplots
    while sample_idx < rows * cols:
        row = sample_idx // cols
        col = sample_idx % cols
        axes[row, col].axis('off')
        sample_idx += 1
    
    plt.tight_layout()
    return fig


# ============================================================================
# TRAINING AND VALIDATION VISUALIZATION CLASSES
# ============================================================================


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
