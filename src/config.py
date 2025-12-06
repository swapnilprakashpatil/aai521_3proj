"""
Configuration file for Flood Impact Assessment Project
Contains paths, constants, and utility functions
"""

from pathlib import Path
import random
import numpy as np
import torch

# ============= Project Paths =============
PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE_PATH = PROJECT_ROOT
BASE_PATH = PROJECT_ROOT / 'dataset/raw/'

# Training data paths
TRAIN_PATH = BASE_PATH / 'train'
GERMANY_TRAIN = TRAIN_PATH / 'Germany_Training_Public'
LOUISIANA_EAST_TRAIN = TRAIN_PATH / 'Louisiana-East_Training_Public'

# Test data paths
TEST_PATH = BASE_PATH / 'test'
LOUISIANA_WEST_TEST = TEST_PATH

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
SAMPLE_OUTPUTS_DIR = OUTPUTS_DIR / 'samples'
RESULTS_DIR = OUTPUTS_DIR / 'results'
MODELS_DIR = OUTPUTS_DIR / 'models'
LOGS_DIR = OUTPUTS_DIR / 'logs'

# Processed data directories
PROCESSED_DIR = PROJECT_ROOT / 'dataset' / 'processed'
PROCESSED_TRAIN_DIR = PROCESSED_DIR / 'train'
PROCESSED_VAL_DIR = PROCESSED_DIR / 'val'
PROCESSED_TEST_DIR = PROCESSED_DIR / 'test'

# ============= Class Definitions =============
# Based on xBD (xView2) challenge format
CLASS_NAMES = {
    0: 'background',
    1: 'no-damage',
    2: 'minor-damage',
    3: 'major-damage',
    4: 'destroyed',
    5: 'un-classified',
    6: 'non-flooded-road'
}

# Alternative interpretation for flood detection
FLOOD_CLASS_NAMES = {
    0: 'background',
    1: 'non-flooded-buildings',
    2: 'flooded-buildings',
    3: 'water',
    4: 'flooded-water',
    5: 'flooded-roads',
    6: 'non-flooded-roads'
}

# Color map for visualization (RGB)
CLASS_COLORS = {
    0: [0, 0, 0],          # black - background
    1: [0, 255, 0],        # green - no damage / non-flooded
    2: [255, 255, 0],      # yellow - minor damage / partially flooded
    3: [255, 165, 0],      # orange - major damage / heavily flooded
    4: [255, 0, 0],        # red - destroyed / completely flooded
    5: [128, 128, 128],    # gray - unclassified
    6: [200, 200, 200]     # light gray - non-flooded roads
}

# ============= Model Configuration =============
IMAGE_SIZE = 1024  # Default tile size
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_CLASSES = 7  # Classes 0-6

# ============= Preprocessing Configuration =============
# Patch extraction
PATCH_SIZE = 512  # Size for patch extraction (optimal for CV segmentation)
PATCH_OVERLAP = 128  # Overlap between patches to ensure boundary coverage
MIN_FLOOD_PIXELS = 100  # AGGRESSIVE: Lowered to 100 (~0.04% of 512x512 patch) - capture even small flood areas

# Normalization settings
# Based on typical satellite imagery ranges
NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean (will be updated with dataset stats)
NORM_STD = [0.229, 0.224, 0.225]   # ImageNet std (will be updated with dataset stats)

# Image enhancement
APPLY_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Advanced preprocessing (cloud removal, deblurring, geometric correction)
# DISABLED: These steps can hurt performance by removing flood pixels and adding artifacts
APPLY_ADVANCED_PREPROCESSING = False  # DISABLED - may remove flood data and add artifacts
REMOVE_CLOUDS = False  # May incorrectly remove bright flood water (sun reflection)
APPLY_DEBLUR = False   # Unnecessary for satellite data, may introduce noise
CORRECT_GEOMETRY = False  # Unnecessary - satellite images already georeferenced

# Class balancing
# Based on EDA: ~20-22% flooded, ~78-80% non-flooded
CLASS_WEIGHTS = {
    0: 1.0,   # background
    1: 1.2,   # non-flooded buildings
    2: 3.5,   # flooded buildings (heavily weighted due to imbalance)
    3: 2.0,   # water
    4: 4.0,   # flooded-water (highest priority)
    5: 3.0,   # flooded roads (heavily weighted)
    6: 1.3    # non-flooded roads
}

# Data augmentation probabilities
AUG_HORIZONTAL_FLIP_PROB = 0.5
AUG_VERTICAL_FLIP_PROB = 0.5
AUG_ROTATE_PROB = 0.5
AUG_BRIGHTNESS_CONTRAST_PROB = 0.3
AUG_GAUSSIAN_NOISE_PROB = 0.2
AUG_BLUR_PROB = 0.15

# Quality control thresholds
MIN_VALID_PIXELS_RATIO = 0.5  # Minimum ratio of valid (non-zero) pixels
MAX_CLOUD_COVERAGE = 0.3      # Maximum acceptable cloud coverage ratio

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # For internal validation; actual test set is Louisiana-West

# ============= Training Configuration =============
# Learning rate and optimization
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP_VAL = 1.0

# Loss configuration
LOSS_TYPE = 'combined'  # 'ce', 'dice', 'focal', 'combined'
LOSS_CE_WEIGHT = 1.0
LOSS_DICE_WEIGHT = 1.0
LOSS_FOCAL_WEIGHT = 0.5
FOCAL_GAMMA = 2.0

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = 'cosine'  # 'step', 'cosine', 'plateau'
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.1
SCHEDULER_T_MAX = 100
SCHEDULER_MIN_LR = 1e-6

# Mixed precision training
USE_AMP = True  # Automatic Mixed Precision

# Model-specific configurations
MODEL_CONFIGS = {
    'unet++': {
        'encoder_name': 'resnet34',
        'encoder_weights': 'imagenet',
        'decoder_channels': [256, 128, 64, 32, 16],
        'decoder_attention_type': None
    },
    'deeplabv3+': {
        'encoder_name': 'resnet50',
        'encoder_weights': 'imagenet',
        'encoder_output_stride': 16,
        'decoder_channels': 256,
        'decoder_atrous_rates': (12, 24, 36)
    },
    'segformer': {
        'segformer_model_name': 'nvidia/segformer-b0-finetuned-ade-512-512',
        'pretrained': True
    }
}

# ============= Utility Functions =============
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_size(bytes_size):
    """Format byte size to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


# ============= File Extensions =============
IMAGE_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
ANNOTATION_EXTENSIONS = ['.geojson', '.json']

# ============= Metadata =============
PROJECT_NAME = "Flood Impact Assessment - Multi-Temporal Satellite Segmentation"
VERSION = "1.0.0"
DESCRIPTION = """
This project implements deep learning models for flood impact assessment using 
multi-temporal satellite imagery. The models perform semantic segmentation to 
identify and classify flood-affected areas, buildings, and infrastructure.
"""

if __name__ == "__main__":
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path: {BASE_PATH}")
    print(f"Device: {get_device()}")
