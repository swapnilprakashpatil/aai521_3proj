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
MIN_FLOOD_PIXELS = 50  # ULTRA-AGGRESSIVE: Lowered to 50 (~0.02% of 512x512 patch) - capture all flood areas

# Oversampling configuration for class balance
OVERSAMPLE_TARGET_RATIO = 0.5  # Target 50% flood patches (was 40%)
OVERSAMPLE_MAX_DUPLICATES = 30  # Maximum duplications per flood patch (was 20)
USE_WEIGHTED_SAMPLING = True  # Enable weighted random sampling during training

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

# Class balancing - OPTIMIZED FOR 60%+ IoU TARGET
# With aggressive oversampling targeting 50% flood pixels, we need higher weights for minority classes
# Formula: weight = (1 / frequency) * adjustment_factor
CLASS_WEIGHTS = {
    0: 0.3,   # background (abundant, heavily reduce weight to prevent bias)
    1: 2.5,   # no-damage (buildings - moderate importance)
    2: 5.0,   # minor-damage (critical for flood detection - INCREASED)
    3: 8.0,   # major-damage (very rare, highest weight - INCREASED)
    4: 10.0,  # destroyed (extremely rare, maximum weight - INCREASED)
    5: 6.0,   # un-classified (rare, high weight - INCREASED)
    6: 2.0    # non-flooded-road (moderate frequency)
}

# Focal loss parameters for hard example mining
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25  # Balancing factor
FOCAL_GAMMA = 2.0   # Focusing parameter (higher = focus more on hard examples)

# Data augmentation probabilities
# INCREASED for flood-positive samples to improve generalization
AUG_HORIZONTAL_FLIP_PROB = 0.7  # Increased from 0.5
AUG_VERTICAL_FLIP_PROB = 0.7    # Increased from 0.5
AUG_ROTATE_PROB = 0.6           # Increased from 0.5
AUG_BRIGHTNESS_CONTRAST_PROB = 0.5  # Increased from 0.3
AUG_GAUSSIAN_NOISE_PROB = 0.3   # Increased from 0.2
AUG_BLUR_PROB = 0.2             # Increased from 0.15
AUG_ELASTIC_TRANSFORM_PROB = 0.3  # NEW: Elastic deformation for flood patterns
AUG_GRID_DISTORTION_PROB = 0.2    # NEW: Grid distortion for geometric variation

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
