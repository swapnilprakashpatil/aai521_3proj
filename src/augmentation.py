"""
Data augmentation pipeline for satellite imagery
Uses Albumentations for efficient augmentation
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Optional


def get_training_augmentation(
    image_size: int = 512,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    brightness_contrast_prob: float = 0.3,
    noise_prob: float = 0.2,
    blur_prob: float = 0.15
) -> A.Compose:
    """
    Get augmentation pipeline for training
    
    Args:
        image_size: Target image size
        horizontal_flip_prob: Probability of horizontal flip
        vertical_flip_prob: Probability of vertical flip
        rotate_prob: Probability of rotation
        brightness_contrast_prob: Probability of brightness/contrast adjustment
        noise_prob: Probability of adding noise
        blur_prob: Probability of blur
        
    Returns:
        Albumentations composition
    """
    transform = A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=horizontal_flip_prob),
        A.VerticalFlip(p=vertical_flip_prob),
        A.RandomRotate90(p=rotate_prob),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5,
            border_mode=0
        ),
        
        # Optical transformations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=brightness_contrast_prob),
        
        # Weather-like augmentations (domain-specific)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=noise_prob),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=blur_prob),
        
        # Simulate atmospheric effects
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.1
        ),
        
        # Grid dropout for robustness
        A.GridDropout(ratio=0.2, p=0.1),
        
        # Ensure proper size
        A.Resize(image_size, image_size),
    ])
    
    return transform


def get_validation_transform(image_size: int = 512) -> A.Compose:
    """
    Get transformation pipeline for validation (no augmentation)
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition
    """
    transform = A.Compose([
        A.Resize(image_size, image_size),
    ])
    
    return transform


def get_flood_specific_augmentation(image_size: int = 512) -> A.Compose:
    """
    Get flood-specific augmentation that simulates different flood conditions
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition
    """
    transform = A.Compose([
        # Standard augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Simulate varying water levels and conditions
        A.OneOf([
            # Darker water (shadows, deep water)
            A.RandomBrightness(limit=(-0.2, -0.05), p=1.0),
            # Brighter water (reflections, shallow water)
            A.RandomBrightness(limit=(0.05, 0.2), p=1.0),
        ], p=0.4),
        
        # Simulate water surface variations
        A.OneOf([
            # Calm water (blur)
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            # Turbulent water (noise)
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
        ], p=0.3),
        
        # Simulate haze/fog often present during floods
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.25, alpha_coef=0.08, p=0.2),
        
        # Color variations (muddy water, clear water, vegetation)
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
        
        # Ensure proper size
        A.Resize(image_size, image_size),
    ])
    
    return transform


def get_test_time_augmentation() -> list:
    """
    Get list of test-time augmentation transforms
    Used for TTA (Test-Time Augmentation) to improve predictions
    
    Returns:
        List of transform compositions
    """
    tta_transforms = [
        # Original
        A.Compose([]),
        
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0)]),
        
        # Vertical flip
        A.Compose([A.VerticalFlip(p=1.0)]),
        
        # Rotate 90
        A.Compose([A.Rotate(limit=90, p=1.0)]),
        
        # Rotate 180
        A.Compose([A.Rotate(limit=180, p=1.0)]),
        
        # Rotate 270
        A.Compose([A.Rotate(limit=270, p=1.0)]),
        
        # Horizontal + Vertical flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0)
        ]),
    ]
    
    return tta_transforms


class DualImageAugmentation:
    """Augmentation for pre and post event image pairs"""
    
    def __init__(self, 
                 transform: A.Compose,
                 apply_to_both: bool = True):
        """
        Initialize dual image augmentation
        
        Args:
            transform: Albumentations transform
            apply_to_both: If True, apply same transform to both images
        """
        self.transform = transform
        self.apply_to_both = apply_to_both
        
    def __call__(self, 
                 pre_image: np.ndarray,
                 post_image: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> Dict:
        """
        Apply augmentation to pre/post image pair
        
        Args:
            pre_image: Pre-event image
            post_image: Post-event image
            mask: Segmentation mask
            
        Returns:
            Dictionary with augmented images and mask
        """
        if self.apply_to_both:
            # Apply same transform to both images (spatial consistency)
            # This is important for change detection
            
            # Stack images for consistent augmentation
            stacked = np.concatenate([pre_image, post_image], axis=2)
            
            if mask is not None:
                augmented = self.transform(image=stacked, mask=mask)
                aug_stacked = augmented['image']
                aug_mask = augmented['mask']
            else:
                augmented = self.transform(image=stacked)
                aug_stacked = augmented['image']
                aug_mask = None
                
            # Split back
            n_channels = pre_image.shape[2]
            aug_pre = aug_stacked[:, :, :n_channels]
            aug_post = aug_stacked[:, :, n_channels:]
            
        else:
            # Apply different transforms (less common)
            if mask is not None:
                aug_pre = self.transform(image=pre_image, mask=mask)
                aug_post = self.transform(image=post_image, mask=mask)
                aug_mask = aug_pre['mask']  # Use mask from first transform
            else:
                aug_pre = self.transform(image=pre_image)
                aug_post = self.transform(image=post_image)
                aug_mask = None
                
            aug_pre = aug_pre['image']
            aug_post = aug_post['image']
            
        return {
            'pre_image': aug_pre,
            'post_image': aug_post,
            'mask': aug_mask
        }


def simulate_water_level_change(
    mask: np.ndarray,
    dilation_range: tuple = (1, 5)
) -> np.ndarray:
    """
    Simulate water level change by dilating/eroding water regions
    This is a domain-specific augmentation for flood data
    
    Args:
        mask: Segmentation mask
        dilation_range: Range of dilation kernel size
        
    Returns:
        Modified mask
    """
    import cv2
    import random
    
    # Random dilation or erosion
    kernel_size = random.randint(*dilation_range)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply to water classes (3, 4)
    water_mask = np.isin(mask, [3, 4])
    
    if random.random() > 0.5:
        # Dilate (water level rise)
        dilated = cv2.dilate(water_mask.astype(np.uint8), kernel, iterations=1)
        mask[dilated > 0] = 4  # Set to flooded-water
    else:
        # Erode (water level drop)
        eroded = cv2.erode(water_mask.astype(np.uint8), kernel, iterations=1)
        mask[water_mask & (eroded == 0)] = 1  # Change to non-flooded
        
    return mask


if __name__ == "__main__":
    # Test augmentations
    test_pre = np.random.rand(512, 512, 3).astype(np.float32)
    test_post = np.random.rand(512, 512, 3).astype(np.float32)
    test_mask = np.random.randint(0, 6, (512, 512), dtype=np.uint8)
    
    # Test training augmentation
    train_aug = get_training_augmentation()
    augmented = train_aug(image=test_pre, mask=test_mask)
    print(f"Augmented image shape: {augmented['image'].shape}")
    print(f"Augmented mask shape: {augmented['mask'].shape}")
    
    # Test dual augmentation
    dual_aug = DualImageAugmentation(train_aug)
    result = dual_aug(test_pre, test_post, test_mask)
    print(f"Dual augmentation - Pre: {result['pre_image'].shape}, Post: {result['post_image'].shape}")
    
    # Test TTA
    tta_list = get_test_time_augmentation()
    print(f"TTA transforms: {len(tta_list)}")
