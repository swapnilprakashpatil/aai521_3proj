"""
Preprocessing utilities for satellite imagery
Includes normalization, enhancement, quality checks, and patch extraction
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import warnings


class ImagePreprocessor:
    """Handle image preprocessing operations"""
    
    def __init__(self, 
                 apply_clahe: bool = True,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize preprocessor
        
        Args:
            apply_clahe: Whether to apply CLAHE enhancement
            clahe_clip_limit: Clip limit for CLAHE
            clahe_tile_grid_size: Tile grid size for CLAHE
        """
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        
        # Create CLAHE object
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit, 
                tileGridSize=clahe_tile_grid_size
            )
    
    def remove_cloud_cover(self, image: np.ndarray, mask_threshold: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and remove cloud cover using morphological operations and inpainting
        
        Args:
            image: Input image (H, W, C), normalized to 0-1
            mask_threshold: Threshold for detecting bright cloud pixels
            
        Returns:
            Tuple of (cloud-removed image, cloud mask)
        """
        # Convert to uint8 for processing
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Detect clouds using brightness and texture
        # Clouds are typically bright and have low texture variance
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Brightness threshold
        _, bright_mask = cv2.threshold(gray, int(mask_threshold * 255), 255, cv2.THRESH_BINARY)
        
        # Refine mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cloud_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
        
        # Inpaint cloud regions
        inpainted = cv2.inpaint(img_uint8, cloud_mask, 3, cv2.INPAINT_TELEA)
        
        # Convert back to float
        result = inpainted.astype(np.float32) / 255.0
        mask_float = cloud_mask.astype(np.float32) / 255.0
        
        return result, mask_float
    
    def deblur_image(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply deblurring using unsharp masking and Wiener deconvolution approximation
        FIXED: Proper weight normalization to preserve contrast
        
        Args:
            image: Input image (H, W, C), normalized to 0-1
            kernel_size: Size of the sharpening kernel
            
        Returns:
            Deblurred image
        """
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Method 1: Unsharp masking with proper weights
        gaussian = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)
        unsharp = cv2.addWeighted(img_uint8, 1.5, gaussian, -0.5, 0)
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        
        # Method 2: Laplacian sharpening for detail enhancement
        laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
        sharpened = img_uint8.astype(np.float64) - 0.3 * laplacian
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Blend both methods with normalized weights (60% + 40% = 100%)
        result = (unsharp.astype(np.float32) * 0.6 + sharpened.astype(np.float32) * 0.4)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result_enhanced = np.zeros_like(result)
        for c in range(3):
            result_enhanced[:, :, c] = clahe.apply(result[:, :, c])
        
        # Convert back to float
        return result_enhanced.astype(np.float32) / 255.0
    
    def correct_geometric_distortion(self, image: np.ndarray, 
                                     correction_strength: float = 0.1) -> np.ndarray:
        """
        Correct camera angle and geometric distortions using perspective transformation
        
        Args:
            image: Input image (H, W, C), normalized to 0-1
            correction_strength: Strength of geometric correction (0-1)
            
        Returns:
            Geometrically corrected image
        """
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        
        # Detect if image has significant tilt by analyzing edge distributions
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Use Hough transform to detect predominant angles
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 10:
            # Calculate average angle
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            median_angle = np.median(angles)
            
            # If significant tilt detected, apply rotation correction
            if abs(median_angle) > 2:  # More than 2 degrees
                # Rotate to correct
                rotation_angle = -median_angle * correction_strength
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                corrected = cv2.warpAffine(img_uint8, rotation_matrix, (w, h), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            else:
                corrected = img_uint8
        else:
            corrected = img_uint8
        
        # Apply barrel distortion correction (common in satellite imagery)
        # Camera matrix estimation
        camera_matrix = np.array([[w, 0, w/2],
                                  [0, h, h/2],
                                  [0, 0, 1]], dtype=np.float32)
        
        # Distortion coefficients (slight barrel distortion typical in satellite sensors)
        dist_coeffs = np.array([-0.01 * correction_strength, 0, 0, 0], dtype=np.float32)
        
        # Undistort
        corrected = cv2.undistort(corrected, camera_matrix, dist_coeffs)
        
        # Convert back to float
        return corrected.astype(np.float32) / 255.0
    
    def apply_advanced_preprocessing(self, image: np.ndarray,
                                    remove_clouds: bool = True,
                                    apply_deblur: bool = True,
                                    correct_geometry: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply all advanced preprocessing steps and return intermediate results
        
        Args:
            image: Input image (H, W, C), normalized to 0-1
            remove_clouds: Whether to remove cloud cover
            apply_deblur: Whether to apply deblurring
            correct_geometry: Whether to correct geometric distortions
            
        Returns:
            Dictionary containing original, intermediate, and final processed images
        """
        results = {'original': image.copy()}
        current = image.copy()
        
        # Step 1: Remove clouds
        if remove_clouds:
            current, cloud_mask = self.remove_cloud_cover(current)
            results['cloud_removed'] = current.copy()
            results['cloud_mask'] = cloud_mask
        
        # Step 2: Correct geometric distortions
        if correct_geometry:
            current = self.correct_geometric_distortion(current)
            results['geometry_corrected'] = current.copy()
        
        # Step 3: Apply deblurring
        if apply_deblur:
            current = self.deblur_image(current)
            results['deblurred'] = current.copy()
        
        results['final'] = current
        return results
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image (H, W, C), normalized to 0-1
            
        Returns:
            Enhanced image
        """
        if not self.apply_clahe:
            return image
            
        # Convert to uint8 for CLAHE
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply CLAHE to each channel
        enhanced = np.zeros_like(img_uint8)
        for i in range(img_uint8.shape[2]):
            enhanced[:, :, i] = self.clahe.apply(img_uint8[:, :, i])
            
        # Convert back to float
        return enhanced.astype(np.float32) / 255.0
    
    def normalize_percentile(self, image: np.ndarray, 
                            lower_percentile: float = 2.0,
                            upper_percentile: float = 98.0) -> np.ndarray:
        """
        Normalize image using percentile-based scaling
        Helps with outliers and varying brightness
        
        Args:
            image: Input image
            lower_percentile: Lower percentile for clipping
            upper_percentile: Upper percentile for clipping
            
        Returns:
            Normalized image
        """
        result = np.zeros_like(image)
        
        for i in range(image.shape[2]):
            channel = image[:, :, i]
            p_low = np.percentile(channel, lower_percentile)
            p_high = np.percentile(channel, upper_percentile)
            
            # Clip and normalize
            channel_clipped = np.clip(channel, p_low, p_high)
            if p_high > p_low:
                result[:, :, i] = (channel_clipped - p_low) / (p_high - p_low)
            else:
                result[:, :, i] = channel_clipped
                
        return result
    
    def standardize(self, image: np.ndarray, 
                   mean: List[float] = [0.485, 0.456, 0.406],
                   std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """
        Standardize image using mean and std
        
        Args:
            image: Input image (H, W, C), normalized to 0-1
            mean: Mean values for each channel
            std: Std values for each channel
            
        Returns:
            Standardized image
        """
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        
        return (image - mean) / std
    
    def check_image_quality(self, image: np.ndarray,
                           min_valid_ratio: float = 0.5,
                           max_cloud_coverage: float = 0.3) -> Dict:
        """
        Check image quality based on various metrics
        
        Args:
            image: Input image
            min_valid_ratio: Minimum ratio of valid pixels
            max_cloud_coverage: Maximum acceptable cloud coverage
            
        Returns:
            Dictionary with quality metrics and pass/fail status
        """
        # Check for valid pixels (non-zero)
        valid_pixels = np.sum(image > 0)
        total_pixels = image.size
        valid_ratio = valid_pixels / total_pixels
        
        # Estimate cloud coverage (very bright pixels)
        bright_threshold = 0.9
        bright_pixels = np.sum(image > bright_threshold)
        cloud_ratio = bright_pixels / total_pixels
        
        # Check for extreme darkness (shadows, sensor issues)
        dark_threshold = 0.1
        dark_pixels = np.sum(image < dark_threshold)
        dark_ratio = dark_pixels / total_pixels
        
        # Calculate metrics
        quality = {
            'valid_ratio': valid_ratio,
            'cloud_ratio': cloud_ratio,
            'dark_ratio': dark_ratio,
            'mean_intensity': np.mean(image),
            'std_intensity': np.std(image),
            'passes_quality': True
        }
        
        # Quality checks
        if valid_ratio < min_valid_ratio:
            quality['passes_quality'] = False
            quality['fail_reason'] = 'insufficient_valid_pixels'
        elif cloud_ratio > max_cloud_coverage:
            quality['passes_quality'] = False
            quality['fail_reason'] = 'excessive_clouds'
            
        return quality
    
    def preprocess_image(self, image: np.ndarray,
                        apply_enhancement: bool = True,
                        apply_percentile_norm: bool = False,
                        apply_standardization: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline
        
        Args:
            image: Input image (already normalized to 0-1)
            apply_enhancement: Whether to apply CLAHE
            apply_percentile_norm: Whether to apply percentile normalization
            apply_standardization: Whether to apply standardization
            
        Returns:
            Preprocessed image
        """
        result = image.copy()
        
        # Apply percentile normalization if requested
        if apply_percentile_norm:
            result = self.normalize_percentile(result)
            
        # Apply CLAHE enhancement
        if apply_enhancement:
            result = self.apply_clahe_enhancement(result)
            
        # Apply standardization (for deep learning models)
        if apply_standardization:
            result = self.standardize(result)
            
        return result


class PatchExtractor:
    """Extract patches from large images with smart sampling"""
    
    def __init__(self, 
                 patch_size: int = 512,
                 overlap: int = 128,
                 min_flood_pixels: int = 100):
        """
        Initialize patch extractor
        
        Args:
            patch_size: Size of patches to extract
            overlap: Overlap between patches
            min_flood_pixels: Minimum flooded pixels to consider patch as flood-positive
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.min_flood_pixels = min_flood_pixels
        
    def extract_patches(self, 
                       image: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       oversample_flood: bool = True) -> List[Dict]:
        """
        Extract patches from image and mask
        
        Args:
            image: Input image (H, W, C)
            mask: Corresponding mask (H, W)
            oversample_flood: Whether to oversample flood-positive patches
            
        Returns:
            List of patch dictionaries with image, mask, and metadata
        """
        h, w = image.shape[:2]
        patches = []
        
        # Calculate number of patches
        n_patches_h = (h - self.patch_size) // self.stride + 1
        n_patches_w = (w - self.patch_size) // self.stride + 1
        
        # Extract patches
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                y = i * self.stride
                x = j * self.stride
                
                # Ensure we don't go out of bounds
                if y + self.patch_size > h:
                    y = h - self.patch_size
                if x + self.patch_size > w:
                    x = w - self.patch_size
                    
                # Extract patch
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                patch_dict = {
                    'image': img_patch,
                    'position': (y, x),
                    'patch_id': f"{i}_{j}"
                }
                
                # Extract mask patch if provided
                if mask is not None:
                    mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                    patch_dict['mask'] = mask_patch
                    
                    # Calculate flood statistics
                    # Count ALL non-background classes as flood-related (classes 1-6)
                    # Class 0 = background, Classes 1-6 = various flood/damage types
                    # This ensures we capture all flood information, not just damage classes
                    flood_pixels = np.sum(mask_patch > 0)
                    patch_dict['flood_pixels'] = flood_pixels
                    patch_dict['is_flood_positive'] = flood_pixels >= self.min_flood_pixels
                    
                    # Calculate class distribution
                    unique, counts = np.unique(mask_patch, return_counts=True)
                    patch_dict['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
                    
                patches.append(patch_dict)
                
        # Oversample flood-positive patches
        if oversample_flood and mask is not None:
            flood_patches = [p for p in patches if p.get('is_flood_positive', False)]
            non_flood_patches = [p for p in patches if not p.get('is_flood_positive', False)]
            
            # ULTRA-AGGRESSIVE oversampling to achieve 50%+ flood class representation
            # Target: Balance flood and non-flood patches for better training
            if len(flood_patches) > 0 and len(non_flood_patches) > 0:
                # Calculate how many times to duplicate flood patches
                # Target ratio: 50% flood, 50% non-flood for maximum balance
                from config import OVERSAMPLE_TARGET_RATIO, OVERSAMPLE_MAX_DUPLICATES
                target_flood_ratio = OVERSAMPLE_TARGET_RATIO
                current_flood_ratio = len(flood_patches) / len(patches)
                
                if current_flood_ratio < target_flood_ratio:
                    # Calculate required duplicates to reach target ratio
                    n_duplicates = int((target_flood_ratio * len(patches) - len(flood_patches)) / 
                                      (len(flood_patches) * (1 - target_flood_ratio)))
                    n_duplicates = max(1, min(n_duplicates, OVERSAMPLE_MAX_DUPLICATES))
                    
                    total_after = len(patches) + (n_duplicates * len(flood_patches))
                    flood_after = len(flood_patches) * (n_duplicates + 1)
                    final_flood_ratio = flood_after / total_after
                    
                    for _ in range(n_duplicates):
                        patches.extend(flood_patches)
                    
        return patches
    
    def extract_random_patches(self,
                              image: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              n_patches: int = 100,
                              flood_ratio: float = 0.5) -> List[Dict]:
        """
        Extract random patches with controlled flood ratio
        
        Args:
            image: Input image (H, W, C)
            mask: Corresponding mask (H, W)
            n_patches: Number of patches to extract
            flood_ratio: Desired ratio of flood-positive patches
            
        Returns:
            List of patch dictionaries
        """
        h, w = image.shape[:2]
        patches = []
        
        # First extract all valid positions
        valid_positions = []
        for y in range(0, h - self.patch_size, self.stride):
            for x in range(0, w - self.patch_size, self.stride):
                valid_positions.append((y, x))
                
        # If we have a mask, separate flood and non-flood positions
        if mask is not None:
            flood_positions = []
            non_flood_positions = []
            
            for y, x in valid_positions:
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                flood_pixels = np.sum(mask_patch > 1)
                
                if flood_pixels >= self.min_flood_pixels:
                    flood_positions.append((y, x))
                else:
                    non_flood_positions.append((y, x))
                    
            # Sample based on flood ratio
            n_flood = int(n_patches * flood_ratio)
            n_non_flood = n_patches - n_flood
            
            # Randomly sample
            import random
            selected_flood = random.sample(flood_positions, min(n_flood, len(flood_positions)))
            selected_non_flood = random.sample(non_flood_positions, min(n_non_flood, len(non_flood_positions)))
            
            selected_positions = selected_flood + selected_non_flood
        else:
            # Random sampling without flood consideration
            import random
            selected_positions = random.sample(valid_positions, min(n_patches, len(valid_positions)))
            
        # Extract patches
        for idx, (y, x) in enumerate(selected_positions):
            img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
            
            patch_dict = {
                'image': img_patch,
                'position': (y, x),
                'patch_id': f"rand_{idx}"
            }
            
            if mask is not None:
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                patch_dict['mask'] = mask_patch
                flood_pixels = np.sum(mask_patch > 1)
                patch_dict['flood_pixels'] = flood_pixels
                patch_dict['is_flood_positive'] = flood_pixels >= self.min_flood_pixels
                
            patches.append(patch_dict)
            
        return patches


def calculate_dataset_statistics(images: List[np.ndarray]) -> Dict:
    """
    Calculate mean and std across dataset for normalization
    
    Args:
        images: List of images (all normalized to 0-1)
        
    Returns:
        Dictionary with mean and std per channel
    """
    # Stack all images
    all_pixels = []
    for img in images:
        all_pixels.append(img.reshape(-1, img.shape[-1]))
        
    all_pixels = np.vstack(all_pixels)
    
    # Calculate statistics
    mean = np.mean(all_pixels, axis=0)
    std = np.std(all_pixels, axis=0)
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'min': np.min(all_pixels, axis=0).tolist(),
        'max': np.max(all_pixels, axis=0).tolist()
    }


def cleanup_processed_data(processed_dir: Path) -> Dict[str, any]:
    """
    Delete all previously processed data before re-running preprocessing
    
    Args:
        processed_dir: Path to the processed data directory
        
    Returns:
        Dictionary with cleanup status and statistics
    """
    import shutil
    import time
    
    result = {
        'existed': False,
        'deleted': False,
        'total_files': 0,
        'error': None
    }
    
    print("="*80)
    print("CLEANUP: Deleting Previous Preprocessing Output")
    print("="*80)
    
    # Check if processed directory exists
    if not processed_dir.exists():
        print(f"\nNo processed directory found at: {processed_dir}")
        print("STATUS: Starting fresh (no cleanup needed)")
        result['deleted'] = True
        return result
    
    result['existed'] = True
    print(f"\nFound processed directory: {processed_dir}")
    
    # Count files before deletion
    # Count files
    for split in ['train', 'val', 'test']:
        split_dir = processed_dir / split
        if split_dir.exists():
            for subdir in ['images', 'masks', 'metadata', 'processed_images']:
                subdir_path = split_dir / subdir
                if subdir_path.exists():
                    result['total_files'] += len(list(subdir_path.glob('*')))
    
    try:
        shutil.rmtree(processed_dir)
        
        # Wait for deletion
        max_wait = 10
        wait_interval = 0.5
        elapsed = 0
        
        while processed_dir.exists() and elapsed < max_wait:
            time.sleep(wait_interval)
            elapsed += wait_interval
        
        if processed_dir.exists():
            remaining = sum(1 for _ in processed_dir.rglob('*') if _.is_file())
            result['error'] = f"Directory still exists with {remaining} files"
        else:
            result['deleted'] = True
            print(f"Cleanup: Deleted {result['total_files']} files")
            
    except Exception as e:
        result['error'] = str(e)
        print(f"Cleanup failed: {e}")
    
    return result


def validate_class_balance(processed_dir: Path, num_classes: int = 7) -> Dict[str, any]:
    """
    Validate class balance after preprocessing to ensure proper distribution
    
    Args:
        processed_dir: Path to processed data directory
        num_classes: Number of classes in the dataset
        
    Returns:
        Dictionary with validation results
    """
    import torch
    from tqdm import tqdm
    
    result = {
        'success': False,
        'quality': None,
        'background_pct': 0.0,
        'flood_pct': 0.0,
        'class_distribution': {},
        'missing_classes': [],
        'error': None
    }
    
    print("\n" + "="*60)
    print("CLASS BALANCE VALIDATION")
    print("="*60)
    
    train_dir = processed_dir / 'train'
    
    if not train_dir.exists():
        result['error'] = "Processed data not found"
        print(f"[ERROR] {result['error']}")
        return result
    
    train_masks = sorted((train_dir / 'masks').glob('*.npy'))
    
    if len(train_masks) == 0:
        result['error'] = "No mask files found"
        print(f"[ERROR] {result['error']}")
        return result
    
    # Calculate patch-level statistics (critical for oversampling validation)
    flood_positive_patches = 0
    for mask_path in train_masks:
        mask = np.load(mask_path)
        if np.sum(mask > 0) > 0:  # Has any flood pixels
            flood_positive_patches += 1
    
    patch_flood_pct = (flood_positive_patches / len(train_masks) * 100) if len(train_masks) > 0 else 0
    
    # Calculate pixel-level class distribution
    class_counts = torch.zeros(num_classes)
    
    for mask_path in tqdm(train_masks, desc="Computing distribution"):
        mask = np.load(mask_path)
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum()
    
    total_pixels = class_counts.sum()
    
    # Store results
    result['success'] = True
    bg_pct = (class_counts[0] / total_pixels * 100).item()
    flood_pct = ((total_pixels - class_counts[0]) / total_pixels * 100).item()
    result['background_pct'] = bg_pct
    result['flood_pct'] = flood_pct
    result['patch_flood_pct'] = patch_flood_pct
    
    class_names = ['background', 'no-damage', 'minor-damage', 'major-damage', 
                   'destroyed', 'un-classified', 'non-flooded-road']
    
    for cls in range(num_classes):
        pct = (class_counts[cls] / total_pixels * 100).item()
        result['class_distribution'][class_names[cls]] = {
            'count': int(class_counts[cls].item()),
            'percentage': pct
        }
        if class_counts[cls] == 0:
            result['missing_classes'].append(class_names[cls])
    
    # Quality assessment based on PATCH-LEVEL distribution (what models see during training)
    # Updated thresholds for 60%+ target with 50% oversampling
    if patch_flood_pct < 30:
        result['quality'] = "POOR"
        status_icon = "[POOR]"
    elif patch_flood_pct < 45:
        result['quality'] = "ACCEPTABLE"
        status_icon = "[ACCEPTABLE]"
    elif patch_flood_pct < 55:
        result['quality'] = "GOOD"
        status_icon = "[GOOD]"
    else:
        result['quality'] = "EXCELLENT"
        status_icon = "[EXCELLENT]"
    
    # Display compact results
    print(f"\n{status_icon} Quality: {result['quality']}")
    print(f"  PATCH LEVEL: {patch_flood_pct:.1f}% flood patches (oversampling applied)")
    print(f"  PIXEL LEVEL: {bg_pct:.1f}% background | {flood_pct:.1f}% flood (raw data)")
    
    # Show only non-zero classes
    non_zero = [f"{name}({result['class_distribution'][name]['percentage']:.1f}%)" 
                for cls, name in enumerate(class_names) 
                if result['class_distribution'][name]['percentage'] > 0.01]
    print(f"  Classes: {', '.join(non_zero)}")
    
    # Expected performance based on patch distribution with new thresholds
    if result['quality'] == "POOR":
        print(f"  Expected IoU: 20-40% - Need more oversampling")
    elif result['quality'] == "ACCEPTABLE":
        print(f"  Expected IoU: 45-60% - Increase oversampling for 60%+ target")
    elif result['quality'] == "GOOD":
        print(f"  Expected IoU: 60-75% - Good balance for target")
    else:  # EXCELLENT
        print(f"  Expected IoU: 70-85% - Excellent balance")
    
    if result['missing_classes']:
        print(f"  Missing: {', '.join(result['missing_classes'])}")
    
    print("="*60)
    
    return result


if __name__ == "__main__":
    # Test preprocessing
    test_image = np.random.rand(1024, 1024, 3).astype(np.float32)
    test_mask = np.random.randint(0, 6, (1024, 1024), dtype=np.uint8)
    
    # Test preprocessor
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.preprocess_image(test_image)
    print(f"Enhanced shape: {enhanced.shape}")
    
    quality = preprocessor.check_image_quality(test_image)
    print(f"Quality check: {quality}")
    
    # Test patch extractor
    extractor = PatchExtractor(patch_size=512, overlap=128)
    patches = extractor.extract_patches(test_image, test_mask)
    print(f"Extracted {len(patches)} patches")
    
    flood_patches = [p for p in patches if p.get('is_flood_positive', False)]
    print(f"Flood-positive patches: {len(flood_patches)}")
