"""
Preprocessing utilities for satellite imagery
Includes normalization, enhancement, quality checks, and patch extraction
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import warnings
from skimage import morphology, filters, exposure, restoration
from skimage.segmentation import chan_vese
from scipy import ndimage


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
    
    def remove_cloud_cover(self, image: np.ndarray, mask_threshold: float = 0.70, aggressive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced cloud detection and removal using scikit-image
        Uses multi-stage detection: brightness, texture analysis, and morphological operations
        
        Args:
            image: Input image (H, W, C), normalized to 0-1
            mask_threshold: Threshold for detecting bright cloud pixels
            aggressive: Use more sensitive detection thresholds
            
        Returns:
            Tuple of (cloud-removed image, cloud mask)
        """
        # Convert to uint8 and extract channels
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Multi-stage cloud detection
        # Stage 1: Brightness-based detection (clouds are bright)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        if aggressive:
            bright_mask = gray > 160  # Lower threshold for more detection
        else:
            bright_mask = gray > int(mask_threshold * 255)
        
        # Stage 2: Blue channel analysis (clouds are blue-white)
        blue_excess = img_uint8[:, :, 2].astype(float) - (img_uint8[:, :, 0].astype(float) + img_uint8[:, :, 1].astype(float)) / 2
        blue_mask = blue_excess > 10
        
        # Stage 3: Texture analysis (clouds have low texture variance)
        from skimage.filters import rank
        from skimage.morphology import disk
        try:
            # Calculate local entropy
            selem = disk(7)
            entropy_img = rank.entropy(gray, selem)
            # Clouds have low entropy (uniform texture)
            texture_mask = entropy_img < np.percentile(entropy_img, 25)
        except:
            # Fallback if rank filters fail
            texture_mask = np.zeros_like(bright_mask)
        
        # Stage 4: Saturation analysis (clouds have low saturation)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        low_sat_mask = hsv[:, :, 1] < 40
        
        # Stage 5: Value analysis (clouds are bright in HSV)
        high_value_mask = hsv[:, :, 2] > 200
        
        # Combine all detection stages
        cloud_mask_combined = (bright_mask & blue_mask) | (bright_mask & low_sat_mask & texture_mask) | (high_value_mask & low_sat_mask)
        cloud_mask_combined = cloud_mask_combined.astype(np.uint8) * 255
        
        # Refine mask using advanced morphological operations
        # Remove small noise
        cloud_mask_refined = morphology.remove_small_objects(
            cloud_mask_combined > 0, min_size=100, connectivity=2
        )
        cloud_mask_refined = morphology.remove_small_holes(
            cloud_mask_refined, area_threshold=200
        )
        
        # Dilate to ensure full cloud coverage
        selem = morphology.disk(5 if aggressive else 3)
        cloud_mask_refined = morphology.dilation(cloud_mask_refined, selem)
        cloud_mask_final = (cloud_mask_refined * 255).astype(np.uint8)
        
        # Advanced inpainting using multiple methods
        if np.sum(cloud_mask_final > 0) > 200:  # Only if significant clouds detected
            # Method 1: Navier-Stokes based inpainting (better for large regions)
            inpainted_ns = cv2.inpaint(img_uint8, cloud_mask_final, 10, cv2.INPAINT_NS)
            
            # Method 2: Fast marching method
            inpainted_telea = cv2.inpaint(img_uint8, cloud_mask_final, 7, cv2.INPAINT_TELEA)
            
            # Blend both methods for best results
            result_uint8 = cv2.addWeighted(inpainted_ns, 0.6, inpainted_telea, 0.4, 0)
            
            # Apply bilateral filter to smooth transitions
            result_uint8 = cv2.bilateralFilter(result_uint8, 7, 75, 75)
        else:
            result_uint8 = img_uint8
        
        # Convert back to float
        result = result_uint8.astype(np.float32) / 255.0
        mask_float = cloud_mask_final.astype(np.float32) / 255.0
        
        return result, mask_float
    
    def deblur_image(self, image: np.ndarray, kernel_size: int = 11, strength: str = 'high') -> np.ndarray:
        """
        Advanced deblurring using Wiener deconvolution, Richardson-Lucy, and unsharp masking
        
        Args:
            image: Input image (H, W, C), normalized to 0-1
            kernel_size: Size of the deconvolution kernel
            strength: Deblurring strength ('medium' or 'high')
            
        Returns:
            Deblurred image
        """
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Create motion blur PSF
        psf = np.zeros((kernel_size, kernel_size))
        psf[kernel_size // 2, :] = 1.0
        psf = psf / psf.sum()
        
        try:
            # Method 1: Wiener deconvolution
            deconvolved = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                deconv_channel = restoration.wiener(channel, psf, balance=0.05)
                deconvolved[:, :, i] = np.clip(deconv_channel, 0, 1)
            deconv_uint8 = (deconvolved * 255).astype(np.uint8)
        except Exception as e:
            deconv_uint8 = img_uint8
        
        try:
            # Method 2: Richardson-Lucy deconvolution
            rl_deconvolved = np.zeros_like(image)
            for i in range(3):
                rl_channel = restoration.richardson_lucy(image[:, :, i], psf, num_iter=15)
                rl_deconvolved[:, :, i] = np.clip(rl_channel, 0, 1)
            rl_uint8 = (rl_deconvolved * 255).astype(np.uint8)
        except Exception as e:
            rl_uint8 = img_uint8
        
        # Method 3: Enhanced unsharp masking
        gaussian_blur = cv2.GaussianBlur(img_uint8, (9, 9), 2.0)
        if strength == 'high':
            unsharp = cv2.addWeighted(img_uint8, 2.5, gaussian_blur, -1.5, 0)
        else:
            unsharp = cv2.addWeighted(img_uint8, 2.0, gaussian_blur, -1.0, 0)
        
        # Method 4: Edge enhancement with Sobel
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edge_enhanced = cv2.addWeighted(img_uint8, 1.0, edges_colored, 0.4, 0)
        
        # Method 5: Adaptive CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        clahe_enhanced = np.zeros_like(img_uint8)
        for i in range(3):
            clahe_enhanced[:, :, i] = clahe.apply(img_uint8[:, :, i])
        
        # Blend all methods: Wiener (25%) + RL (20%) + Unsharp (30%) + Edge (15%) + CLAHE (10%)
        try:
            result = cv2.addWeighted(deconv_uint8, 0.25, rl_uint8, 0.20, 0)
            result = cv2.addWeighted(result, 1.0, unsharp, 0.30, 0)
            result = cv2.addWeighted(result, 1.0, edge_enhanced, 0.15, 0)
            result = cv2.addWeighted(result, 0.9, clahe_enhanced, 0.10, 0)
        except:
            # Fallback to simpler blend
            result = cv2.addWeighted(unsharp, 0.6, edge_enhanced, 0.4, 0)
        
        # Final sharpening with Laplacian
        gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray_result, cv2.CV_64F, ksize=3)
        laplacian_colored = cv2.cvtColor(np.abs(laplacian).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(result, 1.0, laplacian_colored, 0.20, 0)
        
        # Convert back to float
        return np.clip(result.astype(np.float32) / 255.0, 0, 1)
    
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
                    # Count damage/flood classes: 2 (minor-damage), 3 (major-damage), 4 (destroyed)
                    # Classes > 1 indicate some level of damage/flooding (excluding class 5=unclassified, 6=non-flooded-road)
                    flood_pixels = np.sum((mask_patch == 2) | (mask_patch == 3) | (mask_patch == 4))
                    patch_dict['flood_pixels'] = flood_pixels
                    patch_dict['is_flood_positive'] = flood_pixels >= self.min_flood_pixels
                    
                    # Calculate class distribution
                    unique, counts = np.unique(mask_patch, return_counts=True)
                    patch_dict['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
                    
                patches.append(patch_dict)
                
        # Oversample flood-positive patches
        if oversample_flood and mask is not None:
            flood_patches = [p for p in patches if p.get('is_flood_positive', False)]
            
            # Duplicate flood patches to balance classes
            # Based on EDA: ~20% flooded, so we duplicate to get closer to 50/50
            if len(flood_patches) > 0:
                n_duplicates = min(len(patches) // len(flood_patches) - 1, 3)  # Max 3x duplication
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
