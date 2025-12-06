"""
Main preprocessing script to process raw data and export to processed format
Handles train/val/test splits with geo-stratification
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import cv2
from functools import partial

from config import (
    TRAIN_PATH,
    PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR,
    PATCH_SIZE, PATCH_OVERLAP, MIN_FLOOD_PIXELS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    APPLY_CLAHE, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    MIN_VALID_PIXELS_RATIO, MAX_CLOUD_COVERAGE,
    APPLY_ADVANCED_PREPROCESSING, REMOVE_CLOUDS, APPLY_DEBLUR, CORRECT_GEOMETRY
)
from data_loader import DatasetLoader, load_tile_data
from preprocessing import ImagePreprocessor, PatchExtractor, calculate_dataset_statistics

warnings.filterwarnings('ignore')


# ============================================================
# WRAPPER FUNCTION FOR MULTIPROCESSING
# ============================================================
def _process_tile_wrapper(tile_name, region_path, region_name, output_dir,
                         apply_advanced, remove_clouds, apply_deblur, correct_geometry,
                         apply_clahe, clahe_clip_limit, clahe_tile_grid_size,
                         patch_size, patch_overlap, min_flood_pixels,
                         min_valid_ratio, max_cloud_coverage):
    """
    Wrapper function for processing a single tile (must be at module level for ProcessPoolExecutor)
    
    Returns:
        Tuple of (success: bool, patch_metadata: List[Dict], stats: Dict)
    """
    try:
        # Initialize processors (each process needs its own instances)
        image_preprocessor = ImagePreprocessor(
            apply_clahe=apply_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size
        )
        
        patch_extractor = PatchExtractor(
            patch_size=patch_size,
            overlap=patch_overlap,
            min_flood_pixels=min_flood_pixels
        )
        
        # Load tile data
        tile_data = load_tile_data(region_path, tile_name, region_name)
        
        if not tile_data or tile_data['pre_image'] is None:
            return (False, [], {'quality_failed': 0, 'patches': 0, 'flood_positive': 0})
        
        pre_image = tile_data['pre_image']
        post_image = tile_data['post_image']
        mask = tile_data['mask']
        
        # Get original image filenames from metadata
        pre_filename = tile_data.get('pre_metadata', {}).get('filename')
        post_filename = tile_data.get('post_metadata', {}).get('filename')
        
        # Quality check on pre-image
        quality_pre = image_preprocessor.check_image_quality(
            pre_image,
            min_valid_ratio=min_valid_ratio,
            max_cloud_coverage=max_cloud_coverage
        )
        
        if not quality_pre['passes_quality']:
            return (False, [], {'quality_failed': 1, 'patches': 0, 'flood_positive': 0})
        
        # Quality check on post-image if available
        if post_image is not None:
            quality_post = image_preprocessor.check_image_quality(
                post_image,
                min_valid_ratio=min_valid_ratio,
                max_cloud_coverage=max_cloud_coverage
            )
            
            if not quality_post['passes_quality']:
                return (False, [], {'quality_failed': 1, 'patches': 0, 'flood_positive': 0})
        else:
            post_image = pre_image.copy()
            quality_post = quality_pre
        
        # Apply advanced preprocessing
        if apply_advanced:
            pre_advanced = image_preprocessor.apply_advanced_preprocessing(
                pre_image,
                remove_clouds=remove_clouds,
                apply_deblur=apply_deblur,
                correct_geometry=correct_geometry
            )
            
            post_advanced = image_preprocessor.apply_advanced_preprocessing(
                post_image,
                remove_clouds=remove_clouds,
                apply_deblur=apply_deblur,
                correct_geometry=correct_geometry
            )
            
            pre_processed = pre_advanced['final']
            post_processed = post_advanced['final']
        else:
            pre_processed = pre_image.copy()
            post_processed = post_image.copy()
        
        # Apply CLAHE enhancement
        if apply_clahe:
            pre_processed = image_preprocessor.apply_clahe_enhancement(pre_processed)
            post_processed = image_preprocessor.apply_clahe_enhancement(post_processed)
        
        # Save processed full-resolution images
        _save_processed_images_wrapper(
            pre_processed, post_processed, tile_name, region_name,
            pre_filename, post_filename, output_dir
        )
        
        # Concatenate pre and post images (6 channels)
        combined_image = np.concatenate([pre_processed, post_processed], axis=2)
        
        # Extract patches
        patches = patch_extractor.extract_patches(
            combined_image,
            mask=mask,
            oversample_flood=True
        )
        
        # Save patches and create metadata
        patch_metadata_list = []
        flood_positive_count = 0
        
        for patch_idx, patch_dict in enumerate(patches):
            patch_id = f"{region_name}_{tile_name.replace('.geojson', '')}_{patch_idx}"
            
            # Save image (6 channels: 3 pre + 3 post)
            image_path = output_dir / 'images' / f"{patch_id}.npy"
            np.save(image_path, patch_dict['image'].astype(np.float32))
            
            # Save mask
            mask_path = output_dir / 'masks' / f"{patch_id}.npy"
            np.save(mask_path, patch_dict['mask'].astype(np.uint8))
            
            # Create metadata
            is_flood_positive = patch_dict.get('is_flood_positive', False)
            if is_flood_positive:
                flood_positive_count += 1
            
            metadata = {
                'patch_id': patch_id,
                'tile_name': tile_name,
                'region': region_name,
                'position': patch_dict['position'],
                'image_path': str(image_path.relative_to(output_dir.parent)),
                'mask_path': str(mask_path.relative_to(output_dir.parent)),
                'flood_pixels': patch_dict.get('flood_pixels', 0),
                'is_flood_positive': is_flood_positive,
                'class_distribution': patch_dict.get('class_distribution', {}),
                'quality_metrics': {
                    'pre_image': quality_pre,
                    'post_image': quality_post
                }
            }
            
            patch_metadata_list.append(metadata)
        
        stats = {
            'quality_failed': 0,
            'patches': len(patches),
            'flood_positive': flood_positive_count
        }
        
        return (True, patch_metadata_list, stats)
        
    except Exception as e:
        # Return failure on any error
        return (False, [], {'quality_failed': 0, 'patches': 0, 'flood_positive': 0})


def _save_processed_images_wrapper(pre_image, post_image, tile_name, region, 
                                   pre_image_filename, post_image_filename, output_dir):
    """Helper function to save processed images (used by multiprocessing wrapper)"""
    region_dir = output_dir / 'processed_images' / region
    region_dir.mkdir(parents=True, exist_ok=True)
    
    pre_dir = region_dir / 'PRE-event'
    post_dir = region_dir / 'POST-event'
    pre_dir.mkdir(exist_ok=True)
    post_dir.mkdir(exist_ok=True)
    
    pre_output_name = pre_image_filename if pre_image_filename else tile_name.replace('.geojson', '.tif')
    post_output_name = post_image_filename if post_image_filename else tile_name.replace('.geojson', '.tif')
    
    # Convert to uint16
    pre_image_uint16 = (np.clip(pre_image, 0, 1) * 65535).astype(np.uint16)
    post_image_uint16 = (np.clip(post_image, 0, 1) * 65535).astype(np.uint16)
    
    # Save images
    pre_path = pre_dir / pre_output_name
    post_path = post_dir / post_output_name
    cv2.imwrite(str(pre_path), cv2.cvtColor(pre_image_uint16, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(post_path), cv2.cvtColor(post_image_uint16, cv2.COLOR_RGB2BGR))


class DataPreprocessor:
    """Main class for preprocessing and exporting dataset"""
    
    def __init__(self, output_dir: Path = PROCESSED_TRAIN_DIR, max_workers: int = None):
        """
        Initialize preprocessor
        
        Args:
            output_dir: Directory to save processed data
            max_workers: Number of parallel workers (default: CPU count)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for patches
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'masks').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        # Create subdirectories for processed full-resolution images
        (self.output_dir / 'processed_images').mkdir(exist_ok=True)
        # Region-specific directories will be created dynamically as regions are processed
        
        # Initialize processors
        self.image_preprocessor = ImagePreprocessor(
            apply_clahe=APPLY_CLAHE,
            clahe_clip_limit=CLAHE_CLIP_LIMIT,
            clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE
        )
        
        self.patch_extractor = PatchExtractor(
            patch_size=PATCH_SIZE,
            overlap=PATCH_OVERLAP,
            min_flood_pixels=MIN_FLOOD_PIXELS
        )
        
        # Set number of workers (use all cores for ProcessPoolExecutor)
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Statistics
        self.stats = {
            'total_tiles': 0,
            'processed_tiles': 0,
            'failed_tiles': 0,
            'total_patches': 0,
            'flood_positive_patches': 0,
            'quality_failed': 0
        }
        
        self.metadata_list = []
        
    def process_region(self, 
                      region_path: Path,
                      region_name: str,
                      skip_existing: bool = True) -> List[Dict]:
        """
        Process all tiles from a region using parallel processing
        
        Args:
            region_path: Path to region directory
            region_name: Name of the region
            skip_existing: Whether to skip already processed tiles
            
        Returns:
            List of metadata dictionaries for all patches
        """
        print(f"\n{'='*60}")
        print(f"Processing: {region_name}")
        
        # Initialize loader
        loader = DatasetLoader(region_path, region_name)
        tile_list = loader.get_tile_list()
        
        print(f"  Tiles: {len(tile_list)} | Workers: {self.max_workers}")
            
        region_metadata = []
        self.stats['total_tiles'] += len(tile_list)
        
        # Process tiles in parallel using ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create partial function with fixed arguments
            process_func = partial(
                _process_tile_wrapper,
                region_path=region_path,
                region_name=region_name,
                output_dir=self.output_dir,
                apply_advanced=APPLY_ADVANCED_PREPROCESSING,
                remove_clouds=REMOVE_CLOUDS,
                apply_deblur=APPLY_DEBLUR,
                correct_geometry=CORRECT_GEOMETRY,
                apply_clahe=APPLY_CLAHE,
                clahe_clip_limit=CLAHE_CLIP_LIMIT,
                clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE,
                patch_size=PATCH_SIZE,
                patch_overlap=PATCH_OVERLAP,
                min_flood_pixels=MIN_FLOOD_PIXELS,
                min_valid_ratio=MIN_VALID_PIXELS_RATIO,
                max_cloud_coverage=MAX_CLOUD_COVERAGE
            )
            
            # Submit all tasks
            future_to_tile = {
                executor.submit(process_func, tile_name): tile_name 
                for tile_name in tile_list
            }
            
            # Progress tracking
            completed = 0
            total = len(tile_list)
            
            # Create progress bar with compact display
            pbar = tqdm(total=total, desc=f"  {region_name}", unit="tiles", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            # Collect results as they complete
            for future in as_completed(future_to_tile):
                tile_name = future_to_tile[future]
                completed += 1
                
                try:
                    success, patch_metadata, tile_stats = future.result()
                    
                    if success and patch_metadata:
                        region_metadata.extend(patch_metadata)
                        self.stats['processed_tiles'] += 1
                        self.stats['total_patches'] += tile_stats['patches']
                        self.stats['flood_positive_patches'] += tile_stats['flood_positive']
                        self.stats['quality_failed'] += tile_stats['quality_failed']
                    else:
                        self.stats['failed_tiles'] += 1
                        self.stats['quality_failed'] += tile_stats['quality_failed']
                        
                except Exception:
                    self.stats['failed_tiles'] += 1
                
                pbar.update(1)
            
            pbar.close()
            
            # Show region summary
            flood_pct = (self.stats['flood_positive_patches'] / self.stats['total_patches'] * 100) if self.stats['total_patches'] > 0 else 0
            print(f"  ✓ Patches: {self.stats['total_patches']} ({flood_pct:.1f}% flood) | Failed: {self.stats['failed_tiles']}")
                
        return region_metadata
    
    def _process_single_tile(self, region_path: Path, tile_name: str, region_name: str) -> List[Dict]:
        """
        Process a single tile (used for parallel execution)
        
        Args:
            region_path: Path to region directory
            tile_name: Name of the tile
            region_name: Region name
            
        Returns:
            List of patch metadata or empty list if failed
        """
        try:
            # Load tile data
            tile_data = load_tile_data(region_path, tile_name, region_name)
            
            if not tile_data or tile_data['pre_image'] is None:
                return []
            
            # Get original image filenames from metadata
            pre_filename = tile_data.get('pre_metadata', {}).get('filename')
            post_filename = tile_data.get('post_metadata', {}).get('filename')
                
            # Process tile
            patch_metadata = self.process_tile(
                pre_image=tile_data['pre_image'],
                post_image=tile_data['post_image'],
                mask=tile_data['mask'],
                tile_name=tile_name,
                region=region_name,
                pre_image_filename=pre_filename,
                post_image_filename=post_filename
            )
            
            return patch_metadata
            
        except Exception as e:
            # Return empty list on error
            return []
    
    def process_tile(self,
                    pre_image: np.ndarray,
                    post_image: np.ndarray,
                    mask: np.ndarray,
                    tile_name: str,
                    region: str,
                    pre_image_filename: str = None,
                    post_image_filename: str = None) -> List[Dict]:
        """
        Process a single tile
        
        Args:
            pre_image: Pre-event image
            post_image: Post-event image
            mask: Segmentation mask
            tile_name: Name of the tile
            region: Region name
            pre_image_filename: Original pre-event image filename
            post_image_filename: Original post-event image filename
            
        Returns:
            List of patch metadata
        """
        patch_metadata_list = []
        
        # Quality check on pre-image
        quality_pre = self.image_preprocessor.check_image_quality(
            pre_image,
            min_valid_ratio=MIN_VALID_PIXELS_RATIO,
            max_cloud_coverage=MAX_CLOUD_COVERAGE
        )
        
        if not quality_pre['passes_quality']:
            self.stats['quality_failed'] += 1
            return []
            
        # Quality check on post-image if available
        if post_image is not None:
            quality_post = self.image_preprocessor.check_image_quality(
                post_image,
                min_valid_ratio=MIN_VALID_PIXELS_RATIO,
                max_cloud_coverage=MAX_CLOUD_COVERAGE
            )
            
            if not quality_post['passes_quality']:
                self.stats['quality_failed'] += 1
                return []
        else:
            # If no post image, use pre-image as post
            post_image = pre_image.copy()
        
        # Apply advanced preprocessing: cloud removal, deblurring, geometric correction
        if APPLY_ADVANCED_PREPROCESSING:
            pre_advanced = self.image_preprocessor.apply_advanced_preprocessing(
                pre_image,
                remove_clouds=REMOVE_CLOUDS,
                apply_deblur=APPLY_DEBLUR,
                correct_geometry=CORRECT_GEOMETRY
            )
            
            post_advanced = self.image_preprocessor.apply_advanced_preprocessing(
                post_image,
                remove_clouds=REMOVE_CLOUDS,
                apply_deblur=APPLY_DEBLUR,
                correct_geometry=CORRECT_GEOMETRY
            )
            
            # Use the final processed images
            pre_processed = pre_advanced['final']
            post_processed = post_advanced['final']
        else:
            pre_processed = pre_image.copy()
            post_processed = post_image.copy()
        
        # Apply CLAHE enhancement on top
        if APPLY_CLAHE:
            pre_processed = self.image_preprocessor.apply_clahe_enhancement(pre_processed)
            post_processed = self.image_preprocessor.apply_clahe_enhancement(post_processed)
            apply_standardization=False        
        
        # Save processed full-resolution images
        self._save_processed_images(pre_processed, post_processed, tile_name, region, 
                                   pre_image_filename, post_image_filename)
        
        # Concatenate pre and post images (6 channels)
        combined_image = np.concatenate([pre_processed, post_processed], axis=2)
        
        # Extract patches
        patches = self.patch_extractor.extract_patches(
            combined_image,
            mask=mask,
            oversample_flood=True
        )
        
        # Save each patch
        for patch_idx, patch_dict in enumerate(patches):
            patch_id = f"{region}_{tile_name.replace('.geojson', '')}_{patch_idx}"
            
            # Save image (6 channels: 3 pre + 3 post)
            image_path = self.output_dir / 'images' / f"{patch_id}.npy"
            np.save(image_path, patch_dict['image'].astype(np.float32))
            
            # Save mask
            mask_path = self.output_dir / 'masks' / f"{patch_id}.npy"
            np.save(mask_path, patch_dict['mask'].astype(np.uint8))
            
            # Create metadata
            metadata = {
                'patch_id': patch_id,
                'tile_name': tile_name,
                'region': region,
                'position': patch_dict['position'],
                'image_path': str(image_path.relative_to(self.output_dir.parent)),
                'mask_path': str(mask_path.relative_to(self.output_dir.parent)),
                'flood_pixels': patch_dict.get('flood_pixels', 0),
                'is_flood_positive': patch_dict.get('is_flood_positive', False),
                'class_distribution': patch_dict.get('class_distribution', {}),
                'quality_metrics': {
                    'pre_image': quality_pre,
                    'post_image': quality_post if post_image is not None else None
                }
            }
            
            patch_metadata_list.append(metadata)
            self.stats['total_patches'] += 1
            if metadata['is_flood_positive']:
                self.stats['flood_positive_patches'] += 1
                
        return patch_metadata_list
    
    def _save_processed_images(self, pre_image: np.ndarray, post_image: np.ndarray, 
                               tile_name: str, region: str, pre_image_filename: str = None, 
                               post_image_filename: str = None):
        """
        Save processed full-resolution images maintaining folder structure
        
        Args:
            pre_image: Processed pre-event image
            post_image: Processed post-event image
            tile_name: Name of the tile (annotation file)
            region: Region name (full folder name)
            pre_image_filename: Original pre-event image filename (with extension)
            post_image_filename: Original post-event image filename (with extension)
        """
        # Create region-specific directory
        region_dir = self.output_dir / 'processed_images' / region
        region_dir.mkdir(parents=True, exist_ok=True)
        
        # Create PRE-event and POST-event subdirectories
        pre_dir = region_dir / 'PRE-event'
        post_dir = region_dir / 'POST-event'
        pre_dir.mkdir(exist_ok=True)
        post_dir.mkdir(exist_ok=True)
        
        # Use original filenames from CSV if provided
        if pre_image_filename:
            # Use the exact filename from CSV (already has .tif extension)
            pre_output_name = pre_image_filename
        else:
            # Fallback: use tile name
            pre_output_name = tile_name.replace('.geojson', '.tif')
        
        if post_image_filename:
            # Use the exact filename from CSV (already has .tif extension)
            post_output_name = post_image_filename
        else:
            # Fallback: use tile name
            post_output_name = tile_name.replace('.geojson', '.tif')
        
        # Convert float32 [0, 1] to uint16 for TIF format (better precision than uint8)
        # Scale from [0, 1] to [0, 65535]
        pre_image_uint16 = (np.clip(pre_image, 0, 1) * 65535).astype(np.uint16)
        post_image_uint16 = (np.clip(post_image, 0, 1) * 65535).astype(np.uint16)
        
        # Save pre-event image as TIF
        pre_path = pre_dir / pre_output_name
        # OpenCV uses BGR, but our images are RGB, so convert
        cv2.imwrite(str(pre_path), cv2.cvtColor(pre_image_uint16, cv2.COLOR_RGB2BGR))
        
        # Save post-event image as TIF
        post_path = post_dir / post_output_name
        cv2.imwrite(str(post_path), cv2.cvtColor(post_image_uint16, cv2.COLOR_RGB2BGR))
    
    def save_metadata(self, metadata_list: List[Dict], split_name: str = 'train'):
        """
        Save metadata to file
        
        Args:
            metadata_list: List of metadata dictionaries
            split_name: Name of the split (train/val/test)
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj
        
        # Save as JSON
        json_path = self.output_dir / 'metadata' / f'{split_name}_metadata.json'
        with open(json_path, 'w') as f:
            json.dump(convert_numpy_types(metadata_list), f, indent=2)
            
        # Save as pickle for faster loading
        pkl_path = self.output_dir / 'metadata' / f'{split_name}_metadata.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(metadata_list, f)
            
        # Save as CSV for easy inspection
        df = pd.DataFrame(metadata_list)
        # Flatten nested dicts for CSV
        df_flat = df.drop(columns=['quality_metrics', 'class_distribution'], errors='ignore')
        csv_path = self.output_dir / 'metadata' / f'{split_name}_metadata.csv'
        df_flat.to_csv(csv_path, index=False)


def create_geo_stratified_split(
    metadata_list: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create geo-stratified train/val/test splits
    Ensures tiles from same spatial region don't appear in multiple splits
    
    Args:
        metadata_list: List of all patch metadata
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_metadata, val_metadata, test_metadata)
    """
    import random
    
    # Group patches by tile (to avoid spatial leakage)
    tiles_dict = {}
    for metadata in metadata_list:
        tile_key = f"{metadata['region']}_{metadata['tile_name']}"
        if tile_key not in tiles_dict:
            tiles_dict[tile_key] = []
        tiles_dict[tile_key].append(metadata)
        
    # Get list of tiles
    tile_keys = list(tiles_dict.keys())
    random.shuffle(tile_keys)
    
    # Calculate split indices
    n_tiles = len(tile_keys)
    n_train = int(n_tiles * train_ratio)
    n_val = int(n_tiles * val_ratio)
    
    # Split tiles
    train_tiles = tile_keys[:n_train]
    val_tiles = tile_keys[n_train:n_train+n_val]
    test_tiles = tile_keys[n_train+n_val:]
    
    # Collect patches for each split
    train_metadata = []
    val_metadata = []
    test_metadata = []
    
    for tile_key in train_tiles:
        train_metadata.extend(tiles_dict[tile_key])
        
    for tile_key in val_tiles:
        val_metadata.extend(tiles_dict[tile_key])
        
    for tile_key in test_tiles:
        test_metadata.extend(tiles_dict[tile_key])
        
    print(f"\nGeo-stratified split:")
    print(f"  Training: {len(train_tiles)} tiles, {len(train_metadata)} patches")
    print(f"  Validation: {len(val_tiles)} tiles, {len(val_metadata)} patches")
    print(f"  Test: {len(test_tiles)} tiles, {len(test_metadata)} patches")
    
    # Calculate flood ratios
    train_flood = sum(1 for m in train_metadata if m['is_flood_positive'])
    val_flood = sum(1 for m in val_metadata if m['is_flood_positive'])
    test_flood = sum(1 for m in test_metadata if m['is_flood_positive'])
    
    print(f"\nFlood-positive patches:")
    print(f"  Training: {train_flood}/{len(train_metadata)} ({train_flood/len(train_metadata)*100:.1f}%)")
    print(f"  Validation: {val_flood}/{len(val_metadata)} ({val_flood/len(val_metadata)*100:.1f}%)")
    print(f"  Test: {test_flood}/{len(test_metadata)} ({test_flood/len(test_metadata)*100:.1f}%)")
    
    return train_metadata, val_metadata, test_metadata


def copy_processed_full_images(metadata_list: List[Dict], src_dir: Path, dst_dir: Path):
    """
    Copy processed full-resolution images for tiles in metadata_list.
    Uses actual filenames from metadata to ensure correct file matching.
    
    Args:
        metadata_list: List of patch metadata
        src_dir: Source directory (e.g., PROCESSED_TRAIN_DIR)
        dst_dir: Destination directory (e.g., PROCESSED_VAL_DIR or PROCESSED_TEST_DIR)
    """
    from tqdm import tqdm
    
    # Get unique tiles from metadata
    unique_tiles = {}  # (region, tile_name) -> set of actual filenames
    for metadata in metadata_list:
        region = metadata['region']
        tile_name = metadata['tile_name']
        key = (region, tile_name)
        if key not in unique_tiles:
            unique_tiles[key] = set()
    
    # For each unique tile, find and copy all matching processed images
    print(f"  Copying {len(unique_tiles)} unique tiles...")
    for (region, tile_name) in tqdm(unique_tiles.keys(), desc="  Copying full-res images"):
        # Create destination directories
        dst_pre_dir = dst_dir / 'processed_images' / region / 'PRE-event'
        dst_post_dir = dst_dir / 'processed_images' / region / 'POST-event'
        dst_pre_dir.mkdir(parents=True, exist_ok=True)
        dst_post_dir.mkdir(parents=True, exist_ok=True)
        
        # Source directories
        src_pre_dir = src_dir / 'processed_images' / region / 'PRE-event'
        src_post_dir = src_dir / 'processed_images' / region / 'POST-event'
        
        # Find all TIF files that could match this tile
        # The tile_name is the annotation file (e.g., "0_15_63.geojson")
        # We need to find corresponding image files
        tile_base = tile_name.replace('.geojson', '')
        
        # Copy all matching PRE-event images
        if src_pre_dir.exists():
            for src_file in src_pre_dir.glob('*.tif'):
                # Check if this file belongs to this tile (based on coordinate pattern)
                if tile_base in src_file.stem or src_file.stem.endswith(tile_base):
                    dst_file = dst_pre_dir / src_file.name
                    if src_file.exists() and not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
        
        # Copy all matching POST-event images
        if src_post_dir.exists():
            for src_file in src_post_dir.glob('*.tif'):
                # Check if this file belongs to this tile (based on coordinate pattern)
                if tile_base in src_file.stem or src_file.stem.endswith(tile_base):
                    dst_file = dst_post_dir / src_file.name
                    if src_file.exists() and not dst_file.exists():
                        shutil.copy2(src_file, dst_file)


def discover_training_regions(train_path: Path) -> List[Tuple[Path, str]]:
    """
    Discover all training regions from the raw dataset directory
    
    Args:
        train_path: Path to the training data directory
        
    Returns:
        List of tuples (region_path, region_name)
    """
    regions = []
    
    # Look for directories ending with '_Training_Public'
    for item in train_path.iterdir():
        if item.is_dir() and '_Training_Public' in item.name:
            # Use full folder name as region name
            region_name = item.name
            regions.append((item, region_name))
    
    return sorted(regions, key=lambda x: x[1])


def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*80)
    print("FLOOD DETECTION - DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # ========== PROCESS TRAINING DATA ==========
    print(f"\n{'='*80}")
    print("STEP 1: PROCESSING TRAINING DATA")
    print(f"{'='*80}")
    
    # Discover training regions dynamically
    training_regions = discover_training_regions(TRAIN_PATH)
    
    if not training_regions:
        print("\nNo training regions found in:", TRAIN_PATH)
        print("Expected directories with pattern: *_Training_Public")
        return
    
    print(f"\nDiscovered {len(training_regions)} training region(s):")
    for region_path, region_name in training_regions:
        print(f"  - {region_name}: {region_path}")
    
    # Process all discovered training regions
    all_train_metadata = []
    preprocessor_train = DataPreprocessor(output_dir=PROCESSED_TRAIN_DIR)
    
    for region_path, region_name in training_regions:
        region_metadata = preprocessor_train.process_region(
            region_path,
            region_name
        )
        all_train_metadata.extend(region_metadata)
    
    print(f"\n{'='*80}")
    print("TRAINING DATA PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total tiles processed: {preprocessor_train.stats['processed_tiles']}")
    print(f"Total tiles failed: {preprocessor_train.stats['failed_tiles']}")
    print(f"Quality check failures: {preprocessor_train.stats['quality_failed']}")
    print(f"Total patches extracted: {preprocessor_train.stats['total_patches']}")
    print(f"Flood-positive patches: {preprocessor_train.stats['flood_positive_patches']}")
    
    if preprocessor_train.stats['total_patches'] > 0:
        flood_ratio = preprocessor_train.stats['flood_positive_patches'] / preprocessor_train.stats['total_patches']
        print(f"Flood ratio: {flood_ratio*100:.2f}%")
    
    # ========== CREATE TRAIN/VAL SPLIT ==========
    print(f"\n{'='*80}")
    print("STEP 2: CREATING TRAIN/VAL SPLITS")
    print(f"{'='*80}")
    
    # Create geo-stratified train/val split (no test from training data)
    train_metadata, val_metadata, _ = create_geo_stratified_split(
        all_train_metadata,
        train_ratio=TRAIN_RATIO + TEST_RATIO,  # Combine train and test ratios for training data
        val_ratio=VAL_RATIO,
        test_ratio=0.0  # No test split from training data
    )
    
    # Create val directory structure
    PROCESSED_VAL_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_VAL_DIR / 'images').mkdir(exist_ok=True)
    (PROCESSED_VAL_DIR / 'masks').mkdir(exist_ok=True)
    (PROCESSED_VAL_DIR / 'metadata').mkdir(exist_ok=True)
    (PROCESSED_VAL_DIR / 'processed_images').mkdir(exist_ok=True)
    
    # Copy validation files
    for metadata in tqdm(val_metadata, desc="  Copying val patches", unit="files",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'):
        # Extract just the filename from the path
        img_filename = Path(metadata['image_path']).name
        mask_filename = Path(metadata['mask_path']).name
        
        src_img = PROCESSED_TRAIN_DIR / 'images' / img_filename
        src_mask = PROCESSED_TRAIN_DIR / 'masks' / mask_filename
        
        dst_img = PROCESSED_VAL_DIR / 'images' / img_filename
        dst_mask = PROCESSED_VAL_DIR / 'masks' / mask_filename
        
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        if src_mask.exists():
            shutil.copy2(src_mask, dst_mask)
        
        # Update paths
        metadata['image_path'] = str(dst_img.relative_to(PROCESSED_VAL_DIR.parent))
        metadata['mask_path'] = str(dst_mask.relative_to(PROCESSED_VAL_DIR.parent))
    
    # Copy processed full-resolution images for validation
    copy_processed_full_images(val_metadata, PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR)
    
    # Save train and val metadata
    DataPreprocessor(PROCESSED_TRAIN_DIR).save_metadata(train_metadata, 'train')
    DataPreprocessor(PROCESSED_VAL_DIR).save_metadata(val_metadata, 'val')
    
    # ========== CALCULATE DATASET STATISTICS ==========
    print(f"\n{'='*60}")
    print("STEP 3: Dataset Statistics")
    print(f"{'='*60}")
    
    # Sample images for statistics (use subset for speed)
    sample_size = min(100, len(train_metadata))
    sample_metadata = np.random.choice(train_metadata, sample_size, replace=False)
    
    sample_images = []
    for metadata in tqdm(sample_metadata, desc="  Sampling", unit="imgs",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        img_path = PROCESSED_TRAIN_DIR.parent / metadata['image_path']
        if img_path.exists():
            img = np.load(img_path)
            sample_images.append(img)
    
    if sample_images:
        stats = calculate_dataset_statistics(sample_images)
        
        # Save statistics
        stats_path = PROCESSED_TRAIN_DIR.parent / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Mean: {[f'{x:.3f}' for x in stats['mean']]}" 
              f" | Std: {[f'{x:.3f}' for x in stats['std']]}")
    
    # ========== FINAL SUMMARY ==========
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"\nProcessed data saved to:")
    print(f"  Training: {PROCESSED_TRAIN_DIR}")
    print(f"    - Patches: {len(train_metadata)}")
    print(f"    - Flood-positive: {sum(1 for m in train_metadata if m['is_flood_positive'])}")
    print(f"  Validation: {PROCESSED_VAL_DIR}")
    print(f"    - Patches: {len(val_metadata)}")
    print(f"    - Flood-positive: {sum(1 for m in val_metadata if m['is_flood_positive'])}")
    
    print(f"\nOutput structure:")
    print(f"  - images/         : Extracted patches (512x512, 6 channels)")
    print(f"  - masks/          : Segmentation masks for patches")
    print(f"  - processed_images/: Full-resolution processed images")
    
    # Get all unique regions
    all_regions = set()
    for metadata in train_metadata + val_metadata:
        all_regions.add(metadata['region'])
    
    for region in sorted(all_regions):
        print(f"    - {region}/")
        print(f"      - PRE-event/  : Processed pre-event images")
        print(f"      - POST-event/ : Processed post-event images")
    print(f"  - metadata/       : JSON, pickle, and CSV metadata files")
        

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    
    main()