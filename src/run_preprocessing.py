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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from config import (
    GERMANY_TRAIN, LOUISIANA_EAST_TRAIN,
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
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'masks').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
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
        
        # Set number of workers
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        
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
        print(f"Processing region: {region_name}")
        print(f"{'='*60}")
        
        # Initialize loader
        loader = DatasetLoader(region_path, region_name)
        tile_list = loader.get_tile_list()
        
        print(f"Found {len(tile_list)} tiles")
        print(f"Using {self.max_workers} parallel workers")
        
        # Get flood statistics
        flood_stats = loader.get_flood_statistics()
        print(f"\nFlood statistics:")
        for key, value in flood_stats.items():
            print(f"  {key}: {value}")
            
        region_metadata = []
        self.stats['total_tiles'] += len(tile_list)
        
        # Process tiles in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_tile = {
                executor.submit(self._process_single_tile, region_path, tile_name, region_name): tile_name 
                for tile_name in tile_list
            }
            
            # Progress tracking
            completed = 0
            total = len(tile_list)
            
            # Create progress bar
            pbar = tqdm(total=total, desc=f"Processing {region_name}", unit="tile")
            
            # Collect results as they complete
            for future in as_completed(future_to_tile):
                tile_name = future_to_tile[future]
                completed += 1
                
                try:
                    patch_metadata = future.result()
                    if patch_metadata:
                        region_metadata.extend(patch_metadata)
                        self.stats['processed_tiles'] += 1
                    else:
                        self.stats['failed_tiles'] += 1
                        
                except Exception as e:
                    print(f"\nError processing {tile_name}: {e}")
                    self.stats['failed_tiles'] += 1
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'progress': f'{(completed/total)*100:.1f}%',
                    'success': self.stats['processed_tiles'],
                    'failed': self.stats['failed_tiles']
                })
            
            pbar.close()
                
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
                
            # Process tile
            patch_metadata = self.process_tile(
                pre_image=tile_data['pre_image'],
                post_image=tile_data['post_image'],
                mask=tile_data['mask'],
                tile_name=tile_name,
                region=region_name
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
                    region: str) -> List[Dict]:
        """
        Process a single tile
        
        Args:
            pre_image: Pre-event image
            post_image: Post-event image
            mask: Segmentation mask
            tile_name: Name of the tile
            region: Region name
            
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
        )
        
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
        
        print(f"\nMetadata saved:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pkl_path}")
        print(f"  CSV: {csv_path}")


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


def main():
    """Main preprocessing pipeline"""
    print("\n" + "="*80)
    print("FLOOD DETECTION - DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Process training regions
    all_metadata = []
    
    # Process Germany
    preprocessor_train = DataPreprocessor(output_dir=PROCESSED_TRAIN_DIR)
    germany_metadata = preprocessor_train.process_region(
        GERMANY_TRAIN,
        'Germany'
    )
    all_metadata.extend(germany_metadata)
    
    # Process Louisiana-East
    louisiana_metadata = preprocessor_train.process_region(
        LOUISIANA_EAST_TRAIN,
        'Louisiana-East'
    )
    all_metadata.extend(louisiana_metadata)
    
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total tiles processed: {preprocessor_train.stats['processed_tiles']}")
    print(f"Total tiles failed: {preprocessor_train.stats['failed_tiles']}")
    print(f"Quality check failures: {preprocessor_train.stats['quality_failed']}")
    print(f"Total patches extracted: {preprocessor_train.stats['total_patches']}")
    print(f"Flood-positive patches: {preprocessor_train.stats['flood_positive_patches']}")
    
    if preprocessor_train.stats['total_patches'] > 0:
        flood_ratio = preprocessor_train.stats['flood_positive_patches'] / preprocessor_train.stats['total_patches']
        print(f"Flood ratio: {flood_ratio*100:.2f}%")
    
    # Create geo-stratified splits
    print(f"\n{'='*80}")
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print(f"{'='*80}")
    
    train_metadata, val_metadata, test_metadata = create_geo_stratified_split(
        all_metadata,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    # Copy files to appropriate directories
    print(f"\nOrganizing files into split directories...")
    
    # Create val and test directories
    PROCESSED_VAL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    for split_dir in [PROCESSED_VAL_DIR, PROCESSED_TEST_DIR]:
        (split_dir / 'images').mkdir(exist_ok=True)
        (split_dir / 'masks').mkdir(exist_ok=True)
        (split_dir / 'metadata').mkdir(exist_ok=True)
    
    # Move validation files
    for metadata in tqdm(val_metadata, desc="Moving validation files"):
        # Extract just the filename from the path
        img_filename = Path(metadata['image_path']).name
        mask_filename = Path(metadata['mask_path']).name
        
        src_img = PROCESSED_TRAIN_DIR / 'images' / img_filename
        src_mask = PROCESSED_TRAIN_DIR / 'masks' / mask_filename
        
        dst_img = PROCESSED_VAL_DIR / 'images' / img_filename
        dst_mask = PROCESSED_VAL_DIR / 'masks' / mask_filename
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)
        
        # Update paths
        metadata['image_path'] = str(dst_img.relative_to(PROCESSED_VAL_DIR.parent))
        metadata['mask_path'] = str(dst_mask.relative_to(PROCESSED_VAL_DIR.parent))
    
    # Move test files
    for metadata in tqdm(test_metadata, desc="Moving test files"):
        # Extract just the filename from the path
        img_filename = Path(metadata['image_path']).name
        mask_filename = Path(metadata['mask_path']).name
        
        src_img = PROCESSED_TRAIN_DIR / 'images' / img_filename
        src_mask = PROCESSED_TRAIN_DIR / 'masks' / mask_filename
        
        dst_img = PROCESSED_TEST_DIR / 'images' / img_filename
        dst_mask = PROCESSED_TEST_DIR / 'masks' / mask_filename
        
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)
        
        # Update paths
        metadata['image_path'] = str(dst_img.relative_to(PROCESSED_TEST_DIR.parent))
        metadata['mask_path'] = str(dst_mask.relative_to(PROCESSED_TEST_DIR.parent))
    
    # Save metadata for each split
    print(f"\nSaving metadata...")
    
    DataPreprocessor(PROCESSED_TRAIN_DIR).save_metadata(train_metadata, 'train')
    DataPreprocessor(PROCESSED_VAL_DIR).save_metadata(val_metadata, 'val')
    DataPreprocessor(PROCESSED_TEST_DIR).save_metadata(test_metadata, 'test')
    
    # Calculate and save dataset statistics
    print(f"\n{'='*80}")
    print("CALCULATING DATASET STATISTICS")
    print(f"{'='*80}")
    
    # Sample images for statistics (use subset for speed)
    sample_size = min(100, len(train_metadata))
    sample_metadata = np.random.choice(train_metadata, sample_size, replace=False)
    
    print(f"Calculating statistics from {sample_size} sample images...")
    sample_images = []
    for metadata in tqdm(sample_metadata, desc="Loading samples"):
        img_path = PROCESSED_TRAIN_DIR.parent / metadata['image_path']
        img = np.load(img_path)
        sample_images.append(img)
    
    stats = calculate_dataset_statistics(sample_images)
    
    print(f"\nDataset statistics:")
    print(f"  Mean (per channel): {stats['mean']}")
    print(f"  Std (per channel): {stats['std']}")
    print(f"  Min: {stats['min']}")
    print(f"  Max: {stats['max']}")
    
    # Save statistics
    stats_path = PROCESSED_TRAIN_DIR.parent / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_path}")
    
    print(f"\n{'='*80}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nProcessed data saved to:")
    print(f"  Training: {PROCESSED_TRAIN_DIR}")
    print(f"  Validation: {PROCESSED_VAL_DIR}")
    print(f"  Test: {PROCESSED_TEST_DIR}")
    

if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    
    main()
