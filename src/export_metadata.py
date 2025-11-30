"""
Export metadata for preprocessed datasets.
This script can be run separately after preprocessing images to generate metadata files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from tqdm import tqdm
from typing import List, Dict

from config import (
    PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR,
    CLASS_NAMES
)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
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


def scan_processed_directory(base_dir: Path) -> List[Dict]:
    """
    Scan a processed directory and extract metadata from existing files.
    
    Args:
        base_dir: Base directory containing images/ and masks/ subdirectories
    
    Returns:
        List of metadata dictionaries
    """
    print(f"\nScanning directory: {base_dir}")
    
    images_dir = base_dir / 'images'
    masks_dir = base_dir / 'masks'
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"  Warning: images or masks directory not found")
        return []
    
    # Get all image files
    image_files = sorted(images_dir.glob('*.npy'))
    print(f"  Found {len(image_files)} image files")
    
    metadata_list = []
    
    for img_path in tqdm(image_files, desc="  Extracting metadata"):
        patch_id = img_path.stem
        mask_path = masks_dir / f"{patch_id}.npy"
        
        if not mask_path.exists():
            print(f"  Warning: Mask not found for {patch_id}")
            continue
        
        # Load image and mask to extract metadata
        try:
            image = np.load(img_path)
            mask = np.load(mask_path)
            
            # Parse patch ID (format: region_tilename_patchidx)
            parts = patch_id.split('_')
            if len(parts) >= 3:
                # Handle region names with underscores (e.g., Louisiana-East_Training_Public)
                # Last part is patch index, second-to-last might be part of tile name
                patch_idx = parts[-1]
                
                # Find where region name ends
                # Region names end with "_Training_Public" or "_Test_Public"
                region = None
                tile_name = None
                
                for i in range(len(parts) - 1):
                    candidate = '_'.join(parts[:i+1])
                    if candidate.endswith('_Training_Public') or candidate.endswith('_Test_Public'):
                        region = candidate
                        tile_name = '_'.join(parts[i+1:-1])
                        break
                
                if region is None:
                    # Fallback: assume first part is region, middle parts are tile, last is index
                    region = parts[0]
                    tile_name = '_'.join(parts[1:-1])
            else:
                region = 'unknown'
                tile_name = 'unknown'
                patch_idx = 0
            
            # Calculate flood pixels and class distribution
            unique_classes, counts = np.unique(mask, return_counts=True)
            class_distribution = {int(cls): int(count) for cls, count in zip(unique_classes, counts)}
            
            # Count flood-related pixels (classes 1-4: flooded/non-flooded buildings and roads)
            flood_pixels = sum(int(count) for cls, count in zip(unique_classes, counts) if cls in [1, 2, 3, 4])
            is_flood_positive = flood_pixels > 0
            
            # Create metadata entry
            metadata = {
                'patch_id': patch_id,
                'tile_name': tile_name,
                'region': region,
                'position': [0, 0],  # Position not available from filename
                'image_path': str(img_path.relative_to(base_dir.parent)),
                'mask_path': str(mask_path.relative_to(base_dir.parent)),
                'flood_pixels': flood_pixels,
                'is_flood_positive': is_flood_positive,
                'class_distribution': class_distribution,
                'image_shape': list(image.shape),
                'mask_shape': list(mask.shape)
            }
            
            metadata_list.append(metadata)
            
        except Exception as e:
            print(f"  Error processing {patch_id}: {e}")
            continue
    
    return metadata_list


def save_metadata(metadata_list: List[Dict], output_dir: Path, split_name: str):
    """
    Save metadata to JSON, pickle, and CSV formats.
    
    Args:
        metadata_list: List of metadata dictionaries
        output_dir: Output directory
        split_name: Name of the split (train/val/test)
    """
    metadata_dir = output_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Saving {split_name} metadata ({len(metadata_list)} patches)...")
    
    # Save as JSON
    json_path = metadata_dir / f'{split_name}_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(convert_numpy_types(metadata_list), f, indent=2)
    print(f"    JSON: {json_path}")
    
    # Save as pickle for faster loading
    pkl_path = metadata_dir / f'{split_name}_metadata.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(metadata_list, f)
    print(f"    Pickle: {pkl_path}")
    
    # Save as CSV for easy inspection
    df = pd.DataFrame(metadata_list)
    # Flatten nested dicts for CSV
    df_flat = df.drop(columns=['class_distribution', 'image_shape', 'mask_shape'], errors='ignore')
    csv_path = metadata_dir / f'{split_name}_metadata.csv'
    df_flat.to_csv(csv_path, index=False)
    print(f"    CSV: {csv_path}")
    
    # Print statistics
    flood_count = sum(1 for m in metadata_list if m['is_flood_positive'])
    print(f"    Statistics:")
    print(f"      Total patches: {len(metadata_list)}")
    print(f"      Flood-positive: {flood_count} ({flood_count/len(metadata_list)*100:.1f}%)")
    
    # Class distribution
    class_totals = {}
    for metadata in metadata_list:
        for cls, count in metadata['class_distribution'].items():
            class_totals[int(cls)] = class_totals.get(int(cls), 0) + count
    
    print(f"      Class distribution:")
    total_pixels = sum(class_totals.values())
    for cls in sorted(class_totals.keys()):
        count = class_totals[cls]
        pct = (count / total_pixels) * 100
        class_name = CLASS_NAMES.get(cls, f'Class {cls}')
        print(f"        {class_name}: {count:,} pixels ({pct:.2f}%)")


def main():
    """Main metadata export function."""
    print("="*80)
    print("METADATA EXPORT FOR PREPROCESSED DATASETS")
    print("="*80)
    
    # Process each split
    splits = [
        ('train', PROCESSED_TRAIN_DIR),
        ('val', PROCESSED_VAL_DIR),
        ('test', PROCESSED_TEST_DIR)
    ]
    
    for split_name, split_dir in splits:
        if not split_dir.exists():
            print(f"\n{split_name.upper()} directory not found: {split_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*80}")
        
        # Scan directory and extract metadata
        metadata_list = scan_processed_directory(split_dir)
        
        if len(metadata_list) > 0:
            # Save metadata
            save_metadata(metadata_list, split_dir, split_name)
        else:
            print(f"  No metadata extracted for {split_name}")
    
    print(f"\n{'='*80}")
    print("METADATA EXPORT COMPLETE!")
    print(f"{'='*80}")
    print("\n✓ Metadata files generated:")
    print("  - JSON (human-readable)")
    print("  - Pickle (fast loading)")
    print("  - CSV (spreadsheet-compatible)")
    print("\n✓ Ready for model training!")


if __name__ == '__main__':
    main()
