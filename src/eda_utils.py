"""
Utility functions for Exploratory Data Analysis (EDA).
Contains reusable helper functions for file analysis, data validation, and statistics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter
from PIL import Image


def format_size(size_bytes: float) -> str:
    """
    Convert bytes to human-readable file size format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def load_geojson_annotations(annotation_path: Path) -> pd.DataFrame:
    """
    Load GeoJSON annotations and extract feature properties
    
    Args:
        annotation_path: Path to directory containing GeoJSON files
        
    Returns:
        DataFrame with annotation properties
    """
    annotations = []
    if annotation_path.exists():
        for geojson_file in sorted(annotation_path.glob("*.geojson")):
            try:
                with open(geojson_file, 'r') as f:
                    geojson_data = json.load(f)
                    for feature in geojson_data.get('features', []):
                        props = feature.get('properties', {})
                        props['tile_id'] = geojson_file.stem
                        props['geometry_type'] = feature.get('geometry', {}).get('type', 'unknown')
                        annotations.append(props)
            except Exception as e:
                print(f"  ERROR loading {geojson_file.name}: {e}")
    return pd.DataFrame(annotations)


def validate_csv_file_integrity(dataset_name: str, 
                                dataset_path: Path,
                                verbose: bool = True) -> Dict:
    """
    Validate CSV-to-file mapping integrity for a dataset
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to dataset directory
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with integrity validation results
    """
    # Load mapping CSV if it exists
    mapping_csv = list(dataset_path.glob("*mapping.csv"))
    if not mapping_csv:
        return {"error": "No mapping CSV found"}
    
    df_mapping = pd.read_csv(mapping_csv[0])
    
    # Check existence of files
    missing_labels = []
    missing_pre = []
    missing_post1 = []
    missing_post2 = []
    
    # Define directories
    pre_dir = dataset_path / "PRE-event"
    post_dir = dataset_path / "POST-event"
    ann_dir = dataset_path / "annotations"
    
    for idx, row in df_mapping.iterrows():
        # Check label file
        label_file = ann_dir / row['label']
        if not label_file.exists():
            missing_labels.append(row['label'])
        
        # Check pre-event image
        pre_file = pre_dir / row['pre-event image']
        if not pre_file.exists():
            missing_pre.append(row['pre-event image'])
        
        # Check post-event image 1
        post1_file = post_dir / row['post-event image 1']
        if not post1_file.exists():
            missing_post1.append(row['post-event image 1'])
        
        # Check post-event image 2 (if it exists in CSV)
        if 'post-event image 2' in df_mapping.columns and pd.notna(row['post-event image 2']):
            post2_file = post_dir / row['post-event image 2']
            if not post2_file.exists():
                missing_post2.append(row['post-event image 2'])
    
    # Calculate match rates
    total_entries = len(df_mapping)
    label_match_rate = (total_entries - len(missing_labels)) / total_entries * 100 if total_entries > 0 else 0
    pre_match_rate = (total_entries - len(missing_pre)) / total_entries * 100 if total_entries > 0 else 0
    post1_match_rate = (total_entries - len(missing_post1)) / total_entries * 100 if total_entries > 0 else 0
    
    results = {
        "dataset_name": dataset_name,
        "total_entries": total_entries,
        "missing_labels": len(missing_labels),
        "missing_pre": len(missing_pre),
        "missing_post1": len(missing_post1),
        "missing_post2": len(missing_post2),
        "label_match_rate": label_match_rate,
        "pre_match_rate": pre_match_rate,
        "post1_match_rate": post1_match_rate,
        "missing_labels_list": missing_labels[:5],  # First 5 only
        "missing_pre_list": missing_pre[:5],
        "missing_post1_list": missing_post1[:5]
    }
    
    if verbose:
        print(f"\n  Integrity Check Results:")
        print(f"    Label files:           {total_entries - len(missing_labels):4d}/{total_entries} ({label_match_rate:5.1f}%)")
        print(f"    Pre-event images:      {total_entries - len(missing_pre):4d}/{total_entries} ({pre_match_rate:5.1f}%)")
        print(f"    Post-event images (1): {total_entries - len(missing_post1):4d}/{total_entries} ({post1_match_rate:5.1f}%)")
        
        if missing_labels and len(missing_labels) <= 5:
            print(f"\n  WARNING: {len(missing_labels)} label files not found:")
            for label in missing_labels:
                print(f"    - {label}")
    
    return results


def analyze_image_properties(dataset_name: str,
                            dataset_path: Path,
                            sample_size: int = 10) -> Dict:
    """
    Analyze image properties (shape, dtype, value ranges, channels)
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to dataset directory
        sample_size: Number of images to sample for analysis
        
    Returns:
        Dictionary with image analysis results
    """
    pre_dir = dataset_path / "PRE-event"
    post_dir = dataset_path / "POST-event"
    
    pre_images = sorted(list(pre_dir.glob("*.png")) + list(pre_dir.glob("*.tif")))
    post_images = sorted(list(post_dir.glob("*.png")) + list(post_dir.glob("*.tif")))
    
    if len(pre_images) == 0:
        return {"error": "No images found"}
    
    # Sample images
    sample_size = min(sample_size, len(pre_images))
    shapes = []
    dtypes = []
    value_ranges = []
    channel_stats = []
    file_sizes = []
    
    for i in range(sample_size):
        # Load image
        pre_img = np.array(Image.open(pre_images[i]))
        
        shapes.append(pre_img.shape)
        dtypes.append(str(pre_img.dtype))
        value_ranges.append((pre_img.min(), pre_img.max(), pre_img.mean(), pre_img.std()))
        file_sizes.append(pre_images[i].stat().st_size)
        
        # Channel-wise statistics
        if len(pre_img.shape) == 3:
            channel_means = [pre_img[:,:,c].mean() for c in range(pre_img.shape[2])]
            channel_stds = [pre_img[:,:,c].std() for c in range(pre_img.shape[2])]
            channel_stats.append((channel_means, channel_stds))
    
    # Aggregate statistics
    unique_shapes = Counter([str(s) for s in shapes])
    unique_dtypes = Counter(dtypes)
    
    results = {
        "dataset_name": dataset_name,
        "num_samples_analyzed": sample_size,
        "total_images": len(pre_images),
        "shapes": dict(unique_shapes),
        "most_common_shape": shapes[0],
        "dtypes": dict(unique_dtypes),
        "value_ranges": {
            "min": min([v[0] for v in value_ranges]),
            "max": max([v[1] for v in value_ranges]),
            "mean_avg": np.mean([v[2] for v in value_ranges]),
            "std_avg": np.mean([v[3] for v in value_ranges])
        },
        "avg_file_size": np.mean(file_sizes),
        "total_pixels": shapes[0][0] * shapes[0][1] if len(shapes[0]) >= 2 else 0,
        "has_channels": len(shapes[0]) == 3,
        "num_channels": shapes[0][2] if len(shapes[0]) == 3 else 1,
        "channel_stats": channel_stats,
        "shape_consistency": len(unique_shapes) == 1
    }
    
    return results


def compute_csv_summary_statistics(csv_analysis: Dict) -> Dict:
    """
    Compute comprehensive summary statistics from CSV analysis data
    
    Args:
        csv_analysis: Dictionary containing CSV analysis data per dataset
        
    Returns:
        Dictionary with summary statistics
    """
    csv_summary = {
        "total_datasets": len(csv_analysis),
        "total_mappings": 0,
        "total_road_segments": 0,
        "datasets_with_dual_post_images": 0,
        "flood_statistics": {},
        "road_length_statistics": {},
        "travel_time_statistics": {}
    }
    
    for dataset_name, data in csv_analysis.items():
        if 'mapping' in data:
            csv_summary['total_mappings'] += data['mapping']['rows']
            
            # Check for dual post-event images
            df_map = data['mapping']['dataframe']
            if 'post-event image 2' in df_map.columns:
                post2_count = df_map['post-event image 2'].notna().sum()
                if post2_count > 0:
                    csv_summary['datasets_with_dual_post_images'] += 1
        
        if 'reference' in data:
            csv_summary['total_road_segments'] += data['reference']['rows']
            df_ref = data['reference']['dataframe']
            
            # Aggregate flood statistics
            if 'Flooded' in df_ref.columns:
                flood_counts = df_ref['Flooded'].value_counts()
                for status, count in flood_counts.items():
                    if status not in csv_summary['flood_statistics']:
                        csv_summary['flood_statistics'][status] = 0
                    csv_summary['flood_statistics'][status] += count
            
            # Aggregate road length statistics
            if 'length_m' in df_ref.columns:
                # Convert to numeric, handling string values
                valid_lengths = pd.to_numeric(df_ref['length_m'], errors='coerce').dropna()
                if len(valid_lengths) > 0:
                    if 'all_lengths' not in csv_summary['road_length_statistics']:
                        csv_summary['road_length_statistics']['all_lengths'] = []
                    csv_summary['road_length_statistics']['all_lengths'].extend(valid_lengths.tolist())
            
            # Aggregate travel time statistics
            if 'travel_time_s' in df_ref.columns:
                valid_times = df_ref['travel_time_s'].dropna()
                if len(valid_times) > 0:
                    if 'all_times' not in csv_summary['travel_time_statistics']:
                        csv_summary['travel_time_statistics']['all_times'] = []
                    csv_summary['travel_time_statistics']['all_times'].extend(valid_times.tolist())
    
    return csv_summary


def calculate_class_imbalance_metrics(class_distributions: Dict) -> Dict:
    """
    Calculate class imbalance metrics from class distributions
    
    Args:
        class_distributions: Dictionary mapping keys to class count distributions
        
    Returns:
        Dictionary with imbalance metrics per distribution
    """
    imbalance_metrics = {}
    
    for key, distribution in class_distributions.items():
        if distribution:
            counts = list(distribution.values())
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            # Determine severity
            if imbalance_ratio > 10:
                severity = "HIGH"
                recommendation = "Consider class weighting or resampling"
            elif imbalance_ratio > 5:
                severity = "MODERATE"
                recommendation = "May benefit from class weighting"
            else:
                severity = "LOW"
                recommendation = "Relatively balanced"
            
            imbalance_metrics[key] = {
                'max_count': max_count,
                'min_count': min_count,
                'ratio': imbalance_ratio,
                'severity': severity,
                'recommendation': recommendation,
                'distribution': distribution
            }
    
    return imbalance_metrics


def extract_geospatial_coordinates(dataset_path: Path,
                                   max_files: int = 50) -> Optional[Dict]:
    """
    Extract geospatial coordinates from GeoJSON annotation files
    
    Args:
        dataset_path: Path to dataset directory
        max_files: Maximum number of GeoJSON files to process
        
    Returns:
        Dictionary with coordinates and geometries, or None if no data
    """
    ann_path = dataset_path / "annotations"
    if not ann_path.exists():
        return None
    
    all_coords = []
    all_geometries = []
    
    for geojson_file in sorted(list(ann_path.glob("*.geojson")))[:max_files]:
        try:
            with open(geojson_file, 'r') as f:
                geojson_data = json.load(f)
                for feature in geojson_data.get('features', []):
                    geom = feature.get('geometry', {})
                    coords = geom.get('coordinates', [])
                    geom_type = geom.get('type', 'unknown')
                    
                    # Extract coordinates based on geometry type
                    if geom_type == 'Polygon' and coords:
                        for ring in coords:
                            all_coords.extend(ring)
                    elif geom_type == 'LineString' and coords:
                        all_coords.extend(coords)
                    elif geom_type == 'Point' and coords:
                        all_coords.append(coords)
                    
                    all_geometries.append({'type': geom_type, 'coords': coords})
        except Exception:
            pass
    
    if not all_coords:
        return None
    
    coords_array = np.array(all_coords)
    
    return {
        'coordinates': coords_array,
        'geometries': all_geometries,
        'bounds': {
            'min_x': coords_array[:, 0].min(),
            'max_x': coords_array[:, 0].max(),
            'min_y': coords_array[:, 1].min(),
            'max_y': coords_array[:, 1].max()
        },
        'center': {
            'x': coords_array[:, 0].mean(),
            'y': coords_array[:, 1].mean()
        }
    }
