"""
Data loading utilities for flood detection dataset
Handles image loading, normalization, and mask generation from GeoJSON annotations
"""

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json
from shapely.geometry import shape, mapping
from rasterio.features import rasterize
import cv2
import warnings

# Suppress rasterio warnings
warnings.filterwarnings('ignore', message='.*not georeferenced.*')


class DatasetLoader:
    """Load and process satellite imagery and annotations"""
    
    def __init__(self, region_path: Path, region_name: str):
        """
        Initialize dataset loader
        
        Args:
            region_path: Path to region directory (e.g., Germany_Training_Public)
            region_name: Name of the region (Germany, Louisiana-East, etc.)
        """
        self.region_path = Path(region_path)
        self.region_name = region_name
        
        # Find CSV files
        mapping_files = list(self.region_path.glob('*_label_image_mapping.csv'))
        reference_files = list(self.region_path.glob('*_reference.csv'))
        
        self.mapping_csv = mapping_files[0] if mapping_files else None
        self.reference_csv = reference_files[0] if reference_files else None
        
        # Load mappings
        self.mapping_df = None
        self.reference_df = None
        if self.mapping_csv:
            self.mapping_df = pd.read_csv(self.mapping_csv)
        if self.reference_csv:
            self.reference_df = pd.read_csv(self.reference_csv)
            # Convert length_m to numeric if it exists
            if 'length_m' in self.reference_df.columns:
                self.reference_df['length_m'] = pd.to_numeric(self.reference_df['length_m'], errors='coerce')
            # Convert Flooded column from string to boolean
            if 'Flooded' in self.reference_df.columns:
                self.reference_df['Flooded'] = self.reference_df['Flooded'].map({
                    'True': True, 
                    'False': False, 
                    'Null': None,
                    True: True,
                    False: False
                })
            
        # Directories
        self.pre_event_dir = self.region_path / 'PRE-event'
        self.post_event_dir = self.region_path / 'POST-event'
        self.annotations_dir = self.region_path / 'annotations'
        
    def get_tile_list(self) -> List[str]:
        """Get list of all tiles (annotation filenames)"""
        if self.mapping_df is None:
            return []
        return self.mapping_df['label'].tolist()
    
    def get_tile_info(self, tile_name: str) -> Dict:
        """
        Get information about a specific tile
        
        Args:
            tile_name: Name of the annotation file (e.g., '0_15_63.geojson')
            
        Returns:
            Dictionary with pre-event, post-event images, and annotation path
        """
        if self.mapping_df is None:
            return {}
            
        row = self.mapping_df[self.mapping_df['label'] == tile_name]
        if len(row) == 0:
            return {}
            
        row = row.iloc[0]
        
        # Get pre-event image
        pre_image = self.pre_event_dir / row['pre-event image']
        
        # Get post-event images (can be multiple)
        post_images = []
        if pd.notna(row['post-event image 1']):
            post_images.append(self.post_event_dir / row['post-event image 1'])
        if 'post-event image 2' in row.index and pd.notna(row['post-event image 2']):
            post_images.append(self.post_event_dir / row['post-event image 2'])
            
        # Annotation path
        annotation_path = self.annotations_dir / tile_name
        
        return {
            'tile_name': tile_name,
            'pre_image': pre_image,
            'post_images': post_images,
            'annotation': annotation_path,
            'region': self.region_name
        }
    
    def load_image(self, image_path: Path, normalize: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Load a satellite image with proper normalization
        
        Args:
            image_path: Path to the image file
            normalize: Whether to normalize to 0-1 range
            
        Returns:
            Tuple of (image array, metadata dict)
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()  # Shape: (C, H, W)
            
            # Get metadata
            metadata = {
                'dtype': image.dtype,
                'shape': image.shape,
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds,
                'filename': image_path.name  # Add original filename
            }
            
            # Transpose to (H, W, C)
            image = np.transpose(image, (1, 2, 0))
            
            # Handle different dtypes
            if normalize:
                if image.dtype == np.uint16:
                    # Normalize uint16 to 0-1
                    image = image.astype(np.float32) / 65535.0
                elif image.dtype == np.uint8:
                    # Normalize uint8 to 0-1
                    image = image.astype(np.float32) / 255.0
                else:
                    # Already float, clip to 0-1
                    image = np.clip(image.astype(np.float32), 0, 1)
            else:
                image = image.astype(np.float32)
                
            # Handle RGB (take first 3 channels if more exist)
            if image.shape[2] > 3:
                image = image[:, :, :3]
                
            return image, metadata
    
    def load_annotation(self, annotation_path: Path, image_shape: Tuple[int, int], transform=None) -> np.ndarray:
        """
        Load GeoJSON annotation and convert to segmentation mask
        
        Args:
            annotation_path: Path to GeoJSON file
            image_shape: Shape of the corresponding image (H, W)
            transform: Rasterio affine transform for coordinate conversion
            
        Returns:
            Segmentation mask array (H, W) with class labels
        """
        if not annotation_path.exists():
            # Return empty mask if no annotation
            return np.zeros(image_shape, dtype=np.uint8)
            
        # Load GeoJSON
        with open(annotation_path, 'r') as f:
            geojson_data = json.load(f)
            
        # Initialize mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Process features
        if 'features' in geojson_data:
            for feature in geojson_data['features']:
                # Get properties
                props = feature.get('properties', {})
                
                # Determine class based on properties
                # This needs to be adapted based on actual annotation format
                class_id = self._get_class_from_properties(props)
                
                # Get geometry
                geom = feature.get('geometry')
                if geom:
                    try:
                        # Convert to shapely geometry
                        shapely_geom = shape(geom)
                        
                        # Skip empty or invalid geometries
                        if shapely_geom.is_empty or not shapely_geom.is_valid:
                            continue
                        
                        # Convert geographic coordinates to pixel coordinates
                        def geo_to_pixel(coords_list, transform):
                            """Convert geographic coordinates to pixel coordinates"""
                            if transform is None:
                                # Fallback: assume coordinates are already in pixel space
                                return coords_list
                            
                            pixel_coords = []
                            for coord in coords_list:
                                lon, lat = coord[0], coord[1]
                                # Use inverse transform: ~transform * (lon, lat)
                                col, row = ~transform * (lon, lat)
                                pixel_coords.append([int(round(col)), int(round(row))])
                            return pixel_coords
                        
                        # Rasterize based on geometry type
                        if shapely_geom.geom_type == 'Polygon':
                            # For Polygons, get exterior coordinates
                            coords = list(shapely_geom.exterior.coords)
                            pixel_coords = geo_to_pixel(coords, transform)
                            if pixel_coords and len(pixel_coords) > 0:
                                coords_array = np.array(pixel_coords, dtype=np.int32)
                                cv2.fillPoly(mask, [coords_array], class_id)
                        elif shapely_geom.geom_type == 'LineString':
                            # For LineStrings, draw as polyline with thickness
                            coords = list(shapely_geom.coords)
                            pixel_coords = geo_to_pixel(coords, transform)
                            if pixel_coords and len(pixel_coords) > 1:
                                coords_array = np.array(pixel_coords, dtype=np.int32)
                                cv2.polylines(mask, [coords_array], False, class_id, thickness=2)
                        elif shapely_geom.geom_type == 'MultiPolygon':
                            # Handle MultiPolygon
                            for poly in shapely_geom.geoms:
                                coords = list(poly.exterior.coords)
                                pixel_coords = geo_to_pixel(coords, transform)
                                if pixel_coords and len(pixel_coords) > 0:
                                    coords_array = np.array(pixel_coords, dtype=np.int32)
                                    cv2.fillPoly(mask, [coords_array], class_id)
                    except Exception:
                        # Silently skip invalid geometries
                        continue
                        
        return mask
    
    def _get_class_from_properties(self, properties: Dict) -> int:
        """
        Determine class ID from GeoJSON properties
        
        Args:
            properties: Feature properties dictionary
            
        Returns:
            Class ID (integer)
        """
        # Check for flood status (handles "yes", "no", null, True, False)
        flooded_value = properties.get('flooded', properties.get('Flooded', None))
        is_flooded = flooded_value == 'yes' or flooded_value == True or flooded_value == 'True'
        
        # Check for object/feature type
        obj_type = str(properties.get('building', properties.get('highway', ''))).lower()
        
        # Map to class IDs based on flood status and object type
        # Class mapping: 0=background, 1=non-flooded-building, 2=flooded-building
        #                3=water, 4=flooded-water, 5=road
        
        if properties.get('building') and properties.get('building') != 'null':
            # It's a building
            if is_flooded:
                return 2  # flooded-building
            else:
                return 1  # non-flooded-building
        elif properties.get('highway') and properties.get('highway') != 'null':
            # It's a road
            if is_flooded:
                return 5  # flooded road
            else:
                return 6  # non-flooded road
        elif 'water' in obj_type:
            if is_flooded:
                return 4  # flooded-water
            else:
                return 3  # water
        else:
            # Default based on flood status
            if is_flooded:
                return 2  # flooded
            else:
                return 0  # background
                
        return 0  # background
    
    def get_flood_statistics(self) -> Dict:
        """
        Get flood statistics from reference CSV
        
        Returns:
            Dictionary with flood statistics
        """
        if self.reference_df is None:
            return {}
            
        stats = {
            'total_segments': len(self.reference_df),
            'flooded_count': (self.reference_df['Flooded'] == True).sum(),
            'non_flooded_count': (self.reference_df['Flooded'] == False).sum(),
            'null_count': self.reference_df['Flooded'].isna().sum(),
        }
        
        # Calculate percentages
        total = stats['total_segments']
        if total > 0:
            stats['flooded_pct'] = (stats['flooded_count'] / total) * 100
            stats['non_flooded_pct'] = (stats['non_flooded_count'] / total) * 100
            stats['null_pct'] = (stats['null_count'] / total) * 100
            
        # Road statistics
        if 'length_m' in self.reference_df.columns:
            flooded_roads = self.reference_df[self.reference_df['Flooded'] == True]
            stats['flooded_road_length_km'] = flooded_roads['length_m'].sum() / 1000
            stats['total_road_length_km'] = self.reference_df['length_m'].sum() / 1000
            
        return stats


def load_tile_data(region_path: Path, tile_name: str, region_name: str) -> Dict:
    """
    Convenience function to load all data for a single tile
    
    Args:
        region_path: Path to region directory
        tile_name: Name of the tile
        region_name: Name of the region
        
    Returns:
        Dictionary with pre_image, post_image, mask, and metadata
    """
    loader = DatasetLoader(region_path, region_name)
    tile_info = loader.get_tile_info(tile_name)
    
    if not tile_info:
        return {}
        
    # Load pre-event image
    pre_image, pre_meta = loader.load_image(tile_info['pre_image'])
    
    # Load first post-event image
    post_image, post_meta = None, None
    if tile_info['post_images']:
        post_image, post_meta = loader.load_image(tile_info['post_images'][0])
        
        # Ensure post-image matches pre-image size
        if post_image is not None and post_image.shape[:2] != pre_image.shape[:2]:
            post_image = cv2.resize(post_image, (pre_image.shape[1], pre_image.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
        
    # Load annotation mask
    mask = loader.load_annotation(tile_info['annotation'], pre_image.shape[:2], transform=pre_meta.get('transform'))
    
    return {
        'tile_name': tile_name,
        'region': region_name,
        'pre_image': pre_image,
        'post_image': post_image,
        'mask': mask,
        'pre_metadata': pre_meta,
        'post_metadata': post_meta
    }


if __name__ == "__main__":
    # Test the loader
    from config import GERMANY_TRAIN
    
    loader = DatasetLoader(GERMANY_TRAIN, 'Germany')
    print(f"Found {len(loader.get_tile_list())} tiles")
    
    stats = loader.get_flood_statistics()
    print(f"\nFlood Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
