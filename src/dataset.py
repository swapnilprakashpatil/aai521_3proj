"""
PyTorch Dataset and DataLoader for Flood Detection
Handles loading preprocessed patches with augmentation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import pickle
from typing import Dict, List, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FloodDataset(Dataset):
    """Dataset for flood detection segmentation task"""
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        num_classes: int = 7,
        use_augmentation: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Path to processed data directory
            split: 'train', 'val', or 'test'
            transform: Albumentations transform pipeline
            num_classes: Number of segmentation classes
            use_augmentation: Whether to apply augmentation (only for training)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_classes = num_classes
        self.use_augmentation = use_augmentation and (split == 'train')
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata' / f'{split}_metadata.pkl'
        if not metadata_path.exists():
            # Fallback to JSON
            metadata_path = self.data_dir / 'metadata' / f'{split}_metadata.json'
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        
        # Set transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
        
        print(f"Loaded {split} dataset: {len(self.metadata)} samples")
        
        # Calculate class distribution
        self._calculate_class_distribution()
    
    def _get_default_transform(self) -> A.Compose:
        """Get default transforms based on split"""
        if self.use_augmentation:
            # Training augmentation
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5
                ),
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
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.485, 0.456, 0.485, 0.456],
                           std=[0.229, 0.224, 0.229, 0.224, 0.229, 0.224]),
                ToTensorV2()
            ])
        else:
            # Validation/Test: only normalization
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.485, 0.456, 0.485, 0.456],
                           std=[0.229, 0.224, 0.229, 0.224, 0.229, 0.224]),
                ToTensorV2()
            ])
    
    def _calculate_class_distribution(self):
        """Calculate class distribution across dataset"""
        self.class_counts = np.zeros(self.num_classes, dtype=np.int64)
        
        for item in self.metadata:
            class_dist = item.get('class_distribution', {})
            for cls, count in class_dist.items():
                self.class_counts[int(cls)] += count
        
        # Calculate class weights (inverse frequency)
        total_pixels = self.class_counts.sum()
        self.class_weights = np.zeros(self.num_classes, dtype=np.float32)
        
        for i in range(self.num_classes):
            if self.class_counts[i] > 0:
                # Inverse frequency normalized
                self.class_weights[i] = total_pixels / (self.num_classes * self.class_counts[i])
            else:
                self.class_weights[i] = 0.0
        
        print(f"\nClass distribution ({self.split}):")
        for i in range(self.num_classes):
            pct = (self.class_counts[i] / total_pixels) * 100 if total_pixels > 0 else 0
            print(f"  Class {i}: {self.class_counts[i]:,} pixels ({pct:.2f}%), weight: {self.class_weights[i]:.4f}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dictionary with 'image', 'mask', 'patch_id', 'is_flood_positive'
        """
        item = self.metadata[idx]
        
        # Load image and mask
        image_path = self.data_dir.parent / item['image_path']
        mask_path = self.data_dir.parent / item['mask_path']
        
        image = np.load(image_path).astype(np.float32)  # Shape: (H, W, 6)
        mask = np.load(mask_path).astype(np.int64)      # Shape: (H, W)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask,
            'patch_id': item['patch_id'],
            'is_flood_positive': item.get('is_flood_positive', False),
            'region': item.get('region', 'unknown')
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss function"""
        return torch.from_numpy(self.class_weights).float()


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_transform: Optional[A.Compose] = None,
    val_transform: Optional[A.Compose] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        test_dir: Path to test data directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        train_transform: Custom transform for training
        val_transform: Custom transform for validation/test
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = FloodDataset(
        train_dir,
        split='train',
        transform=train_transform,
        use_augmentation=True
    )
    
    val_dataset = FloodDataset(
        val_dir,
        split='val',
        transform=val_transform,
        use_augmentation=False
    )
    
    test_dataset = FloodDataset(
        test_dir,
        split='test',
        transform=val_transform,
        use_augmentation=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for batch norm stability
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from config import PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR, PROCESSED_TEST_DIR
    
    print("Testing FloodDataset...")
    
    # Create dataset
    dataset = FloodDataset(PROCESSED_TRAIN_DIR, split='train')
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Class weights: {dataset.get_class_weights()}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Patch ID: {sample['patch_id']}")
    print(f"  Is flood positive: {sample['is_flood_positive']}")
    print(f"  Region: {sample['region']}")
    
    # Test dataloaders
    print("\n" + "="*60)
    print("Testing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        PROCESSED_TRAIN_DIR,
        PROCESSED_VAL_DIR,
        PROCESSED_TEST_DIR,
        batch_size=4,
        num_workers=0  # Use 0 for testing
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Masks shape: {batch['mask'].shape}")
    print(f"  Batch size: {len(batch['patch_id'])}")
