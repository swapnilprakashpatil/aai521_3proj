"""
Phase III: Process Test Data (Louisiana-West)
Separate script to process test data independently from training data
"""

import numpy as np
from pathlib import Path
import json

from config import (
    TEST_PATH, PROCESSED_TEST_DIR
)
from run_preprocessing import DataPreprocessor

def main():
    """Process test data from Louisiana-West_Test_Public"""
    
    print(f"\n{'='*80}")
    print("PHASE III: PROCESSING TEST DATA (Louisiana-West)")
    print(f"{'='*80}")
    
    # Find Louisiana-West test directory
    test_regions = []
    for item in TEST_PATH.iterdir():
        if item.is_dir() and 'Test_Public' in item.name:
            test_regions.append((item, item.name))
    
    if not test_regions:
        print("\nNo test regions found in:", TEST_PATH)
        print("Skipping test data processing.")
        return
    
    print(f"\nDiscovered {len(test_regions)} test region(s):")
    for region_path, region_name in test_regions:
        print(f"  - {region_name}: {region_path}")
    
    # Process test regions
    all_test_metadata = []
    preprocessor_test = DataPreprocessor(output_dir=PROCESSED_TEST_DIR)
    
    for region_path, region_name in test_regions:
        region_metadata = preprocessor_test.process_region(
            region_path,
            region_name
        )
        all_test_metadata.extend(region_metadata)
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST DATA PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total tiles processed: {preprocessor_test.stats['processed_tiles']}")
    print(f"Total tiles failed: {preprocessor_test.stats['failed_tiles']}")
    print(f"Quality check failures: {preprocessor_test.stats['quality_failed']}")
    print(f"Total patches extracted: {preprocessor_test.stats['total_patches']}")
    print(f"Flood-positive patches: {preprocessor_test.stats['flood_positive_patches']}")
    
    if preprocessor_test.stats['total_patches'] > 0:
        flood_ratio = preprocessor_test.stats['flood_positive_patches'] / preprocessor_test.stats['total_patches']
        print(f"Flood ratio: {flood_ratio*100:.2f}%")
    
    # Save test metadata
    print(f"\nSaving test metadata...")
    preprocessor_test.save_metadata(all_test_metadata, 'test')
    
    print(f"\n{'='*80}")
    print("PHASE III COMPLETE!")
    print(f"{'='*80}")
    print(f"\nProcessed test data saved to: {PROCESSED_TEST_DIR}")
    print(f"  - Patches: {len(all_test_metadata)}")
    if all_test_metadata:
        print(f"  - Flood-positive: {sum(1 for m in all_test_metadata if m['is_flood_positive'])}")
    
    return all_test_metadata


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    
    main()
