# Phase 2 Preprocessing - Quick Start Guide

## Files Created

### Core Modules

1. **`src/config.py`** - Updated with preprocessing parameters
2. **`src/data_loader.py`** - Load images, masks, and CSV metadata
3. **`src/preprocessing.py`** - Image enhancement, quality checks, patch extraction
4. **`src/augmentation.py`** - Albumentations-based augmentation pipeline
5. **`src/run_preprocessing.py`** - Main preprocessing script

### Documentation

6. **`docs/02_preprocessing_summary.md`** - Detailed implementation summary
7. **`notebooks/02_preprocessing.ipynb`** - Interactive demonstration notebook

## Quick Start

### 1. Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run Preprocessing

```powershell
python src/run_preprocessing.py
```

**Expected Time:** 30-60 minutes  
**Output Location:** `dataset/processed/`

### 3. Verify Output

```powershell
# Check directory structure
Get-ChildItem -Recurse dataset/processed/ | Select-Object Name

# Expected structure:
# train/images/*.npy
# train/masks/*.npy
# train/metadata/*.json
# val/images/*.npy
# val/masks/*.npy
# val/metadata/*.json
# test/images/*.npy
# test/masks/*.npy
# test/metadata/*.json
# dataset_statistics.json
```

## Key Features Implemented

### Data Loading

- Handles uint8 and uint16 images
- Automatic normalization (0-1 range)
- GeoJSON annotation parsing
- Multi-temporal (pre/post) support

### Image Enhancement

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Percentile normalization
- Quality checks (valid pixels, clouds, shadows)

### Patch Extraction

- 512×512 patches with 128-pixel overlap
- Smart flood sampling (>100 flood pixels threshold)
- Oversampling of flood-positive patches (up to 3×)
- Class distribution tracking

### Class Imbalance Handling

- **Patch oversampling:** ~20% → ~40-50% flood patches
- **Class weights:** Background=1.0, Flooded=3.5, Flooded-water=4.0
- **Stratified splitting:** Balanced flood ratios across train/val/test

### Data Augmentation

- Geometric: flips, rotations, shift-scale-rotate
- Optical: brightness, contrast, hue-saturation-value
- Domain-specific: fog, shadows, water variations
- Dual-image augmentation for pre/post consistency

### Geo-Stratified Splitting

- Tile-level splitting (prevents spatial leakage)
- Train: 70%, Val: 15%, Test: 15%
- Maintains spatial relationships

### Quality Control

- Valid pixel ratio ≥ 50%
- Cloud coverage ≤ 30%
- Darkness detection
- Comprehensive metadata tracking

## Output Format

### Image Files (.npy)

- **Shape:** (512, 512, 6)
- **Channels:** [R_pre, G_pre, B_pre, R_post, G_post, B_post]
- **Dtype:** float32
- **Range:** 0-1 (normalized) or standardized

### Mask Files (.npy)

- **Shape:** (512, 512)
- **Dtype:** uint8
- **Classes:**
  - 0: background
  - 1: non-flooded buildings
  - 2: flooded buildings
  - 3: water
  - 4: flooded water
  - 5: roads

### Metadata Files

- **JSON:** Human-readable, complete metadata
- **Pickle:** Fast loading for Python
- **CSV:** Flattened data for inspection

## Configuration Parameters

Key settings in `src/config.py`:

```python
PATCH_SIZE = 512           # Patch dimensions
PATCH_OVERLAP = 128        # Overlap between patches
MIN_FLOOD_PIXELS = 100     # Flood-positive threshold

APPLY_CLAHE = True         # Enable CLAHE enhancement
CLAHE_CLIP_LIMIT = 2.0     # CLAHE contrast limit
CLAHE_TILE_GRID_SIZE = (8, 8)  # CLAHE tile size

CLASS_WEIGHTS = {          # For weighted loss
    0: 1.0,   # background
    1: 1.2,   # non-flooded
    2: 3.5,   # flooded
    3: 2.0,   # water
    4: 4.0,   # flooded-water
    5: 1.5    # roads
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

## Troubleshooting

### Memory Issues

- Reduce `PATCH_SIZE` to 384 or 256
- Process regions separately
- Lower batch size for statistics

### Slow Processing

- Reduce `PATCH_OVERLAP` to 64
- Disable CLAHE: `APPLY_CLAHE = False`
- Test on subset of tiles first

### Quality Check Too Strict

- Lower `MIN_VALID_PIXELS_RATIO = 0.3`
- Raise `MAX_CLOUD_COVERAGE = 0.5`

## Next Steps

After preprocessing is complete:

1. **Verify output** using validation notebook
2. **Calculate dataset statistics** (mean, std per channel)
3. **Implement PyTorch Dataset** in Phase 3
4. **Train models** (U-Net++, DeepLabV3+, SegFormer)

## Support

- **Detailed Documentation:** `docs/02_preprocessing_summary.md`
- **Interactive Notebook:** `notebooks/02_preprocessing.ipynb`
- **EDA Reference:** `docs/01_eda_summary_detailed.md`

---

**Status:** Ready for Phase 3 - Model Training
