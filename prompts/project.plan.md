# SpaceNet 8 Flood Detection – Production-Ready AI Project Plan

> **Optimized for Google Colab | Cloud-First Architecture | MLOps Best Practices**

---

## 1. Executive Summary & Project Objectives

### 1.1 Primary Goals

Build a **production-grade, cloud-native Computer Vision system** that:

- Performs **multiclass semantic segmentation** on SpaceNet 8 pre-flood and post-flood satellite imagery
- Provides **quantitative flood impact assessment** with interpretable visualizations and metrics
- Demonstrates **GenAI enhancement pipeline** (super-resolution/denoising) with A/B testing methodology
- Delivers an **interactive deployment** via Streamlit UI with optional REST API endpoints
- Follows **MLOps best practices** including experiment tracking, model versioning, and reproducibility

### 1.2 Technical Constraints & Design Principles

**Environment:**

- Primary development platform: **Google Colab** (GPU-accelerated notebooks)
- Cloud storage: Google Drive integration for datasets and checkpoints
- Version control: Git with DVC for data versioning
- Secondary environment: Local development with VS Code

**Architectural Principles:**

- **Separation of Concerns**: Clear boundaries between data, models, training, and inference
- **SOLID Principles**: Single responsibility, dependency injection, interface segregation
- **DRY (Don't Repeat Yourself)**: Shared utilities and reusable components
- **Configuration as Code**: Externalized configs with Hydra/OmegaConf
- **Fail-Fast**: Input validation, type hints, and comprehensive error handling
- **Testability**: Unit tests, integration tests, and continuous validation

**Scalability Considerations:**

- Modular pipeline allowing easy model swapping
- Configurable batch processing for large-scale inference
- Efficient data loading with caching and prefetching
- Cloud-ready architecture supporting distributed training (future)

---

## 2. High‑Level Architecture

### 2.1 Components

1. **Data & Notebooks**

   - `01_eda.ipynb` – lightweight exploratory data analysis and visualization.
   - `02_preprocessing.ipynb` – dataset construction, patching, and dataloaders.
   - `03_train_model.ipynb` – model training runs.
   - `04_evaluate_visualize.ipynb` – evaluation metrics and visual comparisons.
   - `05_genai_experiment.ipynb` – GenAI enhancement experiment (raw vs enhanced inference).

2. **Python Package (`src/`)**

   - `dataset.py` – PyTorch dataset and helper functions to load tiles.
   - `transforms.py` – augmentation and preprocessing pipeline.
   - `models.py` – segmentation model factory (U‑Net++ as primary model).
   - `train.py` – training loop and metric utilities.
   - `infer.py` – inference helpers for validation, test, and UI.
   - `visualization.py` – helpers for overlays and comparison grids.
   - `genai_enhance.py` – image enhancement wrapper for a single GenAI operation.

3. **UI Layer**
   - **Primary**: `ui/streamlit_app.py` (simple, Python‑only, perfect for a quick demo).
   - **Optional later**: Angular frontend + Python API (FastAPI) as a more “product‑like” extension.

---

## 3. Repository Structure (Colab-Optimized)

```text
spacenet8_flood/
├── .github/
│   └── workflows/              # CI/CD pipelines
│       └── tests.yml
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── model/
│   │   ├── unetpp.yaml
│   │   └── deeplabv3plus.yaml
│   ├── data/
│   │   ├── spacenet8.yaml
│   │   └── augmentation.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── fast_dev.yaml
│   └── experiment/
│       └── genai_enhancement.yaml
├── data/                       # Data directory (gitignored, Colab mounts here)
│   ├── raw/                   # Original SpaceNet8 data
│   ├── processed/             # Preprocessed patches
│   ├── external/              # External datasets
│   └── interim/               # Intermediate data
├── notebooks/                  # Google Colab notebooks
│   ├── 00_colab_setup.ipynb   # **Colab environment setup**
│   ├── 01_eda.ipynb
│   ├── 02_data_pipeline.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_genai_enhancement.ipynb
├── src/                        # Core Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   ├── data_loader.py
│   │   ├── geospatial_utils.py
│   │   └── augmentation_policies.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── unet_plusplus.py
│   │   ├── segmentation_models.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   ├── optimizers.py
│   │   ├── schedulers.py
│   │   └── experiment_tracker.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── postprocessing.py
│   │   ├── ensemble.py
│   │   └── tta.py
│   ├── enhancement/
│   │   ├── __init__.py
│   │   ├── base_enhancer.py
│   │   ├── super_resolution.py
│   │   ├── denoising.py
│   │   └── enhancement_pipeline.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       ├── visualization.py
│       ├── metrics_calculator.py
│       ├── io_utils.py
│       └── colab_utils.py        # **Colab helpers**
├── api/                         # FastAPI backend (optional)
│   ├── __init__.py
│   ├── main.py
│   ├── routers/
│   ├── schemas/
│   └── dependencies.py
├── ui/                          # Streamlit frontend
│   ├── streamlit_app.py
│   ├── pages/
│   ├── components/
│   └── utils.py
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   ├── test_training/
│   └── test_inference/
├── scripts/                     # Utility scripts
│   ├── download_data.py
│   ├── setup_colab.sh          # **Colab setup script**
│   ├── train.py                # CLI training script
│   └── evaluate.py             # CLI evaluation script
├── outputs/                     # Experiment outputs (gitignored)
│   ├── logs/
│   ├── checkpoints/
│   ├── predictions/
│   └── visualizations/
├── reports/                     # Documentation and results
│   ├── figures/
│   ├── metrics/
│   └── presentations/
├── .gitignore
├── .dvcignore
├── .env.example                # Environment variables template
├── pyproject.toml              # Modern Python packaging
├── setup.py                    # Package installation
├── requirements.txt            # Core dependencies
├── requirements-dev.txt        # Development dependencies
├── requirements-colab.txt      # **Colab-specific dependencies**
├── Dockerfile                  # Container deployment
├── docker-compose.yml
├── README.md
└── LICENSE
```

### 3.1 Key Architectural Decisions

**Colab Integration:**

- `notebooks/00_colab_setup.ipynb` handles all environment initialization
- `src/utils/colab_utils.py` provides Colab-specific helpers (Drive mounting, GPU checks)
- `requirements-colab.txt` contains Colab-optimized dependencies
- Data paths use environment variables for flexibility (`COLAB_DATA_PATH`, `LOCAL_DATA_PATH`)

**Configuration Management:**

- Hydra for hierarchical configs
- Environment-specific overrides
- Experiment tracking integration

**MLOps Foundation:**

- DVC for data versioning
- Weights & Biases for experiment tracking
- Structured logging with context
- Automated testing pipeline

---

## 4. Environment & Dependencies (Colab-Optimized)

### 4.1 Dependency Management Strategy

**Three-tier dependency structure:**

1. **`requirements.txt`** - Core production dependencies
2. **`requirements-dev.txt`** - Development tools (pytest, black, mypy, etc.)
3. **`requirements-colab.txt`** - Colab-specific optimizations

### 4.2 Core Dependencies (`requirements.txt`)

```ini
# Deep Learning Framework
torch>=2.1.0,<2.3.0
torchvision>=0.16.0,<0.18.0
torch-summary>=1.4.5

# Segmentation Models
segmentation-models-pytorch>=0.3.3
timm>=0.9.12  # For additional backbones

# Data Processing
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
scikit-image>=0.21.0

# Augmentation
albumentations>=1.3.1
imgaug>=0.4.0

# Geospatial
rasterio>=1.3.9
geopandas>=0.14.0
shapely>=2.0.0
pyproj>=3.6.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
folium>=0.14.0  # Interactive maps

# Configuration & Logging
hydra-core>=1.3.2
omegaconf>=2.3.0
python-dotenv>=1.0.0
pyyaml>=6.0

# MLOps & Tracking
wandb>=0.16.0
tensorboard>=2.15.0
mlflow>=2.9.0

# Utils
tqdm>=4.66.0
rich>=13.7.0  # Beautiful terminal output
typer>=0.9.0  # CLI framework

# API & UI
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.29.0
pydantic>=2.5.0

# GenAI Enhancement
diffusers>=0.24.0
transformers>=4.36.0
```

### 4.3 Colab-Specific Dependencies (`requirements-colab.txt`)

```ini
# Include all core requirements
-r requirements.txt

# Colab-specific optimizations
google-colab>=1.0.0
gdown>=4.7.1  # Download from Google Drive

# Pre-installed in Colab, but pinning versions
ipywidgets>=8.1.0
jupyter>=1.0.0

# Additional Colab utilities
pyngrok>=7.0.0  # For tunneling (optional)
```

### 4.4 Development Dependencies (`requirements-dev.txt`)

```ini
# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
hypothesis>=6.92.0

# Code Quality
black>=23.12.0
isort>=5.13.0
flake8>=6.1.0
mypy>=1.7.0
pylint>=3.0.0

# Pre-commit hooks
pre-commit>=3.6.0

# Documentation
sphinx>=7.2.0
sphinx-rtd-theme>=2.0.0
mkdocs>=1.5.0
mkdocs-material>=9.5.0

# Profiling
line-profiler>=4.1.0
memory-profiler>=0.61.0
```

### 4.5 Colab Environment Setup (`notebooks/00_colab_setup.ipynb`)

**Key initialization steps:**

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Set workspace path
import os
WORKSPACE = '/content/drive/MyDrive/spacenet8_flood'
os.chdir(WORKSPACE)

# 3. Verify GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# 4. Install dependencies
!pip install -q -r requirements-colab.txt

# 5. Install project as package
!pip install -q -e .

# 6. Initialize Weights & Biases
import wandb
wandb.login()

# 7. Environment validation
from src.utils.colab_utils import validate_environment
validate_environment()
```

### 4.6 Environment Validation Utility

**`src/utils/colab_utils.py`:**

```python
"""Google Colab-specific utilities."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict
import torch


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = '/content/drive') -> Path:
    """Mount Google Drive if in Colab."""
    if is_colab():
        from google.colab import drive
        drive.mount(mount_point, force_remount=False)
        return Path(mount_point)
    return Path.cwd()


def get_gpu_info() -> Dict[str, any]:
    """Get GPU information."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info['cuda_available']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9
        info['memory_reserved'] = torch.cuda.memory_reserved(0) / 1e9
        
    return info


def validate_environment() -> None:
    """Validate the complete environment setup."""
    print("=" * 50)
    print("Environment Validation")
    print("=" * 50)
    
    # Check Python version
    print(f"Python: {sys.version.split()[0]}")
    
    # Check GPU
    gpu_info = get_gpu_info()
    if gpu_info['cuda_available']:
        print(f"✓ GPU: {gpu_info['device_name']}")
        print(f"  CUDA: {gpu_info['cuda_version']}")
        print(f"  Memory: {gpu_info['memory_reserved']:.2f} GB reserved")
    else:
        print("✗ No GPU available")
    
    # Check key dependencies
    dependencies = [
        'torch', 'torchvision', 'segmentation_models_pytorch',
        'albumentations', 'rasterio', 'wandb', 'hydra'
    ]
    
    print("\nDependencies:")
    for dep in dependencies:
        try:
            mod = __import__(dep)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {dep}: {version}")
        except ImportError:
            print(f"  ✗ {dep}: NOT INSTALLED")
    
    # Check data directory
    data_dir = Path(os.getenv('DATA_DIR', './data'))
    if data_dir.exists():
        print(f"\n✓ Data directory: {data_dir}")
    else:
        print(f"\n✗ Data directory not found: {data_dir}")
    
    print("=" * 50)
```

---

## 5. Lightweight EDA & Visualization (`01_eda.ipynb`)

### 5.1 Load and Inspect Samples

Goals:

- Understand shapes, channels, and label classes.
- See a few pre‑flood and post‑flood examples with their masks.

Actions:

1. Implement a small helper function (even inline in the notebook) using `rasterio` to:
   - Read pre‑event and post‑event `.tif` images.
   - Read corresponding segmentation mask (classes like water, flooded water, buildings, etc.).
2. Print for a handful of tiles:
   - Image shape (height, width, channels).
   - Data type (e.g., `uint8`).
   - Unique values in mask (class IDs).

### 5.2 Class Distribution Snapshot

For a limited subset (for speed):

- Accumulate pixel counts per class.
- Convert counts to percentages.
- Plot a simple bar chart showing:
  - Background
  - Water
  - Flooded water
  - Building
  - Road
  - Any other relevant classes

This gives a quick sense of class imbalance (typically flood classes are relatively rare).

### 5.3 Pre vs Post Flood Visualization

For a handful of tiles:

- Show a 2×2 grid:
  - Pre‑event image (RGB)
  - Post‑event image (RGB)
  - Ground‑truth mask (color‑coded)
  - Overlay: mask over the post‑event image

Also:

- Compute a difference image:
  - Convert pre and post to float
  - `diff = abs(post - pre)`
  - Display as a heatmap to highlight large changes.

Add a short markdown explanation summarizing what you see visually (more water, submerged areas, changes in brightness, etc.).

---

## 6. Preprocessing & Dataset Construction (`02_preprocessing.ipynb` and `src/dataset.py`)

### 6.1 Patch Strategy

Decide a usable patch size:

- For example, `512 × 512` pixels.
- If the tiles are already near that size, you may:
  - Use them as single patches, or
  - Split larger tiles into a grid of patches.

Store for each training sample:

- Path to pre‑event image.
- Path to post‑event image.
- Path to segmentation mask.

### 6.2 PyTorch Dataset (`src/dataset.py`)

Implement a `Dataset` subclass, for example:

- `FloodSegmentationDataset(Dataset)`:

  - In `__init__`:
    - Accept a list of items (each with paths to pre, post, mask).
    - Accept an Albumentations transform (optional).
  - In `__getitem__`:
    - Load pre and post images.
    - Stack them along channel dimension: 3 (pre) + 3 (post) = 6 channels.
    - Load mask as an integer label map.
    - Apply transform to image and mask (if provided).
    - Return `(image_tensor, mask_tensor)`.

Use Copilot to scaffold this class from a short docstring.

### 6.3 Augmentations (`src/transforms.py`)

Keep augmentations simple but meaningful:

- Random horizontal and vertical flip.
- Random rotation by 90‑degree steps.
- Random brightness and contrast adjustment.
- Small Gaussian noise.

Use Albumentations’ `Compose` for image+mask joint transforms.

### 6.4 Train / Validation Split

Create a Python script or notebook cell to:

- Build a list of tile items for training.
- Split tiles into:
  - Training set (for model learning).
  - Validation set (for model and hyperparameter selection).
- Keep the test set separate (SpaceNet test tiles).

Create PyTorch `DataLoader`s for train and validation in `02_preprocessing.ipynb` to confirm everything works.

---

## 7. Model Training – One Strong Baseline (`03_train_model.ipynb` + `src/models.py` + `src/train.py`)

### 7.1 Model Choice

Use **U‑Net++** from Segmentation Models PyTorch as the primary model:

- Encoder (backbone): `resnet34` or `efficientnet-b4` (pretrained on ImageNet).
- Decoder: U‑Net++.
- Input channels: `6` (Pre RGB + Post RGB stacked).
- Output classes: number of segmentation classes in your masks.

In `src/models.py`:

- Implement a function such as:

  - `get_model(model_name: str, in_channels: int, num_classes: int) -> nn.Module`
  - Start with a case for `"unetpp_resnet34"`.

Optionally (only if there is extra time):

- Add a second model: for example, DeepLabV3+ with a ResNet‑50 encoder, configured similarly.

### 7.2 Loss Functions & Optimizer

Use a simple but robust combination:

- **Cross‑Entropy Loss**:
  - Standard pixel‑wise multi‑class classification loss.
- **Dice Loss**:
  - Measures overlap between predicted regions and ground truth, helping with class imbalance.

Combine them as a weighted sum.

Use `AdamW` optimizer:

- Learning rate around `1e-4` to `3e-4`.
- Weight decay set to a small value (for example, `1e-4`).

Optionally use `ReduceLROnPlateau` scheduler on validation loss.

### 7.3 Training Utilities (`src/train.py`)

Implement:

- `train_one_epoch(model, dataloader, optimizer, criterion, device)`:
  - Standard PyTorch training loop.
- `validate(model, dataloader, criterion, device, num_classes)`:
  - Loop over validation data.
  - Compute validation loss and metrics.
- `compute_iou(preds, targets, num_classes)`:
  - Convert predictions to class labels (argmax).
  - Compute Intersection over Union (IoU) per class and mean IoU.

Let Copilot help by generating these patterns from inline comments.

### 7.4 Training in the Notebook

In `03_train_model.ipynb`:

1. Import dataset, model, and training utilities from `src`.
2. Create train and validation loaders with reasonable batch size (depending on GPU / CPU resources).
3. Instantiate the model with `get_model`.
4. Instantiate optimizer and combined loss function.
5. Implement training loop:

   - For each epoch:
     - Call `train_one_epoch`.
     - Call `validate`.
     - Log:
       - Training loss
       - Validation loss
       - Validation mean IoU
     - Save `checkpoints/best_unetpp.pth` when validation IoU improves.

6. Plot:
   - Training and validation loss curves across epochs.
   - Validation mean IoU curve.

---

## 8. Evaluation & Visualization (`04_evaluate_visualize.ipynb` + `src/infer.py` + `src/visualization.py`)

### 8.1 Inference Helpers (`src/infer.py`)

Implement:

- `load_model_for_inference(checkpoint_path, in_channels, num_classes, device)`:
  - Creates the model.
  - Loads weights from checkpoint.
  - Moves model to correct device.
- `predict_batch(model, images, device)`:
  - Runs a forward pass on a batch of images.
  - Returns predicted label maps (`argmax` over logits).

### 8.2 Metrics on Validation / Test

In `04_evaluate_visualize.ipynb`:

1. Load the best model checkpoint.
2. Run it on the validation (and/or test) dataloader.
3. Compute:
   - IoU per class.
   - Mean IoU across classes.
   - Optionally, per‑class Precision, Recall, F1‑score using simple helper functions.

Summarize metrics in a small table for quick reading.

### 8.3 Visual Comparison Grids (`src/visualization.py`)

Implement a helper such as:

- `plot_tile_comparison(pre_img, post_img, mask_gt, mask_pred, class_colors)`:
  - Builds a grid, for example:
    - Pre‑event image
    - Post‑event image
    - Ground truth mask (colored)
    - Prediction mask (colored)
    - Optional: error map (pixels where prediction ≠ ground truth)

In the evaluation notebook:

- Select about 10–15 tiles:
  - Some with heavy flooding
  - Some with minimal flooding
  - Some urban, some rural
- Use the helper to visualize each tile.
- Save a few comparison figures to `reports/figures/` for presentation.

---

## 9. Generative AI Enhancement Experiment (`05_genai_experiment.ipynb` + `src/genai_enhance.py`)

### 9.1 Scope of GenAI Experiment

Define a clear, small question:

> “If we enhance post‑event images using a super‑resolution or denoising model before feeding them to the segmentation model, does prediction quality improve on selected tiles?”

Important constraints:

- Do **not** retrain the segmentation model for this experiment.
- Only:
  - Enhance images at inference time.
  - Compare predictions and metrics for:
    - Raw images.
    - Enhanced images.

### 9.2 Enhancement Wrapper (`src/genai_enhance.py`)

Choose one enhancement type:

- **Super‑resolution**:
  - For example, a light‑weight Real‑ESRGAN‑like model or another SR implementation.
- **Denoising**:
  - A simple denoiser or light restoration model.

Implement:

- `enhance_image(image: np.ndarray, method: str = "sr") -> np.ndarray`:
  - Takes a single RGB image (for example, the post‑event image).
  - Applies the enhancement model.
  - Returns the enhanced image as a NumPy array.

You can start with just the post‑event image and optionally enhance the pre‑event image for consistency.

### 9.3 Raw vs Enhanced Inference (`05_genai_experiment.ipynb`)

In the GenAI experiment notebook:

1. Load the trained segmentation model (U‑Net++) and device.
2. Choose a subset of test tiles (for example, 5–10 visually interesting cases).
3. For each tile:

   - **Raw pipeline**:
     - Create input tensor from raw pre and post images (stacked).
     - Get predictions using the trained model.
     - Compute tile‑level IoU and at least flood‑class IoU.
   - **Enhanced pipeline**:
     - Enhance the post‑event image (and optionally the pre‑event image).
     - Build input tensor from enhanced images.
     - Run the model again.
     - Compute IoU metrics again.

4. Visualize for each tile:

   - Pre‑event image
   - Post‑event image (raw)
   - Post‑event image (enhanced)
   - Prediction on raw pipeline
   - Prediction on enhanced pipeline

5. Summarize in a simple table:

```text
Tile ID   IoU (Raw)   IoU (Enhanced)
------------------------------------
tile_1    0.xx        0.yy
tile_2    0.xx        0.yy
...
```

6. Add 2–3 sentences of interpretation in markdown:
   - Did enhancement help the segmentation?
   - Did it remain neutral?
   - Did it introduce artifacts that hurt performance?

---

## 10. Simple UI with Streamlit (`ui/streamlit_app.py`)

### 10.1 General Structure

Use Streamlit to create a quick interactive app with:

- A sidebar for navigation and choices.
- A main area for images, masks, and metrics.

Use Streamlit caching:

- `st.cache_resource` to load the model once.
- `st.cache_data` for loading images and masks.

### 10.2 Page: Overview

Content:

- Title: “SpaceNet 8 Flood Detection Demo”.
- Short explanation of:
  - What SpaceNet 8 is.
  - Pre‑flood vs post‑flood imagery.
  - Multiclass segmentation and flood detection.
- Optionally show one static comparison figure.

### 10.3 Page: Tile Explorer

Sidebar controls:

- Area of Interest (AOI) selector: Germany / Louisiana.
- Tile selector: dropdown or slider.
- Model selector (for now, just “U‑Net++ baseline”).

Main layout:

- First column:
  - Pre‑event image.
  - Ground‑truth segmentation mask (color‑coded).
- Second column:
  - Post‑event image.
  - Model prediction mask (color‑coded).

Optionally:

- Show an overlay of the prediction on the post‑event image.
- Display basic metrics for that tile (for example, IoU or F1 for the flooded class).

### 10.4 Page: GenAI Comparison

Sidebar controls:

- AOI and tile selectors (reuse from Tile Explorer).
- Enhancement option: `Raw` vs `Enhanced`.

Main layout:

- Column 1 (Raw):
  - Post‑event image (raw).
  - Prediction on raw image pair.
  - Tile‑level IoU for that prediction.
- Column 2 (Enhanced):
  - Enhanced post‑event image.
  - Prediction on enhanced input.
  - Tile‑level IoU for enhanced prediction.

Below the columns:

- A short textual summary (even dynamically generated) such as:
  - “For this tile, IoU changed from X to Y after enhancement.”

---

## 11. Optional Stretch Enhancements

If time remains:

- Add a second segmentation model (for example, DeepLabV3+), and a small comparison:
  - Show metrics and visuals from both models.
- Add a small chart (for example, `st.bar_chart`) summarizing mean IoU for:
  - Raw vs Enhanced inference.
- Add a Grad‑CAM‑style visualization if your backbone supports it with minimal effort.

---

## 12. Final Deliverables Checklist

- **Code and Models**

  - Trained U‑Net++ model checkpoint in `checkpoints/best_unetpp.pth`.
  - `src/` directory with:
    - `dataset.py` – dataset class.
    - `transforms.py` – augmentations and preprocessing.
    - `models.py` – model factory.
    - `train.py` – training logic.
    - `infer.py` – inference helpers.
    - `visualization.py` – plotting utilities.
    - `genai_enhance.py` – image enhancement wrapper.

- **Notebooks**

  - `01_eda.ipynb` – initial data understanding.
  - `02_preprocessing.ipynb` – dataset and dataloader verification.
  - `03_train_model.ipynb` – training run and learning curves.
  - `04_evaluate_visualize.ipynb` – metrics and visual comparisons.
  - `05_genai_experiment.ipynb` – raw vs enhanced inference experiment.

- **UI**

  - `ui/streamlit_app.py` – interactive demo with:
    - Tile Explorer page.
    - GenAI Comparison page.

- **Documentation**
  - `README.md` with:
    - Environment setup instructions.
    - How to train or reuse the provided model.
    - How to run evaluation notebooks.
    - How to start the Streamlit app:
      - `streamlit run ui/streamlit_app.py`
  - A few exported PNG figures in `reports/figures/` for use in slides or brief written summary.

This compact plan keeps the project realistic for a short build, while still showcasing:

- A modern segmentation model,
- Meaningful quantitative and qualitative evaluation,
- And a clear, focused GenAI enhancement story presented through an interactive UI.
