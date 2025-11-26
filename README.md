# Flood Impact Assessment

Multi-Temporal Satellite Image Segmentation for Flood Detection

## Project Overview

This project implements an end-to-end computer vision pipeline for flood impact assessment using satellite imagery (Pre/Post event). The solution includes:

- Data exploration and preprocessing
- Multiple model architectures (U-Net++, DeepLabV3+, SegFormer)
- GenAI-based data enhancement (Real-ESRGAN, synthetic floods)
- Robustness evaluation
- Interactive Streamlit dashboard

## Quick Start

### Google Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -r requirements.txt
```

### Dataset Location

- **Google Drive Path**: `/content/drive/MyDrive/data/dataset`
- **Training Data**: `train/PRE-event`, `train/POST-event`, `train/annotations`
- **Test Data**: `test/PRE-event`, `test/POST-event`

## Project Structure

```
├── .tracker/          # Progress tracking
├── src/               # Modular Python code
├── notebooks/         # Jupyter notebooks (01-06)
├── ui/                # Streamlit dashboard
├── checkpoints/       # Model weights
├── results/           # Evaluation results
├── sample_outputs/    # Visualizations
├── config/            # Configuration files
└── data/              # Preprocessed datasets
```

## Workflow Steps

Follow `PROJECT_TRACKER.md` for detailed step-by-step execution:

1. **Step 1**: Exploratory Data & Visualization
2. **Step 2**: Data Preprocessing & Augmentation
3. **Step 3**: Model Architecture Portfolio
4. **Step 4**: Model Evaluation & Comparison
5. **Step 5**: Data Enhancement (GenAI) - MANDATORY
6. **Step 6**: Robustness & Advanced Evaluation
7. **Step 7**: UI & Streamlit Dashboard
8. **Step 8**: Final Validation & Documentation

## Success Criteria

- ✅ mIoU ≥ 0.55 (validation set)
- ✅ Flood IoU ≥ 0.60
- ✅ F1-score ≥ 0.50
- ✅ GenAI improvement ≥ +0.03 mIoU
- ✅ All code committed with clear messages
- ✅ Streamlit UI functional

## Key Technologies

- **Deep Learning**: PyTorch 2.x
- **Segmentation**: Segmentation Models PyTorch (smp)
- **GenAI**: Real-ESRGAN, HuggingFace Diffusers
- **Experiment Tracking**: Weights & Biases (wandb)
- **UI**: Streamlit
- **Geospatial**: Rasterio, GeoPandas

## Current Status

See `.tracker/log.md` for progress updates.

## Documentation

- **Plan**: `visionproject.plan.md`
- **Tracker**: `PROJECT_TRACKER.md`
- **Final Report**: `FINAL_REPORT.md` (generated at completion)

## Next Steps

1. Initialize Git repository: `git init`
2. Start Step 1: Review `PROJECT_TRACKER.md` and run `notebooks/01_eda.ipynb`
3. Follow tracker commands for validation checkpoints
