High-Level Roadmap

Phase 1 – EDA & Advanced Visualization

Phase 2 – Preprocessing & Classical Enhancements

Phase 3 – Model Architectures (3 main models + Siamese / Multi-task)

Phase 4 – Training Strategy & Losses

Phase 5 – Model Comparison, Interpretation & Ensembling

Phase 6 – Generative AI (GenAI) Enhancement Pipeline

Phase 7 – Re-training with Enhanced Data

Phase 8 – Advanced Evaluation & Robustness

Phase 9 – UI & Storytelling (Streamlit / Angular)

Phase 10 – Bonus Research Directions

Deliverables & Tech Stack

You can easily map this to “Weeks 1–11” if you want a timeline.

Phase 1 – Advanced EDA & Visualization (Core Weeks 1–2)
1.1 Core EDA (from earlier plan) – Core

Notebook: 01_eda.ipynb

Load sample images and masks from:

Germany: /dataset/train/Germany_Training_Public

Louisiana-East: /dataset/train/Louisiana-East_Training_Public

Check:

Image size, channels, data type

Mask size & class values

Class distribution:

Count pixels per class (water, flooded water, buildings, roads, etc.)

Bar charts for class imbalance

Pre vs Post visualization:

2×2 grids: Pre, Post, Ground Truth (GT) mask, Overlay (mask on Post)

Difference image: abs(Post - Pre) heatmap

Data quality:

Missing/corrupted files

Misalignment

Clouds, haze, shadows

1.2 Interactive / advanced visual analytics – Advanced

Temporal change detection heatmaps

Compute and visualize Δ between pre and post images (difference maps).

Geospatial mapping

Use GeoPandas + Folium or kepler.gl to show where tiles are located and highlight flood-heavy tiles.

Class distribution sunburst charts

Use Plotly to visualize classes nested by AOI (Area of Interest) and tile.

t-SNE / UMAP feature analysis (later in project)

t-SNE = t-Distributed Stochastic Neighbor Embedding

UMAP = Uniform Manifold Approximation and Projection

Once you have embeddings from a model, visualize clusters (Germany vs Louisiana, high vs low flood severity).

Uncertainty maps (pre-EDA approximation)

Early on, you can mark “ambiguous” areas (shadows, low contrast) to anticipate where models might struggle.

1.3 Novel visualizations – Advanced but very cool

Edge detection overlays

Use Canny or Sobel edge detectors to highlight boundaries of water, buildings, flood extents.

Change vector analysis (CVA)

For each pixel, treat [Pre bands] and [Post bands] as vectors and compute magnitude and direction of change → show as heatmap or direction field.

Temporal sequence animations

If multiple temporal snapshots exist (even small number), create GIFs or MP4 animations of flood progression.

Phase 2 – Preprocessing & Classical Enhancements (Week 3)

Notebook: 02_preprocessing.ipynb

2.1 Basic pipeline – Core

Decide on patch size: e.g., 512×512 or 1024×1024.

For each patch:

Input tensor X: concatenated Pre + Post (6 channels for RGB pre & post).

Output mask y: multiclass segmentation mask.

Train/validation/test split:

Train: most of Germany + Louisiana-East

Validation: held-out subset of both

Test: /dataset/test/ (Louisiana) – unseen area.

2.2 Classical image enhancement – Advanced (but simpler than GenAI)

CLAHE (Contrast-Limited Adaptive Histogram Equalization)

Local contrast enhancement to better see flooded edges and dark areas.

Multi-scale image pyramids

Save downsampled and upsampled versions for experiments with different resolutions.

Atmospheric correction & shadow handling (if needed)

Simple methods: per-channel normalization, shadow detection via thresholds.

2.3 Data augmentation & sampling – Core + Advanced

Standard augmentations (Core):

Flips, rotations, brightness/contrast jitter, small noise, light blur (Albumentations).

Domain-specific augmentations (Advanced):

Simulate haze, slight water level changes (mask dilation).

CutMix / MixUp (Advanced)

CutMix: combine two images by cutting and pasting patches.

MixUp: mix pixel values and labels using linear combination.

Smart patch extraction:

Oversample patches with flood boundaries and rare flood classes.

Cross-location validation:

Ensure Germany and Louisiana data are not mixed across train/val/test in a way that leaks location info.

Phase 3 – Model Architectures (Week 4)

Notebook: 03_models_training.ipynb

3.1 Three primary models (your “3 types CV models”) – Core

Use Segmentation Models PyTorch (smp) to speed up.

Model 1: U-Net++ (nested U-Net) – CNN-based

Strong for segmentation with complex boundaries.

Encoder: ResNet-50 or EfficientNet-B4 (pretrained on ImageNet).

Model 2: DeepLabV3+ – CNN with Atrous Spatial Pyramid Pooling

Atrous = dilated convolutions capturing multi-scale context.

Encoder: Xception or ResNet-101.

Good for large flood areas and varied object sizes.

Model 3: SegFormer or UPerNet (Transformer-based) – Vision Transformer

Transformer encoder + lightweight decoder.

Great for global context and large, continuous floods.

3.2 Shared architectural ideas across models – Advanced

Siamese architecture (change detection)

Two encoders (pre & post) sharing weights, features fused before decoding.

Can be combined with U-Net++ / DeepLabV3+ as variants.

Attention mechanisms

SE (Squeeze-and-Excitation) blocks, CBAM (Convolutional Block Attention Module), or spatial attention.

Help model focus on relevant regions (e.g., edges of water).

Multi-task learning

Add extra heads to predict:

Building footprints

Roads

Flood masks

Shared encoder → better representation learning.

Phase 4 – Training Strategy & Losses (Weeks 5–6)

Still mainly in 03_models_training.ipynb + 04_model_comparison.ipynb.

4.1 Loss functions – Core + Advanced

Dice Loss

Measures overlap between predicted and real regions; good for imbalanced segmentation.

Focal Loss

Focuses more on hard-to-classify pixels (e.g., rare flood pixels).

Boundary Loss (Advanced)

Penalizes errors along object boundaries (important for flood edges).

Tversky Loss (Advanced)

Generalization of Dice Loss that can weight false positives and false negatives differently.

Lovász-Softmax Loss (Advanced)

Directly optimizes Intersection-over-Union (IoU) metric.

Combine them as Combo Loss:

Loss = α _ Dice + β _ Focal (+ γ \* Boundary/Tversky/Lovász)

4.2 Training techniques – Advanced, but strong impact

Progressive resizing

Start training at 256×256, then fine-tune at 512×512 or 1024×1024.

Differential learning rates

Lower learning rate for pretrained encoder; higher for decoder and new heads.

Cosine annealing with warm restarts

Learning rate schedule that periodically restarts to avoid local minima.

Stochastic Weight Averaging (SWA)

Keep an average of weights across epochs for smoother, more robust models.

Mixed precision training (with PyTorch AMP)

Faster training using float16 where safe, with little to no accuracy loss.

4.3 Evaluation metrics – Core

Pixel-wise:

IoU (Intersection over Union) per class

Dice coefficient

Precision, Recall, F1-score

Higher-level:

Per-class IoU for water, flooded water, buildings, roads

Per-region metrics: Germany vs Louisiana

Phase 5 – Model Comparison, Interpretation & Ensembling (Week 7)

Notebook: 04_model_comparison.ipynb + 07_visualization_explainability.ipynb.

5.1 Model comparison – Core

Comparison tables:

Parameters, mIoU, Flood IoU, inference time, etc.

Visual grids:

For selected tiles: GT vs U-Net++ vs DeepLabV3+ vs SegFormer.

Error analysis dashboard:

Show false positives/false negatives per class and per AOI.

5.2 Ensembling – Advanced

Weighted averaging

Average logits or probabilities of all 3 models, weight based on validation IoU.

Stacking (meta-learner)

Train a small model on top of per-model outputs to decide final class.

Conditional ensemble

E.g., transformer on rural floodplains, U-Net++ in dense urban cores (if you detect region type).

5.3 Interpretation – Advanced but very nice

Grad-CAM (Gradient-weighted Class Activation Mapping)

Visualize which regions contributed most to a prediction.

Feature map visualizations

Show intermediate activations to understand what features are being learned.

Prediction confidence maps

Visualize per-pixel confidence or uncertainty.

Phase 6 – Generative AI Enhancement Pipeline (Weeks 8–9)

Notebook: 06_genai_enhancement_experiments.ipynb.

6.1 Pre-processing enhancement – Core for your GenAI story

Super-resolution (SR)

Models like Real-ESRGAN or SwinIR to upsample images (e.g., from 512→1024).

Either:

Train on enhanced images directly, or

Enhance only a subset for experiments.

Denoising

NAFNet / Restormer → remove sensor noise / compression artifacts.

Color-related enhancements (if needed):

If you use NIR (Near Infrared) or grayscale bands, you can experiment with colorization (e.g., DeOldify) to improve interpretability.

6.2 Post-processing / filling gaps – Advanced

Inpainting (cloud/missing data):

LaMa or Mask-Aware Transformer (MAT) to fill occluded regions.

Synthetic flood generation

Use diffusion models or GANs to:

Generate “what-if” flood scenarios.

Create rare situations (e.g., extreme flooding) for training.

6.3 Counterfactual & semantic GenAI – Advanced + flashy

Counterfactual generation

“What would this tile look like if flooded/not flooded?”

Useful for explaining model behavior and training robustness.

Semantic image synthesis

From predicted masks back to realistic images → sanity-check segmentation (does the segmentation look like a plausible scene?).

Phase 7 – Re-training with Enhanced Data (Week 10)

Still 06_genai_enhancement_experiments.ipynb + maybe a dedicated 08_ablation_study.ipynb.

7.1 Experiments

For the chosen best architecture (e.g., SegFormer or Siamese U-Net), run:

Train on original data → baseline.

Train on super-resolved data → see resolution effects.

Train on denoised data → see noise robustness.

Train with synthetic augmented data → see if rare cases improve.

Train on combined enhanced dataset → best possible.

7.2 Comparison & significance

Build a matrix of:

(Training regime) × (mIoU, Flood IoU, F1, inference cost).

Optionally use:

Bootstrap confidence intervals for IoU.

McNemar’s test for checking if improvement in error patterns is statistically significant.

Phase 8 – Advanced Evaluation & Robustness (Week 11)
8.1 Robustness testing – Advanced

Out-of-distribution (OOD) testing

Test on tiles or regions not represented in training (different land cover, geography).

Adversarial perturbations (lightweight)

Small, realistic perturbations: brightness shifts, blur, JPEG compression.

Failure case library

Collect and annotate “hard failures,” categorize: shadow areas, dense urban, small rivers, etc.

8.2 Results dashboard & GIS integration – Core for presentation

Web-based viewer (can be Streamlit or Angular):

Adjustable thresholds, toggles for GT vs prediction vs error.

Export predictions as GeoTIFF or similar formats for GIS tools (QGIS/ArcGIS).

Phase 9 – UI & Storytelling (Streamlit / Angular)

This is where your earlier UI request fits perfectly.

9.1 Streamlit app – Core demo

Folder: ui/streamlit_app.py.

Pages:

Overview

Project description, diagrams, dataset info.

Tile Explorer

Select AOI, tile, model.

Show Pre, Post, GT, Prediction side-by-side.

Show per-tile metrics.

Model Comparison

Compare multiple models on same tile (grid of predictions).

GenAI Enhancements

Raw vs Enhanced image, prediction, and metrics changes.

AI Text Summary (optional)

Use a vision–language model (VLM) to generate short narrative: “X% buildings flooded, main road partially unusable…”

This uses your existing Python code directly and is ideal for class/demo.

9.2 Angular + FastAPI backend – Advanced / industry-style

If you want to go further:

Backend (FastAPI):

/tiles, /tile/{id}, /predict, /enhance, /summary endpoints.

Angular frontend:

Pages mirroring Streamlit pages but with richer UI (Material components, charts).

Uses an ApiService to talk to the FastAPI backend.

You can position Streamlit as Phase 1 UI, Angular as Phase 2 productization.

Phase 10 – Bonus Research Directions (if you want a paper or extra credit)

These map directly to your “Bonus” list:

Self-supervised pre-training

SimCLR / MoCo on unlabeled satellite imagery to pretrain encoder.

Foundation model integration

Segment Anything Model (SAM) for initial masks.

CLIP embeddings for region-level classification of flood severity.

Physics-informed models

Incorporate elevation maps and simple flood simulation constraints to penalize physically impossible floods (e.g., water uphill).

Active learning

Use model uncertainty to propose new points for annotation (simulated human-in-the-loop).

Explainable AI (XAI) dashboard

SHAP (SHapley Additive exPlanations) for patch-level contributions.

Counterfactual examples (“change this region to non-flood, how do predictions change?”).

Multi-modal learning

Combine satellite imagery with rainfall data, river flow, social media reports for richer predictions.

Deliverables & Tech Stack (Integrated)
Deliverables

✅ Jupyter notebooks per phase (EDA, preprocessing, training, GenAI, evaluation).

✅ Clean, modular src/ folder with dataset, models, training and inference code.

✅ Streamlit UI (and/or Angular + FastAPI) showcasing:

Tile exploration

Model comparison

GenAI enhancements

✅ Model comparison report (PDF / markdown) with:

Metrics, visual examples, ablation results.

✅ Slide deck for presentation (tell the story: problem → data → models → GenAI → results).

✅ Short video demo of the UI + main findings.

✅ Optional: research-style writeup (for arXiv / class project report).

Tech stack (final consolidated)

Deep learning: PyTorch, possibly PyTorch Lightning for structured training.

Segmentation: Segmentation Models PyTorch (smp).

Visualization: Matplotlib, Seaborn, Plotly, Folium, maybe kepler.gl.

GenAI: HuggingFace Diffusers, Real-ESRGAN, SwinIR, LaMa, etc (as much as your compute allows).

Geospatial: Rasterio, GeoPandas, GDAL.

UI: Streamlit (core demo), optional Angular + FastAPI for advanced front-end.

Experiment tracking: Weights & Biases (wandb) or MLflow.
