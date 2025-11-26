# Exploratory Data Analysis (EDA) Summary

_Flood Impact Assessment on SpaceNet-8–Style Multi-Temporal Imagery_

---

## 1. Objectives of the EDA

This EDA examines a multi-region flood mapping dataset (Germany, Louisiana-East, Louisiana-West) with pre- and post-event satellite imagery and vector labels. The analysis aims to:

- Understand the structure and coverage of the imagery, labels, and CSV metadata.
- Quantify the prevalence and characteristics of flooded vs non-flooded road segments.
- Characterize image-level properties (resolution, intensity distributions, temporal shifts).
- Analyze temporal change patterns between pre- and post-event images.
- Explore geospatial clustering of annotated features.
- Examine textural and spectral differences between flooded and non-flooded areas.

The goal is to derive concrete, plot-driven insights that inform model design for multi-class flood segmentation and downstream road-network–level risk analysis.

---

## 2. Dataset Structure & Coverage

### 2.1 Regions and Folders

The dataset is organized by region:

- `Germany_Training_Public`
- `Louisiana-East_Training_Public`
- `Louisiana-West_Test_Public`

Each region contains:

- `PRE-event/` – pre-flood imagery tiles.
- `POST-event/` – post-flood imagery tiles.
- `annotations/` – GeoJSON labels for training regions.
- CSV files for label–image mapping and road-network metadata.

**Plot(s):**

- Bar charts showing number of PRE and POST images per region.
- Bar chart of number of annotation files per region.

**Interpretation:**

- Germany and Louisiana-East both have substantial numbers of pre- and post-event tiles and a matching annotation file per tile. Bar plots show roughly comparable scale across the two training regions.
- Louisiana-West shows bars only for images (no annotations), confirming its usage as a held-out test region.
- The PRE/POST counts per training region are well balanced, indicating that models can be trained with symmetric temporal inputs (either concatenated channels or change-based features).
- The 1:1 correspondence between tiles and annotation files in the training regions simplifies data loading and strongly suggests that tiling and labeling are consistent.

---

## 3. CSV Metadata & Road Network Flood Statistics

Two main CSV types:

1. `*_label_image_mapping.csv` – links each annotation tile to its PRE and POST image(s).
2. `*_reference.csv` – road/building network metadata with attributes such as `Object` (Road/Building), `Flooded` (True/False/Null), and optional length/travel-time fields.

### 3.1 Class Balance: Flooded vs Non-Flooded

**Plot(s):**

- Bar chart of `Flooded` status counts (True / False / Null) per region.
- Combined bar chart for all training regions.
- Pie chart showing the overall percentage of flooded vs non-flooded segments.

**Key Observations:**

- Across training datasets, non-flooded segments dominate (~78–80% of labeled road segments).
- Flooded segments account for roughly ~20–22%, while Null/Unknown are <1%.

**Interpretation:**

- The bar and pie charts highlight **class imbalance**: flooded segments are clearly under-represented relative to non-flooded segments.
- For a segmentation or road-level classification task, this imbalance naturally biases models toward the majority non-flooded class.
- From a modeling perspective, this supports the need for:
  - Class-weighted loss functions (e.g., weighted cross-entropy, focal loss).
  - Careful sampling strategies (oversampling flooded segments or targeted tile sampling where floods are present).
  - Evaluation metrics beyond accuracy (e.g., precision/recall for the flooded class, F1 score, AUROC).

### 3.2 Road Length and Travel Time Distributions

**Plot(s):**

- Histograms and boxplots of road segment length (meters) for each region.
- Histograms/boxplots of road segment travel time (seconds) for each region.
- Overlaid histograms for flooded vs non-flooded segments.

**Key Observations:**

- Length distributions are right-skewed: many short segments with a long tail of longer roads.
- Median road lengths differ somewhat by region (Germany generally shorter than Louisiana-East), consistent with denser European urban layouts vs more extensive US road geometry.
- Flooded and non-flooded segments have overlapping length distributions but with subtle differences:
  - Flooded segments tend to include a slightly higher proportion of longer segments in some regions.
- Travel times show similar right-skewed distributions, correlating with length.

**Interpretation:**

- The skewed distributions suggest that a few long roads contribute disproportionately to network metrics (e.g., total kilometers at risk), which is relevant for cost or impact analyses.
- The overlap between flooded and non-flooded length distributions implies that **length alone is not a strong discriminator** for flood status, though it may be useful as a secondary feature (e.g., for prioritizing inspection of long segments).
- The presence of longer flooded segments raises practical concerns: longer segments closed due to flooding may have outsized impact on mobility and emergency response.

---

## 4. Image Properties & Pixel Intensity Distributions

### 4.1 Resolution and Data Types

**Plot(s):**

- Bar chart summarizing tile shapes (height × width) across a random sample.
- Bar chart of image dtypes (e.g., `uint8`, `uint16`).
- Table or text summary of sample shapes and dtypes.

**Interpretation:**

- All sampled tiles per region share a consistent resolution (square tiles) and channel count, simplifying batching and model input design.
- The presence of both `uint16` and `uint8` tiles (often due to different sensor sources or pre-processing pipelines) highlights the need for consistent normalization (e.g., scaling to 0–1 or 0–255 and per-channel standardization).
- No significant outliers in image shape were observed; this reduces the need for special handling of irregular tiles.

### 4.2 Pixel Intensity Histograms (PRE vs POST)

**Plot(s):**

- Overlaid intensity histograms (per channel) for PRE vs POST images for Germany.
- Overlaid intensity histograms (per channel) for PRE vs POST images for Louisiana-East.
- Boxplots of mean tile intensity per region and time (PRE vs POST).

**Key Observations:**

- Germany:
  - PRE and POST histograms are similar, with only modest shifts in mean intensity.
  - Slight brightening is visible in post-event histograms in certain channels.
- Louisiana-East:
  - Post-event histograms are noticeably shifted to the right (higher intensities), particularly in channels sensitive to water or soil moisture.
  - There is also a slightly broader spread in post-event intensities, indicating more variability in surface reflectance after the event.

**Interpretation:**

- For Germany, subtle intensity shifts suggest that flood impact may be visually less pronounced or more localized, making **texture and fine-grained change detection** more important than gross brightness changes.
- For Louisiana-East, the marked shift in post-event brightness is consistent with **large, reflective flood surfaces** (water, saturated soil) and possibly cloud cover or atmospheric effects.
- The post-event brightening in the Louisiana-East histograms supports the use of:
  - Difference-based features (POST − PRE),
  - Ratio-based features (POST / PRE),
  - Learned temporal fusion mechanisms that can capture these systematic shifts.

---

## 5. Temporal Change Analysis (PRE vs POST)

### 5.1 Difference Images & Change Magnitude

**Plot(s):**

- 4-panel visualization per sample:
  1. PRE tile
  2. POST tile
  3. Absolute difference image |PRE − POST|
  4. Binary change mask (thresholded difference)
- Histograms of absolute difference values.
- Bar chart of percentage of changed pixels per sampled tile and per region.

**Key Observations:**

- In Germany:
  - Difference images show patchy regions of change, often aligned with river channels or low-lying areas.
  - Histograms of difference values have moderate tails; the percentage of changed pixels is non-trivial but not extreme.
- In Louisiana-East:
  - Difference images show large, contiguous bright regions corresponding to floodplains and water accumulation.
  - Difference histograms have heavier tails; the maximum differences are noticeably higher.
  - Binary change masks highlight a larger proportion of the tile as “changed” compared to Germany.

**Interpretation:**

- The difference plots confirm that **temporal change is a strong signal** for flood detection, especially in Louisiana-East where large homogeneous changes indicate broad flood coverage.
- The moderate change in Germany suggests that models need to be sensitive to more subtle patterns (e.g., water encroachment along narrow channels or partial inundation).
- Using pre/post stacks as model input (e.g., 6-channel input: 3 PRE + 3 POST) or designing explicit change-detection architectures (e.g., Siamese encoders with difference fusion) is well justified.
- The binary change masks, while noisy, align qualitatively with flooded regions and can serve as:
  - A sanity check for label quality,
  - A baseline unsupervised flood indicator,
  - Potential auxiliary supervision in a multi-task learning setup.

---

## 6. Geospatial Patterns & Spatial Density

### 6.1 Spatial Density of Annotations

**Plot(s):**

- Hexbin or 2D histogram plots of annotation centroid coordinates for Germany and Louisiana-East.
- Scatter plots showing spatial distribution of flooded vs non-flooded road segments (where coordinates are available).

**Key Observations:**

- Germany:
  - A dense cluster of annotations appears in a relatively compact geographical area, consistent with smaller spatial extent and tighter focus around specific river basins.
- Louisiana-East:
  - Annotations cover a wider longitudinal extent and appear slightly more dispersed, reflecting broader floodplains and more extensive road networks.
- Where flood status is visualized:
  - Flooded roads tend to cluster near river channels, low-lying areas, and edges of urban layouts.

**Interpretation:**

- Spatial density plots indicate that the dataset is **not uniformly sampled**: certain areas (e.g., around rivers and urban centers) are overrepresented, while remote regions are underrepresented.
- Flooded features clustering near river channels reinforces domain knowledge and validates the plausibility of the labels.
- For model evaluation, geo-stratified cross-validation (splitting by spatial clusters or tiles) is recommended to avoid spatial leakage and overoptimistic performance estimates.

---

## 7. Texture & Frequency Analysis

### 7.1 Texture Features (GLCM, LBP)

**Plot(s):**

- Bar charts of mean GLCM metrics (contrast, homogeneity, energy, correlation) for representative flooded vs non-flooded patches.
- Boxplots comparing GLCM features between flood and non-flood patches.
- Histograms of LBP (Local Binary Pattern) entropy for patches.

**Key Observations:**

- Flooded patches:
  - Show lower GLCM contrast/dissimilarity and higher homogeneity, reflecting smoother textures of water surfaces or uniformly saturated soil.
  - Have lower LBP entropy, indicating more uniform local patterns.
- Non-flooded urban/road patches:
  - Show higher contrast and dissimilarity, with rich edge content from buildings, roads, and vegetation boundaries.
  - Have higher LBP entropy, reflecting complex textures and structural details.

**Interpretation:**

- Texture plots strongly support the hypothesis that **flooded regions are texturally simpler** than non-flooded urban or vegetated regions.
- These findings justify the use of:
  - CNNs with multi-scale filters that can capture both fine-grained and smooth patterns.
  - Potential auxiliary texture descriptors (e.g., LBP histograms or GLCM-derived channels) for post-hoc analysis or as additional model input.
- From an explainability perspective, showing texture differences between correctly and incorrectly classified patches can help stakeholders understand model decisions.

### 7.2 Frequency-Domain Analysis (FFT)

**Plot(s):**

- Log-magnitude FFT images for flooded vs non-flooded patches.
- Boxplots of average FFT magnitude per frequency band.

**Key Observations:**

- Flooded patches tend to have lower high-frequency content (smoother surfaces).
- Non-flooded patches show more energy in mid- and high-frequency bands, consistent with edges and structural detail.

**Interpretation:**

- The frequency-domain plots further confirm that floods simplify local structure, which can be exploited by CNNs (which inherently act as localized frequency analyzers).
- These differences suggest that deeper models may be particularly effective since they can capture hierarchical frequency patterns spanning from low-frequency smooth regions to high-frequency edges.

---

## 8. Colour-Space & Water Signature Analysis

**Plot(s):**

- PRE vs POST visualizations in RGB, HSV, and Lab color spaces.
- Per-channel histograms in HSV (Hue, Saturation, Value).
- Scatter plots of Lab (a*, b*) values for pixels in flooded vs non-flooded regions.

**Key Observations:**

- In HSV:
  - Flooded water often appears as clusters in specific Hue–Value ranges (typically darker value but sometimes brighter depending on sensor and sun angle).
  - Saturation tends to be modest, reflecting relatively uniform spectral signatures.
- In Lab:
  - Water pixels cluster tightly in a narrower region of the a*–b* plane, while urban and vegetated pixels are more scattered.
- Comparing PRE vs POST:
  - Post-event images show expansion of the water-related clusters, particularly in flood-affected tiles.

**Interpretation:**

- Colour-space plots indicate that water (and thus floods) occupies a **more compact region in transformed color spaces** (HSV, Lab) than in raw RGB.
- This supports:
  - Augmenting model input with additional derived channels (e.g., hue, saturation, or simple water indices).
  - Using color-space transformations as pre-processing steps to stabilize the spectral representation across scenes.
- However, overlaps with shadows and dark roofs in these spaces highlight that color alone is not sufficient; temporal and textural cues are essential to disambiguate water from other dark surfaces.

---

## 9. Overall EDA Conclusions & Modeling Implications

### 9.1 Key Empirical Findings

1. **Consistent Tile Geometry, Mixed Dtypes**

   - Tiles per region share consistent resolution and channel count, simplifying model design.
   - Mixed dtypes (`uint8` / `uint16`) require robust normalization.

2. **Moderate to Severe Class Imbalance**

   - Flooded vs non-flooded segments show ~1:4 class ratio.
   - Geometry-related classes (e.g., geometry types) are highly imbalanced, with ratios exceeding 250:1.

3. **Strong Temporal Signal, Region-Dependent**

   - Louisiana-East exhibits pronounced pre/post intensity shifts and large change regions.
   - Germany shows more subtle changes, requiring sensitive change detectors.

4. **Spatial Clustering Around Rivers & Urban Centers**

   - Flooded segments cluster near rivers and floodplains.
   - Sampling and validation strategies must account for spatial autocorrelation.

5. **Texture & Frequency Differences Between Flooded and Non-Flooded Areas**

   - Flooded areas: smoother, lower contrast, lower high-frequency content.
   - Non-flooded areas: higher contrast and more complex textures.

6. **Colour-Space Separability of Water**
   - Water occupies compact regions in HSV/Lab spaces, though not perfectly separable due to confounders like shadows.

### 9.2 Implications for Model Design

Based on the plots and findings:

- **Input Representation**

  - Stack PRE and POST images as multi-channel inputs.
  - Consider adding engineered channels (difference, ratios, water indices, color-space transforms).

- **Architecture Choices**

  - Use segmentation architectures tailored for change detection (e.g., UNet with dual encoders, Siamese UNet, temporal attention).
  - Include multi-scale feature extraction to capture both large floodplains and narrow flooded roads.

- **Training Strategy**

  - Mitigate class imbalance through:
    - Class-weighted or focal loss,
    - Oversampling of flooded tiles,
    - Hard-negative mining where non-flooded areas look similar to flooded ones.
  - Adopt geo-stratified cross-validation splits to reduce spatial leakage.

- **Evaluation & Explainability**
  - Evaluate with metrics sensitive to the minority class (e.g., recall/F1 for flooded pixels).
  - Use difference images, texture maps, and colour clustering plots to visually explain model successes/failures to stakeholders.
