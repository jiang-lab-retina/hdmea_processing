# AP Tracking Algorithm Documentation

## Overview

The AP (Action Potential) Tracking pipeline analyzes retinal ganglion cell (RGC) spike-triggered average (STA) data to:
1. Detect soma (cell body) location
2. Predict axon signal probability using CNN
3. Extract axon trajectory centroids
4. Fit pathway lines to axon trajectories  
5. Calculate the Optic Nerve Head (ONH) location from pathway intersections
6. Compute soma polar coordinates relative to ONH

---

## Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: HDF5 File                             │
│                 (units/{id}/features/eimage_sta/data)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 1. SOMA DETECTION                                │
│   • Find max intensity region in temporal range [5, 27]         │
│   • Refine soma position with Gaussian weighting                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2. CNN PREDICTION                                │
│   • Run 3D CNN on STA data (50, 65, 65)                        │
│   • Output: probability map of axon signal per pixel/frame      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 3. POST-PROCESSING                               │
│   • Threshold filtering (>0.1)                                  │
│   • Connected component analysis (min 3 pixels)                 │
│   • Temporal consistency smoothing                              │
│   • Soma exclusion (5px radius)                                 │
│   • Centroid extraction with trajectory segment detection       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 4. PATHWAY FITTING                               │
│   • Linear regression on axon centroids                         │
│   • Calculate direction from temporal order                     │
│   • Store: slope, intercept, R², direction_angle, start_point   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 5. ONH DETECTION                                 │
│   • Enhanced algorithm with clustering (primary)                │
│   • Legacy weighted mean (fallback)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 6. POLAR COORDINATES                             │
│   • Compute soma position relative to ONH                       │
│   • Apply anatomical angle correction                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: HDF5 File                            │
│            (units/{id}/features/ap_tracking/...)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Algorithm Steps

### Step 1: Soma Detection

The soma is detected by finding the brightest region in the STA data:

1. **Temporal Windowing**: Focus on frames 5-27 where soma signal is strongest
2. **Max Intensity Search**: Find pixel with highest intensity
3. **Refinement**: Apply Gaussian-weighted averaging in a 5-pixel radius for sub-pixel precision

### Step 2: CNN Prediction

A pre-trained 3D Convolutional Neural Network predicts axon signal probability:

- **Input**: STA data `(50 frames, 65 rows, 65 cols)`
- **Output**: Probability map `(50, 65, 65)` with values 0-1
- **Architecture**: 3D convolutions with batch normalization

### Step 3: Post-Processing

Raw CNN predictions are refined through multiple filtering stages:

#### 3.1 Noise Filtering
```python
# Threshold: Remove low-intensity predictions
prediction[prediction < 0.1] = 0

# Connected components: Remove isolated pixels
# Keep clusters with ≥3 connected pixels

# Temporal consistency: Remove sporadic signals
# Signal must appear in neighboring frames
```

#### 3.2 Soma Exclusion
```python
# Exclude 5-pixel radius around detected soma
# Prevents soma signal from contaminating axon trace
```

#### 3.3 Centroid Extraction
```python
# For each frame (starting from frame 10):
#   1. Find largest connected component
#   2. Calculate center of mass → centroid (t, row, col)
#   3. Filter by max displacement (5px) between consecutive frames

# Trajectory segment detection:
#   - Identify "break points" where displacement > threshold
#   - Keep only the longest continuous segment
```

### Step 4: Pathway Fitting

Linear regression fits a line to the axon trajectory:

```python
# Linear fit: row = slope × col + intercept
slope, intercept, r_value, p_value, std_err = linregress(cols, rows)

# Direction calculation from temporal order:
# - Sort centroids by frame number
# - Calculate weighted mean (later frames weighted more)
# - Direction = atan2(d_row, d_col) in degrees [0-360]
```

**Quality Metric**: R² (coefficient of determination)
- R² ≥ 0.8: High-quality linear trajectory
- R² < 0.4: Poor fit, pathway excluded

---

## ONH Detection: Enhanced vs Legacy Algorithm

### Enhanced Algorithm (Primary Method)

The enhanced algorithm uses a 5-step pipeline with robust outlier rejection:

```
Step 1: Calculate Consensus Direction
        ↓
Step 2: Filter by Direction (±45° tolerance)
        ↓
Step 3: Calculate Pairwise Intersections with Distance Filter
        ↓
Step 4: DBSCAN Clustering
        ↓
Step 5: Weighted Mean from Main Cluster
```

#### Step 1: Consensus Direction

Calculate the dominant direction all axons are pointing:

```python
# Weighted circular mean of all pathway directions
# Weights = R² of each pathway
# Result: consensus angle in degrees [0-360]
```

This represents the approximate direction toward the ONH from the MEA center.

#### Step 2: Direction Filtering

Remove pathways pointing away from the consensus:

```python
# For each pathway with direction_angle:
#   angle_diff = abs(direction_angle - consensus_direction)
#   if angle_diff > 180: angle_diff = 360 - angle_diff
#   
#   if angle_diff > 45°:
#       pathway.direction_valid = False  # Exclude from ONH calculation
```

**Rationale**: All RGC axons should point toward the ONH. Pathways pointing opposite directions are likely errors.

#### Step 3: Pairwise Intersections

Calculate where each pair of pathway lines intersect:

```python
# For each pair of pathways (line1, line2):
#   Solve: y = m1*x + c1 = m2*x + c2
#   x_int = (c2 - c1) / (m1 - m2)
#   y_int = m1 * x_int + c1
#
#   # Distance filter: intersection must be within visible area
#   dist = sqrt((x_int - 33)² + (y_int - 33)²)
#   if dist > 98: exclude  # 65 + 33 = 98 pixels from center
```

#### Step 4: DBSCAN Clustering

Identify the main cluster of intersection points:

```python
# DBSCAN parameters:
#   eps = 15 pixels (neighborhood radius)
#   min_samples = 3 points
#
# Result: 
#   main_cluster: Points in largest cluster → used for ONH
#   outliers: Scattered points → excluded
```

**Rationale**: True pairwise intersections should cluster around the real ONH. Outliers from bad fits are spatially scattered.

#### Step 5: Weighted Mean from Cluster

Final ONH location from main cluster:

```python
# Weight each intersection by mean R² of its two source pathways
x_onh = Σ(x_i × w_i) / Σ(w_i)
y_onh = Σ(y_i × w_i) / Σ(w_i)

# Calculate RMSE (error metric)
rmse = sqrt(mean(distance² from each point to weighted mean))
```

### Legacy Algorithm (Fallback Method)

Simple weighted average without any filtering:

```python
# Calculate ALL pairwise intersections (no direction/distance filtering)
# Weight by mean R² of each pair
# Compute weighted average

x_onh = Σ(x_i × r²_mean_i) / Σ(r²_mean_i)
y_onh = Σ(y_i × r²_mean_i) / Σ(r²_mean_i)
```

**Limitation**: No outlier rejection. A single bad pathway can skew the result significantly.

---

## Comparison: Enhanced vs Legacy

| Feature | Enhanced Algorithm | Legacy Algorithm |
|---------|-------------------|------------------|
| **R² Filtering** | Yes (≥0.8, with retry at 0.6, 0.4) | No |
| **Direction Filtering** | Yes (±45° from consensus) | No |
| **Distance Filtering** | Yes (98px from center) | No |
| **Outlier Rejection** | DBSCAN clustering | None |
| **Robustness** | High | Low |
| **Failure Mode** | Falls back to legacy | Returns noisy result |

### Retry Logic

The enhanced algorithm attempts progressively lower R² thresholds:

```
Try R² ≥ 0.8 → If fail...
Try R² ≥ 0.6 → If fail...
Try R² ≥ 0.4 → If fail...
Use legacy method (R² = 0.0, no filtering)
```

The actual R² threshold that succeeded is recorded in the HDF5 output.

---

## Output Data Structure

Results are stored in HDF5 at `units/{unit_id}/features/ap_tracking/`:

```
ap_tracking/
├── DV_position          # Dorso-Ventral position from metadata
├── NT_position          # Naso-Temporal position from metadata  
├── LR_position          # Left/Right eye
├── refined_soma/
│   ├── t                # Soma time (frame)
│   ├── x                # Soma row
│   └── y                # Soma column
├── axon_initial_segment/
│   ├── t, x, y          # AIS position
├── prediction_sta_data  # CNN output (50, 65, 65)
├── post_processed_data/
│   ├── filtered_prediction  # Noise-filtered prediction
│   └── axon_centroids       # (N, 3) array: [t, row, col]
├── ap_pathway/
│   ├── slope            # Line slope
│   ├── intercept        # Line intercept  
│   ├── r_value          # Correlation coefficient
│   └── ...
├── all_ap_intersection/
│   ├── x                # ONH x coordinate (column)
│   ├── y                # ONH y coordinate (row)
│   ├── mse              # Mean squared error
│   ├── rmse             # Root mean squared error
│   ├── method           # "clustered_weighted_mean" or "legacy_weighted_mean"
│   ├── r2_threshold     # Actual R² threshold used (0.0 for legacy)
│   ├── consensus_direction  # Degrees [0-360]
│   ├── n_cluster_points     # Points in main cluster
│   ├── n_total_intersections # Total pairwise intersections
│   └── cluster_points   # (N, 2) array of valid intersection points
└── soma_polar_coordinates/
    ├── radius           # Distance from ONH to soma
    ├── angle            # Angle in radians
    ├── quadrant         # "Q1", "Q2", "Q3", "Q4"
    └── anatomical_quadrant  # e.g., "dorsal-nasal"
```

---

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r2_threshold` | 0.8 | Minimum R² for pathway inclusion |
| `direction_tolerance` | 45° | Max deviation from consensus direction |
| `max_distance_from_center` | 98 | Max intersection distance from (33,33) |
| `cluster_eps` | 15 | DBSCAN neighborhood radius |
| `cluster_min_samples` | 3 | DBSCAN minimum cluster size |
| `centroid_start_frame` | 10 | Exclude centroids before this frame |
| `max_displacement_post` | 5.0 | Max displacement for trajectory continuity |
| `min_points_for_fit` | 10 | Minimum centroids for pathway fitting |

---

## References

- DBSCAN clustering: Ester et al., 1996
- Circular statistics: Mardia & Jupp, 2000
- Spike-triggered average: Chichilnisky, 2001

