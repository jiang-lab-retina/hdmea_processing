# AP Tracking Algorithm Documentation

## Overview

The AP (Action Potential) Tracking pipeline analyzes retinal ganglion cell (RGC) spike-triggered average (STA) data to:
1. Detect soma (cell body) location
2. Predict axon signal probability using CNN
3. Extract axon trajectory centroids
4. Fit pathway lines to axon trajectories  
5. Calculate the Optic Nerve Head (ONH) location using constrained optimization
6. Compute soma polar coordinates relative to ONH for ALL cells (including non-RGCs)

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
│              FIRST PASS: RGC CELLS ONLY                         │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ 1. SOMA DETECTION│ │ 2. CNN PREDICTION│ │ 3. POST-PROCESS  │
│ • Temporal range │ │ • 3D convolutions│ │ • Noise filter   │
│ • Gaussian refine│ │ • Probability map│ │ • Soma exclusion │
│ • AIS detection  │ │                  │ │ • Centroid extract│
└──────────────────┘ └──────────────────┘ └──────────────────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
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
│                 5. ONH DETECTION (Iterative Refinement)         │
│   • Global optimization: minimize perpendicular distance        │
│   • Direction constraint: ONH in forward direction              │
│   • Distance constraint: within 98px of center                  │
│   • Iterative pathway elimination                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              SECOND PASS: NON-RGC CELLS                         │
│   • Soma/AIS detection only (no CNN)                            │
│   • Polar coordinates using ONH from RGCs                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: HDF5 File                            │
│   • metadata/ap_tracking/ (retina-level: ONH, DVNT)             │
│   • units/{id}/features/ap_tracking/ (per-cell data)            │
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

## ONH Detection: Iterative Refinement Algorithm

### Overview

The current ONH detection uses **global optimization with iterative refinement**. This approach:
1. Finds the ONH that minimizes perpendicular distance from all centroids
2. Iteratively removes pathways that push the ONH outside the valid boundary
3. Uses direction and distance constraints to ensure physically plausible results

### Algorithm Steps

```
Step 1: Filter Pathways by R² and Direction
        ↓
Step 2: Exclude First 10% of Centroids
        ↓
Step 3: Global Optimization with Constraints
        ↓
Step 4: Iterative Pathway Elimination (if needed)
        ↓
Step 5: Return ONH with ≥5 Pathways Remaining
```

### Step 1: Initial Filtering

```python
# R² Filter: Keep pathways with R² ≥ 0.8
valid_pathways = [p for p in pathways if p.r_squared >= r2_threshold]

# Direction Filter: Calculate consensus direction
# - Weighted circular mean of all pathway directions (weights = R²)
# - Remove pathways with direction > 45° from consensus
consensus = weighted_circular_mean(directions, weights=r2_values)
valid_pathways = [p for p in valid_pathways 
                  if angle_difference(p.direction, consensus) <= 45°]
```

### Step 2: Centroid Preprocessing

```python
# For each cell, exclude the first 10% of centroids
# Rationale: Initial centroids near soma are less reliable for projection
n_exclude = int(len(centroids) * 0.1)
remaining_centroids = centroids[n_exclude:]

# The new "start point" for projection = first remaining centroid
start_point = remaining_centroids[0]
```

### Step 3: Global Optimization

The ONH is found by minimizing a loss function with scipy:

```python
# Loss Function:
# L(ONH) = Σ (perpendicular_distance(centroid_i, line(ONH → start_point_i)))²
#
# For each cell with start_point and centroids:
#   - Define a line from ONH to start_point
#   - Calculate perpendicular distance from each centroid to this line
#   - Sum squared distances across all cells and centroids

def loss_function(onh_candidate):
    total_error = 0
    for start_point, centroids in cell_data:
        for centroid in centroids:
            dist = perpendicular_distance(centroid, onh_candidate, start_point)
            total_error += dist ** 2
    return total_error
```

**Constraints** (enforced via scipy SLSQP):

1. **Distance Constraint**: ONH must be within 98 pixels of chip center (33, 33)
   ```python
   constraint_1: (onh_x - 33)² + (onh_y - 33)² <= 98²
   ```

2. **Direction Constraint**: ONH must be in the forward direction from cells
   ```python
   # Vector from mean_start_point to ONH
   to_onh = (onh - mean_start_point)
   
   # Consensus direction unit vector
   consensus_vec = (cos(consensus_angle), sin(consensus_angle))
   
   constraint_2: dot(to_onh, consensus_vec) >= 0
   ```

### Step 4: Iterative Pathway Elimination with Escalating Removal

If the optimized ONH is at the boundary (> 95% of max distance), problematic pathways are iteratively removed using an **escalating removal strategy**:

```python
current_removal_rate = 0.10  # Start at 10%
no_progress_count = 0

while distance_to_center(onh) > 0.95 * max_distance:
    # Stop conditions
    if len(remaining_pathways) < min_pathways:  # Default: 5
        break
    if len(remaining_pathways) < initial_count * min_remaining_fraction:  # Default: 20%
        break
    
    # Calculate per-unit projection error
    errors = [calculate_unit_projection_error(onh, pathway) for pathway in remaining_pathways]
    
    # Remove worst-fitting pathways (highest error)
    n_to_remove = max(1, int(len(remaining_pathways) * current_removal_rate))
    worst_pathways = get_worst_n(pathways, errors, n_to_remove)
    remaining_pathways.remove_all(worst_pathways)
    
    # Re-optimize with remaining pathways
    onh = optimize(remaining_pathways)
    
    # Check for progress
    if distance_improvement < 0.5:  # Less than 0.5px improvement
        no_progress_count += 1
    else:
        no_progress_count = 0
        current_removal_rate = 0.10  # Reset on progress
    
    # Escalate removal rate if stuck
    if no_progress_count >= 3:
        if current_removal_rate < 0.20:
            current_removal_rate = 0.20  # Jump to 20%
        else:
            current_removal_rate = min(current_removal_rate + 0.10, 0.50)  # +10%, max 50%
        no_progress_count = 0
```

**Escalation Strategy**:
| Condition | Action |
|-----------|--------|
| Making progress | Use 10% removal rate |
| No progress for 3 iterations | Escalate to 20% |
| Still stuck for 3 more iterations | Escalate to 30% |
| Continue pattern | +10% each time, max 50% |
| Progress resumes | Reset to 10% |

**Stop Conditions**:
1. ONH converges within boundary (< 95% of max distance) ✓ Success
2. Fewer than `min_pathways` (5) remain
3. Fewer than `min_remaining_fraction` (20%) of original pathways remain

**Per-Unit Error Calculation**:
```python
# Calculate mean squared perpendicular distance for a single pathway

def calculate_unit_projection_error(onh, start_point, centroids):
    total_error = 0
    for centroid in centroids:
        dist = perpendicular_distance(centroid, line(onh, start_point))
        total_error += dist ** 2
    return total_error / len(centroids)  # Mean squared error
```

### Error Metric

The final RMSE (Root Mean Squared Error) is calculated:

```python
rmse = sqrt(total_squared_error / n_centroids)
```

Lower RMSE indicates better fit (all centroids align well with projections to ONH).

---

## Second Pass: Non-RGC Cell Processing

After the ONH is determined from RGC cells, **all non-RGC cells** (AC, unknown, other) are processed:

```python
# For each non-RGC cell:
#   1. Detect soma from STA (same algorithm as RGC)
#   2. Detect AIS
#   3. Calculate polar coordinates using ONH from RGCs
#   4. Write to HDF5 (without CNN prediction or pathway data)
```

This ensures all cells have:
- `refined_soma/` - Soma position
- `axon_initial_segment/` - AIS position  
- `soma_polar_coordinates/` - Position relative to ONH

But non-RGC cells do NOT have:
- `prediction_sta_data` - CNN output
- `post_processed_data/` - Filtered predictions and centroids
- `ap_pathway/` - Fitted pathway line

---

## Comparison: Iterative Refinement vs Legacy

| Feature | Iterative Refinement (Current) | Legacy Algorithm |
|---------|-------------------------------|------------------|
| **Method** | Global optimization | Pairwise intersections |
| **Loss Function** | Sum of squared perpendicular distances | N/A (geometric) |
| **R² Filtering** | Yes (≥0.8, retry at 0.6, 0.4) | No |
| **Direction Filtering** | Yes (±45° from consensus) | No |
| **Distance Constraint** | Enforced in optimizer (≤98px) | Post-hoc filter |
| **Direction Constraint** | Enforced in optimizer | None |
| **Outlier Rejection** | Escalating removal (10%→20%→30%...) | DBSCAN clustering |
| **Removal Strategy** | Error-based batch removal | N/A |
| **Minimum Pathways** | 5 or 20% of original | 2 required |
| **Robustness** | Very high | Moderate |
| **Output Field** | `method: "iterative_refinement"` | `method: "legacy_weighted_mean"` |

### Retry Logic

The algorithm attempts progressively lower R² thresholds:

```
Try R² ≥ 0.8 → If fail...
Try R² ≥ 0.6 → If fail...
Try R² ≥ 0.4 → If fail...
Use legacy method (pairwise intersection clustering)
```

The actual R² threshold that succeeded is recorded in the HDF5 output.

---

## Output Data Structure

### Retina-Level Data (shared by all cells)

Stored at `metadata/ap_tracking/`:

```
metadata/ap_tracking/
├── DV_position           # Dorso-Ventral position from Center_xy
├── NT_position           # Naso-Temporal position from Center_xy
├── LR_position           # Left/Right eye ("L" or "R")
├── _processed_at         # ISO timestamp of processing
└── all_ap_intersection/
    ├── x                 # ONH x coordinate (column)
    ├── y                 # ONH y coordinate (row)
    ├── mse               # Mean squared error
    ├── rmse              # Root mean squared error
    ├── method            # "iterative_refinement" or "legacy_weighted_mean"
    ├── r2_threshold      # Actual R² threshold used
    ├── consensus_direction  # Degrees [0-360]
    ├── n_cells_used      # Number of cells in final optimization
    ├── n_centroids_used  # Total centroids used
    ├── centroid_exclude_fraction  # Fraction excluded (0.1 = 10%)
    ├── max_distance_from_center   # Maximum allowed distance (98)
    ├── outlier_unit_ids  # Unit IDs removed for boundary compliance
    └── kept_unit_ids     # Unit IDs kept in final optimization
```

### Per-Cell Data

Stored at `units/{unit_id}/features/ap_tracking/`:

**For RGC cells:**
```
ap_tracking/
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
│   ├── r_squared        # R² value
│   ├── direction_angle  # Direction in degrees
│   └── start_point      # (row, col) of first centroid
└── soma_polar_coordinates/
    ├── radius           # Distance from ONH to soma
    ├── angle            # Angle in radians
    ├── quadrant         # "Q1", "Q2", "Q3", "Q4"
    └── anatomical_quadrant  # e.g., "dorsal-nasal"
```

**For non-RGC cells:**
```
ap_tracking/
├── refined_soma/        # Same as RGC
├── axon_initial_segment/  # Same as RGC
└── soma_polar_coordinates/  # Same as RGC (uses ONH from RGCs)
```

---

## Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r2_threshold` | 0.8 | Minimum R² for pathway inclusion |
| `direction_tolerance` | 45° | Max deviation from consensus direction |
| `max_distance_from_center` | 98 | Max ONH distance from (33,33) |
| `center_point` | (33, 33) | Center of MEA chip |
| `centroid_exclude_fraction` | 0.1 | Exclude first 10% of centroids |
| `centroid_start_frame` | 10 | Exclude centroids before this frame |
| `max_displacement_post` | 5.0 | Max displacement for trajectory continuity |
| `min_points_for_fit` | 10 | Minimum centroids for pathway fitting |
| `min_pathways` | 5 | Minimum pathways for ONH calculation |
| `min_remaining_fraction` | 0.2 | Stop if < 20% of original pathways remain |
| `boundary_tolerance` | 0.95 | Consider "at boundary" if > 95% of max distance |
| `max_iterations` | 50 | Maximum pathway elimination iterations |
| `removal_fraction` | 0.1 | Starting removal rate (10%), escalates on no progress |

---

## References

- Constrained optimization: scipy.optimize.minimize with SLSQP
- Circular statistics: Mardia & Jupp, 2000
- Spike-triggered average: Chichilnisky, 2001
