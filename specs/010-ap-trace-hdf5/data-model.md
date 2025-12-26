# Data Model: Axon Tracking (AP Trace) for HDF5 Pipeline

**Feature Branch**: `010-ap-trace-hdf5`
**Date**: 2025-12-25

## Entities

### 1. STAData

**Description**: 3D Spike-Triggered Average data from eimage movie stimulus.

**Location**: `units/{unit_id}/features/eimage_sta/data`

| Field | Type | Description |
|-------|------|-------------|
| data | np.ndarray[float32] | 3D array shape (50, 65, 65) - (time, row, col) |

**Validation**:
- Shape must be 3D
- Time dimension typically 50 frames
- Spatial dimensions typically 65x65

---

### 2. CellGeometry

**Description**: Geometric properties of the cell derived from STA analysis.

**Location**: `units/{unit_id}/features/eimage_sta/geometry/`

| Field | Type | Description |
|-------|------|-------------|
| center_row | int | Y coordinate of cell center (row index) |
| center_col | int | X coordinate of cell center (column index) |

**Validation**:
- center_row: 0 <= value < sta_height (65)
- center_col: 0 <= value < sta_width (65)

---

### 3. DVNTMetadata

**Description**: Dorsal-Ventral and Nasal-Temporal anatomical position.

**Input Location**: `metadata/gsheet_row/Center_xy`

| Field | Type | Description |
|-------|------|-------------|
| Center_xy | str | Format: "L/R, VD_coord, NT_coord" (e.g., "L, 1.5, -0.8") |

**Derived Fields** (stored in output):
| Field | Type | Description |
|-------|------|-------------|
| DV_position | float | Dorsal-ventral position (-VD_coord, positive=dorsal) |
| NT_position | float | Nasal-temporal position (positive=nasal) |
| LR_position | str | Left/Right eye ("L" or "R") |

---

### 4. RefinedSoma

**Description**: Refined soma (cell body) position in 3D STA space.

**Output Location**: `units/{unit_id}/features/ap_tracking/refined_soma/`

| Field | Type | Description |
|-------|------|-------------|
| t | int | Time index of peak soma signal |
| x | int | X coordinate (row index) |
| y | int | Y coordinate (column index) |

**Algorithm**: `soma_refiner()` - finds minimum value in local neighborhood

---

### 5. AxonInitialSegment

**Description**: Detected axon initial segment (AIS) position.

**Output Location**: `units/{unit_id}/features/ap_tracking/axon_initial_segment/`

| Field | Type | Description |
|-------|------|-------------|
| t | int or None | Time index of AIS detection |
| x | int | X coordinate (row index) |
| y | int | Y coordinate (column index) |

**Algorithm**: `AIS_refiner()` - finds earliest signal near soma

---

### 6. PredictionData

**Description**: CNN model output predicting axon signal probability.

**Output Location**: `units/{unit_id}/features/ap_tracking/prediction_sta_data`

| Field | Type | Description |
|-------|------|-------------|
| prediction_sta_data | np.ndarray[float32] | Shape matches STA (50, 65, 65), values 0-1 |

**Algorithm**: Sliding window CNN inference with 5x5x5 cubes

---

### 7. PostProcessedData

**Description**: Filtered and refined axon detection results.

**Output Location**: `units/{unit_id}/features/ap_tracking/post_processed_data/`

| Field | Type | Description |
|-------|------|-------------|
| filtered_prediction | np.ndarray[float32] | Noise-filtered prediction map |
| axon_centroids | np.ndarray[float32] | Shape (N, 3) - [t, x, y] for each detected point |

**Algorithm**: Temporal filtering, noise removal, centroid extraction

---

### 8. APPathway

**Description**: Fitted line to axon projection for determining AP direction.

**Output Location**: `units/{unit_id}/features/ap_tracking/ap_pathway/`

| Field | Type | Description |
|-------|------|-------------|
| slope | float | Line slope |
| intercept | float | Line y-intercept |
| r_value | float | Correlation coefficient |
| p_value | float | Statistical significance |
| std_err | float | Standard error of estimate |

**Algorithm**: Linear regression on axon centroid projections

---

### 9. APIntersection

**Description**: Optimal intersection point of all AP pathways in a recording.

**Output Location**: `units/{unit_id}/features/ap_tracking/all_ap_intersection/`

| Field | Type | Description |
|-------|------|-------------|
| x | float | X coordinate of intersection |
| y | float | Y coordinate of intersection |

**Algorithm**: Least-squares optimization across all valid pathway fits

**Note**: Same value stored in all units of a recording

---

### 10. SomaPolarCoordinates

**Description**: Soma position in polar coordinates relative to AP intersection.

**Output Location**: `units/{unit_id}/features/ap_tracking/soma_polar_coordinates/`

| Field | Type | Description |
|-------|------|-------------|
| radius | float | Distance from intersection point |
| angle | float | Angle in radians from positive x-axis |
| cartesian_x | float | X distance from intersection |
| cartesian_y | float | Y distance from intersection |
| quadrant | str | Geometric quadrant (e.g., "Q1", "Q2", "Q3", "Q4") |
| anatomical_quadrant | str or None | Anatomical description (e.g., "dorsal-nasal") |

**Algorithm**: Cartesian to polar conversion with anatomical labeling

---

## HDF5 Structure Summary

**Design Principle**: All data is stored as explicit datasets, not HDF5 attributes. This improves compatibility with HDF5 viewers and ensures consistent data access patterns.

```
{recording}.h5
├── units/
│   └── {unit_id}/
│       └── features/
│           ├── eimage_sta/              # INPUT (existing)
│           │   ├── data                 # STA array (50,65,65)
│           │   └── geometry/
│           │       ├── center_row       # Y coordinate (dataset, scalar)
│           │       └── center_col       # X coordinate (dataset, scalar)
│           │
│           └── ap_tracking/             # OUTPUT (new)
│               ├── DV_position          # (dataset, scalar float)
│               ├── NT_position          # (dataset, scalar float)
│               ├── LR_position          # (dataset, variable-length string)
│               │
│               ├── refined_soma/
│               │   ├── t                # (dataset, scalar int)
│               │   ├── x                # (dataset, scalar int)
│               │   └── y                # (dataset, scalar int)
│               │
│               ├── axon_initial_segment/
│               │   ├── t                # (dataset, scalar int or empty)
│               │   ├── x                # (dataset, scalar int)
│               │   └── y                # (dataset, scalar int)
│               │
│               ├── prediction_sta_data  # (dataset, float32 array)
│               │
│               ├── post_processed_data/
│               │   ├── filtered_prediction  # (dataset, float32 array)
│               │   └── axon_centroids       # (dataset, float32 Nx3)
│               │
│               ├── ap_pathway/
│               │   ├── slope            # (dataset, scalar float)
│               │   ├── intercept        # (dataset, scalar float)
│               │   ├── r_value          # (dataset, scalar float)
│               │   ├── p_value          # (dataset, scalar float)
│               │   └── std_err          # (dataset, scalar float)
│               │
│               ├── all_ap_intersection/
│               │   ├── x                # (dataset, scalar float)
│               │   └── y                # (dataset, scalar float)
│               │
│               └── soma_polar_coordinates/
│                   ├── radius           # (dataset, scalar float)
│                   ├── angle            # (dataset, scalar float)
│                   ├── cartesian_x      # (dataset, scalar float)
│                   ├── cartesian_y      # (dataset, scalar float)
│                   ├── quadrant         # (dataset, variable-length string)
│                   └── anatomical_quadrant  # (dataset, variable-length string)
│
├── stimulus/                            # (existing, not modified)
│
└── metadata/
    └── gsheet_row/                      # INPUT (existing)
        └── Center_xy                    # DVNT position string
```

## State Transitions

### Unit Processing States

```
┌─────────────────┐
│   UNPROCESSED   │  (no ap_tracking group exists)
└────────┬────────┘
         │ compute_ap_tracking()
         ▼
┌─────────────────┐
│   IN_PROGRESS   │  (processing started)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌───────┐
│COMPLETE│  │SKIPPED│  (no eimage_sta data)
└───────┘  └───────┘
```

### File Processing States

```
┌─────────────────┐
│  NO_STA_DATA    │  (file lacks eimage_sta)
└─────────────────┘

┌─────────────────┐
│  PARTIAL_FIT    │  (<2 valid AP pathway fits)
│  No polar coords│
└─────────────────┘

┌─────────────────┐
│   COMPLETE      │  (≥2 fits, polar coords calculated)
└─────────────────┘
```

## Validation Rules

| Entity | Rule | Error Handling |
|--------|------|----------------|
| STAData | Must be 3D numpy array | Skip unit, log warning |
| STAData | Time dim >= 5 | Skip unit, log warning |
| CellGeometry | Coordinates within bounds | Use detected soma position |
| DVNTMetadata | Valid format "L/R, num, num" | Set positions to NaN |
| APPathway | R² > 0.8 | Exclude from intersection calc |
| APIntersection | ≥2 valid pathways | Skip polar coords for file |

