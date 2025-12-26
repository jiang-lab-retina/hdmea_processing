# Research: Axon Tracking (AP Trace) for HDF5 Pipeline

**Feature Branch**: `010-ap-trace-hdf5`
**Date**: 2025-12-25

## Research Summary

This document consolidates research findings for migrating the AP trace feature from pkl to HDF5.

---

## R1: HDF5 Data Location for STA Data

**Question**: Where is STA data stored in the new HDF5 structure?

**Finding**: STA data for the eimage movie is stored at:
```
units/{unit_id}/features/eimage_sta/data
```

**Source**: User clarification + HDF5 inspection of test file

**Decision**: Read from `units/{unit_id}/features/eimage_sta/data`

---

## R2: Cell Location for Polar Coordinate Calculations

**Question**: Where is the cell location (soma center) stored for polar coordinate calculations?

**Finding**: Cell location is stored in the geometry subgroup of eimage_sta:
```
units/{unit_id}/features/eimage_sta/geometry/center_row  → Y coordinate
units/{unit_id}/features/eimage_sta/geometry/center_col  → X coordinate
```

**Source**: User clarification referencing `rf_sta_pipeline_explained.md`

**Decision**: Use `center_row` as Y and `center_col` as X (standard NumPy array indexing)

---

## R3: DVNT Position Source and Conversion

**Question**: How should DV/NT/LR positions be derived?

**Finding**: Parse from `metadata/gsheet_row/Center_xy`:
- Format: "L/R, VD_coord, NT_coord" (e.g., "L, 1.5, -0.8")
- Conversion formula (from `auto_label.py:423-453`):
  ```python
  position_string = Center_xy.replace(" ", "")
  LR_position = position_string.split(",")[0].upper()
  VD_coordinate = float(position_string.split(",")[1])  # ventral is positive
  DV_position = -VD_coordinate  # positive = dorsal, negative = ventral
  NT_position = float(position_string.split(",")[2])  # positive = nasal
  ```

**Source**: User clarification + legacy code `auto_label.py` lines 423-453

**Decision**: Implement `parse_dvnt_from_center_xy()` function with this exact conversion

---

## R4: Output Feature Group Structure

**Question**: What should the output HDF5 structure look like?

**Finding**: All AP tracking outputs go under `features/ap_tracking/`. All values are stored as **explicit datasets** (not HDF5 attributes) for consistency and better tooling compatibility.

```
units/{unit_id}/features/ap_tracking/
├── DV_position          (dataset: scalar float64)
├── NT_position          (dataset: scalar float64)
├── LR_position          (dataset: variable-length string)
├── refined_soma/
│   ├── t                (dataset: scalar int64)
│   ├── x                (dataset: scalar int64)
│   └── y                (dataset: scalar int64)
├── axon_initial_segment/
│   ├── t                (dataset: scalar int64 or empty)
│   ├── x                (dataset: scalar int64)
│   └── y                (dataset: scalar int64)
├── prediction_sta_data  (dataset: float32 array 50x65x65)
├── post_processed_data/
│   ├── filtered_prediction  (dataset: float32 array)
│   └── axon_centroids       (dataset: float32 Nx3)
├── ap_pathway/
│   ├── slope            (dataset: scalar float64)
│   ├── intercept        (dataset: scalar float64)
│   ├── r_value          (dataset: scalar float64)
│   ├── p_value          (dataset: scalar float64)
│   └── std_err          (dataset: scalar float64)
├── all_ap_intersection/
│   ├── x                (dataset: scalar float64)
│   └── y                (dataset: scalar float64)
└── soma_polar_coordinates/
    ├── radius           (dataset: scalar float64)
    ├── angle            (dataset: scalar float64, radians)
    ├── cartesian_x      (dataset: scalar float64)
    ├── cartesian_y      (dataset: scalar float64)
    ├── quadrant         (dataset: variable-length string)
    └── anatomical_quadrant (dataset: variable-length string)
```

**Source**: Legacy `apply_model_for_ap_trace()` output structure + user requirements

**Decision**: Store all values as HDF5 datasets (not attributes). Scalar values use shape `()` datasets. Strings use h5py special dtype for variable-length strings.

**Rationale for datasets over attributes**:
1. Consistent access pattern: all data read with `dataset[()]` syntax
2. Better HDF5 viewer compatibility (attributes often hidden)
3. Supports NaN values for missing data (attributes have limitations)
4. Easier to iterate and inspect programmatically

---

## R5: CNN Model Architecture

**Question**: What is the architecture of the CNN model to be loaded?

**Finding**: The model is `CNN3D_WithVelocity` with:
- Input: 5x5x5 cube from STA data (input_dim=(5,5,5))
- Auxiliary features: 2 (velocity, direction - but set to 0 during inference)
- Output: 1 (axon probability)
- Architecture:
  ```python
  class CNN3D_WithVelocity(nn.Module):
      def __init__(self, input_dim=(5,5,5), aux_features=2, num_classes=1):
          # 3D Conv layers
          # Fully connected layers
          # Auxiliary feature fusion
  ```

**Source**: Legacy code `A05_3D_CNN_with_velocity_model.py`

**Decision**: Reimplement the model class in `model_inference.py` to load saved weights

---

## R6: GPU Memory Management

**Question**: How should GPU memory be managed for batch processing?

**Finding**: Adaptive batch sizing based on GPU memory:
```python
if gpu_memory_gb >= 24:
    batch_size = 512
elif gpu_memory_gb >= 16:
    batch_size = 256
elif gpu_memory_gb >= 8:
    batch_size = 128
else:
    batch_size = 64
```

Additional optimizations:
- Use `torch.no_grad()` during inference
- Clear GPU cache periodically: `torch.cuda.empty_cache()`
- Use pinned memory for faster GPU transfer
- Enable cuDNN optimizations: `torch.backends.cudnn.benchmark = True`

**Source**: Legacy code `use_model_for_single_pkl_content()` lines 234-290

**Decision**: Implement same memory management strategy

---

## R7: Missing Data Handling

**Question**: How to handle missing STA data or metadata?

**Finding**:
- **Missing eimage_sta**: Skip unit, log warning
- **Missing Center_xy metadata**: Set DV/NT/LR positions to None/NaN, continue processing
- **Missing geometry data**: Use soma detection algorithm to find center
- **Model file missing**: Raise clear error with path

**Source**: User acceptance scenarios + spec edge cases

**Decision**: Implement defensive handling with clear logging

---

## R8: Polar Coordinate Calculation Prerequisites

**Question**: What are the prerequisites for calculating polar coordinates?

**Finding**: Polar coordinates require:
1. At least 2 valid AP pathway fits (R² > threshold) across all units in the file
2. An optimal intersection point calculated from all valid fits
3. Soma position for each unit

Without sufficient valid fits, skip polar coordinate calculation for the entire file.

**Source**: Legacy code `get_ap_pathway_and_polar_coordinates()` lines 466-532

**Decision**: Calculate intersection at file level, then apply to all units

---

## R9: Post-Processing Parameters

**Question**: What are the default parameters for axon centroid extraction?

**Finding**: Default parameters from legacy:
```python
temporal_window_size = 5      # Time window for filtering
exclude_radius = 5            # Radius around soma to exclude
centroid_threshold = 0.05     # Minimum prediction value
max_displacement = 5          # Max frame-to-frame movement
min_points_for_fit = 10       # Min points for line fitting
r2_threshold = 0.8            # R² threshold for valid fits
```

**Source**: Legacy code `apply_model_for_ap_trace()` function signature

**Decision**: Use same defaults, expose as parameters

---

## R10: Integration with PipelineSession

**Question**: How should AP tracking integrate with deferred save mode?

**Finding**: Follow the pattern from `add_section_time()`:
1. Accept optional `session` parameter
2. If session provided: accumulate results in memory, return session
3. If no session: open HDF5 in r+ mode, write directly, close file

```python
def compute_ap_tracking(
    hdf5_path: Path,
    model_path: Path,
    *,
    session: Optional[PipelineSession] = None,
    force_cpu: bool = False,
    **kwargs
) -> Optional[PipelineSession]:
    """
    If session is provided, accumulate results and return session.
    If session is None, write directly to HDF5 file.
    """
```

**Source**: Constitution IV.B + `section_time.py` patterns

**Decision**: Implement dual-mode: immediate write or session accumulation

---

---

## R11: Data Storage Strategy - Datasets vs Attributes

**Question**: Should scalar values be stored as HDF5 attributes or datasets?

**Finding**: Store ALL values as explicit HDF5 datasets, not attributes.

**Decision**: Use datasets for everything, including scalar values.

**Rationale**:
1. **Consistency**: Uniform access pattern with `dataset[()]` for all data
2. **Tooling compatibility**: HDF5 viewers (HDFView, h5dump) show datasets prominently; attributes often hidden
3. **NaN support**: Datasets support NaN for missing float values; attributes have limitations
4. **Type flexibility**: Variable-length strings work better as datasets
5. **Iteration**: Easier to programmatically iterate over group contents

**Implementation**:
```python
# Scalar values stored as shape () datasets
group.create_dataset("slope", data=0.5)  # Creates scalar dataset

# Read with [()]
value = group["slope"][()]

# Strings use special dtype
dt = h5py.special_dtype(vlen=str)
group.create_dataset("quadrant", data="Q1", dtype=dt)

# Read string
text = group["quadrant"][()].decode("utf-8")
```

**Rejected Alternative**: Using HDF5 attributes for metadata/scalars
- Pro: Slightly less storage overhead
- Con: Inconsistent access patterns, harder to discover, viewer issues

---

## Alternatives Considered

### A1: Single File vs. Subdirectory Module

**Chosen**: Subdirectory module (`features/ap_tracking/`)

**Rejected Alternative**: Single `features/ap_tracking.py` file

**Reason**: 6 distinct algorithm components would create an 800+ line file, violating maintainability. Subdirectory allows per-component testing.

### A2: Reuse Legacy Code vs. Reimplement

**Chosen**: Reimplement algorithms in new module

**Rejected Alternative**: Import from Legacy_code/

**Reason**: Constitution VII explicitly forbids legacy imports. Core algorithms are pure NumPy and easy to reimplement.

### A3: Process All Units vs. Configurable Limit

**Chosen**: Process all units with optional `max_units` limit

**Rejected Alternative**: Fixed batch sizes

**Reason**: Matches legacy behavior, allows testing with subsets

