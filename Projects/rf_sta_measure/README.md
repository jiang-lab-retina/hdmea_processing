# RF-STA Receptive Field Measurement

This module extracts receptive field (RF) geometry from spike-triggered average (STA) data computed with dense noise stimulus. It performs Gaussian, Difference of Gaussians (DoG), and ON/OFF fitting to characterize RF structure.

## Overview

The RF measurement pipeline:

1. **Preprocessing**: Baseline subtraction, padding, Gaussian blur, temporal smoothing
2. **Center Detection**: Find RF center using extreme absolute value method
3. **Fitting**: Apply 2D Gaussian, DoG (center-surround), and ON/OFF models
4. **Export**: Save structured results to HDF5

## Data Flow

```
Input HDF5
    └── units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/data
                                        ↓
                            [Preprocessing Pipeline]
                                        ↓
                            [Center Detection + Fitting]
                                        ↓
Output HDF5
    └── units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/
            ├── center_row, center_col, size_x, size_y, area, equivalent_diameter, peak_frame
            ├── gaussian_fit/
            ├── DoG/
            └── ONOFF_model/
```

## Preprocessing Pipeline

The preprocessing is performed in the following order:

### 1. Baseline Subtraction (First)
- Uses frames 0-10 as baseline
- Subtracts mean baseline from all frames
- Centers the data around zero

### 2. Padding
- Adds 5-pixel padding to spatial dimensions
- Padding value is 0 (since data is already baseline-subtracted)
- Prevents edge effects during filtering

### 3. Gaussian Blur (2D)
- Applies 2D Gaussian filter with $\sigma = 1.5$ pixels
- Smooths spatial noise while preserving RF structure

### 4. Temporal Smoothing
- Applies 1D Gaussian filter along time axis with $\sigma_t = 2.0$ frames
- Reduces temporal noise

## Center Detection

The RF center is found using the **extreme absolute value method**:

1. For each pixel $(x, y)$, compute the extreme value across time (largest absolute magnitude)
2. Create an "extreme map" where each pixel contains its extreme value
3. The RF center is the pixel with the maximum absolute value in this map

This method works well for both ON and OFF cells, as it considers the strongest response regardless of polarity.

## Fitting Models

### 2D Gaussian Fit

Fits a rotated elliptical 2D Gaussian:

$$G(x,y) = A \exp\left(-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}\right)\right) + \text{offset}$$

**Parameters**:
- `center_x`, `center_y`: RF center position
- `sigma_x`, `sigma_y`: RF size (standard deviation in each axis)
- `amplitude`: Peak amplitude
- `theta`: Rotation angle (radians)
- `offset`: Baseline offset
- `r_squared`: Fit quality ($R^2$)

**Constraints**:
- Center constrained within 5 pixels of initial estimate
- Maximum $\sigma$ = 7.5 pixels (15 pixel diameter)

### DoG (Difference of Gaussians) Fit

Models center-surround organization:

$$\text{DoG}(x,y) = A_{\text{exc}} \cdot G(\sigma_{\text{exc}}) - A_{\text{inh}} \cdot G(\sigma_{\text{inh}}) + \text{offset}$$

**Parameters**:
- `center_x`, `center_y`: RF center position
- `sigma_exc`: Excitatory (center) size
- `sigma_inh`: Inhibitory (surround) size
- `amp_exc`, `amp_inh`: Amplitudes
- `offset`: Baseline offset
- `r_squared`: Fit quality

### ON/OFF Model

Fits separate Gaussians to positive (ON) and negative (OFF) components:

**ON parameters**: `on_center_x`, `on_center_y`, `on_sigma_x`, `on_sigma_y`, `on_amplitude`, `on_r_squared`

**OFF parameters**: `off_center_x`, `off_center_y`, `off_sigma_x`, `off_sigma_y`, `off_amplitude`, `off_r_squared`

## HDF5 Output Structure

Results are saved under each unit's STA feature:

```
units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/
    ├── center_row          # RF center Y coordinate (original frame)
    ├── center_col          # RF center X coordinate (original frame)
    ├── size_x              # Bounding box width
    ├── size_y              # Bounding box height
    ├── area                # RF area in pixels²
    ├── equivalent_diameter # Circular equivalent diameter
    ├── peak_frame          # Frame with maximum activity
    │
    ├── gaussian_fit/
    │   ├── center_x
    │   ├── center_y
    │   ├── sigma_x
    │   ├── sigma_y
    │   ├── amplitude
    │   ├── theta
    │   ├── offset
    │   └── r_squared
    │
    ├── DoG/
    │   ├── center_x
    │   ├── center_y
    │   ├── sigma_exc
    │   ├── sigma_inh
    │   ├── amp_exc
    │   ├── amp_inh
    │   ├── offset
    │   └── r_squared
    │
    └── ONOFF_model/
        ├── on_center_x
        ├── on_center_y
        ├── on_sigma_x
        ├── on_sigma_y
        ├── on_amplitude
        ├── on_r_squared
        ├── off_center_x
        ├── off_center_y
        ├── off_sigma_x
        ├── off_sigma_y
        ├── off_amplitude
        └── off_r_squared
```

## Usage

### Option 1: Session Pipeline (Recommended)

Use `rf_session.py` for session-based processing that preserves all existing HDF5 data:

```python
from rf_session import (
    load_hdf5_to_session,
    extract_rf_geometry_session,
    save_rf_geometry_to_hdf5,
    FRAME_RANGE,
    THRESHOLD_FRACTION,
)

# Load HDF5 into session
session = load_hdf5_to_session("path/to/input.h5")

# Extract RF geometry for all units
session = extract_rf_geometry_session(
    session,
    frame_range=FRAME_RANGE,      # (40, 60)
    threshold_fraction=THRESHOLD_FRACTION,  # 0.5
)

# Save to output (preserves all existing data)
save_rf_geometry_to_hdf5(session, "path/to/output.h5")
```

### Option 2: Batch Processing

Use `batch_rf.py` to process multiple files:

```bash
python batch_rf.py
```

This processes all HDF5 files in `eimage_sta_output_20251225/` and exports to `rf_sta_output_20251225/`.

Features:
- Progress tracking
- Skips existing outputs (allows resuming)
- Creates processing log
- Handles errors gracefully

### Option 3: Single Unit Analysis with Plots

Use `rf_sta_measure.py` for detailed analysis with visualization:

```python
from rf_sta_measure import (
    extract_rf_geometry,
    plot_unit_rf_geometry,
    load_sta_data,
)

# Load STA data
sta_data = load_sta_data("path/to/input.h5")

# Process single unit
geometry = extract_rf_geometry(
    sta_data["unit_001"],
    frame_range=(40, 60),
    threshold_fraction=0.5,
)

# Generate plot
plot_unit_rf_geometry(
    "unit_001",
    sta_data["unit_001"],
    geometry,
    output_path="results/unit_001_rf_geometry.png",
)
```

## Configuration Parameters

### Spatial Parameters (`SpatialConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `padding` | 5 | Padding size in pixels |
| `gaussian_sigma` | 1.5 | 2D Gaussian blur sigma |
| `center_fit_radius` | 5.0 | Maximum center offset for fitting |
| `max_rf_diameter` | 15.0 | Maximum RF diameter (pixels) |

### Temporal Parameters (`TemporalConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_t` | 2.0 | Temporal smoothing sigma (frames) |
| `baseline_frames` | 10 | Number of frames for baseline (0-10) |
| `savgol_window` | 7 | Savitzky-Golay filter window |
| `savgol_order` | 3 | Savitzky-Golay polynomial order |

### Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAME_RANGE` | (40, 60) | Frames to analyze for RF |
| `THRESHOLD_FRACTION` | 0.5 | Threshold for RF mask |

## File Structure

```
Projects/rf_sta_measure/
├── rf_sta_measure.py    # Core RF extraction and visualization
├── rf_session.py        # Session-based workflow
├── batch_rf.py          # Batch processing script
├── README.md            # This documentation
├── results/             # Visualization output (from rf_sta_measure.py)
├── export/              # Single file export (from rf_session.py)
└── rf_sta_output_*/     # Batch export (from batch_rf.py)
```

## Dependencies

- numpy
- scipy (ndimage, signal, optimize)
- h5py
- matplotlib
- hdmea.pipeline (PipelineSession, create_session)

## Example Output

The visualization (`rf_sta_measure.py`) generates a 3×3 subplot:

1. **Peak Frame**: STA at peak activity with center marker
2. **Difference Map**: Max-min temporal difference
3. **Extreme Value Map**: Absolute extreme values
4. **Gaussian Fit**: Fitted Gaussian with contours
5. **DoG Fit**: Center-surround model visualization
6. **ON/OFF Comparison**: Separate ON and OFF centers
7. **Temporal Profile**: Time course at RF center
8. **Geometry Summary**: Text summary of measurements
9. **Fit Statistics**: R² values and fit parameters

