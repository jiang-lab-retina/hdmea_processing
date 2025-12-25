# eimage_sta Geometry Extraction Pipeline

This document explains the computational algorithm and usage of the soma geometry extraction module (`ap_sta.py`) for extracting cell body (soma) location and size from spike-triggered average (STA) electrode images.

## Overview

The **eimage_sta** (electrode image spike-triggered average) represents the average electrical activity pattern recorded across a multi-electrode array (MEA) aligned to spike times. This spatial signature reveals the soma location and extent of each recorded neuron.

The geometry extraction pipeline uses the **MaxMin method** to:
1. Identify the soma center location
2. Estimate the soma size (width, height, area)
3. Compute derived metrics (equivalent diameter, aspect ratio)

---

## Computational Algorithm

### Step 1: Temporal Preprocessing

Before analysis, the raw eimage_sta data undergoes preprocessing:

```
Input: eimage_sta[time, rows, cols] - 3D array
```

#### 1.1 Baseline Subtraction
Remove the mean of the first N frames (default: 5) to center the signal around zero:

$$
\text{data}(t, r, c) = \text{eimage\_sta}(t, r, c) - \frac{1}{N}\sum_{i=0}^{N-1} \text{eimage\_sta}(i, r, c)
$$

#### 1.2 Temporal Smoothing
Apply 1D Gaussian smoothing along the time axis ($\sigma_t = 2.0$ frames):

$$
\text{data}_{\text{smooth}}(t, r, c) = G_{\sigma_t} * \text{data}(t, r, c)
$$

### Step 2: Center Detection (MaxMin Method)

The soma center is identified as the electrode with the **largest peak-to-trough amplitude** in the action potential waveform.

#### 2.1 Robust Peak-Trough Detection
For each electrode $(r, c)$, apply Savitzky-Golay filtering (window=7, order=3) to the temporal signal, then find the maximum and minimum values:

$$
\text{diff\_map}(r, c) = \max_t[\text{signal}(t)] - \min_t[\text{signal}(t)]
$$

#### 2.2 Find Center
The soma center is the electrode with maximum difference:

$$
(\text{center\_row}, \text{center\_col}) = \arg\max_{r,c} \text{diff\_map}(r, c)
$$

**Rationale**: The soma location shows the strongest voltage swing during an action potential because it's the site of maximum current density.

### Step 3: Size Estimation

Size is estimated using a **subset of frames** (default: frames 10-14) to focus on the action potential peak.

#### 3.1 Compute Difference Map
Using only the specified frame range:

$$
\text{diff\_map}(r, c) = \max_t[\text{data}(t, r, c)] - \min_t[\text{data}(t, r, c)]
$$

#### 3.2 Apply Gaussian Smoothing
Smooth the difference map ($\sigma = 1.0$) to reduce noise.

#### 3.3 Threshold and Mask
Create a binary mask of high-activity regions:

$$
\text{mask}(r, c) = \begin{cases} 1 & \text{if } \text{diff\_smooth}(r, c) > \theta \cdot \max(\text{diff\_smooth}) \\ 0 & \text{otherwise} \end{cases}
$$

Default threshold fraction $\theta = 0.5$ (half-maximum).

#### 3.4 Connected Component Analysis
Extract the connected component containing the soma center. This isolates the soma from other high-activity regions (e.g., axons).

#### 3.5 Compute Size Metrics

From the soma mask:

| Metric | Formula |
|--------|---------|
| **size_x** (width) | $\max(x) - \min(x) + 1$ |
| **size_y** (height) | $\max(y) - \min(y) + 1$ |
| **area** | $\sum \text{mask}(r, c)$ |
| **equivalent_diameter** | $2\sqrt{\text{area}/\pi}$ |

---

## Output Structure

### HDF5 Storage Path
```
units/{unit_id}/features/eimage_sta/
├── data                    # Original eimage_sta array (time, rows, cols)
└── geometry/               # Extracted geometry
    ├── center_row          # Soma center row coordinate
    ├── center_col          # Soma center column coordinate
    ├── size_x              # Width in electrodes
    ├── size_y              # Height in electrodes
    ├── area                # Area in electrodes²
    ├── equivalent_diameter # √(4·area/π)
    └── diff_map            # 2D max-min difference map
```

### Python Data Structure
```python
@dataclass
class SomaGeometry:
    center_row: float
    center_col: float
    size_x: float
    size_y: float
    area: float
    equivalent_diameter: float
    diff_map: Optional[np.ndarray] = None
```

---

## Usage

### Pipeline Mode (Recommended)

Use with `PipelineSession` for deferred saving - follows the same pattern as other hdmea pipeline functions:

```python
from hdmea.pipeline import create_session, load_recording_with_eimage_sta
from sta_quantification.ap_sta import extract_eimage_sta_geometry

# 1. Create session
session = create_session(dataset_id="2024.03.01-14.40.14-Rec")

# 2. Load recording with eimage_sta computation
session = load_recording_with_eimage_sta(
    cmcr_path="path/to/recording.cmcr",
    cmtr_path="path/to/recording.cmtr",
    duration_s=120.0,
    spike_limit=10000,
    window_range=(-10, 40),
    session=session,
)

# 3. Extract soma geometry
session = extract_eimage_sta_geometry(
    frame_range=(10, 14),      # Frames for size estimation
    threshold_fraction=0.5,    # Threshold for soma mask
    session=session,           # Pass session for deferred mode
)

# 4. Save once at the end
hdf5_path = session.save()
print(f"Completed steps: {session.completed_steps}")
```

### Standalone Mode (Direct HDF5)

For processing existing HDF5 files without session management:

```python
from sta_quantification.ap_sta import extract_eimage_sta_geometry

# Process and save directly to HDF5
extract_eimage_sta_geometry(
    frame_range=(10, 14),
    threshold_fraction=0.5,
    hdf5_path="path/to/file.h5",  # Immediate mode
)
```

### Loading Existing HDF5 into Session

Use the helper script `run_geometry_session.py` or:

```python
from sta_quantification.run_geometry_session import (
    load_hdf5_to_session,
    save_session_to_hdf5,
)
from sta_quantification.ap_sta import extract_eimage_sta_geometry

# Load existing HDF5 into session
session = load_hdf5_to_session("path/to/file.h5")

# Run geometry extraction
session = extract_eimage_sta_geometry(session=session)

# Save back to HDF5
save_session_to_hdf5(session, "path/to/file.h5")
```

### Visualization

```python
from sta_quantification.ap_sta import (
    plot_geometry_results,
    load_geometries_from_hdf5,
)

# Load geometries and generate plots
geometries = load_geometries_from_hdf5("path/to/file.h5")
plot_geometry_results(
    hdf5_path="path/to/file.h5",
    geometries=geometries,
    output_dir="results/",
)
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_range` | (10, 14) | Frames to use for size estimation |
| `threshold_fraction` | 0.5 | Fraction of max for soma threshold |
| `sigma_t` | 2.0 | Temporal smoothing sigma (frames) |
| `baseline_frames` | 5 | Frames for baseline subtraction |
| `savgol_window` | 7 | Savitzky-Golay filter window |
| `savgol_order` | 3 | Savitzky-Golay polynomial order |

To modify defaults, edit `TemporalConfig` in `ap_sta.py`:

```python
@dataclass
class TemporalConfig:
    sigma_t: float = 2.0
    baseline_frames: int = 5
    savgol_window: int = 7
    savgol_order: int = 3
```

---

## Output Visualization

The module generates two types of plots:

### 1. Per-Unit Plot (`{unit_id}_geometry.png`)

A 2×2 subplot showing:
- **Top-left**: Peak activity frame with soma ellipse overlay
- **Top-right**: Difference map with center marker
- **Bottom-left**: Cross-section profiles (horizontal/vertical)
- **Bottom-right**: Geometry summary table

### 2. Summary Plot (`geometry_summary.png`)

Aggregated statistics across all units:
- Scatter plot of soma center locations
- Histogram of equivalent diameters
- Box plot of size distributions
- Summary table with mean/std

---

## File Structure

```
Projects/sta_quantification/
├── ap_sta.py                      # Main module with geometry extraction
├── run_geometry_session.py        # Session workflow demo script
├── eimage_pipeline_explained.md   # This documentation
└── results/                       # Output plots
    ├── unit_001_geometry.png
    ├── unit_002_geometry.png
    ├── ...
    └── geometry_summary.png
```

---

## API Reference

### Core Functions

```python
def extract_eimage_sta_geometry(
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
    session: Optional[PipelineSession] = None,
    hdf5_path: Optional[Path] = None,
) -> Optional[PipelineSession]:
    """
    Extract soma geometry for all units with eimage_sta data.
    
    Returns:
        Updated session (session mode) or None (immediate mode)
    """
```

```python
def extract_soma_geometry(
    eimage_sta: np.ndarray,
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
) -> SomaGeometry:
    """
    Extract geometry from a single eimage_sta array.
    
    Returns:
        SomaGeometry dataclass with all metrics
    """
```

### Helper Functions

```python
def load_geometries_from_hdf5(hdf5_path: Path) -> Dict[str, SomaGeometry]:
    """Load computed geometries from HDF5 file."""

def plot_geometry_results(
    hdf5_path: Path,
    geometries: Dict[str, SomaGeometry],
    output_dir: Path,
) -> None:
    """Generate all visualization plots."""
```

---

## Example Output

```
[extract_eimage_sta_geometry] Processing 152 units...
  unit_001: center=(37.0, 44.0), size=5.0×4.0, Ø=4.5
  unit_002: center=(38.0, 40.0), size=4.0×5.0, Ø=4.5
  unit_003: center=(44.0, 52.0), size=4.0×6.0, Ø=4.9
  ...
[extract_eimage_sta_geometry] Computed geometry for 152 units
```

---

## References

- **Spike-Triggered Average (STA)**: Chichilnisky, E.J. (2001). A simple white noise analysis of neuronal light responses. Network: Computation in Neural Systems.
- **MaxMin Method**: Based on peak-to-peak voltage analysis common in MEA signal processing.
- **Connected Component Analysis**: Used for isolating soma from axonal signals.

