# DSGC Direction Sectioning Algorithm

This document explains the algorithm for sectioning spike times by moving bar direction for Direction-Selective Ganglion Cell (DSGC) analysis.

## Overview

The goal is to extract spike responses for each cell when a moving bar stimulus crosses its receptive field (RF) center, organized by motion direction. This enables analysis of direction selectivity.

**Key components:**
1. **Moving bar stimulus**: 8 directions × 3 repetitions = 24 trials
2. **Per-pixel on/off timing dictionary**: Pre-computed frame indices when bar covers/uncovers each pixel
3. **Cell RF center**: From STA (Spike-Triggered Average) analysis, converted to stimulus space
4. **Spike sectioning**: Extract spikes within each trial's time window

---

## Stimulus Design

### Moving Bar Movie: `moving_h_bar_s5_d8_3x.npy`

| Parameter | Value |
|-----------|-------|
| Shape | (13560, 300, 300) |
| Frame rate | 60 Hz |
| Directions | 8 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°) |
| Repetitions | 3 |
| Bar speed | 5 pixels/frame |
| Bar height | 50 pixels |

### Timing Structure

```
Total frames: 13560
├── Rep 1: 4520 frames
│   ├── Pre-margin: 120 frames
│   └── 8 directions × 550 frames each
├── Rep 2: 4520 frames
│   └── (same structure)
└── Rep 3: 4520 frames
    └── (same structure)
```

### Direction Order (within each repetition)

| Index | Direction |
|-------|-----------|
| 0 | 0° (right) |
| 1 | 45° |
| 2 | 90° (down) |
| 3 | 135° |
| 4 | 180° (left) |
| 5 | 225° |
| 6 | 270° (up) |
| 7 | 315° |

---

## On/Off Dictionary Generation

### Source File
`moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl`

### Algorithm

For each pixel (x, y) in the 300×300 grid:

```python
# 1. Extract intensity trace for this pixel
trace = movie[:, x, y]  # Shape: (13560,)

# 2. Compute derivative to detect edges
trace = np.diff(np.concatenate([[0], trace]))

# 3. For each trial (3 reps × 8 directions = 24 trials)
counter = 0
for rep in range(3):
    counter += 120  # Skip pre-margin
    for dir_idx in range(8):
        start = counter
        end = counter + 550  # 4400/8 frames per direction
        
        # ON: Maximum positive derivative (bar entering pixel)
        on_peak = np.argmax(trace[start:end]) + start
        
        # OFF: Maximum negative derivative (bar leaving pixel)
        off_peak = np.argmax(-trace[start:end]) + start
        
        counter = end
```

### Output Structure

```python
on_off_dict = {
    (row, col): {
        "on_peak_location": [24 frame indices],   # When bar covers pixel
        "off_peak_location": [24 frame indices],  # When bar leaves pixel
    },
    # ... for all 90,000 pixels
}
```

### Dictionary Versions

| File | Resolution | Method |
|------|------------|--------|
| `_on_off_dict.pkl` | 15×15 | Downsampled (20×20 averaging) |
| `_on_off_dict_hd.pkl` | 300×300 | Single pixel derivative |
| `_on_off_dict_area_hd.pkl` | 300×300 | 25×25 area derivative (simulates RF) |

**Current implementation uses `_hd.pkl`** for precise single-pixel detection.

---

## Spike Sectioning Process

### Step 1: Load Prerequisites

```python
# Load on/off dictionary
on_off_dict = load_on_off_dict(dict_path)

# From HDF5 file:
frame_timestamps = hdf5["metadata/frame_timestamps"][:]
section_time = hdf5[f"stimulus/section_time/{movie_name}"][:]
```

### Step 2: Compute Movie Start Frame

The `section_time` contains acquisition sample indices. We need to convert to movie-relative frames:

```python
movie_start_sample = section_time[0, 0]
movie_start_frame = convert_sample_index_to_frame(movie_start_sample, frame_timestamps)
movie_start_frame += PRE_MARGIN_FRAME_NUM  # +60 frames padding
```

**Critical**: `PRE_MARGIN_FRAME_NUM = 60` is a recording padding constant that must be added.

### Step 3: Get Cell Center

```python
# Read from HDF5 (in 15×15 STA grid)
center_row_15 = hdf5[f"units/{unit_id}/features/{sta_feature}/sta_geometry/center_row"][()]
center_col_15 = hdf5[f"units/{unit_id}/features/{sta_feature}/sta_geometry/center_col"][()]

# Convert to 300×300 stimulus space
center_row_300 = int(center_row_15 * 20)  # Scale factor = 20
center_col_300 = int(center_col_15 * 20)

# Clip to valid range [0, 299]
center = (clip(center_row_300, 0, 299), clip(center_col_300, 0, 299))
```

### Step 4: Convert Spike Times to Movie Frames

```python
# Load spike times (acquisition sample indices)
spike_samples = hdf5[f"units/{unit_id}/spike_times_sectioned/{movie_name}/full_spike_times"][:]

# Convert to absolute frame numbers
spike_frames_abs = convert_sample_index_to_frame(spike_samples, frame_timestamps)

# Convert to movie-relative frames
spike_frames = spike_frames_abs - movie_start_frame
```

### Step 5: Section by Direction

For each of the 24 trials:

```python
PADDING = 10  # Frames before/after on/off window

pixel_timing = on_off_dict[cell_center]

for trial_idx in range(24):
    direction_idx = trial_idx % 8
    direction = DIRECTION_LIST[direction_idx]  # [0, 45, 90, ..., 315]
    rep_idx = trial_idx // 8
    
    # Get trial window with padding
    on_time = pixel_timing["on_peak_location"][trial_idx]
    off_time = pixel_timing["off_peak_location"][trial_idx]
    start = on_time - PADDING
    end = off_time + PADDING
    
    # Extract spikes in window
    mask = (spike_frames >= start) & (spike_frames <= end)
    trial_spikes = spike_samples[mask]  # Keep in sample units
```

---

## Output Structure

### HDF5 Organization

```
units/{unit_id}/
└── spike_times_sectioned/
    └── moving_h_bar_s5_d8_3x/
        ├── full_spike_times      # Original data (UNCHANGED)
        └── direction_section/
            ├── @attrs:
            │   ├── direction_list: [0, 45, 90, 135, 180, 225, 270, 315]
            │   ├── n_directions: 8
            │   ├── n_repetitions: 3
            │   ├── padding_frames: 10
            │   ├── cell_center_row: int
            │   └── cell_center_col: int
            │
            ├── 0/                 # Direction 0°
            │   ├── trials/
            │   │   ├── 0          # Rep 1: int64[] spike samples
            │   │   ├── 1          # Rep 2: int64[] spike samples
            │   │   └── 2          # Rep 3: int64[] spike samples
            │   └── section_bounds # int64[3,2]: [[start,end], ...]
            │
            ├── 45/                # Direction 45°
            │   └── ...
            │
            └── ... (315/)
```

### Data Types

| Field | Type | Description |
|-------|------|-------------|
| `trials/{rep}` | int64[] | Spike times in acquisition samples |
| `section_bounds` | int64[3,2] | [start_frame, end_frame] for each rep |

---

## Usage

### Basic Processing

```python
from hdmea.features import section_by_direction

result = section_by_direction(
    "recording.h5",
    movie_name="moving_h_bar_s5_d8_3x",
    padding_frames=10,
)
print(f"Processed {result.units_processed} units")
```

### With Custom On/Off Dictionary

```python
result = section_by_direction(
    "recording.h5",
    on_off_dict_path="path/to/custom_on_off_dict.pkl",
    padding_frames=10,
)
```

### Safe Testing (Copy-on-Write)

```python
result = section_by_direction(
    "source.h5",
    output_path="export/copy.h5",  # Copy source first
    force=True,                     # Overwrite existing
)
```

### Selective Unit Processing

```python
result = section_by_direction(
    "recording.h5",
    unit_ids=["unit_001", "unit_002"],
)
```

---

## Validation Plots

The `dsgc_validation_plots.py` script generates combined figures with:

1. **Movie Frames**: Bar position at on/off times for all 8 directions × 3 trials
2. **Polar Tuning**: Direction selectivity curve with DSI
3. **Trial Consistency**: Heatmap of spike counts (direction × rep)
4. **Spike Raster**: Aligned to stimulus onset
5. **PSTH**: Firing rate histograms by direction

```bash
# Generate for all units
python dsgc_validation_plots.py --all

# Generate for single unit
python dsgc_validation_plots.py unit_001
```

---

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| `DIRECTION_LIST` | [0, 45, 90, 135, 180, 225, 270, 315] | `dsgc_direction.py` |
| `N_REPETITIONS` | 3 | `dsgc_direction.py` |
| `DEFAULT_PADDING_FRAMES` | 10 | `dsgc_direction.py` |
| `COORDINATE_SCALE_FACTOR` | 20 | `dsgc_direction.py` |
| `PRE_MARGIN_FRAME_NUM` | 60 | `section_time.py` |

---

## File Locations

| File | Path |
|------|------|
| Module | `src/hdmea/features/dsgc_direction.py` |
| Test script | `Projects/dsgc_section/dsgc_section.py` |
| Validation plots | `Projects/dsgc_section/dsgc_validation_plots.py` |
| On/off dictionary | `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\` |
| Movie | Same directory, `.npy` files |

---

## References

- Legacy code: `Legacy_code/Data_Processing_2025/Processing_2025/Design_Stimulation_Pattern/Quick_check.py`
- Spec: `specs/011-dsgc-direction-section/spec.md`
- Pipeline log: `docs/pipeline_log.md`

