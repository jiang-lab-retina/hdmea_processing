# Quickstart: DSGC Direction Sectioning

## What This Does

Sections spike times by moving bar direction for direction-selective ganglion cell (DSGC) analysis.

**Input**: Full spike times during moving bar movie + cell RF center  
**Output**: Spike times organized by 8 directions × 3 repetitions

## Quick Usage

```python
from hdmea.features import section_by_direction

# Basic usage (in-place modification)
result = section_by_direction("path/to/recording.h5")
print(f"Processed {result.units_processed} units")

# Safe testing (copy to output path first)
result = section_by_direction(
    "path/to/recording.h5",
    output_path="path/to/export/recording.h5",  # Copies, then modifies copy
)

# Process specific units only
result = section_by_direction(
    "path/to/recording.h5",
    unit_ids=["unit_001", "unit_002"],  # Only these units
    force=True,  # Overwrite existing direction_section
)
```

## Key Concepts

### Directions

8 motion directions in degrees:
```
[0, 45, 90, 135, 180, 225, 270, 315]
```

### Trial Organization

24 total trials: 8 directions × 3 repetitions

| Trial Index | Direction | Repetition |
|-------------|-----------|------------|
| 0-7         | 0°-315°   | 1          |
| 8-15        | 0°-315°   | 2          |
| 16-23       | 0°-315°   | 3          |

### Cell Center Conversion

```python
# From 15×15 grid (STA geometry) to 300×300 (on/off dict)
center_300 = center_15 * 20
# e.g., (7.5, 7.5) → (150, 150)
```

### Frame Alignment

```
PRE_MARGIN_FRAME_NUM = 60  # Must add when converting from section_time
```

## Data Paths

| Data | HDF5 Path |
|------|-----------|
| Spike times | `units/{id}/spike_times_sectioned/moving_h_bar_s5_d8_3x/full_spike_times` |
| Cell center | `units/{id}/features/sta_perfect_dense_noise.../sta_geometry/center_row`, `center_col` |
| Frame timestamps | `metadata/frame_timestamps` |
| Section time | `stimulus/section_time/moving_h_bar_s5_d8_3x` |
| **Output** | `units/{id}/spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section/` |

## Reading Results

```python
import h5py

with h5py.File("recording.h5", "r") as f:
    # Get spikes for direction 90°, rep 1
    path = "units/unit_002/spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section/90/trials/0"
    spikes = f[path][:]
    print(f"{len(spikes)} spikes")
```

## Test File

```
M:\Python_Project\Data_Processing_2027\Projects\ap_trace_hdf5\export_ap_tracking_20251226\2024.08.08-10.40.20-Rec.h5
```

