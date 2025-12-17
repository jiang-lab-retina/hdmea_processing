# Quickstart: Analog Section Time Detection

**Feature**: 005-analog-section-time  
**Date**: 2025-12-17

## Prerequisites

1. Zarr archive from Stage 1 loading with:
   - `stimulus/light_reference/raw_ch1` - Light reference signal
   - `metadata/acquisition_rate` - Sampling rate

2. For analog detection: Know your threshold value (inspect signal first)

---

## Usage: Analog Section Time

### Step 1: Inspect Signal to Determine Threshold

```python
import zarr
import numpy as np
import matplotlib.pyplot as plt

# Load the raw signal
root = zarr.open("artifacts/JIANG009_2025-04-10.zarr", mode="r")
raw_ch1 = np.array(root["stimulus"]["light_reference"]["raw_ch1"])

# Plot derivative to see peak heights
diff_signal = np.diff(raw_ch1)
plt.figure(figsize=(12, 4))
plt.plot(diff_signal)
plt.title("Derivative of raw_ch1 - identify peak heights for threshold")
plt.xlabel("Sample index")
plt.ylabel("Derivative value")
plt.show()

# Note the approximate height of stimulus onset peaks
# Use a value that captures real peaks but rejects noise
```

### Step 2: Run Analog Section Time Detection

```python
from hdmea.io.section_time import add_section_time_analog

success = add_section_time_analog(
    zarr_path="artifacts/JIANG009_2025-04-10.zarr",
    threshold_value=1e5,      # From signal inspection
    movie_name="iprgc_test",  # Identifier for this stimulus
    plot_duration=120.0,      # 2 minutes per trial
    repeat=None,              # Use all detected trials (or set limit)
    force=False,              # Set True to overwrite existing
)

if success:
    print("Section times detected and stored!")
else:
    print("No peaks detected - check threshold value")
```

### Step 3: Verify Results

```python
import zarr
import numpy as np

root = zarr.open("artifacts/JIANG009_2025-04-10.zarr", mode="r")
section_time = np.array(root["stimulus"]["section_time"]["iprgc_test"])

print(f"Detected {len(section_time)} trials")
print(f"Section time array shape: {section_time.shape}")
print(f"First trial: samples {section_time[0, 0]} to {section_time[0, 1]}")

# Convert to time in seconds
acquisition_rate = float(np.array(root["metadata"]["acquisition_rate"]).flat[0])
start_time_s = section_time[0, 0] / acquisition_rate
end_time_s = section_time[0, 1] / acquisition_rate
print(f"First trial: {start_time_s:.2f}s to {end_time_s:.2f}s")
```

---

## Usage: Playlist-Based Section Time

For standard visual stimuli with known playlist:

```python
from hdmea.io.section_time import add_section_time

success = add_section_time(
    zarr_path="artifacts/REC_2023-12-07.zarr",
    playlist_name="set6a",
    repeats=2,
    force=False,
)

if success:
    print("Section times computed from playlist!")
```

**Note**: Output is now in acquisition sample indices (same unit as analog).

---

## Converting Section Time to Other Units

### To Time in Seconds

```python
import zarr
import numpy as np

root = zarr.open("artifacts/recording.zarr", mode="r")
section_time = np.array(root["stimulus"]["section_time"]["movie_name"])
acquisition_rate = float(np.array(root["metadata"]["acquisition_rate"]).flat[0])

# Convert to seconds
section_time_seconds = section_time / acquisition_rate
print(f"Trial 1: {section_time_seconds[0, 0]:.3f}s to {section_time_seconds[0, 1]:.3f}s")
```

### To Display Frame Indices (if needed)

```python
import zarr
import numpy as np

root = zarr.open("artifacts/recording.zarr", mode="r")
section_time = np.array(root["stimulus"]["section_time"]["movie_name"])
frame_timestamps = np.array(root["metadata"]["frame_timestamps"])

# Find nearest frame for each sample index
def sample_to_frame(sample_idx, frame_timestamps):
    return np.abs(frame_timestamps - sample_idx).argmin()

# Convert (note: may not be exact due to irregular frame intervals)
start_frame = sample_to_frame(section_time[0, 0], frame_timestamps)
end_frame = sample_to_frame(section_time[0, 1], frame_timestamps)
```

---

## Slicing Data Using Section Time

### Slice Unit FRIF (Firing Rate Inter-Frame)

```python
import zarr
import numpy as np

root = zarr.open("artifacts/recording.zarr", mode="r")
section_time = np.array(root["stimulus"]["section_time"]["iprgc_test"])
acquisition_rate = float(np.array(root["metadata"]["acquisition_rate"]).flat[0])

# Get FRIF for a unit
unit_frif = np.array(root["units"]["unit_000"]["features"]["frif"]["FRIF"])
frif_x_axis = np.array(root["units"]["unit_000"]["features"]["frif"]["FRIF_x_axis"])

# Convert section time to seconds for comparison with FRIF x-axis
trial_start_s = section_time[0, 0] / acquisition_rate
trial_end_s = section_time[0, 1] / acquisition_rate

# Find FRIF indices within trial window
mask = (frif_x_axis >= trial_start_s) & (frif_x_axis < trial_end_s)
trial_frif = unit_frif[mask]
trial_times = frif_x_axis[mask]
```

### Slice Raw Light Reference

```python
# Section time IS in acquisition sample indices - direct slicing!
raw_ch1 = np.array(root["stimulus"]["light_reference"]["raw_ch1"])

trial_start = section_time[0, 0]
trial_end = section_time[0, 1]
trial_signal = raw_ch1[trial_start:trial_end]
```

---

## Common Issues

### No Peaks Detected

1. **Threshold too high**: Lower the threshold value
2. **Signal is flat**: Check that raw_ch1 contains actual light transitions
3. **Wrong channel**: Ensure raw_ch1 is the light intensity channel (not frame sync)

### Threshold Selection Tips

- Plot `np.diff(raw_ch1)` to visualize peak heights
- Start with a low threshold and increase until noise peaks disappear
- Typical range: 1e4 to 1e7 depending on signal amplitude

### Overwriting Existing Data

```python
# Use force=True to overwrite
add_section_time_analog(
    zarr_path="...",
    threshold_value=1e5,
    movie_name="iprgc_test",
    force=True,  # Overwrite existing
)
```
