# Quickstart: Load Section Time Metadata

**Feature**: 004-load-section-time
**Date**: 2025-12-16

## Prerequisites

1. **Existing Zarr archive** created by `load_recording()`
2. **Configuration files** (or use defaults):
   - `playlist.csv` - Playlist definitions
   - `movie_length.csv` - Movie durations

## Installation

The feature is part of the hdmea package:

```bash
pip install -e ".[dev]"
```

## Basic Usage

### Import

```python
from hdmea.io.section_time import add_section_time
# Or via pipeline module:
from hdmea.pipeline import add_section_time
```

### Simple Example

```python
# Add section times using default configuration paths
success = add_section_time(
    zarr_path="artifacts/REC_2023-12-07.zarr",
    playlist_name="set6a",
)

if success:
    print("Section times added successfully!")
else:
    print("Failed to add section times. Check logs.")
```

### With Repeats

```python
# Playlist was shown 3 times during recording
success = add_section_time(
    zarr_path="artifacts/REC_2023-12-07.zarr",
    playlist_name="set6a",
    repeats=3,
)
```

### Custom Configuration Paths

```python
# Use local configuration files
success = add_section_time(
    zarr_path="artifacts/REC_2023-12-07.zarr",
    playlist_name="custom_playlist",
    playlist_file_path="./config/my_playlist.csv",
    movie_length_file_path="./config/my_movie_lengths.csv",
)
```

### Custom Frame Parameters

```python
# Adjust frame margins and padding
success = add_section_time(
    zarr_path="artifacts/REC_2023-12-07.zarr",
    playlist_name="set6a",
    pre_margin_frame_num=30,   # Frames before movie start
    post_margin_frame_num=60,  # Frames after movie end
    pad_frame=120,             # Padding between movies
)
```

### Overwriting Existing Data

```python
# Force overwrite if section_time already exists
success = add_section_time(
    zarr_path="artifacts/REC_2023-12-07.zarr",
    playlist_name="set6a",
    force=True,
)
```

## Complete Pipeline Example

```python
from hdmea.pipeline import load_recording, extract_features
from hdmea.io.section_time import add_section_time

# Step 1: Load recording
result = load_recording(
    cmcr_path="M:\\data\\2023.12.07-09.37.02-Rec.cmcr",
    cmtr_path="M:\\data\\2023.12.07-09.37.02-Rec-.cmtr",
    dataset_id="REC_2023-12-07",
)

# Step 2: Add section timing metadata
add_section_time(
    zarr_path=result.zarr_path,
    playlist_name="set6a",
    repeats=2,
)

# Step 3: Extract features (now can use section timing)
extract_features(
    zarr_path=result.zarr_path,
    features=["frif", "step_up", "chirp"],
)
```

## Accessing Results

```python
import zarr

# Open the Zarr archive
root = zarr.open("artifacts/REC_2023-12-07.zarr", mode="r")

# List available movies
section_time_group = root["stimulus"]["section_time"]
print("Movies with section times:", list(section_time_group.keys()))

# Get frame boundaries for a specific movie
step_up_times = section_time_group["step_up_5s_5i_3x"][:]
print(f"Step-up movie boundaries: {step_up_times}")
# Output: [[120, 2040], [5520, 7440]]  # (n_repeats, 2)

# Get averaged light template
template = root["stimulus"]["light_template"]["step_up_5s_5i_3x"][:]
print(f"Template shape: {template.shape}")

# Check which playlist was used
print(f"Playlist: {root.attrs['section_time_playlist']}")
print(f"Repeats: {root.attrs['section_time_repeats']}")
```

## Error Handling

```python
from hdmea.io.section_time import add_section_time

try:
    success = add_section_time(
        zarr_path="artifacts/REC_2023-12-07.zarr",
        playlist_name="set6a",
    )
    
    if not success:
        # Check logs for details (missing files, invalid playlist, etc.)
        print("Section time loading failed. Check the log output above.")
        
except FileExistsError:
    print("Section time already exists. Use force=True to overwrite.")
```

## Configuration Files

### playlist.csv Format

```csv
playlist_name,movie_names
set6a,"['step_up_5s_5i_3x.mov', 'chirp_10s.mov', 'moving_bar.mov']"
set6b,"['dense_noise.mov', 'green_blue.mov']"
```

### movie_length.csv Format

```csv
movie_name,movie_length
step_up_5s_5i_3x,1800
chirp_10s,600
moving_bar,3600
dense_noise,18000
green_blue,2400
```

## Logging

Enable logging to see detailed progress:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now add_section_time will log progress
add_section_time(...)
```

Log output example:
```
hdmea.io.section_time - INFO - Computing section times for playlist 'set6a' with 2 repeat(s)
hdmea.io.section_time - INFO - Wrote section_time for 3 movies
hdmea.io.section_time - INFO - Wrote light_template for 3 movies
hdmea.io.section_time - INFO - Section times added successfully to artifacts/REC_2023-12-07.zarr
```

## Troubleshooting

### "Playlist file not found"

Check that the playlist CSV exists at the specified path (or default network path).

### "Playlist 'X' not found"

The playlist name doesn't exist in the CSV. Check available playlists:
```python
import pandas as pd
playlist = pd.read_csv("path/to/playlist.csv")
print("Available playlists:", playlist["playlist_name"].tolist())
```

### "Movie 'X' not found in movie_length"

A movie in the playlist doesn't have a corresponding entry in movie_length.csv. The function will skip that movie and continue with others.

### "No frame_time or frame_timestamps found"

The Zarr archive is missing required timing metadata. Ensure the recording was loaded with `load_recording()` which extracts frame timing from the light reference signal.

