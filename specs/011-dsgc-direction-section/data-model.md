# Data Model: DSGC Direction Sectioning

## Overview

This document defines the data structures and transformations for sectioning spike times by moving bar direction.

## Coordinate Systems

### 15×15 Grid (STA Geometry)

- Used by: `sta_geometry/center_row`, `sta_geometry/center_col`
- Range: `[0, 14]` (15 bins)
- Origin: Top-left corner
- Represents: Coarse receptive field center

### 300×300 Pixel Grid (On/Off Dictionary)

- Used by: On/off timing dictionary keys
- Range: `[0, 299]` (300 pixels)
- Origin: Top-left corner
- Represents: Stimulus pixel coordinates

### Conversion Formula

```
pixel_300 = pixel_15 * 20
```

| 15×15 Value | 300×300 Value |
|-------------|---------------|
| 0           | 0             |
| 7           | 140           |
| 14          | 280           |

**Note**: Clip result to [0, 299] if conversion exceeds bounds.

## Time Units

### Sampling Index

- Used by: `full_spike_times`, `section_time`, `frame_timestamps`
- Rate: 20,000 samples/second (20 kHz)
- Type: `int64`

### Frame Index (Movie-Relative)

- Used by: On/off dictionary values
- Rate: 60 frames/second (60 fps)
- Type: `int` (list elements in dictionary)
- Reference: Frame 0 = first frame of actual movie content

### Conversion: Sample → Movie-Relative Frame

```python
# Step 1: Sample index → Absolute frame
absolute_frame = np.searchsorted(frame_timestamps, sample_index, side='right') - 1

# Step 2: Get movie start from section_time
movie_start_sample = section_time[0, 0]
movie_start_absolute_frame = np.searchsorted(frame_timestamps, movie_start_sample, side='right') - 1

# Step 3: Add margin offset to get actual movie start
PRE_MARGIN_FRAME_NUM = 60
movie_start_frame = movie_start_absolute_frame + PRE_MARGIN_FRAME_NUM

# Step 4: Compute movie-relative frame
movie_relative_frame = absolute_frame - movie_start_frame
```

## Input Data Structures

### HDF5 File Structure

```
/metadata/
    frame_timestamps          Dataset[int64] - shape (N,)
                              Sample index for each display frame

/stimulus/
    section_time/
        moving_h_bar_s5_d8_3x Dataset[int64] - shape (1, 2)
                              [[start_sample, end_sample]] for movie section

/units/{unit_id}/
    spike_times_sectioned/
        moving_h_bar_s5_d8_3x/
            full_spike_times  Dataset[int64] - shape (M,)
                              All spike times during movie (sample indices)
    
    features/
        sta_perfect_dense_noise_15x15_15hz_r42_3min/
            sta_geometry/
                center_row    Dataset[float] - scalar
                              RF center row in 15×15 grid
                center_col    Dataset[float] - scalar
                              RF center col in 15×15 grid
```

### On/Off Dictionary (Pickle)

```python
Dict[Tuple[int, int], Dict[str, List[int]]]

# Structure:
{
    (row, col): {
        'on_peak_location': [frame_0, frame_1, ..., frame_23],
        'off_peak_location': [frame_0, frame_1, ..., frame_23],
    },
    # ... 90,000 entries (300 × 300)
}
```

**Trial Index Mapping**:

| Index | Direction | Repetition |
|-------|-----------|------------|
| 0     | 0°        | 1          |
| 1     | 45°       | 1          |
| 2     | 90°       | 1          |
| 3     | 135°      | 1          |
| 4     | 180°      | 1          |
| 5     | 225°      | 1          |
| 6     | 270°      | 1          |
| 7     | 315°      | 1          |
| 8     | 0°        | 2          |
| ...   | ...       | ...        |
| 23    | 315°      | 3          |

**Direction Index Formula**:
```python
direction_list = [0, 45, 90, 135, 180, 225, 270, 315]
direction_idx = trial_idx % 8
direction_deg = direction_list[direction_idx]
repetition = trial_idx // 8  # 0, 1, or 2
```

## Output Data Structures

### Direction Section (HDF5)

```
/units/{unit_id}/
    spike_times_sectioned/
        moving_h_bar_s5_d8_3x/
            full_spike_times      # UNCHANGED (source data)
            
            direction_section/
                {direction}/          # "0", "45", "90", etc.
                    trials/
                        0             Dataset[int64] - spike samples for rep 1
                        1             Dataset[int64] - spike samples for rep 2
                        2             Dataset[int64] - spike samples for rep 3
                    section_bounds    Dataset[int64] - shape (3, 2)
                                      [[start_sample, end_sample], ...] per trial
                    cell_center       Dataset[int64] - shape (2,)
                                      [row, col] in 300×300 space
                    padding_frames    Dataset[int64] - scalar
                                      Padding applied to on/off window
```

### Attributes (on direction_section group)

| Attribute | Type | Description |
|-----------|------|-------------|
| `direction_list` | `[int]` | `[0, 45, 90, 135, 180, 225, 270, 315]` |
| `n_directions` | `int` | `8` |
| `n_repetitions` | `int` | `3` |
| `source_movie` | `str` | `"moving_h_bar_s5_d8_3x"` |

## Processing Pipeline

```
1. Load Prerequisites
   ├── Read frame_timestamps from /metadata/
   ├── Read section_time for moving_h_bar_s5_d8_3x
   └── Load on/off dictionary from pickle file

2. For Each Unit
   ├── Read center_row, center_col from sta_geometry
   ├── Convert to 300×300 coordinates: (row*20, col*20)
   ├── Read full_spike_times
   └── Convert spike samples to movie-relative frames

3. For Each Trial (0-23)
   ├── Get on_frame, off_frame from dictionary at cell center
   ├── Apply padding: [on - padding, off + padding]
   ├── Extract spikes in window
   ├── Determine direction and repetition
   └── Store by direction key

4. Save Results
   ├── Create direction_section group
   ├── For each direction: save 3 trial datasets
   └── Save section_bounds and metadata
```

## Example Values

### Sample Unit Data

```python
unit_id = "unit_002"
center_row_15 = 7.5   # in 15×15
center_col_15 = 7.5   # in 15×15

# Convert to 300×300
center_row_300 = int(7.5 * 20)  # = 150
center_col_300 = int(7.5 * 20)  # = 150

cell_center = (150, 150)
```

### Sample On/Off Values

```python
on_off_dict[(150, 150)] = {
    'on_peak_location': [
        601, 1156, 1701, 2256, 2801, 3356, 3901, 4456,   # Rep 1
        5121, 5676, 6221, 6776, 7321, 7876, 8421, 8976,  # Rep 2
        9641, 10196, 10741, 11296, 11841, 12396, 12941, 13496  # Rep 3
    ],
    'off_peak_location': [
        621, 1175, 1721, 2275, 2821, 3375, 3921, 4475,   # Rep 1
        5141, 5695, 6241, 6795, 7341, 7895, 8441, 8995,  # Rep 2
        9661, 10215, 10761, 11315, 11861, 12415, 12961, 13515  # Rep 3
    ]
}
```

### Sample Section Bounds (with padding=10)

```python
# Direction 0° (indices 0, 8, 16)
direction_0_bounds = [
    [591, 631],    # Rep 1: on=601-10, off=621+10
    [5111, 5151],  # Rep 2: on=5121-10, off=5141+10
    [9631, 9671],  # Rep 3: on=9641-10, off=9661+10
]
```

