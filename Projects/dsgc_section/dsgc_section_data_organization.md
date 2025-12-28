# Goal: 
Seperate the spikes for each unit when a moving bar come across its center. 

# DSGC Section: Data Organization Summary

This document summarizes the data structures and alignment concepts needed to section spike times for direction-selective ganglion cell (DSGC) analysis using moving bar stimuli.



---

## 1. Moving Bar Stimulus Movie
This moive data should not be used in the processing, the dimension is useful for understanding the alignment.
**File:** `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\moving_h_bar_s5_d8_3x.npy`

| Property | Value |
|----------|-------|
| **Shape** | `(13560, 300, 300)` |
| **Dtype** | `uint8` |
| **Axes** | `(frames, height, width)` |

**Filename Interpretation:**
- `moving_h_bar`: Moving horizontal bar stimulus
- `s5`: Speed 5 (pixels per frame or similar)
- `d8`: 8 motion directions
- `3x`: 3 repetitions per direction

**Timing:**
- Total: 13,560 frames
- 8 directions × 3 repetitions = **24 trials**


---

## 2. On/Off Time Dictionary (Per-Pixel Timing)

This is the key information to be used for lignment and sectioning. 

the 8 directions follows the following sequence
direction_list = [0, 45, 90, 135, 180, 225, 270, 315]

**File:** `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl`

### Structure

```
dict  (90,000 entries)
│
├── (row, col)  ← Tuple key: pixel coordinate (0-299, 0-299)
│   └── dict
│       ├── "on_peak_location"  : list[24]  ← Frame indices when bar covers pixel
│       └── "off_peak_location" : list[24]  ← Frame indices when bar leaves pixel
│
└── ... (300 × 300 = 90,000 pixels)
```

### List Structure (24 elements = 8 directions × 3 reps)

| Index | Direction | Repetition |
|-------|-----------|------------|
| 0–7   | Dir 1–8   | Rep 1      |
| 8–15  | Dir 1–8   | Rep 2      |
| 16–23 | Dir 1–8   | Rep 3      |

### Example: Pixel (150, 150)

```python
data[(150, 150)] = {
    'on_peak_location': [
        # Rep 1 (Dir 1-8)
        601, 1156, 1701, 2256, 2801, 3356, 3901, 4456,
        # Rep 2 (Dir 1-8)
        5121, 5676, 6221, 6776, 7321, 7876, 8421, 8976,
        # Rep 3 (Dir 1-8)
        9641, 10196, 10741, 11296, 11841, 12396, 12941, 13496
    ],
    'off_peak_location': [
        # Rep 1 (Dir 1-8)
        621, 1175, 1721, 2275, 2821, 3375, 3921, 4475,
        # ... (same pattern, each ~20 frames after on_peak)
    ]
}
```

**Key Properties:**
- **Bar coverage duration per pixel:** ~20 frames (off - on)
- **Frame indices are movie-relative** (0 to 13559)
- **Different pixels have different on/off times** (bar is moving)

---

## 3. Spike Times Data (HDF5)

This is the test file to be used. Do not overwrite it but save the result in export folder. 

**File:** `M:\Python_Project\Data_Processing_2027\Projects\ap_trace_hdf5\export_ap_tracking_20251226\2024.08.08-10.40.20-Rec.h5`

### Path Structure

```
units/
└── unit_002/
    └── spike_times_sectioned/
        └── moving_h_bar_s5_d8_3x/
            └── full_spike_times    ← Dataset we need to section
```

### `full_spike_times` Properties

| Property | Value |
|----------|-------|
| **Shape** | `(5558,)` |
| **Dtype** | `int64` |
| **Unit** | Acquisition sample indices (20 kHz) |
| **Range** | 3,990,373 to 8,586,695 samples |

**Note:** These are **sample indices**, not frame indices. They need conversion using /metadata/frame_timestamps in the hdf5 file. 

---

## 4. Frame Alignment and Margin Handling

### The PRE_MARGIN_FRAME_NUM Problem

When recordings are made, extra frames are captured before and after each movie:
The only relevent number is PRE_MARGIN_FRAME_NUM as we are handling the pre-sectioned data.

```
Recording timeline:
|<-- pad_frame (180) -->|<-- movie content (13560 frames) -->|<-- pad_frame -->|
                        ^
         pre_margin (60) | post_margin (120)
                        |
                  Actual movie frame 0
```

### Constants

```python
PRE_MARGIN_FRAME_NUM = 60   # Frames before movie in recording
POST_MARGIN_FRAME_NUM = 120 # Frames after movie in recording
DEFAULT_PAD_FRAME = 180     # Total padding between movies
```

### section_time vs movie_array Alignment

| Data Source | Frame 0 Reference |
|-------------|-------------------|
| `section_time` (HDF5) | Includes pre-margin (60 frames before movie) |
| `movie_array` (.npy) | Actual movie start (no margins) |
| `on_off_dict` (.pkl) | Actual movie start (no margins) |

### Conversion Formula

```python
# Get section start from HDF5
section_time = hdf5_file[f"stimulus/section_time/{movie_name}"][:]
movie_start_sample = section_time[0, 0]

# Convert to frame and add margin offset
movie_start_frame = convert_sample_index_to_frame(movie_start_sample, frame_timestamps) + PRE_MARGIN_FRAME_NUM

# Convert spike samples to movie-relative frames
spike_frames_absolute = convert_sample_index_to_frame(spike_samples, frame_timestamps)
spike_frames_movie_relative = spike_frames_absolute - movie_start_frame
```

**Critical:** After this conversion, `spike_frames_movie_relative` uses the same frame reference as `on_off_dict` (frame 0 = first movie frame).

---

## 5. Cell Center Location

To section spikes by bar crossing, you need the cell's receptive field center in pixel coordinates.

**Cell center** are located from hdf5 file under each unit features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/center_row and features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/center_col, which is in 15x15 grid, and need to be converted to 300 x 300 grid. 

**Expected format:** `(row, col)` tuple matching the on_off_dict keys (0-299, 0-299 pixel space)



---

## 6. Proposed Sectioning Algorithm

### Goal
For each direction/repetition trial, extract spikes that occur during the bar's coverage window at the cell's receptive field center.

get the section with padding frame with default 10 frames.

### Input

```python
DSGC_TRAIL_PADDING_FRAME = 10
# Cell center (pixel coordinates)
cell_center = (row, col)  # e.g., (150, 150)

# On/off times for this pixel
on_times = on_off_dict[cell_center]['on_peak_location'] -DSGC_TRAIL_PADDING_FRAME  # list[24]
off_times = on_off_dict[cell_center]['off_peak_location'] + DSGC_TRAIL_PADDING_FRAME # list[24]

# Spike times (already converted to movie-relative frames)
spike_frames = [...]  # array of movie-relative frame indices
```

### Output Structure

1. record the section start and end sampling index of each trial
2. save sectioned spikes in unit of sampling index
3. save all these information under each unit group spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section
4. organize the output as direction with subgroup 3 trials for each direction. direction_list = [0, 45, 90, 135, 180, 225, 270, 315], use the direction (such as "45" as the key for the data)





---

## 7. Unit Conversion Reference

| From | To | Method |
|------|----|--------|
| Acquisition samples | Absolute frames | `convert_sample_index_to_frame(samples, frame_timestamps)` |
| Absolute frames | Movie-relative frames | `frames - movie_start_frame` |



---

## 8. File Paths Summary

| Data | Path |
|------|------|
| Movie (.npy) | `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\moving_h_bar_s5_d8_3x.npy` |
| On/Off Dict (.pkl) | `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl` |
| Spike Data (.h5) | `...\export_ap_tracking_20251226\*.h5` under `units/{unit_id}/spike_times_sectioned/{movie_name}/full_spike_times` |
| Frame timestamps | HDF5 file under `stimulus/frame_time/default` or `stimulus/frame_timestamps` |
| Section time | HDF5 file under `stimulus/section_time/{movie_name}` |

---

## 9. Key Assumptions

1. **Frame rate:** 60 fps (display rate)
2. **Sampling rate:** 20 kHz (acquisition rate)
3. **Movie and on_off_dict share the same frame reference** (frame 0 = first movie frame)
4. **PRE_MARGIN_FRAME_NUM = 60** is fixed and must be added when converting from section_time
5. **Cell center** are located from hdf5 file under each unit features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/center_row and features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/center_col, which is in 15x15 grid, and need to be converted to 300 x 300 grid. And it must be in the 300×300 pixel coordinate space


