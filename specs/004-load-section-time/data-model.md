# Data Model: Load Section Time Metadata

**Feature**: 004-load-section-time
**Date**: 2025-12-16

## Entities

### 1. Playlist (Input)

**Source**: `playlist.csv`

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `playlist_name` | str | Unique playlist identifier | Primary key, indexed |
| `movie_names` | str | Python list as string | Must be valid Python list syntax |

**Example**:
```csv
playlist_name,movie_names
set6a,"['step_up_5s_5i_3x.mov', 'chirp_10s.mov', 'moving_bar.mov']"
set6b,"['dense_noise.mov', 'green_blue.mov']"
```

**Parsing**:
```python
movie_list = eval(playlist.loc[playlist_name]["movie_names"])
movie_list = [x.split(".")[0] for x in movie_list]  # Strip extensions
```

---

### 2. MovieLength (Input)

**Source**: `movie_length.csv`

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `movie_name` | str | Movie identifier (no extension) | Primary key, indexed |
| `movie_length` | int | Duration in frames | > 0 |

**Example**:
```csv
movie_name,movie_length
step_up_5s_5i_3x,1800
chirp_10s,600
moving_bar,3600
```

---

### 3. SectionTime (Output)

**Storage**: Zarr dataset under `stimulus/section_time/{movie_name}`

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| data | int64 | (n_repeats, 2) | Start/end frame pairs for each repeat |

**Example** (movie with 2 repeats):
```
stimulus/section_time/step_up_5s_5i_3x
  shape: (2, 2)
  data: [[120, 2040], [5520, 7440]]
        # [start1, end1], [start2, end2]
```

**Derivation**:
- `start_frame = cumulative_count + pad_frame - pre_margin_frame_num`
- `end_frame = cumulative_count + pad_frame + post_margin_frame_num + movie_length + 1`

---

### 4. LightTemplate (Output)

**Storage**: Zarr dataset under `stimulus/light_template/{movie_name}`

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| data | float32 | (n_samples,) | Averaged light reference segment |

**Example**:
```
stimulus/light_template/step_up_5s_5i_3x
  shape: (1920,)
  dtype: float32
  # Averaged across repeats using nanmean
```

**Derivation**:
1. Extract light reference segment for each repeat
2. Use `zip_longest` to align variable-length segments (fill with NaN)
3. Compute `nanmean` across repeats

---

## Relationships

```
┌─────────────────┐         ┌──────────────────┐
│    Playlist     │         │   MovieLength    │
│  (playlist.csv) │         │ (movie_length.csv)│
├─────────────────┤         ├──────────────────┤
│ playlist_name ──┼────┐    │ movie_name ──────┼──┐
│ movie_names     │    │    │ movie_length     │  │
└─────────────────┘    │    └──────────────────┘  │
                       │                          │
                       │    ┌─────────────────────┘
                       ▼    ▼
              ┌────────────────────────┐
              │   _get_movie_start_    │
              │      end_frame()       │
              └───────────┬────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  SectionTime    │ │  LightTemplate  │ │   movie_list    │
│ (Zarr dataset)  │ │ (Zarr dataset)  │ │  (return value) │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Zarr Group Structure

### Before `add_section_time()`

```
{dataset}.zarr/
├── units/
│   └── {unit_id}/
├── stimulus/
│   ├── light_reference/
│   │   ├── raw_ch1
│   │   └── raw_ch2
│   └── frame_time/
│       └── default
└── metadata/
    ├── acquisition_rate
    ├── sample_interval
    ├── frame_timestamps
    └── frame_time
```

### After `add_section_time()`

```
{dataset}.zarr/
├── units/
│   └── {unit_id}/
├── stimulus/
│   ├── light_reference/
│   │   ├── raw_ch1
│   │   └── raw_ch2
│   ├── frame_time/
│   │   └── default
│   ├── section_time/           # NEW
│   │   ├── step_up_5s_5i_3x    # int64, shape (n_repeats, 2)
│   │   ├── chirp_10s           # int64, shape (n_repeats, 2)
│   │   └── moving_bar          # int64, shape (n_repeats, 2)
│   └── light_template/         # NEW
│       ├── step_up_5s_5i_3x    # float32, shape (n_samples,)
│       ├── chirp_10s           # float32, shape (n_samples,)
│       └── moving_bar          # float32, shape (n_samples,)
└── metadata/
    ├── acquisition_rate
    ├── sample_interval
    ├── frame_timestamps
    └── frame_time
```

### Root Attributes (after)

```python
root.attrs["section_time_playlist"] = "set6a"   # NEW
root.attrs["section_time_repeats"] = 2          # NEW
```

---

## Validation Rules

### Input Validation

| Rule | Enforcement |
|------|-------------|
| `playlist_name` must exist in playlist.csv | Log error, return False |
| `movie_name` must exist in movie_length.csv | Log warning, skip movie |
| `repeats >= 1` | Treat 0 or negative as 1 |
| Zarr must exist | Log error, return False |
| `frame_time` or `frame_timestamps` must exist | Log error, return False |

### Output Validation

| Rule | Enforcement |
|------|-------------|
| `section_time` group must not exist (unless force=True) | Raise FileExistsError |
| All frame boundaries must be non-negative | Implicit (algorithm) |
| Light template must have at least 1 sample | Skip if empty |

---

## State Transitions

### Section Time State Machine

```
┌─────────────────┐
│   NOT_EXISTS    │ ◄───── Initial state
└────────┬────────┘
         │ add_section_time(force=False or True)
         ▼
┌─────────────────┐
│     EXISTS      │
└────────┬────────┘
         │
         ├── add_section_time(force=False) ──► FileExistsError
         │
         └── add_section_time(force=True) ──► Overwrite, stays EXISTS
```

---

## Data Types Summary

| Entity | Storage | Dtype | Shape | Unit |
|--------|---------|-------|-------|------|
| SectionTime | Zarr | int64 | (n_repeats, 2) | frames |
| LightTemplate | Zarr | float32 | (n_samples,) | arbitrary (raw ADC) |
| playlist_name | attr | str | scalar | - |
| repeats | attr | int | scalar | - |

