# Research: Load Section Time Metadata

**Feature**: 004-load-section-time
**Date**: 2025-12-16

## Overview

This document captures research findings for implementing section time loading functionality. The feature ports the legacy algorithm from `load_raw_data.py` to the new hdmea package.

---

## 1. Legacy Algorithm Analysis

### Source Reference

Legacy code location: `Legacy_code/Data_Processing_2024/Processing_2024/load_raw_data.py`
- Lines 123-158: `get_movie_start_end_frame()`
- Lines 160-222: `add_section_time_auto()`

### Algorithm Steps

1. **Load Configuration Files**
   - `playlist.csv`: Maps playlist names → list of movie filenames
   - `movie_length.csv`: Maps movie names → frame count

2. **Parse Movie Sequence**
   - Get movie list from playlist by name
   - Parse Python list string with `eval()`
   - Strip file extensions (`.mov`, `.avi`, etc.)
   - Apply repeat multiplier

3. **Compute Frame Boundaries**
   - For each movie in sequence:
     - `start_frame = cumulative_count + pad - pre_margin`
     - `end_frame = cumulative_count + pad + post_margin + length + 1`
   - Accumulate: `count += 2 * pad + length + 1`

4. **Extract Light Templates**
   - Convert frame numbers to time indices using `frame_time` array
   - Extract light reference signal segment for each movie section

5. **Average Repeated Movies**
   - Use `itertools.zip_longest(*templates, fillvalue=np.nan)`
   - Compute `np.nanmean()` across repeats

### Constants (from legacy)

```python
PRE_MARGIN_FRAME_NUM = 60
POST_MARGIN_FRAME_NUM = 120
DEFAULT_PAD_FRAME = 180
```

### Decision: Algorithm Implementation

**Chosen**: Direct port with explicit parameters
**Rationale**: Legacy algorithm is well-tested and understood; explicit parameters allow flexibility
**Alternatives Rejected**:
- Template-matching approach: More complex, not needed for structured playlists
- Peak detection: Unreliable for some movie types

---

## 2. CSV File Format

### playlist.csv Structure

| Column | Type | Description |
|--------|------|-------------|
| `playlist_name` | str | Unique playlist identifier (used as index) |
| `movie_names` | str | Python list string, e.g., `"['movie1.mov', 'movie2.mov']"` |

**Note**: `movie_names` column contains a Python list as a string, requiring `eval()` to parse.

### movie_length.csv Structure

| Column | Type | Description |
|--------|------|-------------|
| `movie_name` | str | Movie identifier without extension (used as index) |
| `movie_length` | int | Duration in frames |

### Default File Paths

```python
DEFAULT_PLAYLIST_PATH = "//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/playlist.csv"
DEFAULT_MOVIE_LENGTH_PATH = "//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/movie_length.csv"
```

### Decision: CSV Parsing

**Chosen**: pandas with `set_index()` for lookup by name
**Rationale**: Simple, matches legacy approach, efficient for small files
**Alternatives Rejected**:
- JSON config: Would require migration of existing CSV files
- Database: Overkill for static configuration

---

## 3. Zarr Integration

### Existing Infrastructure

The `hdmea.io.zarr_store` module provides:
- `open_recording_zarr(path, mode)` - Open existing Zarr
- `write_stimulus(root, light_reference, frame_times, section_times)` - Write stimulus data

### Zarr Structure (Target)

```
{dataset}.zarr/
├── stimulus/
│   ├── light_reference/
│   │   ├── raw_ch1
│   │   └── raw_ch2
│   ├── frame_time/
│   │   └── default
│   ├── section_time/          # NEW
│   │   ├── {movie_name_1}     # shape: (n_repeats, 2) - start/end pairs
│   │   ├── {movie_name_2}
│   │   └── ...
│   └── light_template/        # NEW
│       ├── {movie_name_1}     # shape: (n_samples,) - averaged template
│       ├── {movie_name_2}
│       └── ...
└── metadata/
    ├── frame_time             # Required for conversion
    └── frame_timestamps       # Alternative source
```

### Decision: Storage Format

**Chosen**: Create groups under `stimulus/section_time/` and `stimulus/light_template/`
**Rationale**: 
- Matches existing stimulus data organization
- Allows independent access to each movie's data
- Compatible with existing Zarr navigation tools

**Alternatives Rejected**:
- Single flat array: Loses movie-level organization
- Separate Zarr file: Violates "single artifact per recording" principle

---

## 4. Frame-to-Time Conversion

### Legacy Approach

```python
def convert_frame_to_time(frame, frame_time):
    frame = np.array(frame).astype(int)
    return frame_time[frame]
```

### Data Sources

1. **Primary**: `metadata/frame_time` - Array of timestamps in seconds
2. **Fallback**: `metadata/frame_timestamps` / `metadata/acquisition_rate` - Compute from sample indices

### Decision: Conversion Strategy

**Chosen**: Try `frame_time` first, fallback to computing from `frame_timestamps / acquisition_rate`
**Rationale**: Handles both legacy and new Zarr formats

---

## 5. Overwrite Protection

### Requirement

From clarification: "Require `force=True` parameter to overwrite; error by default"

### Implementation

```python
if "section_time" in stimulus_group and not force:
    raise FileExistsError(
        f"section_time already exists in {zarr_path}. "
        "Use force=True to overwrite."
    )
```

### Decision: Error Type

**Chosen**: `FileExistsError`
**Rationale**: Consistent with standard Python semantics for file operations
**Alternatives Rejected**:
- Custom exception: Adds complexity without benefit
- Return False: Would be inconsistent with other error cases that also return False

---

## 6. Error Handling Strategy

### Error Categories

| Category | Severity | Response |
|----------|----------|----------|
| Missing config file | Recoverable | Log warning, return `False` |
| Invalid playlist name | Recoverable | Log error with suggestions, return `False` |
| Missing movie in database | Partial | Log warning, skip movie, continue |
| Missing Zarr | Fatal | Log error, return `False` |
| Missing frame_time | Fatal | Log error, return `False` |
| Existing section_time | Conditional | Raise `FileExistsError` unless `force=True` |

### Decision: Return Type

**Chosen**: Return `bool` (True = success, False = failure)
**Rationale**: Simple, allows caller to handle gracefully in pipelines
**Alternatives Rejected**:
- Raise exceptions for all errors: Makes pipeline orchestration harder
- Return detailed result object: Overkill for this simple operation

---

## 7. Dependencies

### Required (already in project)

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.5 | CSV reading |
| numpy | ≥1.24 | Numerical operations |
| zarr | ≥2.0 | Zarr I/O |

### Standard Library

- `itertools` - `zip_longest` for template averaging
- `pathlib` - Path handling
- `logging` - Logging
- `typing` - Type hints

### Decision: No New Dependencies

**Chosen**: Use only existing project dependencies
**Rationale**: Feature is simple enough with current stack

