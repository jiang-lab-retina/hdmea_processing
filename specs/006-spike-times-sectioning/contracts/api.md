# API Contracts: Spike Times Unit Conversion and Stimulation Sectioning

**Date**: 2025-12-17  
**Spec**: [../spec.md](../spec.md)  
**Plan**: [../plan.md](../plan.md)

---

## Overview

This document specifies the API contracts for the spike times feature. Changes include modifications to existing functions and one new pipeline step.

---

## Modified APIs

### 1. `load_recording()` - Internal Change

**Location**: `src/hdmea/pipeline/runner.py`

**Change**: Add spike timestamp conversion from nanoseconds to sample indices before writing to Zarr.

**Signature** (unchanged):
```python
def load_recording(
    cmcr_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    dataset_id: Optional[str] = None,
    *,
    output_dir: Union[str, Path] = "artifacts",
    force: bool = False,
    allow_overwrite: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> LoadResult:
```

**Behavioral Change**:
- Before: `spike_times` written as raw nanoseconds from CMTR
- After: `spike_times` converted to sample indices using `timestamp_ns * acquisition_rate / 1e9`

---

## New APIs

### 2. `section_spike_times()` - New Pipeline Step

**Location**: `src/hdmea/io/spike_sectioning.py` (new file)

**Purpose**: Extract spike timestamps within stimulation trial boundaries and store BOTH combined and per-trial formats.

**Signature**:
```python
def section_spike_times(
    zarr_path: Union[str, Path],
    *,
    movie_names: Optional[List[str]] = None,
    trial_repeats: int = 3,
    pad_margin: Tuple[float, float] = (2.0, 0.0),
    force: bool = False,
) -> SectionResult:
    """
    Section spike times by stimulation periods defined in section_time.
    
    For each unit in the Zarr archive, extracts spikes falling within
    trial periods (with optional padding) and stores them in TWO formats:
    - `full_spike_times`: All spikes from all trials combined
    - `trials_spike_times/{idx}`: Spikes per trial based on section_time boundaries
    
    Args:
        zarr_path: Path to Zarr archive containing recording data.
            Must have `units/{unit_id}/spike_times` arrays (in sample units)
            and `stimulus/section_time/{movie_name}` trial boundaries.
        movie_names: Optional list of movies to process. If None, processes
            all movies found in section_time.
        trial_repeats: Number of trials to process per movie (default=3).
            Uses first N trials from section_time.
        pad_margin: Tuple of (pre_margin_s, post_margin_s) in seconds.
            Default=(2.0, 0.0) - 2 seconds before trial start, 0 after trial end.
            Converted to samples:
            - pre_samples = int(pad_margin[0] * acquisition_rate)
            - post_samples = int(pad_margin[1] * acquisition_rate)
            Trial boundaries become [start - pre_samples, end + post_samples].
        force: If True, overwrite existing sectioned data. If False (default),
            raise FileExistsError if any unit already has sectioned data.
    
    Returns:
        SectionResult with:
            - success: bool indicating overall success
            - units_processed: int count of units processed
            - movies_processed: List[str] of movie names processed
            - trial_repeats: int - trial_repeats value used
            - warnings: List[str] of any warnings generated
    
    Raises:
        FileNotFoundError: If zarr_path does not exist
        MissingInputError: If required data missing:
            - units/{unit_id}/spike_times (for any unit)
            - stimulus/section_time/{movie_name}
        FileExistsError: If sectioned data exists and force=False
    
    Example:
        >>> from hdmea.io.spike_sectioning import section_spike_times
        >>> 
        >>> result = section_spike_times(
        ...     zarr_path="artifacts/JIANG009_2025-04-10.zarr",
        ...     trial_repeats=3,
        ...     pad_margin=(2.0, 0.0),  # 2s pre, 0s post
        ...     force=False,
        ... )
        >>> print(f"Processed {result.units_processed} units")
        >>> print(f"Movies: {result.movies_processed}")
        >>> print(f"Padding: {result.pre_samples} pre, {result.post_samples} post")
    """
```

**Return Type**:
```python
@dataclass
class SectionResult:
    """Result of spike times sectioning operation."""
    success: bool
    units_processed: int
    movies_processed: List[str]
    trial_repeats: int
    pad_margin: Tuple[float, float]
    pre_samples: int   # Computed: int(pad_margin[0] * acquisition_rate)
    post_samples: int  # Computed: int(pad_margin[1] * acquisition_rate)
    warnings: List[str] = field(default_factory=list)
```

---

### 3. Helper Functions

#### `_section_unit_spikes()`

**Location**: `src/hdmea/io/spike_sectioning.py`

**Purpose**: Section spikes for a single unit (internal helper).

```python
def _section_unit_spikes(
    spike_times: np.ndarray,
    section_time: np.ndarray,
    trial_repeats: int = 3,
    pre_samples: int = 0,
    post_samples: int = 0,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Extract spikes within padded trial boundaries.
    
    Args:
        spike_times: Array of spike times in sample indices (uint64)
        section_time: Array of shape (N_trials, 2) with [start, end] samples
        trial_repeats: Number of trials to process
        pre_samples: Padding in samples to extend before trial start (default=0)
        post_samples: Padding in samples to extend after trial end (default=0)
    
    Returns:
        Tuple of:
            - full_spike_times: All spikes from all processed trials (sorted)
            - trials_spike_times: Dict mapping trial_idx -> spike array
    
    Note:
        Padded boundaries are clamped: start >= 0, end <= max(spike_times)
    """
```

#### `_write_sectioned_spikes()`

**Location**: `src/hdmea/io/spike_sectioning.py`

**Purpose**: Write sectioned spike data to Zarr (internal helper).

```python
def _write_sectioned_spikes(
    unit_group: zarr.Group,
    movie_name: str,
    full_spike_times: np.ndarray,
    trials_spike_times: Dict[int, np.ndarray],
    trial_repeats: int,
    force: bool = False,
) -> None:
    """
    Write sectioned spike times to unit group in Zarr.
    
    Creates structure:
        spike_times_sectioned/{movie_name}/
            full_spike_times          # All trials combined
            trials_spike_times/
                0                     # Trial 0 spikes
                1                     # Trial 1 spikes
                ...
    
    Args:
        unit_group: Zarr group for the unit (units/{unit_id})
        movie_name: Name of the movie/stimulus
        full_spike_times: All spikes from all trials (sorted)
        trials_spike_times: Dict mapping trial_idx -> spike array
        trial_repeats: Number of trials (for metadata)
        force: Whether to overwrite existing data
    
    Raises:
        FileExistsError: If data exists and force=False
    """
```

---

## Public API Summary

| Function | Module | Type | Description |
|----------|--------|------|-------------|
| `load_recording()` | `hdmea.pipeline.runner` | Modified | Now converts spike_times to sample units |
| `section_spike_times()` | `hdmea.io.spike_sectioning` | New | Sections spike_times (both combined and per-trial) |

---

## Usage Flow

```python
from hdmea.pipeline import load_recording
from hdmea.io.spike_sectioning import section_spike_times
from hdmea.io.section_time import add_section_time

# Step 1: Load recording (spike_times now in sample units)
result = load_recording(
    cmcr_path="path/to/recording.cmcr.h5",
    cmtr_path="path/to/recording.cmtr.h5",
    dataset_id="JIANG009_2025-04-10",
)

# Step 2: Add section_time (if not already present)
add_section_time(
    zarr_path=result.zarr_path,
    playlist_name="playlist_set6a",
)

# Step 3: Section spike times by stimulation
section_result = section_spike_times(
    zarr_path=result.zarr_path,
    trial_repeats=3,          # Process first 3 trials
    pad_margin=(2.0, 0.0),    # 2s pre-margin, 0s post-margin
    force=False,
)

# Access sectioned data
import zarr
root = zarr.open(result.zarr_path, mode="r")
unit = root["units/unit_000"]

# Combined spikes
full_spikes = unit["spike_times_sectioned/movie_A/full_spike_times"][:]

# Per-trial spikes
trial_0_spikes = unit["spike_times_sectioned/movie_A/trials_spike_times/0"][:]
trial_1_spikes = unit["spike_times_sectioned/movie_A/trials_spike_times/1"][:]
```

---

## Error Handling

| Error | Condition | Resolution |
|-------|-----------|------------|
| `FileNotFoundError` | Zarr path doesn't exist | Check path; run load_recording first |
| `MissingInputError` | No section_time data | Run add_section_time() first |
| `MissingInputError` | No spike_times for unit | Re-run load_recording with force=True |
| `FileExistsError` | Sectioned data exists | Use force=True to overwrite |
