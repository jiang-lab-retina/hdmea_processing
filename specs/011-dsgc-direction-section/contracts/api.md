# API Contract: DSGC Direction Sectioning

## Overview

This document defines the public API for the DSGC direction sectioning module.

## Module Location

```
src/hdmea/features/dsgc_direction.py
```

## Public Functions

### `section_by_direction`

Main entry point for direction-based spike sectioning.

```python
def section_by_direction(
    hdf5_path: Union[str, Path],
    *,
    movie_name: str = "moving_h_bar_s5_d8_3x",
    on_off_dict_path: Optional[Union[str, Path]] = None,
    padding_frames: int = 10,
    force: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    unit_ids: Optional[List[str]] = None,
) -> DirectionSectionResult:
    """
    Section spike times by moving bar direction for all units.
    
    Extracts spikes occurring when the moving bar crosses each unit's
    receptive field center, organized by motion direction.
    
    Args:
        hdf5_path: Path to HDF5 recording file.
        movie_name: Name of moving bar movie. Default: "moving_h_bar_s5_d8_3x".
        on_off_dict_path: Path to on/off timing pickle file.
            Default: stimuli_dir/{movie_name}_on_off_dict_area_hd.pkl
        padding_frames: Frames to add before/after on/off window. Default: 10.
        force: If True, overwrite existing direction_section data. Default: False.
        output_path: Optional path to write results. If provided, copies source
            HDF5 to this path before modifying. Default: None (in-place).
        unit_ids: Optional list of unit IDs to process. Default: None (all units).
    
    Returns:
        DirectionSectionResult with processing statistics.
    
    Raises:
        FileNotFoundError: If HDF5 file or on/off dict not found.
        ValueError: If no units found with required data.
    
    Example:
        >>> from hdmea.features import section_by_direction
        >>> result = section_by_direction("recording.h5", padding_frames=15)
        >>> print(f"Processed {result.units_processed} units")
    """
```

### `section_unit_by_direction`

Process a single unit (for advanced use cases).

```python
def section_unit_by_direction(
    spike_frames: np.ndarray,
    cell_center: Tuple[int, int],
    on_off_dict: Dict[Tuple[int, int], Dict[str, List[int]]],
    padding_frames: int = 10,
) -> Dict[int, Dict[str, Any]]:
    """
    Section spike frames by direction for a single unit.
    
    Args:
        spike_frames: Movie-relative frame indices of spikes.
        cell_center: (row, col) in 300×300 pixel coordinates.
        on_off_dict: Per-pixel on/off timing dictionary.
        padding_frames: Frames to pad before/after on/off window.
    
    Returns:
        Dictionary keyed by direction (0, 45, ..., 315) containing:
            - 'trials': List of 3 spike arrays (one per repetition)
            - 'bounds': List of 3 (start, end) frame tuples
    
    Example:
        >>> result = section_unit_by_direction(spikes, (150, 150), on_off_dict)
        >>> print(f"Direction 0°: {len(result[0]['trials'][0])} spikes in rep 1")
    """
```

## Data Classes

### `DirectionSectionResult`

```python
@dataclass
class DirectionSectionResult:
    """Result of direction sectioning computation.
    
    Attributes:
        hdf5_path: Path to the HDF5 file processed.
        movie_name: Name of the moving bar movie.
        units_processed: Number of units successfully processed.
        units_skipped: Number of units skipped (missing data).
        padding_frames: Padding applied to trial windows.
        elapsed_seconds: Total computation time.
        warnings: List of warning messages generated.
        skipped_units: List of unit IDs that were skipped.
    """
    hdf5_path: Path
    movie_name: str
    units_processed: int
    units_skipped: int
    padding_frames: int
    elapsed_seconds: float
    warnings: List[str] = field(default_factory=list)
    skipped_units: List[str] = field(default_factory=list)
```

## Constants

```python
# Direction sequence (degrees)
DIRECTION_LIST: List[int] = [0, 45, 90, 135, 180, 225, 270, 315]

# Number of repetitions per direction
N_REPETITIONS: int = 3

# Total trials (8 directions × 3 reps)
N_TRIALS: int = 24

# Default padding (frames)
DEFAULT_PADDING_FRAMES: int = 10

# Frame alignment constant
PRE_MARGIN_FRAME_NUM: int = 60

# Coordinate scaling factor (15×15 → 300×300)
COORDINATE_SCALE_FACTOR: int = 20

# Default paths
DEFAULT_STIMULI_DIR: Path = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations"
)

# STA geometry feature name
STA_GEOMETRY_FEATURE: str = "sta_perfect_dense_noise_15x15_15hz_r42_3min"
```

## Helper Functions

### `convert_center_15_to_300`

```python
def convert_center_15_to_300(
    center_row_15: float,
    center_col_15: float,
) -> Tuple[int, int]:
    """
    Convert cell center from 15×15 grid to 300×300 pixel coordinates.
    
    Args:
        center_row_15: Row coordinate in 15×15 grid.
        center_col_15: Column coordinate in 15×15 grid.
    
    Returns:
        (row, col) tuple in 300×300 space, clipped to [0, 299].
    """
```

### `get_cell_center`

```python
def get_cell_center(
    hdf5_file: h5py.File,
    unit_id: str,
    sta_feature_name: str = STA_GEOMETRY_FEATURE,
) -> Optional[Tuple[int, int]]:
    """
    Read cell center from HDF5 and convert to 300×300 coordinates.
    
    Args:
        hdf5_file: Open HDF5 file handle.
        unit_id: Unit identifier.
        sta_feature_name: Name of STA feature containing geometry.
    
    Returns:
        (row, col) in 300×300 space, or None if not found.
    """
```

### `load_on_off_dict`

```python
def load_on_off_dict(
    dict_path: Union[str, Path],
) -> Dict[Tuple[int, int], Dict[str, List[int]]]:
    """
    Load the per-pixel on/off timing dictionary.
    
    Args:
        dict_path: Path to pickle file.
    
    Returns:
        Dictionary with (row, col) keys and on/off frame lists.
    
    Raises:
        FileNotFoundError: If file not found.
        ValueError: If file structure is invalid.
    """
```

## HDF5 Output Structure

```
units/{unit_id}/spike_times_sectioned/{movie_name}/direction_section/
├── attrs:
│   ├── direction_list = [0, 45, 90, 135, 180, 225, 270, 315]
│   ├── n_directions = 8
│   ├── n_repetitions = 3
│   ├── padding_frames = 10
│   ├── cell_center_row = 150
│   └── cell_center_col = 150
│
├── 0/                          # Direction 0°
│   ├── trials/
│   │   ├── 0                   # Dataset[int64]: spike samples rep 1
│   │   ├── 1                   # Dataset[int64]: spike samples rep 2
│   │   └── 2                   # Dataset[int64]: spike samples rep 3
│   └── section_bounds          # Dataset[int64, (3,2)]: [[start,end], ...]
│
├── 45/                         # Direction 45°
│   └── ...
│
└── 315/                        # Direction 315°
    └── ...
```

## Usage Examples

### Basic Usage

```python
from hdmea.features import section_by_direction

# Process all units in a recording
result = section_by_direction(
    "path/to/recording.h5",
    padding_frames=10,
)

print(f"Processed: {result.units_processed}")
print(f"Skipped: {result.units_skipped}")
print(f"Time: {result.elapsed_seconds:.1f}s")
```

### Reading Results

```python
import h5py

with h5py.File("path/to/recording.h5", "r") as f:
    unit_id = "unit_002"
    movie = "moving_h_bar_s5_d8_3x"
    direction = "90"
    
    # Get spikes for direction 90°, repetition 1
    path = f"units/{unit_id}/spike_times_sectioned/{movie}/direction_section/{direction}/trials/0"
    spikes = f[path][:]
    
    # Get section bounds
    bounds_path = f"units/{unit_id}/spike_times_sectioned/{movie}/direction_section/{direction}/section_bounds"
    bounds = f[bounds_path][:]  # shape (3, 2)
    
    print(f"Direction {direction}°, Rep 1: {len(spikes)} spikes")
    print(f"Window: samples {bounds[0, 0]} to {bounds[0, 1]}")
```

### Custom Processing

```python
from hdmea.features.dsgc_direction import (
    section_unit_by_direction,
    load_on_off_dict,
    convert_center_15_to_300,
)
import numpy as np

# Load on/off dictionary
on_off = load_on_off_dict("path/to/on_off_dict.pkl")

# Convert center
center = convert_center_15_to_300(7.5, 7.5)  # → (150, 150)

# Custom spike frames (already movie-relative)
spike_frames = np.array([605, 610, 615, 620, 625])

# Section by direction
result = section_unit_by_direction(
    spike_frames,
    center,
    on_off,
    padding_frames=10,
)

# Access results
for direction in [0, 45, 90, 135, 180, 225, 270, 315]:
    for rep in range(3):
        spikes = result[direction]['trials'][rep]
        start, end = result[direction]['bounds'][rep]
        print(f"Dir {direction}° Rep {rep+1}: {len(spikes)} spikes in [{start}, {end}]")
```

## Error Handling

| Error Type | Condition | Message Format |
|------------|-----------|----------------|
| `FileNotFoundError` | HDF5 file not found | `"HDF5 file not found: {path}"` |
| `FileNotFoundError` | On/off dict not found | `"On/off dictionary not found: {path}"` |
| `ValueError` | No units with data | `"No units found with required data"` |
| `KeyError` | Cell center missing | `"STA geometry not found for {unit_id}"` |
| `KeyError` | Spike times missing | `"No full_spike_times for {unit_id}/{movie}"` |

## Logging

The module uses standard Python logging under logger name `hdmea.features.dsgc_direction`.

| Level | Message Examples |
|-------|------------------|
| INFO | `"Loaded on/off dictionary: 90000 pixels"` |
| INFO | `"Processing 42 units"` |
| INFO | `"Direction sectioning complete: 40 processed, 2 skipped"` |
| WARNING | `"No STA geometry for unit_005, skipping"` |
| WARNING | `"Cell center (305, 150) clipped to (299, 150)"` |
| DEBUG | `"Unit unit_002: center=(150, 150), 5558 spikes"` |

