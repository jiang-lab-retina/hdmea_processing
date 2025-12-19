# API Contract: STA Computation

**Feature**: 009-sta-computation  
**Date**: 2025-12-18

## Public API

### `compute_sta`

Main entry point for STA computation.

```python
def compute_sta(
    hdf5_path: Union[str, Path],
    *,
    cover_range: Tuple[int, int] = (-60, 0),
    use_multiprocessing: bool = True,
    stimuli_dir: Optional[Path] = None,
    force: bool = False,
) -> STAResult:
    """
    Compute Spike Triggered Average for all units using noise movie stimulus.
    
    Args:
        hdf5_path: Path to HDF5 recording file.
        cover_range: Frame window relative to spike (start, end). 
                     Negative values indicate frames before spike.
                     Default: (-60, 0) = 60 frames before spike.
        use_multiprocessing: If True, process units in parallel using 80% of CPU cores.
                             If False, process sequentially.
        stimuli_dir: Directory containing stimulus .npy files.
                     Default: M:\\Python_Project\\Data_Processing_2025\\Design_Stimulation_Pattern\\Data\\Stimulations\\
        force: If True, overwrite existing STA results. Default: False.
    
    Returns:
        STAResult with summary statistics and list of processed units.
    
    Raises:
        ValueError: If no noise movie found, or multiple noise movies found.
        ValueError: If cover_range[0] >= cover_range[1].
        FileNotFoundError: If stimulus .npy file not found.
        RuntimeError: If HDF5 file is not readable/writable.
    
    Example:
        >>> from hdmea.features import compute_sta
        >>> result = compute_sta("artifacts/recording.h5", cover_range=(-60, 0))
        >>> print(f"Processed {result.units_processed} units")
    """
```

### Return Type: `STAResult`

```python
@dataclass
class STAResult:
    """Result of STA computation."""
    
    hdf5_path: Path
    movie_name: str
    units_processed: int
    units_failed: int
    cover_range: Tuple[int, int]
    elapsed_seconds: float
    warnings: List[str]
```

## Internal API

### `_compute_sta_for_unit`

Worker function for single-unit STA computation.

```python
def _compute_sta_for_unit(
    unit_id: str,
    spike_frames: np.ndarray,
    movie_array: np.ndarray,
    cover_range: Tuple[int, int],
) -> Tuple[str, np.ndarray, int, int]:
    """
    Compute STA for a single unit.
    
    Args:
        unit_id: Unit identifier.
        spike_frames: Spike times converted to movie frame indices.
        movie_array: Stimulus movie array (frames, height, width).
        cover_range: Frame window (start_offset, end_offset).
    
    Returns:
        Tuple of (unit_id, sta_array, n_spikes_used, n_spikes_excluded).
    """
```

### `_find_noise_movie`

Detect the noise movie from available movies.

```python
def _find_noise_movie(hdf5_file: h5py.File, unit_id: str) -> str:
    """
    Find the noise movie name by searching for 'noise' in movie names.
    
    Args:
        hdf5_file: Open HDF5 file handle.
        unit_id: Unit ID to check (all units should have same movies).
    
    Returns:
        Noise movie name.
    
    Raises:
        ValueError: If zero or multiple noise movies found.
    """
```

### `_convert_spikes_to_frames`

Convert spike times from sampling indices to frame numbers.

```python
def _convert_spikes_to_frames(
    spike_samples: np.ndarray,
    acquisition_rate: float,
    frame_rate: float,
) -> np.ndarray:
    """
    Convert spike times from sampling indices to movie frame numbers.
    
    Args:
        spike_samples: Spike times in sampling indices (acquisition rate).
        acquisition_rate: Data acquisition rate in Hz (e.g., 20000).
        frame_rate: Movie display frame rate in Hz (e.g., 15).
    
    Returns:
        Spike times as frame numbers (rounded to nearest frame).
    """
```

### `_load_stimulus_movie`

Load and validate stimulus movie from .npy file.

```python
def _load_stimulus_movie(
    movie_name: str,
    stimuli_dir: Path,
) -> np.ndarray:
    """
    Load stimulus movie from .npy file.
    
    Args:
        movie_name: Name of the movie (matches HDF5 movie name).
        stimuli_dir: Directory containing .npy files.
    
    Returns:
        Movie array (frames, height, width) with original dtype.
    
    Raises:
        FileNotFoundError: If .npy file not found.
    
    Logs:
        Warning if dtype is not uint8.
    """
```

### `_write_sta_to_hdf5`

Save computed STA to HDF5 file.

```python
def _write_sta_to_hdf5(
    hdf5_file: h5py.File,
    unit_id: str,
    movie_name: str,
    sta: np.ndarray,
    n_spikes_used: int,
    n_spikes_excluded: int,
    cover_range: Tuple[int, int],
    force: bool = False,
) -> None:
    """
    Write STA array to HDF5 file.
    
    Args:
        hdf5_file: Open HDF5 file in write mode.
        unit_id: Unit identifier.
        movie_name: Movie name for grouping.
        sta: Computed STA array.
        n_spikes_used: Number of spikes included.
        n_spikes_excluded: Number of spikes excluded.
        cover_range: Frame range used.
        force: If True, overwrite existing. If False, raise on existing.
    """
```

## Error Handling

| Error Type | Condition | Recovery |
|------------|-----------|----------|
| ValueError | No noise movie found | Abort with clear message |
| ValueError | Multiple noise movies | Abort, list found movies |
| ValueError | Invalid cover_range | Abort before processing |
| FileNotFoundError | Stimulus .npy missing | Abort with expected path |
| Per-unit failure | Any exception | Retry once, then skip unit |

## Thread Safety

- HDF5 writes use single-writer model (no concurrent writes)
- Shared memory is read-only in workers
- Progress bar updates are thread-safe via tqdm

