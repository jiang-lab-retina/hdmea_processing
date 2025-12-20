# API Contract: Electrode Image STA (eimage_sta)

**Date**: 2025-12-19  
**Feature**: 001-eimage-sta-feature  
**Version**: 1.0.0

## Public API

### compute_eimage_sta

Main entry point for computing eimage_sta for all units in a recording.

```python
def compute_eimage_sta(
    hdf5_path: Union[str, Path],
    cmcr_path: Union[str, Path],
    *,
    cutoff_hz: float = 100.0,
    filter_order: int = 2,
    pre_samples: int = 10,
    post_samples: int = 40,
    spike_limit: int = 10000,
    duration_s: float = 120.0,
    use_cache: bool = False,
    cache_path: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> EImageSTAResult:
    """
    Compute Electrode Image STA for all units using sensor data from CMCR.
    
    Args:
        hdf5_path: Path to HDF5 file containing units with spike_times.
        cmcr_path: Path to CMCR file containing sensor data.
        cutoff_hz: High-pass filter cutoff frequency in Hz.
        filter_order: Butterworth filter order.
        pre_samples: Number of samples before spike in window.
        post_samples: Number of samples after spike in window.
        spike_limit: Maximum spikes to use per unit (-1 for no limit).
        duration_s: Duration of sensor data to process in seconds.
        use_cache: If True, cache filtered data for reuse.
        cache_path: Path to cache file (auto-generated if None and use_cache=True).
        force: If True, overwrite existing eimage_sta features.
    
    Returns:
        EImageSTAResult with processing summary.
    
    Raises:
        FileNotFoundError: If hdf5_path or cmcr_path does not exist.
        ValueError: If no units found in HDF5 file.
        DataLoadError: If sensor data cannot be loaded from CMCR.
        RuntimeError: If HDF5 file is not writable.
    
    Example:
        >>> result = compute_eimage_sta(
        ...     "artifacts/recording.h5",
        ...     "O:/data/recording.cmcr",
        ...     cutoff_hz=100.0,
        ...     use_cache=True,
        ... )
        >>> print(f"Processed {result.units_processed} units")
    """
```

---

### EImageSTAResult

Result dataclass returned by compute_eimage_sta.

```python
@dataclass
class EImageSTAResult:
    """Result of eimage_sta computation.
    
    Attributes:
        hdf5_path: Path to the HDF5 file processed.
        cmcr_path: Path to the CMCR file used.
        units_processed: Number of units successfully processed.
        units_failed: Number of units that failed.
        elapsed_seconds: Total computation time.
        filter_time_seconds: Time spent on filtering.
        warnings: List of warning messages generated.
        failed_units: List of unit IDs that failed.
    """
    hdf5_path: Path
    cmcr_path: Path
    units_processed: int
    units_failed: int
    elapsed_seconds: float
    filter_time_seconds: float
    warnings: List[str] = field(default_factory=list)
    failed_units: List[str] = field(default_factory=list)
```

---

### EImageSTAExtractor

Feature extractor class for registry integration.

```python
@FeatureRegistry.register("eimage_sta")
class EImageSTAExtractor(FeatureExtractor):
    """
    Electrode Image STA feature extractor.
    
    Computes spike-triggered average of sensor data across the electrode array.
    Unlike visual STA (sta.py), this captures the electrical footprint of the
    neuron's activity pattern across the MEA.
    
    Attributes:
        name: "eimage_sta"
        version: "1.0.0"
        runtime_class: "slow" (requires CMCR sensor data access)
        required_inputs: ["spike_times"]
        output_schema: See below
    """
    
    name = "eimage_sta"
    version = "1.0.0"
    runtime_class = "slow"
    description = "Electrode image spike-triggered average from sensor data"
    
    required_inputs = ["spike_times"]
    
    output_schema = {
        "data": {
            "dtype": "float32",
            "shape": "(window_length, rows, cols)",
            "description": "Average electrode activity around spikes",
        },
        "n_spikes": {
            "dtype": "int64",
            "description": "Number of spikes used in average",
        },
        "n_spikes_excluded": {
            "dtype": "int64",
            "description": "Spikes excluded due to edge effects",
        },
    }
    
    def extract(
        self,
        unit_data: HDF5Group,
        stimulus_data: HDF5Group,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[HDF5Group] = None,
    ) -> Dict[str, Any]:
        """Extract eimage_sta for a single unit.
        
        Note: This extractor requires additional context (filtered sensor data)
        that is typically pre-computed. Use compute_eimage_sta() for the full
        workflow, or pass filtered_data in config.
        """
```

---

## Internal API

### apply_highpass_filter_3d

Vectorized high-pass filter for 3D sensor data.

```python
def apply_highpass_filter_3d(
    sensor_data: np.ndarray,
    cutoff_hz: float,
    sampling_rate: float,
    filter_order: int = 2,
) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to 3D sensor array.
    
    Uses vectorized scipy.signal.filtfilt along time axis for all electrodes.
    
    Args:
        sensor_data: 3D array (time, rows, cols) of sensor readings.
        cutoff_hz: Filter cutoff frequency in Hz.
        sampling_rate: Data sampling rate in Hz.
        filter_order: Butterworth filter order (default: 2).
    
    Returns:
        Filtered array as float32, same shape as input.
    
    Performance:
        ~30s for 120s of data at 20kHz, 64x64 electrodes (vs. ~60min legacy).
    """
```

### compute_sta_for_unit

Vectorized STA computation for single unit.

```python
def compute_sta_for_unit(
    filtered_data: np.ndarray,
    spike_samples: np.ndarray,
    pre_samples: int = 10,
    post_samples: int = 40,
    spike_limit: int = -1,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute eimage STA for a single unit using vectorized window extraction.
    
    Args:
        filtered_data: 3D array (time, rows, cols) of filtered sensor data.
        spike_samples: 1D array of spike times as sample indices.
        pre_samples: Samples before spike in window.
        post_samples: Samples after spike in window.
        spike_limit: Max spikes to use (-1 for no limit).
    
    Returns:
        Tuple of (sta_array, n_spikes_used, n_spikes_excluded).
        sta_array has shape (window_length, rows, cols).
    
    Performance:
        ~0.5s per unit with 10,000 spikes (vs. ~10s legacy).
    """
```

### load_sensor_data

Load sensor data from CMCR with memory-mapped access.

```python
def load_sensor_data(
    cmcr_path: Union[str, Path],
    duration_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Load sensor data from CMCR file.
    
    Uses McsPy for lazy/memory-mapped access to large sensor arrays.
    
    Args:
        cmcr_path: Path to CMCR file.
        duration_samples: Number of time samples to load (None = all).
    
    Returns:
        3D array (time, rows, cols) as int16.
    
    Raises:
        FileNotFoundError: If CMCR file not found.
        DataLoadError: If sensor data not available in file.
    """
```

---

## HDF5 Write Contract

### Feature Group Structure

```python
def write_eimage_sta_to_hdf5(
    hdf5_file: h5py.File,
    unit_id: str,
    sta: np.ndarray,
    n_spikes: int,
    n_spikes_excluded: int,
    config: EImageSTAConfig,
    force: bool = False,
) -> None:
    """
    Write eimage_sta to HDF5 file.
    
    Creates group structure: units/{unit_id}/features/eimage_sta/
    
    Dataset:
        data: float32 array (window_length, rows, cols)
    
    Attributes:
        n_spikes: int
        n_spikes_excluded: int
        pre_samples: int
        post_samples: int
        cutoff_hz: float
        filter_order: int
        sampling_rate: float
        spike_limit: int
        version: str
    """
```

---

## Error Codes

| Error | Cause | Resolution |
|-------|-------|------------|
| `FileNotFoundError` | CMCR or HDF5 file not found | Check file paths |
| `DataLoadError` | Sensor data not in CMCR | Use different CMCR file |
| `ValueError("No units found")` | HDF5 has no units group | Run spike sorting first |
| `RuntimeError("HDF5 not writable")` | File open in read mode | Open with 'r+' mode |
| `MemoryError` | Insufficient RAM | Reduce spike_limit or duration_s |

