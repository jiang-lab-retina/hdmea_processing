# API Contract: JSON-Based Spike Sectioning

**Branch**: `008-json-spike-sectioning` | **Date**: 2024-12-18

## Modified Functions

### `section_spike_times` (existing function - signature preserved)

```python
def section_spike_times(
    hdf5_path: Union[str, Path],
    *,
    movie_names: Optional[List[str]] = None,
    trial_repeats: int = 3,  # DEPRECATED: will use JSON config values
    pad_margin: Tuple[float, float] = (2.0, 0.0),
    force: bool = False,
    config_dir: Optional[Union[str, Path]] = None,  # NEW PARAMETER
) -> SectionResult:
    """
    Section spike times by stimulation periods using JSON configuration.
    
    Args:
        hdf5_path: Path to HDF5 file containing recording data.
        movie_names: Optional list of movies to process. If None, processes
            all movies found in section_time.
        trial_repeats: DEPRECATED - ignored, uses JSON config 'repeat' value.
        pad_margin: Tuple of (pre_margin_s, post_margin_s) in seconds.
        force: If True, overwrite existing sectioned data.
        config_dir: Path to stimuli config directory. Defaults to 
            'config/stimuli/' relative to project root.
    
    Returns:
        SectionResult with success status and processing metadata.
    
    Raises:
        FileNotFoundError: If hdf5_path does not exist.
        FileExistsError: If sectioned data exists and force=False.
        ValueError: If any movie lacks corresponding JSON config file.
        ValueError: If JSON config has invalid section_kwargs.
    """
```

### `_section_unit_spikes` (internal function - modified signature)

```python
def _section_unit_spikes(
    spike_times: np.ndarray,
    section_frame_start: int,  # CHANGED: was section_time array
    trial_boundaries: List[Tuple[int, int]],  # NEW: list of (start_sample, end_sample)
    pre_samples: int = 0,
    post_samples: int = 0,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Extract spikes within trial boundaries for a single unit.
    
    Args:
        spike_times: Array of spike times in sample indices (uint64).
        section_frame_start: Start frame of the movie section.
        trial_boundaries: List of (start_sample, end_sample) tuples for each trial.
        pre_samples: Padding in samples before trial start.
        post_samples: Padding in samples after trial end.
    
    Returns:
        Tuple of:
            - full_spike_times: All spikes from all trials (sorted, unique).
            - trials_spike_times: Dict mapping trial_idx -> spike array.
    """
```

## New Functions

### `_load_stimuli_config`

```python
def _load_stimuli_config(
    movie_name: str,
    config_dir: Path,
) -> Dict[str, Any]:
    """
    Load and validate stimulus configuration from JSON file.
    
    Args:
        movie_name: Name of the movie/stimulus (matches JSON filename).
        config_dir: Directory containing stimulus JSON configs.
    
    Returns:
        Dict containing validated section_kwargs with keys:
            - start_frame: int
            - trial_length_frame: int
            - repeat: int
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid or missing required fields.
    """
```

### `_validate_all_configs`

```python
def _validate_all_configs(
    movie_names: List[str],
    config_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Validate that all movies have valid JSON configs before processing.
    
    Args:
        movie_names: List of movie names to validate.
        config_dir: Directory containing stimulus JSON configs.
    
    Returns:
        Dict mapping movie_name -> validated section_kwargs.
    
    Raises:
        ValueError: If any movie lacks config or has invalid config.
            Error message lists ALL missing/invalid configs.
    """
```

### `_calculate_trial_boundaries`

```python
def _calculate_trial_boundaries(
    section_kwargs: Dict[str, Any],
    section_frame_start: int,
    frame_timestamps: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Calculate trial boundaries in sample indices from JSON config.
    
    Args:
        section_kwargs: Dict with start_frame, trial_length_frame, repeat.
        section_frame_start: Frame number where movie section starts.
        frame_timestamps: Array mapping frame indices to sample indices.
    
    Returns:
        List of (start_sample, end_sample) tuples, one per trial.
    
    Note:
        Formula per trial n:
        start_frame = section_frame_start + PRE_MARGIN_FRAME_NUM + 
                      section_kwargs['start_frame'] + (n * trial_length_frame)
        start_sample = frame_timestamps[start_frame]
    """
```

## Data Types

### StimuliConfigDict (TypedDict)

```python
class StimuliConfigDict(TypedDict):
    start_frame: int
    trial_length_frame: int
    repeat: int
```

### ConfigValidationError

```python
class ConfigValidationError(ValueError):
    """Raised when stimulus config validation fails.
    
    Attributes:
        missing_configs: List of movie names without config files.
        invalid_configs: Dict mapping movie name to validation error message.
    """
    missing_configs: List[str]
    invalid_configs: Dict[str, str]
```

## Error Messages

| Scenario | Error Type | Message Format |
|----------|------------|----------------|
| Missing JSON file | `ValueError` | `"Missing stimulus config files: {list of names}. Expected at {config_dir}"` |
| Missing section_kwargs | `ValueError` | `"Config '{name}' missing 'section_kwargs' object"` |
| Missing required field | `ValueError` | `"Config '{name}' section_kwargs missing required field: {field}"` |
| Invalid field type | `ValueError` | `"Config '{name}' section_kwargs.{field} must be {expected_type}, got {actual_type}"` |
| Invalid field value | `ValueError` | `"Config '{name}' section_kwargs.{field} must be {constraint}, got {value}"` |
| Frame out of range | `ValueError` | `"Config '{name}' trial {n} start frame {frame} exceeds available frames {max_frame}"` |

## Backward Compatibility

| Aspect | Status | Notes |
|--------|--------|-------|
| Function signature | ✅ Compatible | New `config_dir` param has default |
| Return type | ✅ Unchanged | `SectionResult` dataclass unchanged |
| HDF5 output structure | ✅ Unchanged | Same nested structure |
| `trial_repeats` param | ⚠️ Deprecated | Ignored, uses JSON `repeat` value |

