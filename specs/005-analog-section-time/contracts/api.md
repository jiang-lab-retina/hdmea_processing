# API Contracts: Analog Section Time Detection

**Feature**: 005-analog-section-time  
**Date**: 2025-12-17

## Functions

### add_section_time_analog()

Detect stimulus onsets from analog light reference signal and store as section times.

```python
def add_section_time_analog(
    zarr_path: Union[str, Path],
    threshold_value: float,
    *,
    movie_name: str = "iprgc_test",
    plot_duration: float = 120.0,
    repeat: Optional[int] = None,
    force: bool = False,
) -> bool:
    """
    Add section time by detecting peaks in raw_ch1 light reference signal.
    
    Detects actual stimulus onsets from the recorded light signal and stores
    them as acquisition sample indices under stimulus/section_time/{movie_name}.
    
    Args:
        zarr_path: Path to Zarr archive containing recording data
        threshold_value: Required. Peak height threshold for find_peaks().
            User must inspect signal to determine appropriate value.
        movie_name: Identifier for this stimulus type (default: "iprgc_test").
            Used as key under stimulus/section_time/{movie_name}
        plot_duration: Duration of each section in seconds (default: 120.0).
            End sample = onset_sample + (plot_duration * acquisition_rate)
        repeat: If specified, limit to first N detected trials.
        force: If True, overwrite existing section_time for this movie_name.
    
    Returns:
        True if section times were successfully added, False if no peaks detected
    
    Raises:
        FileNotFoundError: If zarr_path does not exist
        MissingInputError: If stimulus/light_reference/raw_ch1 missing
        MissingInputError: If metadata/acquisition_rate missing
        ValueError: If threshold_value not provided or plot_duration <= 0
        FileExistsError: If section_time/{movie_name} exists and force=False
    
    Output Format:
        stimulus/section_time/{movie_name}: int64[N, 2] array
        - Each row: [start_sample, end_sample] in acquisition sample indices
        - N = number of detected trials (or min(detected, repeat) if repeat specified)
    
    Example:
        >>> from hdmea.io.section_time import add_section_time_analog
        >>> 
        >>> success = add_section_time_analog(
        ...     zarr_path="artifacts/JIANG009_2025-04-10.zarr",
        ...     threshold_value=1e5,  # Determined from signal inspection
        ...     movie_name="iprgc_test",
        ...     plot_duration=120.0,  # 2 minute windows
        ...     repeat=3,  # Use first 3 trials only
        ... )
    """
```

**Contract Changes from Current**:
- Removes `frame_timestamps` requirement
- Output unit changes from display frame indices to acquisition sample indices
- Detection uses `raw_ch1` directly (not downsampled signals)

---

### add_section_time()

Compute section times from playlist metadata.

```python
def add_section_time(
    zarr_path: Union[str, Path],
    playlist_name: str,
    *,
    playlist_file_path: Optional[Union[str, Path]] = None,
    movie_length_file_path: Optional[Union[str, Path]] = None,
    repeats: int = 1,
    pad_frame: int = 180,
    pre_margin_frame_num: int = 60,
    post_margin_frame_num: int = 120,
    force: bool = False,
) -> bool:
    """
    Add section time metadata to a Zarr recording from playlist CSV.
    
    Computes frame boundaries for each movie in the playlist and converts
    them to acquisition sample indices using frame_timestamps.
    
    Args:
        zarr_path: Path to Zarr archive
        playlist_name: Name of playlist in playlist.csv
        playlist_file_path: Path to playlist CSV (uses default if None)
        movie_length_file_path: Path to movie_length CSV (uses default if None)
        repeats: Number of playlist repeats
        pad_frame: Padding frames between movies
        pre_margin_frame_num: Frames before movie start to include
        post_margin_frame_num: Frames after movie end to include
        force: If True, overwrite existing section_time data
    
    Returns:
        True if section times were successfully added, False otherwise
    
    Raises:
        FileExistsError: If section_time data already exists and force=False
    
    Output Format:
        stimulus/section_time/{movie_name}: int64[N, 2] array
        - Each row: [start_sample, end_sample] in acquisition sample indices
        - Converted from display frames via: sample = frame_timestamps[frame]
    
    Example:
        >>> from hdmea.io.section_time import add_section_time
        >>> 
        >>> success = add_section_time(
        ...     zarr_path="artifacts/REC_2023-12-07.zarr",
        ...     playlist_name="set6a",
        ...     repeats=2,
        ... )
    """
```

**Contract Changes from Current**:
- Output unit changes from display frame indices to acquisition sample indices
- Internal conversion: `start_sample = frame_timestamps[start_frame]`

---

## Helper Functions (Internal)

### _detect_analog_peaks()

```python
def _detect_analog_peaks(
    signal: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Detect stimulus onset peaks in analog signal derivative.
    
    Args:
        signal: 1D array of light reference signal (float32)
        threshold: Minimum peak height in derivative
    
    Returns:
        Array of peak indices (int64) where derivative exceeds threshold
    """
```

**No changes** - existing implementation is correct.

---

### _convert_frame_to_sample_index()

```python
def _convert_frame_to_sample_index(
    frame: np.ndarray,
    frame_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Convert display frame indices to acquisition sample indices.
    
    Args:
        frame: Array of display frame numbers
        frame_timestamps: Array mapping frames to sample indices
    
    Returns:
        Array of acquisition sample indices
    """
```

**No changes** - but now used for final output conversion in `add_section_time()`.

---

## Zarr Schema

### Output: stimulus/section_time/{movie_name}

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [N, 2],
  "data_type": "int64",
  "attributes": {
    "unit": "acquisition_samples",
    "created_by": "add_section_time_analog | add_section_time"
  }
}
```

### Required Input: stimulus/light_reference/raw_ch1

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [M],
  "data_type": "float32"
}
```

### Required Input: metadata/acquisition_rate

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [1],
  "data_type": "float64"
}
```

---

## Error Codes

| Error | Condition | User Action |
|-------|-----------|-------------|
| `MissingInputError("raw_ch1")` | raw_ch1 not in Zarr | Ensure Stage 1 completed |
| `MissingInputError("acquisition_rate")` | acquisition_rate missing | Ensure Stage 1 completed |
| `MissingInputError("frame_timestamps")` | Playlist mode, no timestamps | Ensure Stage 1 completed |
| `FileExistsError` | Section exists, force=False | Use `force=True` to overwrite |
| `ValueError("threshold")` | threshold_value not provided | Inspect signal, provide threshold |
| `ValueError("plot_duration")` | plot_duration ≤ 0 | Use positive duration in seconds |
