# API Contract: Load Section Time Metadata

**Feature**: 004-load-section-time
**Date**: 2025-12-16

## Public API

### `hdmea.io.section_time.add_section_time`

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
    Add section time metadata to a Zarr recording.

    Computes frame boundaries for each movie in the playlist and stores
    them under stimulus/section_time/{movie_name}. Also extracts and
    averages light templates for each section.

    Args:
        zarr_path: Path to Zarr archive.
        playlist_name: Name of playlist in playlist.csv.
        playlist_file_path: Path to playlist CSV (uses default if None).
        movie_length_file_path: Path to movie_length CSV (uses default if None).
        repeats: Number of playlist repeats.
        pad_frame: Padding frames between movies.
        pre_margin_frame_num: Frames before movie start to include.
        post_margin_frame_num: Frames after movie end to include.
        force: If True, overwrite existing section_time data.

    Returns:
        True if section times were successfully added, False otherwise.

    Raises:
        FileExistsError: If section_time data already exists and force=False.

    Example:
        >>> from hdmea.io.section_time import add_section_time
        >>> success = add_section_time(
        ...     zarr_path="artifacts/REC_2023-12-07.zarr",
        ...     playlist_name="set6a",
        ...     repeats=2,
        ... )
    """
```

---

## Internal API (Private)

### `hdmea.io.section_time._load_playlist_csv`

```python
def _load_playlist_csv(
    playlist_file_path: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """Load and index playlist CSV by playlist_name."""
```

### `hdmea.io.section_time._load_movie_length_csv`

```python
def _load_movie_length_csv(
    movie_length_file_path: Union[str, Path]
) -> Optional[pd.DataFrame]:
    """Load and index movie_length CSV by movie_name."""
```

### `hdmea.io.section_time._convert_frame_to_time`

```python
def _convert_frame_to_time(
    frame: np.ndarray,
    frame_time: np.ndarray
) -> np.ndarray:
    """Convert frame numbers to time values using frame_time array."""
```

### `hdmea.io.section_time._get_movie_start_end_frame`

```python
def _get_movie_start_end_frame(
    playlist_name: str,
    repeats: int,
    all_playlists: pd.DataFrame,
    movies_length: pd.DataFrame,
    frame_time: np.ndarray,
    light_reference_raw: Optional[np.ndarray] = None,
    pad_frame: int = 180,
    pre_margin_frame_num: int = 60,
    post_margin_frame_num: int = 120,
) -> Tuple[List[str], Dict[str, List[List[int]]], Dict[str, List[np.ndarray]]]:
    """
    Compute start/end frame for each movie in a playlist.

    Returns:
        Tuple of (movie_list, movie_start_end_frame, movie_light_template)
    """
```

---

## Constants

```python
# Frame timing constants (from legacy)
PRE_MARGIN_FRAME_NUM: int = 60
POST_MARGIN_FRAME_NUM: int = 120
DEFAULT_PAD_FRAME: int = 180

# Default configuration file paths
DEFAULT_PLAYLIST_PATH: str = "//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/playlist.csv"
DEFAULT_MOVIE_LENGTH_PATH: str = "//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/movie_length.csv"
```

---

## Return Behaviors

| Condition | Return Value | Side Effect |
|-----------|--------------|-------------|
| Success | `True` | Zarr modified with section_time and light_template |
| Playlist file not found | `False` | Warning logged |
| Movie length file not found | `False` | Warning logged |
| Playlist name not in CSV | `False` | Error logged with available names |
| Zarr not found | `False` | Error logged |
| No frame_time in metadata | `False` | Error logged |
| No valid movies computed | `False` | Warning logged |
| Section time exists, force=False | N/A | Raises `FileExistsError` |
| Section time exists, force=True | `True` | Existing data overwritten |

---

## Zarr Output Schema

### section_time/{movie_name}

| Property | Value |
|----------|-------|
| dtype | int64 |
| shape | (n_repeats, 2) |
| description | Start/end frame pairs for each repeat |

### light_template/{movie_name}

| Property | Value |
|----------|-------|
| dtype | float32 |
| shape | (n_samples,) |
| description | Averaged light reference segment |

### Root Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| section_time_playlist | str | Playlist name used |
| section_time_repeats | int | Number of repeats |

---

## Pipeline Integration

### Export from `hdmea.pipeline`

```python
# In hdmea/pipeline/__init__.py
from hdmea.io.section_time import add_section_time

__all__ = [
    # ... existing exports ...
    "add_section_time",
]
```

### Usage in Pipeline

```python
from hdmea.pipeline import load_recording, extract_features, add_section_time

result = load_recording(...)
add_section_time(zarr_path=result.zarr_path, playlist_name="set6a")
extract_features(zarr_path=result.zarr_path, features=["frif"])
```

