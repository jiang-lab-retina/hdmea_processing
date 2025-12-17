"""
Section time loading for visual stimulation experiments.

Computes frame boundaries for each movie in a playlist by parsing
playlist and movie_length CSV configuration files.
"""

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from scipy.signal import find_peaks

from hdmea.io.zarr_store import open_recording_zarr
from hdmea.utils.exceptions import MissingInputError


logger = logging.getLogger(__name__)


# =============================================================================
# Constants (from legacy code)
# =============================================================================

PRE_MARGIN_FRAME_NUM: int = 60
POST_MARGIN_FRAME_NUM: int = 120
DEFAULT_PAD_FRAME: int = 180

# Default paths for configuration files
DEFAULT_PLAYLIST_PATH: str = "//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/playlist.csv"
DEFAULT_MOVIE_LENGTH_PATH: str = "//Jiangfs1/fs_1_2_data/Python_Project/Design_Stimulation_Pattern/Data/movie_length.csv"


# =============================================================================
# Helper Functions
# =============================================================================

def _convert_frame_to_sample_index(frame: np.ndarray, frame_timestamps: np.ndarray) -> np.ndarray:
    """
    Convert frame numbers to acquisition sample indices using frame_timestamps array.
    
    This matches the legacy convert_frame_to_time() behavior where frame_time
    contained sample indices, not seconds. The raw light reference is sampled
    at acquisition rate, so we need sample indices to slice into it.
    
    Args:
        frame: Array of display frame numbers
        frame_timestamps: Array of sample indices for each display frame
    
    Returns:
        Array of sample indices corresponding to frame numbers
    """
    frame = np.array(frame).astype(int)
    # Clip to valid range
    frame = np.clip(frame, 0, len(frame_timestamps) - 1)
    return frame_timestamps[frame]


def _sample_to_nearest_frame(
    sample_idx: Union[int, np.ndarray],
    frame_timestamps: np.ndarray,
) -> Union[int, np.ndarray]:
    """
    Convert acquisition sample index(es) to nearest display frame index(es).
    
    Uses binary search for O(log n) lookup per sample. This is the inverse
    operation of _convert_frame_to_sample_index().
    
    Args:
        sample_idx: Sample index or array of sample indices
        frame_timestamps: Array of sample indices for each display frame
    
    Returns:
        Nearest frame index or array of frame indices
    """
    sample_idx = np.asarray(sample_idx)
    scalar_input = sample_idx.ndim == 0
    sample_idx = np.atleast_1d(sample_idx)
    
    # Use searchsorted for O(log n) lookup
    insert_pos = np.searchsorted(frame_timestamps, sample_idx)
    
    # Handle edge cases
    result = np.empty_like(insert_pos)
    
    for i, (pos, sample) in enumerate(zip(insert_pos, sample_idx)):
        if pos == 0:
            result[i] = 0
        elif pos >= len(frame_timestamps):
            result[i] = len(frame_timestamps) - 1
        else:
            # Find nearest between pos-1 and pos
            before = frame_timestamps[pos - 1]
            after = frame_timestamps[pos]
            if (sample - before) <= (after - sample):
                result[i] = pos - 1
            else:
                result[i] = pos
    
    if scalar_input:
        return int(result[0])
    return result


def _detect_analog_peaks(
    signal: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Detect peaks in signal derivative above threshold.
    
    Finds sharp transitions (rising edges) in the analog signal by
    computing the derivative and finding peaks above the threshold.
    
    Args:
        signal: Raw analog signal array
        threshold: Minimum peak height in derivative
    
    Returns:
        Array of sample indices where peaks detected
    """
    # Compute derivative to detect transitions
    derivative = np.diff(signal.astype(np.float64))
    
    # Find peaks above threshold
    peaks, _ = find_peaks(derivative, height=threshold)
    
    return peaks


def _load_playlist_csv(playlist_file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load playlist CSV file.
    
    Args:
        playlist_file_path: Path to playlist.csv
    
    Returns:
        DataFrame indexed by playlist_name, or None if file not found
    """
    path = Path(playlist_file_path)
    if not path.exists():
        logger.warning(f"Playlist file not found: {path}")
        return None
    
    try:
        playlist = pd.read_csv(path)
        playlist = playlist.set_index("playlist_name")
        return playlist
    except Exception as e:
        logger.error(f"Failed to load playlist CSV: {e}")
        return None


def _load_movie_length_csv(movie_length_file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load movie length CSV file.
    
    Args:
        movie_length_file_path: Path to movie_length.csv
    
    Returns:
        DataFrame indexed by movie_name, or None if file not found
    """
    path = Path(movie_length_file_path)
    if not path.exists():
        logger.warning(f"Movie length file not found: {path}")
        return None
    
    try:
        movies_length = pd.read_csv(path)
        movies_length = movies_length.set_index("movie_name")
        return movies_length
    except Exception as e:
        logger.error(f"Failed to load movie length CSV: {e}")
        return None


def _get_movie_start_end_frame(
    playlist_name: str,
    repeats: int,
    all_playlists: pd.DataFrame,
    movies_length: pd.DataFrame,
    frame_timestamps: np.ndarray,
    light_reference_raw: Optional[np.ndarray] = None,
    pad_frame: int = DEFAULT_PAD_FRAME,
    pre_margin_frame_num: int = PRE_MARGIN_FRAME_NUM,
    post_margin_frame_num: int = POST_MARGIN_FRAME_NUM,
) -> Tuple[List[str], Dict[str, List[List[int]]], Dict[str, List[np.ndarray]]]:
    """
    Compute start/end frame for each movie in a playlist.
    
    Args:
        playlist_name: Name of playlist in all_playlists
        repeats: Number of times the playlist is repeated
        all_playlists: DataFrame with playlist definitions
        movies_length: DataFrame with movie durations
        frame_timestamps: Array of sample indices for each display frame
        light_reference_raw: Optional raw_ch1 light reference signal for template extraction
        pad_frame: Padding frames between movies
        pre_margin_frame_num: Frames before movie start to include
        post_margin_frame_num: Frames after movie end to include
    
    Returns:
        Tuple of (movie_list, movie_start_end_frame, movie_light_template)
    """
    # Parse movie list from playlist
    movie_list_str = all_playlists.loc[playlist_name]["movie_names"]
    movie_list = eval(movie_list_str)  # Parse Python list string
    movie_list = [x.split(".")[0] for x in movie_list] * repeats
    
    movie_start_end_frame: Dict[str, List[List[int]]] = {}
    movie_light_template: Dict[str, List[np.ndarray]] = {}
    frame_count = 0
    
    for movie_name in movie_list:
        if movie_name not in movies_length.index:
            logger.warning(f"Movie '{movie_name}' not found in movie_length database, skipping")
            continue
        
        movie_length = int(movies_length.loc[movie_name]["movie_length"])
        
        # Calculate frame boundaries
        start_frame = frame_count + pad_frame - pre_margin_frame_num
        end_frame = frame_count + pad_frame + post_margin_frame_num + movie_length + 1
        start_frame, end_frame = int(start_frame), int(end_frame)
        
        # Extract light template from raw_ch1 if available
        single_movie_light_template = None
        if light_reference_raw is not None and len(frame_timestamps) > 0:
            # Convert display frame numbers to acquisition sample indices
            start_sample_idx = int(_convert_frame_to_sample_index(np.array([start_frame]), frame_timestamps)[0])
            end_sample_idx = int(_convert_frame_to_sample_index(np.array([end_frame]), frame_timestamps)[0])
            
            # Clip to available raw data
            start_sample_idx = max(0, start_sample_idx)
            end_sample_idx = min(len(light_reference_raw), end_sample_idx)
            
            if start_sample_idx < end_sample_idx:
                single_movie_light_template = light_reference_raw[start_sample_idx:end_sample_idx]
        
        # Accumulate results for repeated movies
        if movie_name in movie_start_end_frame:
            movie_start_end_frame[movie_name].append([start_frame, end_frame])
            if single_movie_light_template is not None:
                movie_light_template[movie_name].append(single_movie_light_template)
        else:
            movie_start_end_frame[movie_name] = [[start_frame, end_frame]]
            if single_movie_light_template is not None:
                movie_light_template[movie_name] = [single_movie_light_template]
        
        # Update frame count for next movie
        frame_count += (2 * pad_frame + movie_length + 1)
    
    return movie_list, movie_start_end_frame, movie_light_template


# =============================================================================
# Main API
# =============================================================================

def add_section_time(
    zarr_path: Union[str, Path],
    playlist_name: str,
    *,
    playlist_file_path: Optional[Union[str, Path]] = None,
    movie_length_file_path: Optional[Union[str, Path]] = None,
    repeats: int = 1,
    pad_frame: int = DEFAULT_PAD_FRAME,
    pre_margin_frame_num: int = PRE_MARGIN_FRAME_NUM,
    post_margin_frame_num: int = POST_MARGIN_FRAME_NUM,
    force: bool = False,
) -> bool:
    """
    Add section time metadata to a Zarr recording from playlist CSV.
    
    Computes frame boundaries for each movie in the playlist, converts them
    to acquisition sample indices using frame_timestamps, and stores them
    under stimulus/section_time/{movie_name}. Also extracts and averages
    light templates for each section.
    
    Note: Output is in acquisition sample indices (unified with analog
    section time) for consistent downstream processing. To convert to time:
    ``time_seconds = sample_index / acquisition_rate``
    
    Args:
        zarr_path: Path to Zarr archive
        playlist_name: Name of playlist in playlist.csv
        playlist_file_path: Path to playlist CSV (uses default if None)
        movie_length_file_path: Path to movie_length CSV (uses default if None)
        repeats: Number of playlist repeats
        pad_frame: Padding frames between movies (display frames)
        pre_margin_frame_num: Display frames before movie start to include
        post_margin_frame_num: Display frames after movie end to include
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
        >>> success = add_section_time(
        ...     zarr_path="artifacts/REC_2023-12-07.zarr",
        ...     playlist_name="set6a",
        ...     repeats=2,
        ... )
        >>> if success:
        ...     print("Section times added!")
    """
    zarr_path = Path(zarr_path)
    
    # Use default paths if not provided
    if playlist_file_path is None:
        playlist_file_path = DEFAULT_PLAYLIST_PATH
    if movie_length_file_path is None:
        movie_length_file_path = DEFAULT_MOVIE_LENGTH_PATH
    
    # Treat repeats <= 0 as 1
    if repeats <= 0:
        repeats = 1
    
    # Load configuration files
    playlist = _load_playlist_csv(playlist_file_path)
    if playlist is None:
        logger.error("Failed to load playlist configuration")
        return False
    
    movies_length = _load_movie_length_csv(movie_length_file_path)
    if movies_length is None:
        logger.error("Failed to load movie length configuration")
        return False
    
    # Check playlist name exists
    if playlist_name not in playlist.index:
        available = list(playlist.index)
        logger.error(f"Playlist '{playlist_name}' not found. Available: {available}")
        return False
    
    # Open Zarr
    try:
        root = open_recording_zarr(zarr_path, mode="r+")
    except FileNotFoundError:
        logger.error(f"Zarr not found: {zarr_path}")
        return False
    
    # Check for existing section_time
    stimulus_group = root["stimulus"]
    if "section_time" in stimulus_group and not force:
        raise FileExistsError(
            f"section_time already exists in {zarr_path}. "
            "Use force=True to overwrite."
        )
    
    # Get frame_timestamps (sample indices) from metadata
    # frame_timestamps maps display frame numbers to acquisition sample indices
    frame_timestamps = None
    if "metadata" in root and "frame_timestamps" in root["metadata"]:
        frame_timestamps = np.array(root["metadata"]["frame_timestamps"])
    
    if frame_timestamps is None or len(frame_timestamps) == 0:
        logger.error("No frame_timestamps found in metadata")
        return False
    
    # Get raw_ch1 light reference for template extraction
    # raw_ch1 contains the light intensity signal (legacy: light_reference_raw[0, :])
    # raw_ch2 is the frame sync channel, not used for templates
    light_reference_raw = None
    if "stimulus" in root and "light_reference" in root["stimulus"]:
        lr_group = root["stimulus"]["light_reference"]
        if "raw_ch1" in lr_group:
            light_reference_raw = np.array(lr_group["raw_ch1"])
    
    # Compute section times
    logger.info(f"Computing section times for playlist '{playlist_name}' with {repeats} repeat(s)")
    
    movie_list, movie_start_end_frame, movie_light_template = _get_movie_start_end_frame(
        playlist_name=playlist_name,
        repeats=repeats,
        all_playlists=playlist,
        movies_length=movies_length,
        frame_timestamps=frame_timestamps,
        light_reference_raw=light_reference_raw,
        pad_frame=pad_frame,
        pre_margin_frame_num=pre_margin_frame_num,
        post_margin_frame_num=post_margin_frame_num,
    )
    
    if not movie_start_end_frame:
        logger.warning("No section times computed - no valid movies found")
        return False
    
    # Prepare section_time data
    # Convert frame indices to acquisition sample indices for unified output unit
    section_time_auto: Dict[str, np.ndarray] = {}
    template_auto: Dict[str, np.ndarray] = {}
    
    for movie_name in movie_start_end_frame:
        # Convert frame boundaries to acquisition sample indices
        frame_pairs = np.array(movie_start_end_frame[movie_name], dtype=np.int64)
        # frame_pairs shape: (N, 2) where each row is [start_frame, end_frame]
        # Convert using frame_timestamps lookup
        sample_pairs = np.zeros_like(frame_pairs)
        for i, (start_frame, end_frame) in enumerate(frame_pairs):
            # Clip to valid frame range
            start_frame = np.clip(start_frame, 0, len(frame_timestamps) - 1)
            end_frame = np.clip(end_frame, 0, len(frame_timestamps) - 1)
            sample_pairs[i, 0] = frame_timestamps[start_frame]
            sample_pairs[i, 1] = frame_timestamps[end_frame]
        section_time_auto[movie_name] = sample_pairs
        
        # Average light templates if available
        if movie_name in movie_light_template and movie_light_template[movie_name]:
            templates = movie_light_template[movie_name]
            # Use zip_longest to handle variable lengths
            template_list = list(itertools.zip_longest(*templates, fillvalue=np.nan))
            template_auto[movie_name] = np.nanmean(template_list, axis=1).astype(np.float32)
    
    # Write to Zarr
    # Create/overwrite section_time group
    if "section_time" in stimulus_group:
        del stimulus_group["section_time"]
    st_group = stimulus_group.create_group("section_time")
    
    for movie_name, times in section_time_auto.items():
        st_group.create_dataset(
            movie_name,
            data=times,
            shape=times.shape,
            dtype=times.dtype,
        )
    
    logger.info(f"Wrote section_time for {len(section_time_auto)} movies")
    
    # Create/overwrite light_template group
    if template_auto:
        if "light_template" in stimulus_group:
            del stimulus_group["light_template"]
        lt_group = stimulus_group.create_group("light_template")
        
        for movie_name, template in template_auto.items():
            lt_group.create_dataset(
                movie_name,
                data=template,
                shape=template.shape,
                dtype=template.dtype,
            )
        
        logger.info(f"Wrote light_template for {len(template_auto)} movies")
    
    # Store metadata about section time computation
    root.attrs["section_time_playlist"] = playlist_name
    root.attrs["section_time_repeats"] = repeats
    
    logger.info(f"Section times added successfully to {zarr_path}")
    return True


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
    
    Unlike add_section_time() which uses playlist CSV files to predict timing,
    this function detects actual stimulus onsets from the recorded light signal.
    This is suitable for experiments where exact timing must be determined
    post-hoc (e.g., ipRGC stimulation tests).
    
    Uses raw_ch1 (full acquisition rate signal) for peak detection, providing
    maximum temporal resolution. Detected peaks are stored directly as
    acquisition sample indices. Also extracts and averages the light reference
    traces across all detected trials to create a stimulus template.
    
    Args:
        zarr_path: Path to Zarr archive containing recording data
        threshold_value: Required. Peak height threshold for find_peaks().
            User must inspect signal to determine appropriate value.
        movie_name: Identifier for this stimulus type (default: "iprgc_test").
            Used as key under stimulus/section_time/{movie_name}
        plot_duration: Duration of each section in seconds (default: 120.0).
            End sample = onset_sample + (plot_duration * acquisition_rate)
        repeat: If specified, limit to first N detected trials for section_time.
            Note: Light template uses all detected trials regardless of repeat.
        force: If True, overwrite existing section_time for this movie_name.
            Default False raises FileExistsError if data exists.
    
    Returns:
        True if section times were successfully added, False on failure
        (e.g., no peaks detected)
    
    Raises:
        FileNotFoundError: If zarr_path does not exist
        MissingInputError: If required data missing:
            - stimulus/light_reference/raw_ch1
            - metadata/acquisition_rate
        ValueError: If threshold_value not provided or plot_duration <= 0
        FileExistsError: If section_time/{movie_name} exists and force=False
    
    Output Format:
        stimulus/section_time/{movie_name}: int64[N, 2] array
        - Each row: [start_sample, end_sample] in acquisition sample indices
        - N = number of detected trials (or min(detected, repeat) if repeat specified)
        
        stimulus/light_template/{movie_name}: float32[M] array
        - Averaged light reference trace across all detected trials
        - M = duration_samples = int(plot_duration * acquisition_rate)
    
    Example:
        >>> from hdmea.io.section_time import add_section_time_analog
        >>> 
        >>> # First inspect signal to determine threshold
        >>> # (e.g., plot np.diff(raw_ch1) and identify peak heights)
        >>> 
        >>> success = add_section_time_analog(
        ...     zarr_path="artifacts/JIANG009_2025-04-10.zarr",
        ...     threshold_value=1e5,  # Determined from signal inspection
        ...     movie_name="iprgc_test",
        ...     plot_duration=120.0,  # 2 minute windows
        ...     repeat=3,  # Use first 3 trials only
        ... )
        >>> if success:
        ...     print("Section times and light template stored!")
    """
    zarr_path = Path(zarr_path)
    
    # Validate parameters
    if threshold_value is None:
        raise ValueError("threshold_value is required (no default)")
    if plot_duration <= 0:
        raise ValueError(f"plot_duration must be positive, got {plot_duration}")
    if not movie_name:
        raise ValueError("movie_name cannot be empty")
    
    logger.info(f"Detecting analog section times for '{movie_name}' in {zarr_path}")
    
    # Open Zarr
    try:
        root = open_recording_zarr(zarr_path, mode="r+")
    except FileNotFoundError:
        logger.error(f"Zarr not found: {zarr_path}")
        raise
    
    # Validate required inputs exist
    # Check raw_ch1
    if "stimulus" not in root or "light_reference" not in root["stimulus"]:
        raise MissingInputError(
            "stimulus/light_reference group not found in Zarr",
            missing_input="stimulus/light_reference",
        )
    
    lr_group = root["stimulus"]["light_reference"]
    if "raw_ch1" not in lr_group:
        raise MissingInputError(
            "raw_ch1 signal not found in stimulus/light_reference. "
            "This function requires the raw light reference at acquisition rate.",
            missing_input="stimulus/light_reference/raw_ch1",
        )
    
    # Check acquisition_rate
    if "metadata" not in root or "acquisition_rate" not in root["metadata"]:
        raise MissingInputError(
            "acquisition_rate not found in metadata",
            missing_input="metadata/acquisition_rate",
        )
    
    # Load required data
    raw_ch1 = np.array(lr_group["raw_ch1"])
    acquisition_rate = float(np.array(root["metadata"]["acquisition_rate"]).flat[0])
    
    logger.info(f"Loaded raw_ch1: {len(raw_ch1)} samples at {acquisition_rate} Hz")
    
    # Check for existing section_time for this movie_name
    stimulus_group = root["stimulus"]
    if "section_time" in stimulus_group:
        st_group = stimulus_group["section_time"]
        if movie_name in st_group and not force:
            raise FileExistsError(
                f"section_time/{movie_name} already exists in {zarr_path}. "
                "Use force=True to overwrite."
            )
    
    # Detect peaks in raw_ch1 signal
    # Peaks are returned as acquisition sample indices (at ~20 kHz)
    logger.info(f"Detecting peaks with threshold={threshold_value}")
    onset_samples = _detect_analog_peaks(raw_ch1, threshold_value)
    
    if len(onset_samples) == 0:
        logger.warning(f"No peaks detected with threshold={threshold_value}")
        return False
    
    logger.info(f"Detected {len(onset_samples)} stimulus onsets")
    
    # Calculate duration in samples
    duration_samples = int(plot_duration * acquisition_rate)
    max_sample = len(raw_ch1) - 1
    
    # Extract light reference traces from ALL detected trials for template averaging
    # (before applying repeat limit)
    all_traces = []
    for onset in onset_samples:
        end_sample = min(onset + duration_samples, max_sample)
        trace = raw_ch1[onset:end_sample]
        all_traces.append(trace)
    
    # Average across all trials (use zip_longest for variable lengths)
    if all_traces:
        template_list = list(itertools.zip_longest(*all_traces, fillvalue=np.nan))
        light_template = np.nanmean(template_list, axis=1).astype(np.float32)
        logger.info(f"Computed light template from {len(all_traces)} trials: {len(light_template)} samples")
    else:
        light_template = None
    
    # Apply repeat limit if specified (for section_time storage only)
    if repeat is not None and repeat > 0:
        onset_samples = onset_samples[:repeat]
        logger.info(f"Limited section_time to first {repeat} trials: {len(onset_samples)} onsets")
    
    # Calculate end samples based on plot_duration
    end_samples = onset_samples + duration_samples
    
    # Clip end samples to signal length
    n_truncated = np.sum(end_samples > max_sample)
    if n_truncated > 0:
        logger.warning(
            f"{n_truncated} section(s) truncated at signal boundary "
            f"(end sample clipped to {max_sample:,})"
        )
    end_samples = np.clip(end_samples, 0, max_sample)
    
    # Build section_time array: shape (N, 2) with [start_sample, end_sample]
    # Store directly as acquisition sample indices (no frame conversion)
    section_time = np.column_stack([onset_samples, end_samples]).astype(np.int64)
    
    logger.info(f"Computed section_time: {section_time.shape[0]} trials, "
                f"sample range [{section_time[:, 0].min()}, {section_time[:, 1].max()}]")
    
    # Write to Zarr
    if "section_time" not in stimulus_group:
        st_group = stimulus_group.create_group("section_time")
    else:
        st_group = stimulus_group["section_time"]
        # Delete existing movie_name dataset if force=True
        if movie_name in st_group:
            del st_group[movie_name]
    
    st_group.create_dataset(
        movie_name,
        data=section_time,
        shape=section_time.shape,
        dtype=section_time.dtype,
    )
    
    logger.info(f"Wrote section_time/{movie_name} with {len(section_time)} trials")
    
    # Write light_template
    if light_template is not None:
        if "light_template" not in stimulus_group:
            lt_group = stimulus_group.create_group("light_template")
        else:
            lt_group = stimulus_group["light_template"]
            if movie_name in lt_group:
                del lt_group[movie_name]
        
        lt_group.create_dataset(
            movie_name,
            data=light_template,
            shape=light_template.shape,
            dtype=light_template.dtype,
        )
        logger.info(f"Wrote light_template/{movie_name} with {len(light_template)} samples")
    
    # Store metadata about analog section time computation
    root.attrs[f"section_time_analog_{movie_name}"] = {
        "threshold_value": threshold_value,
        "plot_duration": plot_duration,
        "n_trials": len(section_time),
    }
    
    logger.info(f"Analog section times added successfully to {zarr_path}")
    return True

