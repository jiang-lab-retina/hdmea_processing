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

from hdmea.io.zarr_store import open_recording_zarr


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
    Add section time metadata to a Zarr recording.
    
    Computes frame boundaries for each movie in the playlist and stores
    them under stimulus/section_time/{movie_name}. Also extracts and
    averages light templates for each section.
    
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
    section_time_auto: Dict[str, np.ndarray] = {}
    template_auto: Dict[str, np.ndarray] = {}
    
    for movie_name in movie_start_end_frame:
        # Convert frame boundaries to numpy array
        section_time_auto[movie_name] = np.array(movie_start_end_frame[movie_name], dtype=np.int64)
        
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

