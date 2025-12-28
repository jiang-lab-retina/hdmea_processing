"""
Direction-based spike sectioning for DSGC (Direction-Selective Ganglion Cell) analysis.

This module provides functions to section spike times by moving bar direction,
extracting responses when the bar crosses each cell's receptive field center.

Key features:
    - Per-pixel on/off timing from pre-computed dictionary
    - 8 directions × 3 repetitions = 24 trials
    - Configurable padding around on/off windows
    - Safe output with force parameter and output_path option

Example:
    >>> from hdmea.features import section_by_direction
    >>> result = section_by_direction("recording.h5", padding_frames=10)
    >>> print(f"Processed {result.units_processed} units")
"""

from __future__ import annotations

import logging
import pickle
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from hdmea.io.section_time import convert_sample_index_to_frame, PRE_MARGIN_FRAME_NUM

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DIRECTION_LIST: List[int] = [0, 45, 90, 135, 180, 225, 270, 315]
"""The 8 motion directions in degrees."""

N_DIRECTIONS: int = 8
"""Number of motion directions."""

N_REPETITIONS: int = 3
"""Number of repetitions per direction."""

N_TRIALS: int = 24
"""Total number of trials (8 directions × 3 repetitions)."""

DEFAULT_PADDING_FRAMES: int = 10
"""Default padding frames before/after on/off window."""

COORDINATE_SCALE_FACTOR: int = 20
"""Scaling factor from 15×15 grid to 300×300 pixel space."""

DEFAULT_STIMULI_DIR: Path = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations"
)
"""Default directory containing stimulus files."""

STA_GEOMETRY_FEATURE: str = "sta_perfect_dense_noise_15x15_15hz_r42_3min"
"""Name of the STA feature containing geometry data."""

DEFAULT_MOVIE_NAME: str = "moving_h_bar_s5_d8_3x"
"""Default moving bar movie name."""

INVALID_CENTER_VALUE: float = -5.0
"""Sentinel value indicating STA fitting failure (no valid RF center)."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DirectionSectionResult:
    """Result of direction sectioning computation.
    
    Attributes:
        hdf5_path: Path to the HDF5 file processed.
        movie_name: Name of the moving bar movie.
        units_processed: Number of units successfully processed.
        units_skipped: Number of units skipped (missing data or existing results).
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


# =============================================================================
# Helper Functions
# =============================================================================

def load_on_off_dict(
    dict_path: Union[str, Path],
) -> Dict[Tuple[int, int], Dict[str, List[int]]]:
    """
    Load the per-pixel on/off timing dictionary.
    
    Args:
        dict_path: Path to pickle file containing on/off timing data.
    
    Returns:
        Dictionary with (row, col) tuple keys and on/off frame lists.
        Structure: {(row, col): {'on_peak_location': [...], 'off_peak_location': [...]}}
    
    Raises:
        FileNotFoundError: If file not found.
        ValueError: If file structure is invalid.
    """
    dict_path = Path(dict_path)
    
    if not dict_path.exists():
        raise FileNotFoundError(
            f"On/off dictionary not found: {dict_path}. "
            f"Expected pickle file with per-pixel timing data."
        )
    
    with open(dict_path, "rb") as f:
        data = pickle.load(f)
    
    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(f"On/off dictionary must be a dict, got {type(data)}")
    
    # Check a sample entry
    if len(data) > 0:
        sample_key = next(iter(data.keys()))
        sample_val = data[sample_key]
        
        if not isinstance(sample_key, tuple) or len(sample_key) != 2:
            raise ValueError(f"Dictionary keys must be (row, col) tuples, got {sample_key}")
        
        if not isinstance(sample_val, dict):
            raise ValueError(f"Dictionary values must be dicts, got {type(sample_val)}")
        
        if "on_peak_location" not in sample_val or "off_peak_location" not in sample_val:
            raise ValueError(
                f"Dictionary values must have 'on_peak_location' and 'off_peak_location' keys"
            )
    
    logger.info(f"Loaded on/off dictionary: {len(data)} pixels")
    return data


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
    row_300 = int(center_row_15 * COORDINATE_SCALE_FACTOR)
    col_300 = int(center_col_15 * COORDINATE_SCALE_FACTOR)
    
    # Clip to valid range
    row_300 = max(0, min(299, row_300))
    col_300 = max(0, min(299, col_300))
    
    return (row_300, col_300)


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
        (row, col) in 300×300 space, or None if:
            - STA geometry path not found
            - center_row/center_col datasets missing
            - center values are invalid (-5.0 sentinel, indicating STA fitting failure)
    """
    geometry_path = f"units/{unit_id}/features/{sta_feature_name}/sta_geometry"
    
    if geometry_path not in hdf5_file:
        return None
    
    geometry_group = hdf5_file[geometry_path]
    
    if "center_row" not in geometry_group or "center_col" not in geometry_group:
        return None
    
    center_row_15 = float(geometry_group["center_row"][()])
    center_col_15 = float(geometry_group["center_col"][()])
    
    # Check for invalid sentinel value (STA fitting failure)
    if center_row_15 == INVALID_CENTER_VALUE or center_col_15 == INVALID_CENTER_VALUE:
        logger.warning(
            f"Unit {unit_id} has invalid STA center ({center_row_15}, {center_col_15}) - "
            f"STA fitting likely failed. Skipping."
        )
        return None
    
    # Convert to 300×300
    row_300, col_300 = convert_center_15_to_300(center_row_15, center_col_15)
    
    # Log warning if clipping occurred (for edge cases, not failure)
    expected_row = int(center_row_15 * COORDINATE_SCALE_FACTOR)
    expected_col = int(center_col_15 * COORDINATE_SCALE_FACTOR)
    
    if expected_row != row_300 or expected_col != col_300:
        logger.warning(
            f"Cell center for {unit_id} clipped: ({expected_row}, {expected_col}) → ({row_300}, {col_300})"
        )
    
    return (row_300, col_300)


def section_unit_by_direction(
    spike_frames: np.ndarray,
    cell_center: Tuple[int, int],
    on_off_dict: Dict[Tuple[int, int], Dict[str, List[int]]],
    padding_frames: int = DEFAULT_PADDING_FRAMES,
) -> Dict[int, Dict[str, Any]]:
    """
    Section spike frames by direction for a single unit.
    
    For each of the 24 trials (8 directions × 3 repetitions), extracts spikes
    that fall within the on/off window (with padding) at the cell's RF center.
    
    Args:
        spike_frames: Movie-relative frame indices of spikes.
        cell_center: (row, col) pixel coordinate of cell's RF center in 300×300 space.
        on_off_dict: Per-pixel on/off timing dictionary.
        padding_frames: Frames to pad before/after on/off window.
    
    Returns:
        Dictionary keyed by direction (0, 45, ..., 315) containing:
            - 'trials': List of 3 spike frame arrays (one per repetition)
            - 'bounds': List of 3 (start_frame, end_frame) tuples
    
    Raises:
        KeyError: If cell_center not found in on_off_dict.
    """
    if cell_center not in on_off_dict:
        raise KeyError(f"Cell center {cell_center} not found in on/off dictionary")
    
    pixel_timing = on_off_dict[cell_center]
    on_times = pixel_timing["on_peak_location"]
    off_times = pixel_timing["off_peak_location"]
    
    # Initialize result structure
    result: Dict[int, Dict[str, Any]] = {}
    for direction in DIRECTION_LIST:
        result[direction] = {
            "trials": [],
            "bounds": [],
        }
    
    # Process each trial
    for trial_idx in range(N_TRIALS):
        # Map trial index to direction and repetition
        direction_idx = trial_idx % N_DIRECTIONS
        direction = DIRECTION_LIST[direction_idx]
        
        # Get window bounds with padding
        on_frame = on_times[trial_idx]
        off_frame = off_times[trial_idx]
        start_frame = on_frame - padding_frames
        end_frame = off_frame + padding_frames
        
        # Extract spikes in window
        mask = (spike_frames >= start_frame) & (spike_frames <= end_frame)
        trial_spikes = spike_frames[mask]
        
        # Store results
        result[direction]["trials"].append(trial_spikes)
        result[direction]["bounds"].append((start_frame, end_frame))
    
    return result


def section_by_direction(
    hdf5_path: Union[str, Path],
    *,
    movie_name: str = DEFAULT_MOVIE_NAME,
    on_off_dict_path: Optional[Union[str, Path]] = None,
    padding_frames: int = DEFAULT_PADDING_FRAMES,
    force: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    unit_ids: Optional[List[str]] = None,
) -> DirectionSectionResult:
    """
    Section spike times by moving bar direction for all units.
    
    Extracts spikes occurring when the moving bar crosses each unit's
    receptive field center, organized by motion direction.
    
    The trial window is defined as [on_time - padding, off_time + padding]
    where on_time and off_time are the frame indices when the bar covers
    and leaves the cell's RF center pixel.
    
    Args:
        hdf5_path: Path to HDF5 recording file.
        movie_name: Name of moving bar movie. Default: "moving_h_bar_s5_d8_3x".
        on_off_dict_path: Path to on/off timing pickle file.
            Default: stimuli_dir/{movie_name}_on_off_dict_area_hd.pkl
        padding_frames: Frames to add before/after on/off window. Default: 10.
            Must be >= 0. Padding is applied symmetrically.
        force: If True, overwrite existing direction_section data. Default: False.
        output_path: Optional path to write results. If provided, copies source
            HDF5 to this path before modifying. Default: None (in-place).
        unit_ids: Optional list of unit IDs to process. Default: None (all units).
    
    Returns:
        DirectionSectionResult with processing statistics.
    
    Raises:
        FileNotFoundError: If HDF5 file or on/off dict not found.
        ValueError: If no units found with required data, or invalid parameters.
    
    Example:
        >>> from hdmea.features import section_by_direction
        >>> result = section_by_direction("recording.h5", padding_frames=10)
        >>> print(f"Processed {result.units_processed} units")
    """
    start_time = time.time()
    
    hdf5_path = Path(hdf5_path)
    warnings_list: List[str] = []
    skipped_units: List[str] = []
    
    # Validate parameters
    if padding_frames < 0:
        raise ValueError(f"padding_frames must be >= 0, got {padding_frames}")
    
    logger.info(f"Direction sectioning with padding={padding_frames} frames")
    
    # Determine on/off dict path
    if on_off_dict_path is None:
        on_off_dict_path = DEFAULT_STIMULI_DIR / f"{movie_name}_on_off_dict_area_hd.pkl"
    else:
        on_off_dict_path = Path(on_off_dict_path)
    
    # Load on/off dictionary
    on_off_dict = load_on_off_dict(on_off_dict_path)
    
    # Handle output_path: copy source to output
    target_path = hdf5_path
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Copying source to {output_path}")
        shutil.copy2(hdf5_path, output_path)
        target_path = output_path
    
    # Open HDF5 file for reading prerequisites
    with h5py.File(target_path, "r") as hdf5_file:
        # Get frame_timestamps
        frame_timestamps_path = None
        for path in ["metadata/frame_timestamps", "stimulus/frame_time/default"]:
            if path in hdf5_file:
                frame_timestamps_path = path
                break
        
        if frame_timestamps_path is None:
            raise ValueError(
                f"No frame timestamps found in {target_path}. "
                "Expected 'metadata/frame_timestamps' or 'stimulus/frame_time/default'."
            )
        
        frame_timestamps = hdf5_file[frame_timestamps_path][:]
        
        # Get section_time for movie start
        section_time_path = f"stimulus/section_time/{movie_name}"
        if section_time_path not in hdf5_file:
            raise ValueError(
                f"No section_time found for movie '{movie_name}' in {target_path}."
            )
        
        section_time = hdf5_file[section_time_path][:]
        movie_start_sample = section_time[0, 0]
        movie_start_frame = int(convert_sample_index_to_frame(
            np.array([movie_start_sample]), frame_timestamps
        )[0]) + PRE_MARGIN_FRAME_NUM
        
        logger.info(f"Movie '{movie_name}' starts at frame {movie_start_frame}")
        
        # Get list of units
        if "units" not in hdf5_file:
            raise ValueError(f"No 'units' group found in {target_path}")
        
        all_unit_ids = list(hdf5_file["units"].keys())
        
        # Filter by unit_ids if provided
        if unit_ids is not None:
            # Validate specified units exist
            for uid in unit_ids:
                if uid not in all_unit_ids:
                    msg = f"Specified unit '{uid}' not found in HDF5 file"
                    warnings_list.append(msg)
                    logger.warning(msg)
            
            target_unit_ids = [uid for uid in unit_ids if uid in all_unit_ids]
        else:
            target_unit_ids = all_unit_ids
        
        logger.info(f"Processing {len(target_unit_ids)} units")
        
        # Collect unit data for processing
        units_data: Dict[str, Dict[str, Any]] = {}
        
        for unit_id in target_unit_ids:
            # Get cell center
            cell_center = get_cell_center(hdf5_file, unit_id, STA_GEOMETRY_FEATURE)
            if cell_center is None:
                msg = f"No STA geometry for {unit_id}, skipping"
                warnings_list.append(msg)
                skipped_units.append(unit_id)
                logger.warning(msg)
                continue
            
            # Get full_spike_times
            spike_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/full_spike_times"
            if spike_path not in hdf5_file:
                msg = f"No full_spike_times for {unit_id}/{movie_name}, skipping"
                warnings_list.append(msg)
                skipped_units.append(unit_id)
                logger.warning(msg)
                continue
            
            spike_samples = hdf5_file[spike_path][:]
            
            # Convert spike samples to movie-relative frames
            spike_frames_absolute = convert_sample_index_to_frame(spike_samples, frame_timestamps)
            spike_frames = spike_frames_absolute - movie_start_frame
            
            units_data[unit_id] = {
                "cell_center": cell_center,
                "spike_frames": spike_frames,
                "spike_samples": spike_samples,  # Keep for bounds calculation
            }
    
    # Process units and write results
    units_processed = 0
    
    with h5py.File(target_path, "r+") as hdf5_file:
        for unit_id, data in units_data.items():
            section_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/direction_section"
            
            # Check if already exists
            if section_path in hdf5_file:
                if not force:
                    msg = f"direction_section already exists for {unit_id}, skipping (use force=True to overwrite)"
                    logger.info(msg)
                    skipped_units.append(unit_id)
                    continue
                else:
                    # Delete existing group
                    del hdf5_file[section_path]
                    logger.debug(f"Deleted existing direction_section for {unit_id}")
            
            # Section spikes by direction
            try:
                section_result = section_unit_by_direction(
                    data["spike_frames"],
                    data["cell_center"],
                    on_off_dict,
                    padding_frames,
                )
            except KeyError as e:
                msg = f"Cell center {data['cell_center']} not in on/off dict for {unit_id}, skipping"
                warnings_list.append(msg)
                skipped_units.append(unit_id)
                logger.warning(msg)
                continue
            
            # Create direction_section group
            section_group = hdf5_file.require_group(section_path)
            
            # Add attributes
            section_group.attrs["direction_list"] = DIRECTION_LIST
            section_group.attrs["n_directions"] = N_DIRECTIONS
            section_group.attrs["n_repetitions"] = N_REPETITIONS
            section_group.attrs["padding_frames"] = padding_frames
            section_group.attrs["cell_center_row"] = data["cell_center"][0]
            section_group.attrs["cell_center_col"] = data["cell_center"][1]
            
            # Get frame_timestamps again for bounds conversion
            frame_timestamps = hdf5_file[frame_timestamps_path][:]
            
            # Write each direction
            for direction in DIRECTION_LIST:
                dir_group = section_group.require_group(str(direction))
                trials_group = dir_group.require_group("trials")
                
                dir_data = section_result[direction]
                bounds_list = []
                
                for rep_idx, (trial_spikes, bounds) in enumerate(
                    zip(dir_data["trials"], dir_data["bounds"])
                ):
                    # Convert frame bounds back to sample indices for storage
                    start_frame, end_frame = bounds
                    
                    # Store spike times (convert frames back to samples if needed,
                    # or store the original samples that fell in the window)
                    # For now, store empty array if no spikes
                    if len(trial_spikes) == 0:
                        trials_group.create_dataset(
                            str(rep_idx),
                            data=np.array([], dtype=np.int64),
                            dtype=np.int64
                        )
                    else:
                        # Find original sample indices for spikes in this window
                        spike_frames = data["spike_frames"]
                        spike_samples = data["spike_samples"]
                        mask = (spike_frames >= start_frame) & (spike_frames <= end_frame)
                        trial_samples = spike_samples[mask]
                        
                        trials_group.create_dataset(
                            str(rep_idx),
                            data=trial_samples,
                            dtype=np.int64
                        )
                    
                    bounds_list.append([start_frame, end_frame])
                
                # Save section_bounds
                dir_group.create_dataset(
                    "section_bounds",
                    data=np.array(bounds_list, dtype=np.int64),
                    dtype=np.int64
                )
            
            units_processed += 1
            logger.debug(f"Processed {unit_id}: center={data['cell_center']}")
    
    elapsed = time.time() - start_time
    
    result = DirectionSectionResult(
        hdf5_path=target_path,
        movie_name=movie_name,
        units_processed=units_processed,
        units_skipped=len(skipped_units),
        padding_frames=padding_frames,
        elapsed_seconds=elapsed,
        warnings=warnings_list,
        skipped_units=skipped_units,
    )
    
    logger.info(
        f"Direction sectioning complete: {units_processed} processed, "
        f"{len(skipped_units)} skipped, {elapsed:.1f}s elapsed"
    )
    
    return result

