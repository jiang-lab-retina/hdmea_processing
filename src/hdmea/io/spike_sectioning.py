"""
Spike times sectioning module for HD-MEA pipeline.

Provides functionality to section spike timestamps by stimulation periods
(trials) using JSON configuration files from config/stimuli/. Each stimulus
type has its own JSON file defining trial parameters (start_frame,
trial_length_frame, repeat). Supports both combined (full_spike_times)
and per-trial (trials_spike_times) storage formats.

JSON Config Format:
    Each stimulus JSON file in config/stimuli/{movie_name}.json must contain:
    
    {
        "section_kwargs": {
            "start_frame": int,        # First trial start frame (relative to movie content)
            "trial_length_frame": int, # Duration of each trial in frames
            "repeat": int              # Number of trial repetitions
        }
    }

Example:
    >>> from hdmea.io.spike_sectioning import section_spike_times
    >>> result = section_spike_times(
    ...     hdf5_path="artifacts/JIANG009_2025-04-10.h5",
    ...     pad_margin=(2.0, 0.0),  # 2s pre-margin, 0s post-margin
    ... )
    >>> print(f"Processed {result.units_processed} units")
    
    # With custom config directory:
    >>> result = section_spike_times(
    ...     hdf5_path="recording.h5",
    ...     config_dir="path/to/custom/configs/",
    ... )
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import h5py
import numpy as np
from tqdm import tqdm

from hdmea.io.section_time import _convert_frame_to_sample_index, _convert_sample_index_to_frame

from hdmea.io.hdf5_store import open_recording_hdf5
from hdmea.io.section_time import PRE_MARGIN_FRAME_NUM, POST_MARGIN_FRAME_NUM

logger = logging.getLogger(__name__)


# Windows file locking workaround: retry operations that may fail due to file handles
MAX_RETRIES = 3
RETRY_DELAY = 0.1  # seconds (only used on actual retry, not every operation)

# Default directory for stimulus JSON configuration files
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config" / "stimuli"


class StimuliConfigDict(TypedDict):
    """Type definition for stimulus section_kwargs from JSON config files."""
    start_frame: int
    trial_length_frame: int
    repeat: int


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class SectionResult:
    """Result of spike times sectioning operation.
    
    Attributes:
        success: Whether the operation completed successfully.
        units_processed: Number of units that were processed.
        movies_processed: List of movie names that were processed.
        trial_repeats: Number of trials processed per movie.
        pad_margin: Tuple of (pre_margin_s, post_margin_s) in seconds.
        pre_samples: Pre-margin padding converted to samples.
        post_samples: Post-margin padding converted to samples.
        warnings: List of warning messages generated during processing.
    """
    success: bool
    units_processed: int
    movies_processed: List[str]
    trial_repeats: int
    pad_margin: Tuple[float, float]
    pre_samples: int
    post_samples: int
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Config Loading Functions
# =============================================================================

def _load_stimuli_config(
    movie_name: str,
    config_dir: Path,
) -> StimuliConfigDict:
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
    config_path = config_dir / f"{movie_name}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Stimulus config file not found: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config '{movie_name}': {e}")
    
    # Validate section_kwargs exists
    if "section_kwargs" not in config_data:
        raise ValueError(f"Config '{movie_name}' missing 'section_kwargs' object")
    
    section_kwargs = config_data["section_kwargs"]
    
    # Validate required fields
    required_fields = ["start_frame", "trial_length_frame", "repeat"]
    for field in required_fields:
        if field not in section_kwargs:
            raise ValueError(
                f"Config '{movie_name}' section_kwargs missing required field: {field}"
            )
    
    # Validate field types and values
    if not isinstance(section_kwargs["start_frame"], int):
        raise ValueError(
            f"Config '{movie_name}' section_kwargs.start_frame must be int, "
            f"got {type(section_kwargs['start_frame']).__name__}"
        )
    if section_kwargs["start_frame"] < 0:
        raise ValueError(
            f"Config '{movie_name}' section_kwargs.start_frame must be >= 0, "
            f"got {section_kwargs['start_frame']}"
        )
    
    if not isinstance(section_kwargs["trial_length_frame"], int):
        raise ValueError(
            f"Config '{movie_name}' section_kwargs.trial_length_frame must be int, "
            f"got {type(section_kwargs['trial_length_frame']).__name__}"
        )
    if section_kwargs["trial_length_frame"] <= 0:
        raise ValueError(
            f"Config '{movie_name}' section_kwargs.trial_length_frame must be > 0, "
            f"got {section_kwargs['trial_length_frame']}"
        )
    
    if not isinstance(section_kwargs["repeat"], int):
        raise ValueError(
            f"Config '{movie_name}' section_kwargs.repeat must be int, "
            f"got {type(section_kwargs['repeat']).__name__}"
        )
    if section_kwargs["repeat"] < 1:
        raise ValueError(
            f"Config '{movie_name}' section_kwargs.repeat must be >= 1, "
            f"got {section_kwargs['repeat']}"
        )
    
    return StimuliConfigDict(
        start_frame=section_kwargs["start_frame"],
        trial_length_frame=section_kwargs["trial_length_frame"],
        repeat=section_kwargs["repeat"],
    )


def _validate_all_configs(
    movie_names: List[str],
    config_dir: Path,
) -> Dict[str, StimuliConfigDict]:
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
    missing_configs: List[str] = []
    invalid_configs: Dict[str, str] = {}
    valid_configs: Dict[str, StimuliConfigDict] = {}
    
    for movie_name in movie_names:
        try:
            config = _load_stimuli_config(movie_name, config_dir)
            valid_configs[movie_name] = config
        except FileNotFoundError:
            missing_configs.append(movie_name)
        except ValueError as e:
            invalid_configs[movie_name] = str(e)
    
    # Build comprehensive error message if any issues found
    errors: List[str] = []
    
    if missing_configs:
        errors.append(
            f"Missing stimulus config files: {missing_configs}. "
            f"Expected at {config_dir}"
        )
    
    if invalid_configs:
        for movie_name, error_msg in invalid_configs.items():
            errors.append(f"Invalid config for '{movie_name}': {error_msg}")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return valid_configs


def _calculate_trial_boundaries(
    section_kwargs: StimuliConfigDict,
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
    start_frame = section_kwargs["start_frame"]
    trial_length_frame = section_kwargs["trial_length_frame"]
    repeat = section_kwargs["repeat"]
    
    boundaries: List[Tuple[int, int]] = []
    max_frame = len(frame_timestamps) - 1
    
    for n in range(repeat):
        # Calculate trial start frame
        trial_start_frame = (
            section_frame_start + PRE_MARGIN_FRAME_NUM + start_frame + 
            (n * trial_length_frame)
        )
        trial_end_frame = trial_start_frame + trial_length_frame
        
        # Clip to valid frame range
        trial_start_frame = min(trial_start_frame, max_frame)
        trial_end_frame = min(trial_end_frame, max_frame)
        
        # Convert to sample indices
        trial_start_sample = int(frame_timestamps[trial_start_frame])
        trial_end_sample = int(frame_timestamps[trial_end_frame])
        
        boundaries.append((trial_start_sample, trial_end_sample))
    
    return boundaries


# =============================================================================
# Helper Functions
# =============================================================================

def _section_unit_spikes(
    spike_times: np.ndarray,
    trial_boundaries: List[Tuple[int, int]],
    pre_samples: int = 0,
    post_samples: int = 0,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Extract spikes within trial boundaries for a single unit.
    
    Args:
        spike_times: Array of spike times in sample indices (uint64).
        trial_boundaries: List of (start_sample, end_sample) tuples for each trial,
            pre-computed from JSON config via _calculate_trial_boundaries().
        pre_samples: Padding in samples to extend before trial start.
        post_samples: Padding in samples to extend after trial end.
    
    Returns:
        Tuple of:
            - full_spike_times: All spikes from all trials combined (sorted, unique).
            - trials_spike_times: Dict mapping trial_idx -> spike array.
    
    Note:
        Padded boundaries are clamped: start >= 0.
    """
    # Handle empty spike_times
    if len(spike_times) == 0:
        empty_array = np.array([], dtype=np.int64)
        trials_dict = {i: empty_array.copy() for i in range(len(trial_boundaries))}
        return empty_array, trials_dict
    
    trials_spikes: Dict[int, np.ndarray] = {}
    trial_arrays: List[np.ndarray] = []
    
    for trial_idx, (trial_start, trial_end) in enumerate(trial_boundaries):
        # Apply padding and clamp to valid range
        padded_start = max(0, trial_start - pre_samples)
        padded_end = trial_end + post_samples
        
        # Extract spikes within [padded_start, padded_end)
        mask = (spike_times >= padded_start) & (spike_times < padded_end)
        trial_spikes = spike_times[mask].astype(np.int64)
        
        # Store per-trial
        trials_spikes[trial_idx] = trial_spikes
        trial_arrays.append(trial_spikes)
    
    # Combine all trials efficiently using numpy concatenation
    if trial_arrays:
        combined = np.concatenate(trial_arrays)
        full_spike_times = np.unique(combined)  # unique also sorts
    else:
        full_spike_times = np.array([], dtype=np.int64)
    
    return full_spike_times, trials_spikes


def _write_sectioned_spikes(
    unit_group: h5py.Group,
    movie_name: str,
    full_spike_times: np.ndarray,
    trials_spike_times: Dict[int, np.ndarray],
    trial_boundaries: List[Tuple[int, int]],
    pad_margin: Tuple[float, float],
    pre_samples: int,
    post_samples: int,
    trial_repeats: int,
    force: bool = False,
) -> None:
    """
    Write sectioned spike times to unit group in HDF5.
    
    Creates structure:
        spike_times_sectioned/{movie_name}/
            full_spike_times          # All trials combined
            trials_spike_times/
                0                     # Trial 0 spikes
                1                     # Trial 1 spikes
                ...
            trials_start_end          # Trial boundaries as sample indices
    
    Args:
        unit_group: HDF5 group for the unit (units/{unit_id}).
        movie_name: Name of the movie/stimulus.
        full_spike_times: All spikes from all trials (sorted).
        trials_spike_times: Dict mapping trial_idx -> spike array.
        trial_boundaries: List of (start_sample, end_sample) tuples for each trial.
        pad_margin: Tuple of (pre_margin_s, post_margin_s) in seconds.
        pre_samples: Pre-margin in samples.
        post_samples: Post-margin in samples.
        trial_repeats: Number of trials parameter.
        force: Whether to overwrite existing data.
    
    Raises:
        FileExistsError: If data exists and force=False.
    """
    # Helper for Windows file locking retries (only sleeps on actual retry)
    def _retry_operation(operation, *args, **kwargs):
        """Retry an operation with delays for Windows file locking issues."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return operation(*args, **kwargs)
            except (PermissionError, OSError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise last_error
        raise last_error
    
    # Ensure spike_times_sectioned group exists
    if "spike_times_sectioned" not in unit_group:
        sectioned_group = _retry_operation(unit_group.create_group, "spike_times_sectioned")
    else:
        sectioned_group = unit_group["spike_times_sectioned"]
    
    # Check if movie already exists
    if movie_name in sectioned_group:
        if not force:
            raise FileExistsError(
                f"spike_times_sectioned/{movie_name} already exists for unit. "
                "Use force=True to overwrite."
            )
        # Delete existing movie group with retry
        try:
            del sectioned_group[movie_name]
        except Exception:
            pass  # May already be deleted
    
    # Create movie group
    movie_group = _retry_operation(sectioned_group.create_group, movie_name)
    
    # Write full_spike_times array
    movie_group.create_dataset("full_spike_times", data=full_spike_times, dtype=np.int64)
    
    # Create trials_spike_times group
    trials_group = movie_group.create_group("trials_spike_times")
    
    # Write per-trial arrays
    for trial_idx, trial_spikes in trials_spike_times.items():
        trials_group.create_dataset(str(trial_idx), data=trial_spikes, dtype=np.int64)
    
    # Write trial boundaries as (n_trials, 2) array with [start_sample, end_sample] per row
    boundaries_array = np.array(trial_boundaries, dtype=np.int64)
    movie_group.create_dataset("trials_start_end", data=boundaries_array, dtype=np.int64)
    
    # Write metadata attributes
    movie_group.attrs["n_trials"] = len(trials_spike_times)
    movie_group.attrs["trial_repeats"] = trial_repeats
    movie_group.attrs["pad_margin"] = list(pad_margin)
    movie_group.attrs["pre_samples"] = pre_samples
    movie_group.attrs["post_samples"] = post_samples
    movie_group.attrs["created_at"] = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Main Function
# =============================================================================

def section_spike_times(
    hdf5_path: Union[str, Path],
    *,
    movie_names: Optional[List[str]] = None,
    trial_repeats: int = 3,
    pad_margin: Tuple[float, float] = (2.0, 0.0),
    force: bool = False,
    config_dir: Optional[Union[str, Path]] = None,
) -> SectionResult:
    """
    Section spike times by stimulation periods using JSON configuration files.
    
    For each unit in the HDF5 file, extracts spikes falling within
    trial periods (with optional padding) and stores them in TWO formats:
    - `full_spike_times`: All spikes from all trials combined
    - `trials_spike_times/{idx}`: Spikes per trial based on JSON config boundaries
    
    Trial boundaries are determined by JSON config files in config/stimuli/,
    where each movie has a corresponding {movie_name}.json file containing
    section_kwargs with start_frame, trial_length_frame, and repeat values.
    
    Args:
        hdf5_path: Path to HDF5 file containing recording data.
            Must have `units/{unit_id}/spike_times` arrays (in sample units)
            and `stimulus/section_time/{movie_name}` trial boundaries.
        movie_names: Optional list of movies to process. If None, processes
            all movies found in section_time.
        trial_repeats: DEPRECATED - ignored, uses JSON config 'repeat' value.
        pad_margin: Tuple of (pre_margin_s, post_margin_s) in seconds.
            Default=(2.0, 0.0) - 2 seconds before trial start, 0 after trial end.
            Converted to samples using acquisition_rate.
        force: If True, overwrite existing sectioned data. If False (default),
            raise FileExistsError if any unit already has sectioned data.
        config_dir: Path to directory containing stimulus JSON config files.
            Defaults to 'config/stimuli/' relative to project root.
    
    Returns:
        SectionResult with success, units_processed, movies_processed, etc.
    
    Raises:
        FileNotFoundError: If hdf5_path does not exist.
        FileExistsError: If sectioned data exists and force=False.
        ValueError: If any movie lacks corresponding JSON config file.
        ValueError: If JSON config has invalid or missing section_kwargs.
    
    Example:
        >>> result = section_spike_times(
        ...     hdf5_path="artifacts/JIANG009_2025-04-10.h5",
        ...     pad_margin=(2.0, 0.0),
        ... )
        >>> print(f"Processed {result.units_processed} units")
    """
    import warnings as warnings_module
    
    hdf5_path = Path(hdf5_path)
    warning_messages: List[str] = []
    
    # Resolve config directory
    if config_dir is None:
        resolved_config_dir = DEFAULT_CONFIG_DIR
    else:
        resolved_config_dir = Path(config_dir)
    
    logger.info(f"Using stimulus config directory: {resolved_config_dir}")
    
    # Deprecation warning for trial_repeats parameter
    if trial_repeats != 3:  # Only warn if explicitly changed from default
        warnings_module.warn(
            "trial_repeats parameter is deprecated and will be ignored. "
            "Trial count is now determined by JSON config 'repeat' value.",
            DeprecationWarning,
            stacklevel=2,
        )
    
    # Validate path exists
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    # Open HDF5
    root = open_recording_hdf5(hdf5_path, mode="r+")
    
    # Get acquisition_rate from metadata
    if "metadata" not in root:
        warning_messages.append("No metadata group found, using default acquisition_rate")
        acquisition_rate = 20000.0
    else:
        metadata = root["metadata"]
        if "acquisition_rate" in metadata:
            acquisition_rate = float(metadata["acquisition_rate"][...])
        else:
            acquisition_rate = 20000.0
            warning_messages.append("acquisition_rate not in metadata, using default 20000 Hz")
    
    # Convert pad_margin to samples
    pre_samples = int(pad_margin[0] * acquisition_rate)
    post_samples = int(pad_margin[1] * acquisition_rate)
    
    logger.info(
        f"Sectioning spike times with trial_repeats={trial_repeats}, "
        f"pad_margin={pad_margin} ({pre_samples} pre, {post_samples} post samples)"
    )
    frame_timestamps = np.array(root["metadata"]["frame_timestamps"])

    logger.info(f"Frame timestamps shape: {frame_timestamps.shape}")
    logger.info(f"Frame timestamps dtype: {frame_timestamps.dtype}")
    logger.info(f"Frame timestamps min: {frame_timestamps.min()}")
    logger.info(f"Frame timestamps max: {frame_timestamps.max()}")
    
    # Check for section_time
    if "stimulus" not in root or "section_time" not in root["stimulus"]:
        warning_messages.append("No section_time data found in HDF5")
        logger.warning("No section_time data found - cannot section spike times")
        root.close()
        return SectionResult(
            success=True,
            units_processed=0,
            movies_processed=[],
            trial_repeats=trial_repeats,
            pad_margin=pad_margin,
            pre_samples=pre_samples,
            post_samples=post_samples,
            warnings=warning_messages,
        )
    
    section_time_group = root["stimulus"]["section_time"]
    
    # Determine movies to process
    available_movies = list(section_time_group.keys())
    if movie_names is None:
        movies_to_process = available_movies
    else:
        movies_to_process = [m for m in movie_names if m in available_movies]
        missing = set(movie_names) - set(available_movies)
        if missing:
            warning_messages.append(f"Movies not found in section_time: {missing}")
            logger.warning(f"Movies not found: {missing}")
    
    if not movies_to_process:
        warning_messages.append("No movies to process")
        root.close()
        return SectionResult(
            success=True,
            units_processed=0,
            movies_processed=[],
            trial_repeats=trial_repeats,
            pad_margin=pad_margin,
            pre_samples=pre_samples,
            post_samples=post_samples,
            warnings=warning_messages,
        )
    
    logger.info(f"Processing movies: {movies_to_process}")
    
    # Validate all JSON configs upfront (fail-fast)
    try:
        stimuli_configs = _validate_all_configs(movies_to_process, resolved_config_dir)
        logger.info(f"Loaded {len(stimuli_configs)} stimulus configs from {resolved_config_dir}")
        for movie_name, cfg in stimuli_configs.items():
            logger.info(
                f"  {movie_name}: start_frame={cfg['start_frame']}, "
                f"trial_length_frame={cfg['trial_length_frame']}, repeat={cfg['repeat']}"
            )
    except ValueError as e:
        root.close()
        raise ValueError(f"Stimulus config validation failed:\n{e}")
    
    # Pre-load all section_time arrays (cache outside unit loop for performance)
    section_time_cache: Dict[str, np.ndarray] = {}
    for movie_name in movies_to_process:
        section_time_cache[movie_name] = section_time_group[movie_name][:]
    
    # Pre-compute trial boundaries for each movie using JSON config
    trial_boundaries_cache: Dict[str, List[Tuple[int, int]]] = {}
    for movie_name in movies_to_process:
        section_time = section_time_cache[movie_name]
        if len(section_time) == 0:
            continue
        
        # Get frame start from section_time (convert first sample to frame)
        section_frame_start = int(_convert_sample_index_to_frame(
            np.array([section_time[0, 0]]), frame_timestamps
        )[0])
        
        # Calculate trial boundaries using JSON config
        boundaries = _calculate_trial_boundaries(
            section_kwargs=stimuli_configs[movie_name],
            section_frame_start=section_frame_start,
            frame_timestamps=frame_timestamps,
        )
        trial_boundaries_cache[movie_name] = boundaries
        logger.debug(f"Movie '{movie_name}' trial boundaries: {len(boundaries)} trials")
    
    # Check for units
    if "units" not in root:
        warning_messages.append("No units group found in HDF5")
        root.close()
        return SectionResult(
            success=True,
            units_processed=0,
            movies_processed=[],
            trial_repeats=trial_repeats,
            pad_margin=pad_margin,
            pre_samples=pre_samples,
            post_samples=post_samples,
            warnings=warning_messages,
        )
    
    units_group = root["units"]
    unit_ids = list(units_group.keys())
    
    if not unit_ids:
        warning_messages.append("No units found in HDF5")
        root.close()
        return SectionResult(
            success=True,
            units_processed=0,
            movies_processed=[],
            trial_repeats=trial_repeats,
            pad_margin=pad_margin,
            pre_samples=pre_samples,
            post_samples=post_samples,
            warnings=warning_messages,
        )
    
    logger.info(f"Processing {len(unit_ids)} units with parallel I/O")
    
    # Debug: Log first unit's spike_times range and section_time range for first movie
    first_unit = units_group[unit_ids[0]]
    if "spike_times" in first_unit:
        st = first_unit["spike_times"][:]
        if len(st) > 0:
            logger.info(f"DEBUG: First unit spike_times range: [{st.min():,} - {st.max():,}] (count={len(st)})")
    first_movie = movies_to_process[0]
    first_section = section_time_cache[first_movie]
    if len(first_section) > 0:
        logger.info(f"DEBUG: First movie '{first_movie}' section_time: trial 0 = [{first_section[0, 0]:,} - {first_section[0, 1]:,}]")
    
    # If force=True, pre-delete all existing spike_times_sectioned groups
    if force:
        logger.info("Clearing existing spike_times_sectioned data...")
        for unit_id in tqdm(unit_ids, desc="Clearing existing data", leave=False):
            try:
                if "spike_times_sectioned" in units_group[unit_id]:
                    del units_group[unit_id]["spike_times_sectioned"]
            except Exception as e:
                warning_messages.append(f"Could not delete {unit_id}/spike_times_sectioned: {e}")
    
    # PHASE 1: Compute all sectioned spikes in PARALLEL (CPU-bound, no I/O conflicts)
    units_processed = 0
    movies_actually_processed: set = set()
    
    # Data structure to hold computed results: {unit_id: {movie_name: (full_spikes, trials_spikes)}}
    ComputedData = Dict[str, Tuple[np.ndarray, Dict[int, np.ndarray]]]
    computed_results: Dict[str, Tuple[ComputedData, List[str]]] = {}
    
    def compute_unit_sections(unit_id: str) -> Tuple[str, ComputedData, List[str]]:
        """Compute sectioned spikes for one unit (no I/O writes)."""
        unit_group = units_group[unit_id]
        unit_warnings: List[str] = []
        unit_data: ComputedData = {}
        
        if "spike_times" not in unit_group:
            return unit_id, {}, [f"Unit {unit_id} has no spike_times"]
        
        spike_times = unit_group["spike_times"][:]
        
        for movie_name in movies_to_process:
            # Use pre-computed trial boundaries from JSON config
            if movie_name not in trial_boundaries_cache:
                unit_warnings.append(f"Movie {movie_name} has no trial boundaries computed")
                continue
            
            trial_boundaries = trial_boundaries_cache[movie_name]
            
            full_spike_times, trials_spike_times = _section_unit_spikes(
                spike_times=spike_times,
                trial_boundaries=trial_boundaries,
                pre_samples=pre_samples,
                post_samples=post_samples,
            )
            
            unit_data[movie_name] = (full_spike_times, trials_spike_times)
        
        return unit_id, unit_data, unit_warnings
    
    # Use ThreadPoolExecutor for parallel COMPUTATION (80% of CPU cores)
    cpu_count = os.cpu_count() or 4
    max_workers = max(1, min(int(cpu_count * 0.8), len(unit_ids)))
    logger.info(f"Phase 1: Computing with {max_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_unit_sections, uid): uid for uid in unit_ids}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Computing sections", leave=False):
            unit_id = futures[future]
            try:
                uid, unit_data, unit_warnings = future.result()
                warning_messages.extend(unit_warnings)
                if unit_data:
                    computed_results[uid] = (unit_data, unit_warnings)
            except Exception as e:
                warning_messages.append(f"Unit {unit_id} compute failed: {e}")
    
    # PHASE 2: Write all results SEQUENTIALLY (avoids Zarr metadata conflicts)
    logger.info(f"Phase 2: Writing {len(computed_results)} units sequentially")
    
    for unit_id in tqdm(computed_results.keys(), desc="Writing sections", leave=False):
        unit_data, _ = computed_results[unit_id]
        unit_group = units_group[unit_id]
        
        try:
            for movie_name, (full_spike_times, trials_spike_times) in unit_data.items():
                _write_sectioned_spikes(
                    unit_group=unit_group,
                    movie_name=movie_name,
                    full_spike_times=full_spike_times,
                    trials_spike_times=trials_spike_times,
                    trial_boundaries=trial_boundaries_cache[movie_name],
                    pad_margin=pad_margin,
                    pre_samples=pre_samples,
                    post_samples=post_samples,
                    trial_repeats=trial_repeats,
                    force=force,
                )
                movies_actually_processed.add(movie_name)
            units_processed += 1
        except FileExistsError:
            raise
        except Exception as e:
            warning_messages.append(f"Unit {unit_id} write failed: {e}")
    
    logger.info(
        f"Sectioning complete: {units_processed} units, "
        f"{len(movies_actually_processed)} movies"
    )
    
    # Close HDF5 file
    root.close()
    
    return SectionResult(
        success=True,
        units_processed=units_processed,
        movies_processed=list(movies_actually_processed),
        trial_repeats=trial_repeats,
        pad_margin=pad_margin,
        pre_samples=pre_samples,
        post_samples=post_samples,
        warnings=warning_messages,
    )

