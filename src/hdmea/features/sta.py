"""
Spike Triggered Average (STA) computation for HD-MEA recordings.

This module provides functions to compute STA from noise movie stimuli,
which is used for receptive field mapping and cell characterization.

Key features:
    - Automatic noise movie detection
    - Spike time to frame number conversion
    - Vectorized window extraction for performance
    - Multiprocessing support with shared memory
    - Progress bar for long computations
    
Example:
    >>> from hdmea.features import compute_sta
    >>> result = compute_sta("artifacts/recording.h5", cover_range=(-60, 0))
    >>> print(f"Processed {result.units_processed} units")
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from hdmea.pipeline.session import PipelineSession

import h5py
import numpy as np
from tqdm import tqdm

from hdmea.io.section_time import convert_sample_index_to_frame, PRE_MARGIN_FRAME_NUM

logger = logging.getLogger(__name__)

# Default stimuli directory
DEFAULT_STIMULI_DIR = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations"
)


@dataclass
class STAResult:
    """Result of STA computation.
    
    Attributes:
        hdf5_path: Path to the HDF5 file processed.
        movie_name: Name of the noise movie used.
        units_processed: Number of units successfully processed.
        units_failed: Number of units that failed after retry.
        cover_range: Frame range used for averaging.
        elapsed_seconds: Total computation time.
        warnings: List of warning messages generated.
        failed_units: List of unit IDs that failed.
    """
    hdf5_path: Path
    movie_name: str
    units_processed: int
    units_failed: int
    cover_range: Tuple[int, int]
    elapsed_seconds: float
    warnings: List[str] = field(default_factory=list)
    failed_units: List[str] = field(default_factory=list)


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
    npy_path = stimuli_dir / f"{movie_name}.npy"
    
    if not npy_path.exists():
        raise FileNotFoundError(
            f"Stimulus file not found: {npy_path}. "
            f"Expected movie '{movie_name}' in directory '{stimuli_dir}'"
        )
    
    movie = np.load(npy_path)
    
    if movie.dtype != np.uint8:
        logger.warning(
            f"Stimulus movie '{movie_name}' has dtype {movie.dtype}, expected uint8. "
            f"This may affect STA computation precision."
        )
    
    logger.info(f"Loaded stimulus movie '{movie_name}': shape={movie.shape}, dtype={movie.dtype}")
    return movie


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
    
    Creates the group structure: units/{unit_id}/features/{movie_name}/sta
    
    Args:
        hdf5_file: Open HDF5 file in write mode.
        unit_id: Unit identifier.
        movie_name: Movie name for grouping.
        sta: Computed STA array.
        n_spikes_used: Number of spikes included in average.
        n_spikes_excluded: Number of spikes excluded due to edge effects.
        cover_range: Frame range used for computation.
        force: If True, overwrite existing. If False, skip if exists.
    
    Raises:
        RuntimeError: If STA exists and force=False.
    """
    unit_group = hdf5_file[f"units/{unit_id}"]
    
    # Create features group if not exists
    if "features" not in unit_group:
        unit_group.create_group("features")
    
    features_group = unit_group["features"]
    
    # Create movie group if not exists
    if movie_name not in features_group:
        features_group.create_group(movie_name)
    
    movie_group = features_group[movie_name]
    
    # Check if STA already exists
    if "sta" in movie_group:
        if force:
            del movie_group["sta"]
            logger.debug(f"Overwriting existing STA for {unit_id}/{movie_name}")
        else:
            logger.debug(f"STA already exists for {unit_id}/{movie_name}, skipping")
            return
    
    # Write STA dataset with metadata as attributes
    sta_dataset = movie_group.create_dataset("sta", data=sta, dtype=np.float32)
    sta_dataset.attrs["n_spikes"] = n_spikes_used
    sta_dataset.attrs["n_spikes_excluded"] = n_spikes_excluded
    sta_dataset.attrs["cover_range"] = cover_range
    sta_dataset.attrs["dtype_warning"] = sta.dtype != np.uint8
    
    logger.debug(
        f"Wrote STA for {unit_id}/{movie_name}: "
        f"shape={sta.shape}, n_spikes={n_spikes_used}"
    )


def _find_noise_movie(
    hdf5_file: h5py.File,
    unit_id: str,
) -> str:
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
    sectioned_path = f"units/{unit_id}/spike_times_sectioned"
    
    if sectioned_path not in hdf5_file:
        raise ValueError(
            f"No spike_times_sectioned found for unit {unit_id}. "
            f"Run spike sectioning first."
        )
    
    movies = list(hdf5_file[sectioned_path].keys())
    noise_movies = [m for m in movies if "noise" in m.lower()]
    
    if len(noise_movies) == 0:
        raise ValueError(
            f"No noise movie found in spike_times_sectioned. "
            f"Available movies: {movies}"
        )
    
    if len(noise_movies) > 1:
        raise ValueError(
            f"Multiple noise movies found: {noise_movies}. "
            f"Cannot determine which to use."
        )
    
    return noise_movies[0]


def _find_noise_movie_from_session(
    section_time_data: Dict[str, np.ndarray],
) -> Optional[str]:
    """
    Find the noise movie name from session section_time data.
    
    Args:
        section_time_data: Dictionary of movie_name -> section_time arrays.
    
    Returns:
        Noise movie name, or None if not found.
    """
    movies = list(section_time_data.keys())
    noise_movies = [m for m in movies if "noise" in m.lower()]
    
    if len(noise_movies) == 0:
        return None
    
    if len(noise_movies) > 1:
        logger.warning(f"Multiple noise movies found: {noise_movies}. Using first one.")
    
    return noise_movies[0]


def _compute_sta_for_unit(
    spike_frames: np.ndarray,
    movie_array: np.ndarray,
    cover_range: Tuple[int, int],
) -> Tuple[np.ndarray, int, int]:
    """
    Compute STA for a single unit using vectorized operations.
    
    Args:
        spike_frames: Spike times converted to movie frame indices.
        movie_array: Stimulus movie array (frames, height, width).
        cover_range: Frame window (start_offset, end_offset).
    
    Returns:
        Tuple of (sta_array, n_spikes_used, n_spikes_excluded).
        sta_array has shape (window_length, height, width).
    """
    movie_length = movie_array.shape[0]
    window_length = cover_range[1] - cover_range[0]
    
    # Pre-compute valid spike mask (vectorized edge handling)
    valid_mask = (
        (spike_frames + cover_range[0] >= 0) &
        (spike_frames + cover_range[1] <= movie_length)
    )
    
    valid_spikes = spike_frames[valid_mask]
    n_spikes_used = len(valid_spikes)
    n_spikes_excluded = len(spike_frames) - n_spikes_used
    
    if n_spikes_used == 0:
        # No valid spikes - return NaN STA
        logger.warning("No valid spikes for STA computation (all excluded by edge effects)")
        sta = np.full(
            (window_length, movie_array.shape[1], movie_array.shape[2]),
            np.nan,
            dtype=np.float32
        )
        return sta, 0, n_spikes_excluded
    
    # Build window indices (n_spikes, window_length)
    window_offsets = np.arange(cover_range[0], cover_range[1])
    all_indices = valid_spikes[:, np.newaxis] + window_offsets  # (n_spikes, window_length)
    
    # Extract all windows at once using fancy indexing
    # movie_array[all_indices] gives shape (n_spikes, window_length, height, width)
    windows = movie_array[all_indices]
    
    # Compute STA as mean across spikes (axis=0)
    sta = windows.mean(axis=0).astype(np.float32)
    
    return sta, n_spikes_used, n_spikes_excluded


def _get_worker_count() -> int:
    """
    Calculate number of worker processes (80% of CPU count, minimum 1).
    
    Returns:
        Number of workers to use for multiprocessing.
    """
    n_cpus = cpu_count()
    n_workers = max(1, int(n_cpus * 0.8))
    return n_workers


# Global variables for multiprocessing workers
_shared_movie: Optional[np.ndarray] = None
_shared_cover_range: Optional[Tuple[int, int]] = None


def _init_worker(movie_array: np.ndarray, cover_range: Tuple[int, int]) -> None:
    """Initialize worker process with shared data."""
    global _shared_movie, _shared_cover_range
    _shared_movie = movie_array
    _shared_cover_range = cover_range


def _worker_compute_sta(args: Tuple[str, np.ndarray]) -> Tuple[str, np.ndarray, int, int, Optional[str]]:
    """
    Worker function for parallel STA computation.
    
    Args:
        args: Tuple of (unit_id, spike_frames).
    
    Returns:
        Tuple of (unit_id, sta_array, n_spikes_used, n_spikes_excluded, error_msg).
        error_msg is None on success, otherwise contains error description.
    """
    global _shared_movie, _shared_cover_range
    unit_id, spike_frames = args
    
    try:
        sta, n_used, n_excluded = _compute_sta_for_unit(
            spike_frames, _shared_movie, _shared_cover_range
        )
        return (unit_id, sta, n_used, n_excluded, None)
    except Exception as e:
        return (unit_id, None, 0, 0, str(e))


def _compute_with_retry(
    args: Tuple[str, np.ndarray],
    max_retries: int = 1,
) -> Tuple[str, np.ndarray, int, int, Optional[str]]:
    """
    Compute STA with retry logic.
    
    Args:
        args: Tuple of (unit_id, spike_frames).
        max_retries: Maximum number of retry attempts.
    
    Returns:
        Same as _worker_compute_sta.
    """
    unit_id = args[0]
    
    for attempt in range(max_retries + 1):
        result = _worker_compute_sta(args)
        if result[4] is None:  # No error
            return result
        
        if attempt < max_retries:
            logger.debug(f"Unit {unit_id} failed attempt {attempt + 1}, retrying...")
    
    return result


def compute_sta(
    hdf5_path: Optional[Union[str, Path]] = None,
    *,
    cover_range: Tuple[int, int] = (-60, 0),
    use_multiprocessing: bool = True,
    stimuli_dir: Optional[Path] = None,
    force: bool = False,
    session: Optional["PipelineSession"] = None,
) -> Union[STAResult, "PipelineSession"]:
    """
    Compute Spike Triggered Average for all units using noise movie stimulus.
    
    Uses frame_timestamps from HDF5 file for accurate spike-to-frame conversion.
    
    Supports deferred saving via the optional `session` parameter.
    When session is provided, STA results are stored in session.units
    instead of HDF5.
    
    Args:
        hdf5_path: Path to HDF5 recording file. Required if session=None.
        cover_range: Frame window relative to spike (start, end). 
                     Negative values indicate frames before spike.
                     Default: (-60, 0) = 60 frames before spike.
        use_multiprocessing: If True, process units in parallel using 80% of CPU cores.
                             If False, process sequentially.
        stimuli_dir: Directory containing stimulus .npy files.
                     Default: M:\\Python_Project\\Data_Processing_2025\\Design_Stimulation_Pattern\\Data\\Stimulations\\
        force: If True, overwrite existing STA results. Default: False.
        session: Optional PipelineSession for deferred saving.
    
    Returns:
        STAResult if session is None (immediate save mode).
        PipelineSession if session is provided (deferred save mode).
    
    Raises:
        ValueError: If no noise movie found, or multiple noise movies found.
        ValueError: If cover_range[0] >= cover_range[1].
        FileNotFoundError: If stimulus .npy file not found.
        RuntimeError: If HDF5 file is not readable/writable.
    
    Example (immediate save):
        >>> from hdmea.features import compute_sta
        >>> result = compute_sta("artifacts/recording.h5", cover_range=(-60, 0))
        >>> print(f"Processed {result.units_processed} units")
    
    Example (deferred save):
        >>> session = load_recording(..., session=session)
        >>> session = add_section_time(..., session=session)
        >>> session = section_spike_times(..., session=session)
        >>> session = compute_sta(cover_range=(-60, 0), session=session)
        >>> session.save()
    """
    import time
    start_time = time.time()
    
    stimuli_dir = stimuli_dir or DEFAULT_STIMULI_DIR
    warnings_list: List[str] = []
    failed_units: List[str] = []
    
    # Validate cover_range
    if cover_range[0] >= cover_range[1]:
        raise ValueError(
            f"Invalid cover_range: {cover_range}. "
            f"Start ({cover_range[0]}) must be less than end ({cover_range[1]})."
        )
    
    # =========================================================================
    # Session-based mode: read from session, write to session
    # =========================================================================
    if session is not None:
        if "section_spike_times" not in session.completed_steps:
            logger.error("Session does not contain sectioned spike data")
            session.warnings.append("Session does not contain sectioned spike data. Run section_spike_times first.")
            return session
        
        unit_ids = list(session.units.keys())
        if not unit_ids:
            session.warnings.append("No units found in session")
            session.mark_step_complete("compute_sta")
            return session
        
        # Find noise movie from session section_time
        section_time_data = session.stimulus.get("section_time", {})
        movie_name = _find_noise_movie_from_session(section_time_data)
        
        if movie_name is None:
            session.warnings.append("No noise movie found in session section_time")
            return session
        
        logger.info(f"Found noise movie (deferred): {movie_name}")
        
        # Load stimulus movie
        movie_array = _load_stimulus_movie(movie_name, stimuli_dir)
        
        # Get frame_timestamps
        frame_timestamps = None
        if "frame_times" in session.stimulus and "frame_timestamps" in session.stimulus["frame_times"]:
            frame_timestamps = np.array(session.stimulus["frame_times"]["frame_timestamps"])
        elif "frame_timestamps" in session.metadata:
            frame_timestamps = np.array(session.metadata["frame_timestamps"])
        
        if frame_timestamps is None:
            session.warnings.append("No frame_timestamps found in session")
            return session
        
        # Get movie start frame
        section_time = section_time_data.get(movie_name)
        if section_time is None or len(section_time) == 0:
            session.warnings.append(f"No section_time for movie '{movie_name}'")
            return session
        
        movie_start_sample = section_time[0, 0]
        movie_start_frame = int(convert_sample_index_to_frame(
            np.array([movie_start_sample]), frame_timestamps
        )[0]) + PRE_MARGIN_FRAME_NUM
        
        # Pre-load spike data from session
        unit_spike_data: Dict[str, np.ndarray] = {}
        for unit_id in unit_ids:
            unit_data = session.units[unit_id]
            
            # Check for sectioned spikes
            if "spike_times_sectioned" not in unit_data:
                warnings_list.append(f"No spike_times_sectioned for {unit_id}")
                continue
            
            if movie_name not in unit_data["spike_times_sectioned"]:
                warnings_list.append(f"No sectioned data for {unit_id}/{movie_name}")
                continue
            
            sectioned = unit_data["spike_times_sectioned"][movie_name]
            if "trials_spike_times" not in sectioned or 0 not in sectioned["trials_spike_times"]:
                warnings_list.append(f"No trials_spike_times for {unit_id}/{movie_name}")
                continue
            
            spike_samples = np.array(sectioned["trials_spike_times"][0])
            spike_frames_absolute = convert_sample_index_to_frame(spike_samples, frame_timestamps)
            spike_frames = spike_frames_absolute - movie_start_frame
            unit_spike_data[unit_id] = spike_frames
        
        # Compute STA (simplified sequential for session mode)
        logger.info(f"Processing {len(unit_spike_data)} units (deferred) with cover_range={cover_range}")
        
        _init_worker(movie_array, cover_range)
        
        units_processed = 0
        for unit_id, spikes in tqdm(unit_spike_data.items(), desc="Computing STA (deferred)"):
            uid, sta, n_used, n_excluded, error = _compute_with_retry((unit_id, spikes))
            
            if error is not None:
                logger.error(f"Failed to compute STA for {unit_id}: {error}")
                failed_units.append(unit_id)
                continue
            
            # Store in session
            session.add_feature(
                unit_id,
                f"sta_{movie_name}",
                sta,
                {
                    "n_spikes_used": n_used,
                    "n_spikes_excluded": n_excluded,
                    "cover_range": list(cover_range),
                    "movie_name": movie_name,
                },
            )
            units_processed += 1
        
        session.warnings.extend(warnings_list)
        session.mark_step_complete("compute_sta")
        
        elapsed = time.time() - start_time
        logger.info(f"STA computation complete (deferred): {units_processed} units, {elapsed:.1f}s")
        return session
    
    # =========================================================================
    # Immediate save mode: read from HDF5, write to HDF5
    # =========================================================================
    if hdf5_path is None:
        raise ValueError("hdf5_path is required when session is not provided")
    
    hdf5_path = Path(hdf5_path)
    
    # Phase 1: Load all data (HDF5 is not multiprocess-safe for reading)
    with h5py.File(hdf5_path, "r") as hdf5_file:
        # Get list of units
        if "units" not in hdf5_file:
            raise ValueError(f"No 'units' group found in {hdf5_path}")
        
        unit_ids = list(hdf5_file["units"].keys())
        if not unit_ids:
            raise ValueError(f"No units found in {hdf5_path}")
        
        # Find noise movie using first unit
        movie_name = _find_noise_movie(hdf5_file, unit_ids[0])
        logger.info(f"Found noise movie: {movie_name}")
        
        # Load stimulus movie
        movie_array = _load_stimulus_movie(movie_name, stimuli_dir)
        
        # Load frame_timestamps for accurate spike-to-frame conversion
        # Try multiple possible paths for frame timestamps
        frame_timestamps_path = None
        for path in ["stimulus/frame_time/default", "stimulus/frame_timestamps"]:
            if path in hdf5_file:
                frame_timestamps_path = path
                break
        
        if frame_timestamps_path is not None:
            frame_timestamps = hdf5_file[frame_timestamps_path][:]
            logger.info(f"Loaded frame_timestamps from '{frame_timestamps_path}': {len(frame_timestamps)} frames")
        else:
            raise ValueError(
                f"No frame timestamps found in {hdf5_path}. "
                "Expected 'stimulus/frame_time/default' or 'stimulus/frame_timestamps'. "
                "This is required for accurate spike-to-frame conversion."
            )
        
        # Get movie start frame from section_time to zero spike frames
        section_time_path = f"stimulus/section_time/{movie_name}"
        if section_time_path not in hdf5_file:
            raise ValueError(
                f"No section_time found for movie '{movie_name}' in {hdf5_path}. "
                "This is required to determine movie start frame."
            )
        section_time = hdf5_file[section_time_path][:]
        movie_start_sample = section_time[0, 0]  # First trial, start sample
        movie_start_frame = int(convert_sample_index_to_frame(
            np.array([movie_start_sample]), frame_timestamps
        )[0]) + PRE_MARGIN_FRAME_NUM # PRE_MARGIN_FRAME_NUM = 60 which is hardcoded in recording padding.
        logger.info(f"Movie '{movie_name}' starts at frame {movie_start_frame} (sample {movie_start_sample})")
        
        # Pre-load all spike data
        unit_spike_data: Dict[str, np.ndarray] = {}
        for unit_id in unit_ids:
            spike_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/trials_spike_times/0"
            
            if spike_path not in hdf5_file:
                logger.warning(f"No spike times found for {unit_id}/{movie_name}, skipping")
                warnings_list.append(f"No spike times for {unit_id}")
                continue
            
            spike_samples = hdf5_file[spike_path][:]
            # Convert to absolute frames, then zero to movie start
            spike_frames_absolute = convert_sample_index_to_frame(spike_samples, frame_timestamps)
            spike_frames = spike_frames_absolute - movie_start_frame
            unit_spike_data[unit_id] = spike_frames
    
    # Phase 2: Compute STA (parallel or sequential)
    n_workers = _get_worker_count() if use_multiprocessing else 1
    logger.info(
        f"Processing {len(unit_spike_data)} units with cover_range={cover_range}, "
        f"workers={n_workers}"
    )
    
    results: List[Tuple[str, np.ndarray, int, int, Optional[str]]] = []
    work_items = [(uid, spikes) for uid, spikes in unit_spike_data.items()]
    
    if use_multiprocessing and n_workers > 1:
        # Parallel processing with Windows spawn handling
        try:
            # Use spawn context explicitly for Windows compatibility
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(movie_array, cover_range),
            ) as pool:
                results = list(tqdm(
                    pool.imap(_compute_with_retry, work_items),
                    total=len(work_items),
                    desc="Computing STA"
                ))
        except RuntimeError as e:
            if "freeze_support" in str(e) or "bootstrapping phase" in str(e):
                logger.warning(
                    "Multiprocessing failed on Windows. This typically happens when "
                    "compute_sta() is called outside of 'if __name__ == \"__main__\":' block. "
                    "Falling back to sequential processing."
                )
                # Fall back to sequential processing
                _init_worker(movie_array, cover_range)
                for item in tqdm(work_items, desc="Computing STA (sequential)"):
                    results.append(_compute_with_retry(item))
            else:
                raise
    else:
        # Sequential processing
        _init_worker(movie_array, cover_range)
        for item in tqdm(work_items, desc="Computing STA"):
            results.append(_compute_with_retry(item))
    
    # Phase 3: Write results to HDF5 (sequential - HDF5 is not multiprocess-safe)
    units_processed = 0
    with h5py.File(hdf5_path, "r+") as hdf5_file:
        for unit_id, sta, n_used, n_excluded, error in results:
            if error is not None:
                logger.error(f"Failed to compute STA for {unit_id}: {error}")
                failed_units.append(unit_id)
                continue
            
            _write_sta_to_hdf5(
                hdf5_file, unit_id, movie_name, sta,
                n_used, n_excluded, cover_range, force
            )
            units_processed += 1
    
    elapsed = time.time() - start_time
    
    result = STAResult(
        hdf5_path=hdf5_path,
        movie_name=movie_name,
        units_processed=units_processed,
        units_failed=len(failed_units),
        cover_range=cover_range,
        elapsed_seconds=elapsed,
        warnings=warnings_list,
        failed_units=failed_units,
    )
    
    logger.info(
        f"STA computation complete: {units_processed} units processed, "
        f"{len(failed_units)} failed, {elapsed:.1f}s elapsed"
    )
    
    return result

