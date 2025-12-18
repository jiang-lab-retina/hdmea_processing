"""
Spike times sectioning module for HD-MEA pipeline.

Provides functionality to section spike timestamps by stimulation periods
(trials) defined in section_time data. Stores combined full_spike_times
(all trials merged).

Example:
    >>> from hdmea.io.spike_sectioning import section_spike_times
    >>> result = section_spike_times(
    ...     hdf5_path="artifacts/JIANG009_2025-04-10.h5",
    ...     trial_repeats=3,
    ...     pad_margin=(2.0, 0.0),  # 2s pre-margin, 0s post-margin
    ... )
    >>> print(f"Processed {result.units_processed} units")
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

from hdmea.io.hdf5_store import open_recording_hdf5

logger = logging.getLogger(__name__)


# Windows file locking workaround: retry operations that may fail due to file handles
MAX_RETRIES = 3
RETRY_DELAY = 0.1  # seconds (only used on actual retry, not every operation)


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
# Helper Functions
# =============================================================================

def _section_unit_spikes(
    spike_times: np.ndarray,
    section_time: np.ndarray,
    trial_repeats: int = 3,
    pre_samples: int = 0,
    post_samples: int = 0,
) -> np.ndarray:
    """
    Extract spikes within padded trial boundaries for a single unit.
    
    Args:
        spike_times: Array of spike times in sample indices (uint64).
        section_time: Array of shape (N_trials, 2) with [start, end] samples.
        trial_repeats: Number of trials to process (uses first N).
        pre_samples: Padding in samples to extend before trial start.
        post_samples: Padding in samples to extend after trial end.
    
    Returns:
        full_spike_times: All spikes from all processed trials (sorted, unique).
    
    Note:
        Padded boundaries are clamped: start >= 0, end <= max(spike_times).
    """
    # Handle empty spike_times
    if len(spike_times) == 0:
        return np.array([], dtype=np.int64)
    
    trial_arrays = []
    
    n_trials = min(trial_repeats, len(section_time))
    
    for trial_idx in range(n_trials):
        trial_start, trial_end = section_time[trial_idx]
        
        # Apply padding and clamp to valid range
        padded_start = max(0, int(trial_start) - pre_samples)
        padded_end = int(trial_end) + post_samples
        
        # Extract spikes within [padded_start, padded_end)
        mask = (spike_times >= padded_start) & (spike_times < padded_end)
        trial_spikes = spike_times[mask].astype(np.int64)
        
        trial_arrays.append(trial_spikes)
    
    # Combine all trials efficiently using numpy concatenation
    if trial_arrays:
        combined = np.concatenate(trial_arrays)
        full_spike_times = np.unique(combined)  # unique also sorts
    else:
        full_spike_times = np.array([], dtype=np.int64)
    
    return full_spike_times


def _write_sectioned_spikes(
    unit_group: h5py.Group,
    movie_name: str,
    full_spike_times: np.ndarray,
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
    
    Args:
        unit_group: HDF5 group for the unit (units/{unit_id}).
        movie_name: Name of the movie/stimulus.
        full_spike_times: All spikes from all trials (sorted).
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
    
    # Write metadata attributes
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
) -> SectionResult:
    """
    Section spike times by stimulation periods defined in section_time.
    
    For each unit in the HDF5 file, extracts spikes falling within
    trial periods (with optional padding) and stores them as:
    - `full_spike_times`: All spikes from all trials combined
    
    Args:
        hdf5_path: Path to HDF5 file containing recording data.
            Must have `units/{unit_id}/spike_times` arrays (in sample units)
            and `stimulus/section_time/{movie_name}` trial boundaries.
        movie_names: Optional list of movies to process. If None, processes
            all movies found in section_time.
        trial_repeats: Number of trials to process per movie (default=3).
            Uses first N trials from section_time.
        pad_margin: Tuple of (pre_margin_s, post_margin_s) in seconds.
            Default=(2.0, 0.0) - 2 seconds before trial start, 0 after trial end.
            Converted to samples using acquisition_rate.
        force: If True, overwrite existing sectioned data. If False (default),
            raise FileExistsError if any unit already has sectioned data.
    
    Returns:
        SectionResult with success, units_processed, movies_processed, etc.
    
    Raises:
        FileNotFoundError: If hdf5_path does not exist.
        FileExistsError: If sectioned data exists and force=False.
    
    Example:
        >>> result = section_spike_times(
        ...     hdf5_path="artifacts/JIANG009_2025-04-10.h5",
        ...     trial_repeats=3,
        ...     pad_margin=(2.0, 0.0),
        ... )
        >>> print(f"Processed {result.units_processed} units")
    """
    hdf5_path = Path(hdf5_path)
    warnings: List[str] = []
    
    # Validate path exists
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
    # Open HDF5
    root = open_recording_hdf5(hdf5_path, mode="r+")
    
    # Get acquisition_rate from metadata
    if "metadata" not in root:
        warnings.append("No metadata group found, using default acquisition_rate")
        acquisition_rate = 20000.0
    else:
        metadata = root["metadata"]
        if "acquisition_rate" in metadata:
            acquisition_rate = float(metadata["acquisition_rate"][...])
        else:
            acquisition_rate = 20000.0
            warnings.append("acquisition_rate not in metadata, using default 20000 Hz")
    
    # Convert pad_margin to samples
    pre_samples = int(pad_margin[0] * acquisition_rate)
    post_samples = int(pad_margin[1] * acquisition_rate)
    
    logger.info(
        f"Sectioning spike times with trial_repeats={trial_repeats}, "
        f"pad_margin={pad_margin} ({pre_samples} pre, {post_samples} post samples)"
    )
    
    # Check for section_time
    if "stimulus" not in root or "section_time" not in root["stimulus"]:
        warnings.append("No section_time data found in HDF5")
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
            warnings=warnings,
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
            warnings.append(f"Movies not found in section_time: {missing}")
            logger.warning(f"Movies not found: {missing}")
    
    if not movies_to_process:
        warnings.append("No movies to process")
        root.close()
        return SectionResult(
            success=True,
            units_processed=0,
            movies_processed=[],
            trial_repeats=trial_repeats,
            pad_margin=pad_margin,
            pre_samples=pre_samples,
            post_samples=post_samples,
            warnings=warnings,
        )
    
    logger.info(f"Processing movies: {movies_to_process}")
    
    # Pre-load all section_time arrays (cache outside unit loop for performance)
    section_time_cache: Dict[str, np.ndarray] = {}
    for movie_name in movies_to_process:
        section_time_cache[movie_name] = section_time_group[movie_name][:]
    
    # Check for units
    if "units" not in root:
        warnings.append("No units group found in HDF5")
        root.close()
        return SectionResult(
            success=True,
            units_processed=0,
            movies_processed=[],
            trial_repeats=trial_repeats,
            pad_margin=pad_margin,
            pre_samples=pre_samples,
            post_samples=post_samples,
            warnings=warnings,
        )
    
    units_group = root["units"]
    unit_ids = list(units_group.keys())
    
    if not unit_ids:
        warnings.append("No units found in HDF5")
        root.close()
        return SectionResult(
            success=True,
            units_processed=0,
            movies_processed=[],
            trial_repeats=trial_repeats,
            pad_margin=pad_margin,
            pre_samples=pre_samples,
            post_samples=post_samples,
            warnings=warnings,
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
                warnings.append(f"Could not delete {unit_id}/spike_times_sectioned: {e}")
    
    # PHASE 1: Compute all sectioned spikes in PARALLEL (CPU-bound, no I/O conflicts)
    units_processed = 0
    movies_actually_processed: set = set()
    
    # Data structure to hold computed results: {unit_id: {movie_name: full_spikes}}
    ComputedData = Dict[str, np.ndarray]
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
            section_time = section_time_cache[movie_name]
            
            if len(section_time) == 0:
                unit_warnings.append(f"Movie {movie_name} has no trials in section_time")
                continue
            
            full_spike_times = _section_unit_spikes(
                spike_times=spike_times,
                section_time=section_time,
                trial_repeats=trial_repeats,
                pre_samples=pre_samples,
                post_samples=post_samples,
            )
            
            unit_data[movie_name] = full_spike_times
        
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
                warnings.extend(unit_warnings)
                if unit_data:
                    computed_results[uid] = (unit_data, unit_warnings)
            except Exception as e:
                warnings.append(f"Unit {unit_id} compute failed: {e}")
    
    # PHASE 2: Write all results SEQUENTIALLY (avoids Zarr metadata conflicts)
    logger.info(f"Phase 2: Writing {len(computed_results)} units sequentially")
    
    for unit_id in tqdm(computed_results.keys(), desc="Writing sections", leave=False):
        unit_data, _ = computed_results[unit_id]
        unit_group = units_group[unit_id]
        
        try:
            for movie_name, full_spike_times in unit_data.items():
                _write_sectioned_spikes(
                    unit_group=unit_group,
                    movie_name=movie_name,
                    full_spike_times=full_spike_times,
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
            warnings.append(f"Unit {unit_id} write failed: {e}")
    
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
        warnings=warnings,
    )

