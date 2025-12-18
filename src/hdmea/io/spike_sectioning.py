"""
Spike times sectioning module for HD-MEA pipeline.

Provides functionality to section spike timestamps by stimulation periods
(trials) defined in section_time data. Supports both combined (full_spike_times)
and per-trial (trials_spike_times) storage formats.

Example:
    >>> from hdmea.io.spike_sectioning import section_spike_times
    >>> result = section_spike_times(
    ...     zarr_path="artifacts/JIANG009_2025-04-10.zarr",
    ...     trial_repeats=3,
    ...     pad_margin=(2.0, 0.0),  # 2s pre-margin, 0s post-margin
    ... )
    >>> print(f"Processed {result.units_processed} units")
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from tqdm import tqdm

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
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Extract spikes within padded trial boundaries for a single unit.
    
    Args:
        spike_times: Array of spike times in sample indices (uint64).
        section_time: Array of shape (N_trials, 2) with [start, end] samples.
        trial_repeats: Number of trials to process (uses first N).
        pre_samples: Padding in samples to extend before trial start.
        post_samples: Padding in samples to extend after trial end.
    
    Returns:
        Tuple of:
            - full_spike_times: All spikes from all processed trials (sorted, unique).
            - trials_spike_times: Dict mapping trial_idx -> spike array.
    
    Note:
        Padded boundaries are clamped: start >= 0, end <= max(spike_times).
    """
    # Handle empty spike_times
    if len(spike_times) == 0:
        empty_array = np.array([], dtype=np.int64)
        n_trials = min(trial_repeats, len(section_time))
        trials_dict = {i: empty_array.copy() for i in range(n_trials)}
        return empty_array, trials_dict
    
    trials_spikes = {}
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
    unit_group: zarr.Group,
    movie_name: str,
    full_spike_times: np.ndarray,
    trials_spike_times: Dict[int, np.ndarray],
    pad_margin: Tuple[float, float],
    pre_samples: int,
    post_samples: int,
    trial_repeats: int,
    force: bool = False,
) -> None:
    """
    Write sectioned spike times to unit group in Zarr.
    
    Creates structure:
        spike_times_sectioned/{movie_name}/
            full_spike_times          # All trials combined
            trials_spike_times/
                0                     # Trial 0 spikes
                1                     # Trial 1 spikes
                ...
    
    Args:
        unit_group: Zarr group for the unit (units/{unit_id}).
        movie_name: Name of the movie/stimulus.
        full_spike_times: All spikes from all trials (sorted).
        trials_spike_times: Dict mapping trial_idx -> spike array.
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
    # Note: When force=True, existing groups are pre-deleted before parallel processing
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
        # Delete existing movie group with retry (shouldn't happen if pre-deleted, but safe fallback)
        try:
            _retry_operation(lambda: sectioned_group.__delitem__(movie_name))
        except Exception:
            pass  # May already be deleted by pre-deletion step
    
    # Create movie group
    movie_group = _retry_operation(sectioned_group.create_group, movie_name)
    
    # Write full_spike_times array
    _retry_operation(
        movie_group.create_dataset,
        "full_spike_times",
        data=full_spike_times,
        shape=full_spike_times.shape,
        dtype=np.int64,
        overwrite=True,
    )
    
    # Create trials_spike_times group
    trials_group = _retry_operation(movie_group.create_group, "trials_spike_times")
    
    # Write per-trial arrays
    for trial_idx, trial_spikes in trials_spike_times.items():
        _retry_operation(
            trials_group.create_dataset,
            str(trial_idx),
            data=trial_spikes,
            shape=trial_spikes.shape,
            dtype=np.int64,
            overwrite=True,
        )
    
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
    zarr_path: Union[str, Path],
    *,
    movie_names: Optional[List[str]] = None,
    trial_repeats: int = 3,
    pad_margin: Tuple[float, float] = (2.0, 0.0),
    force: bool = False,
) -> SectionResult:
    """
    Section spike times by stimulation periods defined in section_time.
    
    For each unit in the Zarr archive, extracts spikes falling within
    trial periods (with optional padding) and stores them in TWO formats:
    - `full_spike_times`: All spikes from all trials combined
    - `trials_spike_times/{idx}`: Spikes per trial based on section_time boundaries
    
    Args:
        zarr_path: Path to Zarr archive containing recording data.
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
        FileNotFoundError: If zarr_path does not exist.
        FileExistsError: If sectioned data exists and force=False.
    
    Example:
        >>> result = section_spike_times(
        ...     zarr_path="artifacts/JIANG009_2025-04-10.zarr",
        ...     trial_repeats=3,
        ...     pad_margin=(2.0, 0.0),
        ... )
        >>> print(f"Processed {result.units_processed} units")
    """
    zarr_path = Path(zarr_path)
    warnings: List[str] = []
    
    # Validate path exists
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr archive not found: {zarr_path}")
    
    # Open Zarr
    root = zarr.open(str(zarr_path), mode="r+")
    
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
        warnings.append("No section_time data found in Zarr")
        logger.warning("No section_time data found - cannot section spike times")
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
        warnings.append("No units group found in Zarr")
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
        warnings.append("No units found in Zarr")
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
    
    # If force=True, pre-delete all existing spike_times_sectioned directories SEQUENTIALLY
    # Uses shutil.rmtree for reliable deletion on Windows (zarr del can fail)
    if force:
        logger.info("Clearing existing spike_times_sectioned data...")
        for unit_id in tqdm(unit_ids, desc="Clearing existing data", leave=False):
            # Use filesystem deletion instead of zarr API (more reliable on Windows)
            sectioned_path = zarr_path / "units" / unit_id / "spike_times_sectioned"
            if sectioned_path.exists():
                try:
                    shutil.rmtree(str(sectioned_path))
                except Exception as e:
                    # Retry with delay for Windows file locking
                    time.sleep(RETRY_DELAY * 2)
                    try:
                        shutil.rmtree(str(sectioned_path))
                    except Exception:
                        warnings.append(f"Could not delete {unit_id}/spike_times_sectioned: {e}")
        
        # Small delay to ensure filesystem sync before writes
        time.sleep(0.2)
    
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
            section_time = section_time_cache[movie_name]
            
            if len(section_time) == 0:
                unit_warnings.append(f"Movie {movie_name} has no trials in section_time")
                continue
            
            full_spike_times, trials_spike_times = _section_unit_spikes(
                spike_times=spike_times,
                section_time=section_time,
                trial_repeats=trial_repeats,
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
            for movie_name, (full_spike_times, trials_spike_times) in unit_data.items():
                _write_sectioned_spikes(
                    unit_group=unit_group,
                    movie_name=movie_name,
                    full_spike_times=full_spike_times,
                    trials_spike_times=trials_spike_times,
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

