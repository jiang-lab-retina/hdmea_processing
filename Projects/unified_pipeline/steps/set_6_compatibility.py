"""
Set6 Compatibility Step

Creates set6-compatible data from set6a recordings by extracting the last N repeats
from green_blue_3s_3i_3x_64_128_255 and storing them as green_blue_3s_3i_3x.

This enables set6a recordings to be analyzed using set6 pipelines.
"""

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hdmea.pipeline.session import PipelineSession

logger = logging.getLogger(__name__)


def set_6_compatibility_step(
    *,
    source_movie: str = "green_blue_3s_3i_3x_64_128_255",
    target_movie: str = "green_blue_3s_3i_3x",
    repeat_slice: Tuple[Optional[int], Optional[int]] = (-3, None),
    session: "PipelineSession",
) -> "PipelineSession":
    """
    Create set6-compatible data by extracting specific repeats from set6a data.
    
    Takes the last N repeats from source_movie and stores them as target_movie,
    enabling set6a recordings to be analyzed with set6 pipelines.
    
    Args:
        source_movie: Source movie name (e.g., "green_blue_3s_3i_3x_64_128_255")
        target_movie: Target movie name (e.g., "green_blue_3s_3i_3x")
        repeat_slice: Tuple of (start, end) for slice notation. (-3, None) = last 3.
        session: PipelineSession object
    
    Returns:
        Updated PipelineSession with set6-compatible data added
    
    Example:
        >>> session = set_6_compatibility_step(
        ...     source_movie="green_blue_3s_3i_3x_64_128_255",
        ...     target_movie="green_blue_3s_3i_3x",
        ...     repeat_slice=(-3, None),
        ...     session=session,
        ... )
    """
    logger.info(f"Step: Creating set6-compatible {target_movie} from {source_movie}...")
    
    slice_start, slice_end = repeat_slice
    
    # =========================================================================
    # Copy stimulus-level data
    # =========================================================================
    
    # Copy section_time (stimulus level)
    if "section_time" in session.stimulus and source_movie in session.stimulus["section_time"]:
        source_section_time = session.stimulus["section_time"][source_movie]
        # Apply slice to get the target repeats
        target_section_time = source_section_time[slice_start:slice_end]
        session.stimulus["section_time"][target_movie] = target_section_time
        logger.debug(f"  Copied section_time: {source_section_time.shape} -> {target_section_time.shape}")
    else:
        logger.warning(f"  No section_time found for {source_movie}")
    
    # Copy light_template (same for all repeats at intensity 255)
    if "light_template" in session.stimulus and source_movie in session.stimulus["light_template"]:
        source_template = session.stimulus["light_template"][source_movie]
        session.stimulus["light_template"][target_movie] = source_template.copy()
        logger.debug(f"  Copied light_template: shape {source_template.shape}")
    
    # =========================================================================
    # Copy unit-level sectioned spike times
    # =========================================================================
    
    units_processed = 0
    units_skipped = 0
    
    for unit_id, unit_data in session.units.items():
        # Check if source movie exists in spike_times_sectioned
        if "spike_times_sectioned" not in unit_data:
            units_skipped += 1
            continue
        
        sectioned = unit_data["spike_times_sectioned"]
        
        if source_movie not in sectioned:
            units_skipped += 1
            continue
        
        source_data = sectioned[source_movie]
        
        # Create target movie data structure
        target_data = {}
        
        # Copy trials_start_end with slice
        if "trials_start_end" in source_data:
            source_start_end = source_data["trials_start_end"]
            target_data["trials_start_end"] = source_start_end[slice_start:slice_end]
        
        # Copy trials_spike_times with re-indexing
        if "trials_spike_times" in source_data:
            source_trials = source_data["trials_spike_times"]
            
            # Get the indices we want (e.g., for slice(-3, None) with 9 repeats: 6, 7, 8)
            total_trials = len(source_trials)
            
            # Calculate actual indices from slice
            if slice_start is None:
                start_idx = 0
            elif slice_start < 0:
                start_idx = total_trials + slice_start
            else:
                start_idx = slice_start
            
            if slice_end is None:
                end_idx = total_trials
            elif slice_end < 0:
                end_idx = total_trials + slice_end
            else:
                end_idx = slice_end
            
            # Copy trials with new indices (0, 1, 2, ...)
            target_trials = {}
            new_idx = 0
            for old_idx in range(start_idx, end_idx):
                if old_idx in source_trials:
                    target_trials[new_idx] = source_trials[old_idx].copy()
                    new_idx += 1
            
            target_data["trials_spike_times"] = target_trials
        
        # Compute full_spike_times by combining all target trials
        if "trials_spike_times" in target_data:
            all_spikes = []
            for trial_spikes in target_data["trials_spike_times"].values():
                if len(trial_spikes) > 0:
                    all_spikes.append(trial_spikes)
            
            if all_spikes:
                combined = np.concatenate(all_spikes)
                target_data["full_spike_times"] = np.sort(combined)
            else:
                target_data["full_spike_times"] = np.array([], dtype=np.uint64)
        
        # Store in session
        sectioned[target_movie] = target_data
        units_processed += 1
    
    logger.info(f"  Processed {units_processed} units, skipped {units_skipped}")
    
    # Mark step complete
    session.mark_step_complete("set_6_compatibility")
    
    return session
