"""
Step Wrapper: Section Spike Times for Analog-Detected Stimuli

Sections spike times using sample indices directly from section_time data,
without frame conversion. This is designed for stimuli detected via analog
signal (raw_ch1) that occur outside the video frame period.

Unlike section_spike_times which uses JSON config files and frame_timestamps,
this step uses the section_time boundaries stored by add_section_time_analog.
"""

import logging
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
from tqdm import tqdm

from hdmea.pipeline import PipelineSession

logger = logging.getLogger(__name__)

STEP_NAME = "section_spike_times_analog"


def section_spike_times_analog_step(
    *,
    movie_name: str = "iprgc_test",
    pad_margin: Tuple[float, float] = (2.0, 0.0),
    session: PipelineSession,
) -> PipelineSession:
    """
    Section spike times for analog-detected stimuli using sample indices directly.
    
    This step is designed for stimuli like ipRGC tests that occur outside the
    video frame period. It uses the section_time boundaries stored by
    add_section_time_analog (which are in sample indices) directly, without
    trying to convert to/from frame indices.
    
    Args:
        movie_name: Name of the movie to section (must exist in section_time)
        pad_margin: Padding (pre_s, post_s) in seconds. Default: (2.0, 0.0) = 2s pre-margin
        session: Pipeline session (required, keyword-only)
    
    Returns:
        Updated session with sectioned spike times for the specified movie
    
    Output stored in session.units[unit_id]:
        spike_times_sectioned/{movie_name}/full_spike_times: All spikes combined (absolute sample indices)
        spike_times_sectioned/{movie_name}/trials_spike_times: Dict of per-trial spikes (absolute sample indices)
        spike_times_sectioned/{movie_name}/trials_start_end: (n_trials, 2) array of trial boundaries
    
    Example:
        >>> session = section_spike_times_analog_step(
        ...     movie_name="iprgc_test",
        ...     pad_margin=(2.0, 0.0),  # 2 second pre-margin
        ...     session=session,
        ... )
    """
    # =========================================================================
    # 1. Check if step already completed
    # =========================================================================
    step_key = f"{STEP_NAME}:{movie_name}"
    if step_key in session.completed_steps:
        logger.info(f"Skipping {step_key} - already completed")
        return session
    
    logger.info(f"Sectioning spike times (analog) for '{movie_name}'...")
    
    # =========================================================================
    # 2. Validate section_time exists for this movie
    # =========================================================================
    if "section_time" not in session.stimulus:
        msg = "No section_time data found in session"
        logger.error(msg)
        session.warnings.append(f"{STEP_NAME}: {msg}")
        session.mark_step_complete(f"{step_key}:failed")
        return session
    
    section_time_data = session.stimulus["section_time"]
    
    if movie_name not in section_time_data:
        msg = f"Movie '{movie_name}' not found in section_time"
        logger.error(msg)
        session.warnings.append(f"{STEP_NAME}: {msg}")
        session.mark_step_complete(f"{step_key}:failed")
        return session
    
    # Get section_time array: shape (N, 2) with [start_sample, end_sample]
    section_time = np.asarray(section_time_data[movie_name])
    n_trials = len(section_time)
    
    if n_trials == 0:
        msg = f"No trials found in section_time for '{movie_name}'"
        logger.warning(msg)
        session.warnings.append(f"{STEP_NAME}: {msg}")
        session.mark_step_complete(f"{step_key}:failed")
        return session
    
    logger.info(f"  Found {n_trials} trials for '{movie_name}'")
    logger.info(f"  Sample range: [{section_time[0, 0]}, {section_time[-1, 1]}]")
    
    # =========================================================================
    # 3. Convert padding from seconds to samples
    # =========================================================================
    acquisition_rate = float(session.metadata.get('acquisition_rate', 20000.0))
    pre_pad = int(pad_margin[0] * acquisition_rate)
    post_pad = int(pad_margin[1] * acquisition_rate)
    
    logger.info(f"  Padding: pre={pad_margin[0]:.1f}s ({pre_pad} samples), "
               f"post={pad_margin[1]:.1f}s ({post_pad} samples)")
    
    # =========================================================================
    # 4. Section spike times for each unit
    # =========================================================================
    units_processed = 0
    total_spikes = 0
    
    for unit_id in tqdm(session.units.keys(), desc=f"Sectioning {movie_name}", leave=False):
        unit_data = session.units[unit_id]
        
        if "spike_times" not in unit_data:
            continue
        
        spike_times = np.asarray(unit_data["spike_times"])
        
        if len(spike_times) == 0:
            continue
        
        # Initialize spike_times_sectioned if not present
        if "spike_times_sectioned" not in unit_data:
            unit_data["spike_times_sectioned"] = {}
        
        # Section spikes for this movie
        all_spikes = []
        trials_spikes = {}
        
        for trial_idx, (start_sample, end_sample) in enumerate(section_time):
            # Apply padding
            trial_start = start_sample - pre_pad
            trial_end = end_sample + post_pad
            
            # Find spikes within this trial window
            mask = (spike_times >= trial_start) & (spike_times < trial_end)
            trial_spikes = spike_times[mask]
            
            # Store as absolute sample indices (not relative to trial start)
            trials_spikes[str(trial_idx)] = trial_spikes
            all_spikes.extend(trial_spikes.tolist())
        
        # Store results with trials_start_end and metadata (matching section_spike_times format)
        unit_data["spike_times_sectioned"][movie_name] = {
            "full_spike_times": np.array(all_spikes, dtype=np.int64),
            "trials_spike_times": trials_spikes,
            "trials_start_end": section_time.astype(np.int64),  # (n_trials, 2) array
            "_attrs": {
                "n_trials": n_trials,
                "pad_margin": list(pad_margin),  # in seconds
                "pre_samples": pre_pad,
                "post_samples": post_pad,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        
        units_processed += 1
        total_spikes += len(all_spikes)
    
    # =========================================================================
    # 4. Summary
    # =========================================================================
    logger.info(f"  Sectioned {units_processed} units, {total_spikes} total spikes")
    logger.info(f"  Trial boundaries stored in trials_start_end for each unit")
    
    session.mark_step_complete(step_key)
    
    return session

