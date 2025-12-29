"""
Step Wrapper: Add Section Time from Analog Signal

Detects stimulus onsets from raw light reference signal (raw_ch1) and stores
section times in the PipelineSession for deferred HDF5 saving.

This is a session-compatible version of hdmea.io.section_time.add_section_time_analog
that works with PipelineSession for pipeline integration.
"""

import itertools
import logging
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from hdmea.pipeline import PipelineSession
from Projects.unified_pipeline.config import SectionTimeAnalogConfig, red_warning

logger = logging.getLogger(__name__)

STEP_NAME = "add_section_time_analog"


def add_section_time_analog_step(
    *,
    config: SectionTimeAnalogConfig,
    session: PipelineSession,
) -> PipelineSession:
    """
    Add section time by detecting peaks in raw_ch1 light reference signal.
    
    This step detects actual stimulus onsets from the recorded light signal,
    suitable for experiments where exact timing must be determined post-hoc
    (e.g., ipRGC stimulation tests).
    
    Reads raw_ch1 from session.stimulus['light_reference']['raw_ch1'] and
    stores results in session.stimulus for deferred HDF5 saving.
    
    Args:
        config: SectionTimeAnalogConfig with detection parameters
        session: Pipeline session (required, keyword-only)
    
    Returns:
        Updated session with section_time and light_template in session.stimulus
    
    Raises:
        ValueError: If required data (raw_ch1, acquisition_rate) not found in session
    
    Output stored in session.stimulus:
        section_time/{movie_name}: int64[N, 2] array
            - Each row: [start_sample, end_sample] in acquisition sample indices
            
        light_template/{movie_name}: float32[M] array
            - Averaged light reference trace across all detected trials
    
    Example:
        >>> from Projects.unified_pipeline.config import SectionTimeAnalogConfig
        >>> config = SectionTimeAnalogConfig(threshold_value=1e5)
        >>> session = add_section_time_analog_step(config=config, session=session)
    """
    # =========================================================================
    # 1. Check if step already completed (skip if so)
    # =========================================================================
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME} - already completed")
        return session
    
    logger.info(f"Adding section time (analog) for '{config.movie_name}'...")
    
    # =========================================================================
    # 2. Validate required data exists in session
    # =========================================================================
    # Check for light_reference/raw_ch1
    if 'light_reference' not in session.stimulus:
        msg = "stimulus/light_reference not found in session"
        logger.error(msg)
        session.warnings.append(f"{STEP_NAME}: {msg}")
        session.mark_step_complete(f"{STEP_NAME}:failed")
        return session
    
    light_ref = session.stimulus['light_reference']
    if 'raw_ch1' not in light_ref:
        msg = "raw_ch1 not found in stimulus/light_reference"
        logger.error(msg)
        session.warnings.append(f"{STEP_NAME}: {msg}")
        session.mark_step_complete(f"{STEP_NAME}:failed")
        return session
    
    # Check for acquisition_rate in metadata
    if 'acquisition_rate' not in session.metadata:
        msg = "acquisition_rate not found in metadata"
        logger.error(msg)
        session.warnings.append(f"{STEP_NAME}: {msg}")
        session.mark_step_complete(f"{STEP_NAME}:failed")
        return session
    
    # =========================================================================
    # 3. Load required data from session
    # =========================================================================
    raw_ch1 = np.asarray(light_ref['raw_ch1'])
    acquisition_rate = float(session.metadata['acquisition_rate'])
    
    logger.info(f"  Loaded raw_ch1: {len(raw_ch1)} samples at {acquisition_rate} Hz")
    
    # =========================================================================
    # 3b. Determine search range (after last video frame)
    # =========================================================================
    # Get frame_timestamps to find where video recording ends
    search_start_sample = 0
    if 'frame_timestamps' in session.metadata:
        frame_timestamps = np.asarray(session.metadata['frame_timestamps'])
        last_frame_sample = int(frame_timestamps[-1])
        search_start_sample = last_frame_sample
        logger.info(f"  Searching after last video frame: sample {last_frame_sample:,} "
                   f"({last_frame_sample / acquisition_rate:.1f}s)")
    elif 'frame_times' in session.stimulus and 'default' in session.stimulus['frame_times']:
        frame_times = np.asarray(session.stimulus['frame_times']['default'])
        last_frame_sample = int(frame_times[-1])
        search_start_sample = last_frame_sample
        logger.info(f"  Searching after last video frame: sample {last_frame_sample:,} "
                   f"({last_frame_sample / acquisition_rate:.1f}s)")
    else:
        logger.warning("  No frame_timestamps found - searching entire signal")
    
    # =========================================================================
    # 4. Check for existing section_time (unless force=True)
    # =========================================================================
    if 'section_time' not in session.stimulus:
        session.stimulus['section_time'] = {}
    
    if config.movie_name in session.stimulus['section_time'] and not config.force:
        msg = f"section_time/{config.movie_name} already exists. Use force=True to overwrite."
        logger.warning(msg)
        session.warnings.append(f"{STEP_NAME}: {msg}")
        session.mark_step_complete(f"{STEP_NAME}:skipped")
        return session
    
    # =========================================================================
    # 5. Detect peaks in raw_ch1 signal (only after last video frame)
    # =========================================================================
    # Extract the portion of the signal to search
    signal_to_search = raw_ch1[search_start_sample:]
    
    # Calculate minimum distance between peaks (with soft margin from config)
    min_distance_s = config.plot_duration * config.min_peak_distance_margin
    min_distance_samples = int(min_distance_s * acquisition_rate)
    
    logger.info(f"  Detecting peaks with threshold={config.threshold_value:.0e}, "
               f"min_distance={min_distance_s:.0f}s ({min_distance_samples:,} samples)")
    logger.info(f"  Searching in samples [{search_start_sample:,}, {len(raw_ch1):,}]")
    
    # Compute derivative and find peaks
    diff_signal = np.diff(signal_to_search.astype(np.float64))
    peaks_relative, peak_props = find_peaks(
        diff_signal, 
        height=config.threshold_value,
        distance=min_distance_samples
    )
    
    if len(peaks_relative) == 0:
        msg = f"No peaks detected with threshold={config.threshold_value:.0e}"
        logger.warning(red_warning(msg))
        session.warnings.append(f"{STEP_NAME}: {msg}")
        session.mark_step_complete(f"{STEP_NAME}:failed")
        return session
    
    # Convert back to absolute sample indices
    onset_samples = peaks_relative + search_start_sample
    peak_heights = peak_props['peak_heights']
    
    logger.info(f"  Detected {len(onset_samples)} stimulus onsets")
    logger.info(f"  Peak heights: {[f'{h:.0f}' for h in peak_heights]}")
    
    # =========================================================================
    # 6. Calculate duration and extract light traces
    # =========================================================================
    duration_samples = int(config.plot_duration * acquisition_rate)
    max_sample = len(raw_ch1) - 1
    
    # Extract light reference traces from ALL detected trials for template averaging
    all_traces = []
    for onset in onset_samples:
        end_sample = min(onset + duration_samples, max_sample)
        trace = raw_ch1[onset:end_sample]
        all_traces.append(trace)
    
    # Average across all trials (use zip_longest for variable lengths)
    if all_traces:
        template_list = list(itertools.zip_longest(*all_traces, fillvalue=np.nan))
        light_template = np.nanmean(template_list, axis=1).astype(np.float32)
        logger.info(f"  Computed light template from {len(all_traces)} trials: {len(light_template)} samples")
    else:
        light_template = None
    
    # =========================================================================
    # 7. Apply repeat limit if specified
    # =========================================================================
    if config.repeat is not None and config.repeat > 0:
        onset_samples = onset_samples[:config.repeat]
        logger.info(f"  Limited section_time to first {config.repeat} trials: {len(onset_samples)} onsets")
    
    # Calculate end samples
    end_samples = onset_samples + duration_samples
    
    # Clip end samples to signal length
    n_truncated = np.sum(end_samples > max_sample)
    if n_truncated > 0:
        logger.warning(
            f"  {n_truncated} section(s) truncated at signal boundary "
            f"(end sample clipped to {max_sample:,})"
        )
    end_samples = np.clip(end_samples, 0, max_sample)
    
    # Build section_time array: shape (N, 2) with [start_sample, end_sample]
    section_time = np.column_stack([onset_samples, end_samples]).astype(np.int64)
    
    logger.info(f"  Computed section_time: {section_time.shape[0]} trials, "
                f"sample range [{section_time[:, 0].min()}, {section_time[:, 1].max()}]")
    
    # =========================================================================
    # 8. Store results in session.stimulus
    # =========================================================================
    session.stimulus['section_time'][config.movie_name] = section_time
    logger.info(f"  Stored section_time/{config.movie_name}")
    
    # Store light_template
    if light_template is not None:
        if 'light_template' not in session.stimulus:
            session.stimulus['light_template'] = {}
        session.stimulus['light_template'][config.movie_name] = light_template
        logger.info(f"  Stored light_template/{config.movie_name}")
    
    # =========================================================================
    # 9. Mark step complete
    # =========================================================================
    session.mark_step_complete(STEP_NAME)
    logger.info(f"  Section time (analog) added successfully")
    
    return session

