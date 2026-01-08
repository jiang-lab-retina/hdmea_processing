"""
Step Wrappers: Extract Soma Geometry, Extract RF Geometry

These steps extract geometric features from STA data.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from hdmea.pipeline import PipelineSession

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "Projects/sta_quantification"))
sys.path.insert(0, str(project_root / "Projects/rf_sta_measure"))

logger = logging.getLogger(__name__)


def extract_soma_geometry_step(
    *,
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
    session: PipelineSession,
) -> PipelineSession:
    """
    Extract soma geometry from eimage_sta data.
    
    This is Step 6 of the pipeline.
    
    Args:
        frame_range: Frames to use for size estimation
        threshold_fraction: Threshold fraction for soma mask
        session: Pipeline session (required)
    
    Returns:
        Updated session with soma geometry
    """
    step_name = "extract_soma_geometry"
    
    if step_name in session.completed_steps:
        logger.info(f"Skipping {step_name} - already completed")
        return session
    
    logger.info(f"Step 6: Extracting soma geometry...")
    
    try:
        from ap_sta import extract_eimage_sta_geometry
        
        session = extract_eimage_sta_geometry(
            frame_range=frame_range,
            threshold_fraction=threshold_fraction,
            session=session,
        )
        
        session.mark_step_complete(step_name)
        logger.info(f"  Soma geometry extracted")
        
    except ImportError as e:
        logger.warning(f"  Cannot import geometry extraction: {e}")
        session.warnings.append(f"{step_name}: Import error - {e}")
        session.completed_steps.add(f"{step_name}:skipped")
    
    except Exception as e:
        logger.error(f"  Error extracting geometry: {e}")
        session.warnings.append(f"{step_name}: {e}")
        session.completed_steps.add(f"{step_name}:failed")
    
    return session


def extract_rf_geometry_step(
    *,
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
    sta_feature_name: str = "sta_perfect_dense_noise_15x15_15hz_r42_3min",
    cover_range: Tuple[int, int] = (-60, 0),
    frame_rate: float = 15.0,
    enable_lnl_fitting: bool = True,
    session: PipelineSession,
) -> PipelineSession:
    """
    Extract RF-STA geometry (Gaussian, DoG, ON/OFF fits) and optionally LNL model.
    
    This is Step 7 of the pipeline.
    
    Args:
        frame_range: Frames to use for analysis
        threshold_fraction: Threshold fraction for fitting
        sta_feature_name: Name of the STA feature to analyze
        cover_range: Frame window for STA/LNL (default: (-60, 0))
        frame_rate: Stimulus frame rate in Hz (default: 15.0)
        enable_lnl_fitting: If True, also fit LNL model when stimulus data available
        session: Pipeline session (required)
    
    Returns:
        Updated session with RF geometry (and LNL fits if enabled)
    """
    step_name = "extract_rf_geometry"
    
    if step_name in session.completed_steps:
        logger.info(f"Skipping {step_name} - already completed")
        return session
    
    logger.info(f"Step 7: Extracting RF-STA geometry...")
    
    try:
        from rf_session import extract_rf_geometry_session
        
        # Prepare stimulus data for LNL fitting if enabled
        movie_array: Optional[np.ndarray] = None
        spike_frames_dict: Optional[Dict[str, np.ndarray]] = None
        
        if enable_lnl_fitting:
            movie_array, spike_frames_dict = _prepare_lnl_data(session, cover_range)
            if movie_array is not None:
                logger.info(f"  LNL fitting enabled: movie shape={movie_array.shape}, {len(spike_frames_dict)} units with spikes")
            else:
                logger.info(f"  LNL fitting disabled: stimulus data not available")
        
        session = extract_rf_geometry_session(
            session,
            frame_range=frame_range,
            threshold_fraction=threshold_fraction,
            movie_array=movie_array,
            spike_frames_dict=spike_frames_dict,
            cover_range=cover_range,
            frame_rate=frame_rate,
        )
        
        # Add rf_sta_geometry metadata (matches reference file structure)
        session.metadata['rf_sta_geometry'] = {
            'sta_feature_name': sta_feature_name,
            'frame_range': list(frame_range),
            'threshold_fraction': threshold_fraction,
            'lnl_fitting_enabled': enable_lnl_fitting and movie_array is not None,
        }
        
        session.mark_step_complete(step_name)
        logger.info(f"  RF geometry extracted")
        
    except ImportError as e:
        logger.warning(f"  Cannot import RF geometry extraction: {e}")
        session.warnings.append(f"{step_name}: Import error - {e}")
        session.completed_steps.add(f"{step_name}:skipped")
    
    except Exception as e:
        logger.error(f"  Error extracting RF geometry: {e}")
        session.warnings.append(f"{step_name}: {e}")
        session.completed_steps.add(f"{step_name}:failed")
    
    return session


def _prepare_lnl_data(
    session: PipelineSession,
    cover_range: Tuple[int, int] = (-60, 0),
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Prepare stimulus movie and spike frames for LNL fitting.
    
    Reuses the same logic as compute_sta to load the stimulus movie
    and convert spike times to frame indices.
    
    Args:
        session: PipelineSession with stimulus and spike data
        cover_range: Frame window for alignment
    
    Returns:
        Tuple of (movie_array, spike_frames_dict) or (None, None) if data unavailable
    """
    try:
        from hdmea.features.sta import (
            _load_stimulus_movie,
            _find_noise_movie_from_session,
            DEFAULT_STIMULI_DIR,
        )
        from hdmea.io.section_time import convert_sample_index_to_frame, PRE_MARGIN_FRAME_NUM
    except ImportError as e:
        logger.debug(f"Cannot import STA utilities for LNL fitting: {e}")
        return None, None
    
    # Find noise movie from session section_time
    section_time_data = session.stimulus.get("section_time", {})
    movie_name = _find_noise_movie_from_session(section_time_data)
    
    if movie_name is None:
        logger.debug("No noise movie found in session section_time")
        return None, None
    
    # Load stimulus movie
    try:
        movie_array = _load_stimulus_movie(movie_name, DEFAULT_STIMULI_DIR)
    except FileNotFoundError as e:
        logger.debug(f"Cannot load stimulus movie: {e}")
        return None, None
    
    # Get frame_timestamps
    frame_timestamps = None
    if "frame_times" in session.stimulus and "frame_timestamps" in session.stimulus["frame_times"]:
        frame_timestamps = np.array(session.stimulus["frame_times"]["frame_timestamps"])
    elif "frame_timestamps" in session.metadata:
        frame_timestamps = np.array(session.metadata["frame_timestamps"])
    
    if frame_timestamps is None:
        logger.debug("No frame_timestamps found in session")
        return None, None
    
    # Get movie start frame
    section_time = section_time_data.get(movie_name)
    if section_time is None or len(section_time) == 0:
        logger.debug(f"No section_time for movie '{movie_name}'")
        return None, None
    
    movie_start_sample = section_time[0, 0]
    movie_start_frame = int(convert_sample_index_to_frame(
        np.array([movie_start_sample]), frame_timestamps
    )[0]) + PRE_MARGIN_FRAME_NUM
    
    # Extract spike frames for each unit
    spike_frames_dict: Dict[str, np.ndarray] = {}
    
    for unit_id, unit_data in session.units.items():
        # Check for sectioned spikes
        if "spike_times_sectioned" not in unit_data:
            continue
        
        if movie_name not in unit_data["spike_times_sectioned"]:
            continue
        
        sectioned = unit_data["spike_times_sectioned"][movie_name]
        if "trials_spike_times" not in sectioned or 0 not in sectioned["trials_spike_times"]:
            continue
        
        spike_samples = np.array(sectioned["trials_spike_times"][0])
        spike_frames_absolute = convert_sample_index_to_frame(spike_samples, frame_timestamps)
        spike_frames = (spike_frames_absolute - movie_start_frame).astype(np.int32)
        spike_frames_dict[unit_id] = spike_frames
    
    if len(spike_frames_dict) == 0:
        logger.debug("No spike data found for any unit")
        return None, None
    
    return movie_array, spike_frames_dict

