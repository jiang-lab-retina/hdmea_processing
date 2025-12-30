"""
Step Wrapper: Section by Direction (DSGC) - Deferred Mode

Sections spike responses by stimulus direction for DSGC analysis.
Works entirely in session/deferred mode.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np

from hdmea.pipeline import PipelineSession
from Projects.unified_pipeline.config import (
    DSGC_MOVIE_NAME,
    DSGC_ON_OFF_DICT_PATH,
    DSGC_PADDING_FRAMES,
    DSGCConfig,
    red_warning,
    PRE_MARGIN_FRAME_NUM,
)

logger = logging.getLogger(__name__)

STEP_NAME = "section_by_direction"

# Direction list (in degrees) matching the stimulus
DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8
N_REPETITIONS = 3
N_TRIALS = 24  # 8 directions × 3 repetitions

# Coordinate scale factor from 15x15 STA grid to 300x300 stimulus space
COORDINATE_SCALE_FACTOR = 20


def load_on_off_dict(path: Path) -> Dict[Tuple[int, int], Dict[str, List[int]]]:
    """
    Load on/off timing dictionary from pickle file.
    
    Returns:
        Dict with (row, col) keys and {'on_peak_location': [...], 'off_peak_location': [...]} values
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def convert_center_15_to_300(center_row_15: float, center_col_15: float) -> Tuple[int, int]:
    """
    Convert STA center from 15×15 grid to 300×300 stimulus pixel space.
    
    Args:
        center_row_15: Row coordinate in 15×15 STA space (0-14)
        center_col_15: Column coordinate in 15×15 STA space (0-14)
    
    Returns:
        (row_300, col_300) in 300×300 stimulus space
    """
    row_300 = int(center_row_15 * COORDINATE_SCALE_FACTOR + COORDINATE_SCALE_FACTOR // 2)
    col_300 = int(center_col_15 * COORDINATE_SCALE_FACTOR + COORDINATE_SCALE_FACTOR // 2)
    
    # Clip to valid range
    row_300 = max(0, min(299, row_300))
    col_300 = max(0, min(299, col_300))
    
    return (row_300, col_300)


def get_sta_geometry_from_session(session: PipelineSession, unit_id: str) -> Optional[Tuple[int, int]]:
    """
    Get STA geometry center (row, col) in 300×300 stimulus space from session.
    
    Uses gaussian_fit center coordinates for more accurate positioning:
    - gaussian_fit/center_y -> row (in 15×15 space)
    - gaussian_fit/center_x -> col (in 15×15 space)
    
    Returns:
        (row_300, col_300) in stimulus pixel space, or None if not found
    """
    unit_data = session.units.get(unit_id, {})
    features = unit_data.get('features', {})
    
    # Try sta_perfect_dense_noise gaussian_fit center first (most accurate)
    for sta_name in ['sta_perfect_dense_noise_15x15_15hz_r42_3min', 'sta']:
        sta_feature = features.get(sta_name, {})
        sta_geometry = sta_feature.get('sta_geometry', {})
        gaussian_fit = sta_geometry.get('gaussian_fit', {})
        
        # Use gaussian_fit: center_y is row, center_x is col
        center_row = gaussian_fit.get('center_y')
        center_col = gaussian_fit.get('center_x')
        
        if center_row is not None and center_col is not None:
            # Handle numpy arrays
            if isinstance(center_row, np.ndarray):
                center_row = float(center_row.flat[0]) if center_row.size > 0 else None
            if isinstance(center_col, np.ndarray):
                center_col = float(center_col.flat[0]) if center_col.size > 0 else None
            
            # Skip if NaN
            if center_row is not None and center_col is not None:
                if not (np.isnan(center_row) or np.isnan(center_col)):
                    # Convert from 15×15 to 300×300 space
                    return convert_center_15_to_300(center_row, center_col)
        
        # Fallback to sta_geometry center_row/center_col if gaussian_fit not available
        center_row = sta_geometry.get('center_row')
        center_col = sta_geometry.get('center_col')
        
        if center_row is not None and center_col is not None:
            # Handle numpy arrays
            if isinstance(center_row, np.ndarray):
                center_row = float(center_row.flat[0]) if center_row.size > 0 else None
            if isinstance(center_col, np.ndarray):
                center_col = float(center_col.flat[0]) if center_col.size > 0 else None
            
            if center_row is not None and center_col is not None:
                if not (np.isnan(center_row) or np.isnan(center_col)):
                    # Convert from 15×15 to 300×300 space
                    return convert_center_15_to_300(center_row, center_col)
    
    # Try eimage_sta corrected coordinates if electrode_alignment was done
    eimage_sta = features.get('eimage_sta', {})
    geometry = eimage_sta.get('geometry', {})
    
    # First try corrected coordinates (already in 300×300 space)
    corrected_row = geometry.get('center_corrected_row')
    corrected_col = geometry.get('center_corrected_col')
    
    if corrected_row is not None and corrected_col is not None:
        if isinstance(corrected_row, np.ndarray):
            corrected_row = float(corrected_row.flat[0]) if corrected_row.size > 0 else None
        if isinstance(corrected_col, np.ndarray):
            corrected_col = float(corrected_col.flat[0]) if corrected_col.size > 0 else None
        
        if corrected_row is not None and corrected_col is not None:
            if not (np.isnan(corrected_row) or np.isnan(corrected_col)):
                # Already in 300×300 space, just round to int
                return (int(round(corrected_row)), int(round(corrected_col)))
    
    return None


def section_by_direction_step(
    *,
    movie_name: Optional[str] = None,
    on_off_dict_path: Optional[Path] = None,
    padding_frames: Optional[int] = None,
    config: Optional[DSGCConfig] = None,
    session: PipelineSession,
) -> PipelineSession:
    """
    Section spike responses by stimulus direction (deferred mode).
    
    This is Step 11 of the pipeline (final step).
    
    Works entirely in session/deferred mode - reads data from session
    and stores results back to session.
    
    Args:
        movie_name: Name of the movie stimulus
        on_off_dict_path: Path to on/off dictionary file
        padding_frames: Number of padding frames
        config: DSGC configuration
        session: Pipeline session (required)
    
    Returns:
        Updated session with DSGC sectioning results
    """
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME} - already completed")
        return session
    
    logger.info(f"Step 11: Section by direction (DSGC, deferred)...")
    
    # Use config or individual parameters
    if config is None:
        config = DSGCConfig()
    
    if movie_name is None:
        movie_name = config.movie_name
    if on_off_dict_path is None:
        on_off_dict_path = config.on_off_dict_path
    if padding_frames is None:
        padding_frames = config.padding_frames
    
    # Check if on_off_dict exists
    if not on_off_dict_path.exists():
        logger.warning(red_warning(f"  On/off dict not found: {on_off_dict_path}"))
        session.warnings.append(f"{STEP_NAME}: On/off dict not found")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
        return session
    
    try:
        from tqdm import tqdm
        
        # Load on/off dictionary
        on_off_dict = load_on_off_dict(on_off_dict_path)
        logger.info(f"  Loaded on/off dictionary: {len(on_off_dict)} pixels")
        
        # Get frame_timestamps from session
        frame_timestamps = session.metadata.get('frame_timestamps')
        if frame_timestamps is None:
            frame_time_data = session.stimulus.get('frame_time', {})
            if isinstance(frame_time_data, dict):
                frame_timestamps = frame_time_data.get('default')
            elif isinstance(frame_time_data, np.ndarray):
                frame_timestamps = frame_time_data
        
        if frame_timestamps is None:
            logger.warning(red_warning(f"  No frame_timestamps in session"))
            session.warnings.append(f"{STEP_NAME}: No frame_timestamps")
            session.completed_steps.add(f"{STEP_NAME}:skipped")
            return session
        
        if isinstance(frame_timestamps, dict):
            frame_timestamps = frame_timestamps.get('data', frame_timestamps)
        
        frame_timestamps = np.asarray(frame_timestamps)
        
        # Get section_time for movie
        section_time_data = session.stimulus.get('section_time', {})
        if isinstance(section_time_data, np.ndarray):
            logger.warning(red_warning(f"  section_time is not a dict"))
            session.warnings.append(f"{STEP_NAME}: section_time format error")
            session.completed_steps.add(f"{STEP_NAME}:skipped")
            return session
        
        section_time = section_time_data.get(movie_name) if isinstance(section_time_data, dict) else None
        if section_time is None:
            logger.warning(red_warning(f"  No section_time for movie '{movie_name}'"))
            session.warnings.append(f"{STEP_NAME}: No section_time for {movie_name}")
            session.completed_steps.add(f"{STEP_NAME}:skipped")
            return session
        
        if isinstance(section_time, dict):
            section_time = section_time.get('data', section_time)
        
        section_time = np.asarray(section_time) # TODO: section_time contains PRE_MARGIN_FRAME_NUM, so we need to subtract it when converting to movie-relative frames
        movie_start_sample = section_time[0, 0]
        
        # Find movie start frame
        movie_start_frame = np.searchsorted(frame_timestamps, movie_start_sample)
        movie_start_frame = movie_start_frame + PRE_MARGIN_FRAME_NUM
        logger.info(f"  Movie '{movie_name}' starts at frame {movie_start_frame}")
        
        # Process each unit
        units_processed = 0
        units_skipped = 0
        
        unit_ids = list(session.units.keys())
        logger.info(f"  Processing {len(unit_ids)} units")
        
        for unit_id in tqdm(unit_ids, desc="DSGC sectioning"):
            unit_data = session.units.get(unit_id, {})
            
            # Get STA geometry for center position
            center = get_sta_geometry_from_session(session, unit_id)
            if center is None:
                logger.debug(f"  No STA geometry for {unit_id}, skipping")
                units_skipped += 1
                continue
            
            center_row, center_col = center
            
            # Get on/off times for this pixel
            # Key format: (row, col) - matches on_off_dict keys
            pixel_key = (center_row, center_col)
            if pixel_key not in on_off_dict:
                logger.debug(f"  No on/off data for pixel {center} in {unit_id}")
                units_skipped += 1
                continue
            
            pixel_timing = on_off_dict[pixel_key]
            on_times = pixel_timing['on_peak_location']
            off_times = pixel_timing['off_peak_location']
            
            # Get spike times for this movie from spike_times_sectioned
            sectioned_data = unit_data.get('spike_times_sectioned', {})
            if isinstance(sectioned_data, np.ndarray):
                logger.debug(f"  spike_times_sectioned is array (not dict) for {unit_id}, skipping")
                units_skipped += 1
                continue
            
            movie_sectioned = sectioned_data.get(movie_name) if isinstance(sectioned_data, dict) else None
            if movie_sectioned is None:
                logger.debug(f"  No sectioned spike times for movie {movie_name} in {unit_id}")
                units_skipped += 1
                continue
            
            # Get full spike times for this movie
            if isinstance(movie_sectioned, dict):
                spike_times_movie = movie_sectioned.get('full_spike_times')
            elif isinstance(movie_sectioned, np.ndarray):
                spike_times_movie = movie_sectioned
            else:
                logger.debug(f"  Unexpected format for sectioned data in {unit_id}")
                units_skipped += 1
                continue
            
            if spike_times_movie is None:
                logger.debug(f"  No full_spike_times for movie {movie_name} in {unit_id}")
                units_skipped += 1
                continue
            
            spike_times_movie = np.asarray(spike_times_movie)
            
            # Convert spike times from samples to movie-relative frames
            # First convert to absolute frames
            spike_frames = np.searchsorted(frame_timestamps, spike_times_movie, side='right') - 1
            # Then make movie-relative by subtracting movie start
            spike_frames_movie = spike_frames - movie_start_frame
            
            # Section spikes by direction (8 directions × 3 repetitions = 24 trials)
            # Store results matching the reference format
            # The reference uses direction as string keys (e.g., "0", "45", "90")
            # with "trials" subgroup containing "0", "1", "2" for each repetition
            direction_section_data: Dict[str, Any] = {}
            
            # Add metadata as attributes (will be stored by session saver)
            direction_section_data['_attrs'] = {
                'direction_list': DIRECTION_LIST,
                'n_directions': N_DIRECTIONS,
                'n_repetitions': N_REPETITIONS,
                'padding_frames': padding_frames,
                'cell_center_row': center_row,
                'cell_center_col': center_col,
            }
            
            # Initialize storage for each direction
            for direction in DIRECTION_LIST:
                direction_section_data[str(direction)] = {
                    'trials': {},  # Will contain "0", "1", "2" keys
                    'section_bounds': [],  # Will be (3, 2) array
                }
            
            # Process each of the 24 trials
            for trial_idx in range(N_TRIALS):
                # Map trial index to direction and repetition
                direction_idx = trial_idx % N_DIRECTIONS
                rep_idx = trial_idx // N_DIRECTIONS
                direction = DIRECTION_LIST[direction_idx]
                
                # Get window bounds with padding
                on_frame = on_times[trial_idx]
                off_frame = off_times[trial_idx]
                start_frame = on_frame - padding_frames
                end_frame = off_frame + padding_frames
                
                # Extract spikes in window (movie-relative frames)
                mask = (spike_frames_movie >= start_frame) & (spike_frames_movie <= end_frame)
                trial_spikes = spike_frames_movie[mask]
                
                # Store results - each trial as separate array under "trials/{rep_idx}"
                direction_section_data[str(direction)]['trials'][str(rep_idx)] = trial_spikes.astype(np.int64)
                direction_section_data[str(direction)]['section_bounds'].append([int(start_frame), int(end_frame)])
            
            # Convert bounds lists to arrays
            for direction in DIRECTION_LIST:
                bounds = direction_section_data[str(direction)]['section_bounds']
                direction_section_data[str(direction)]['section_bounds'] = np.array(bounds, dtype=np.int64)
            
            # Store results in session at the correct location:
            # spike_times_sectioned/{movie_name}/direction_section
            if 'spike_times_sectioned' not in session.units[unit_id]:
                session.units[unit_id]['spike_times_sectioned'] = {}
            
            if movie_name not in session.units[unit_id]['spike_times_sectioned']:
                session.units[unit_id]['spike_times_sectioned'][movie_name] = {}
            
            session.units[unit_id]['spike_times_sectioned'][movie_name]['direction_section'] = direction_section_data
            
            units_processed += 1
        
        logger.info(f"  Processed: {units_processed} units, Skipped: {units_skipped}")
        session.mark_step_complete(STEP_NAME)
        
    except ImportError as e:
        logger.warning(red_warning(f"  Cannot import required module: {e}"))
        session.warnings.append(f"{STEP_NAME}: Import error - {e}")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(red_warning(f"  Error in DSGC section: {e}"))
        logger.error(f"  Traceback:\n{tb}")
        session.warnings.append(f"{STEP_NAME}: {e}")
        session.completed_steps.add(f"{STEP_NAME}:failed")
    
    return session
