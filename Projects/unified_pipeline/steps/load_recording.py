"""
Step Wrapper: Load Recording with EImage STA

Loads CMCR/CMTR recording files and computes eimage_sta for all units.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

from hdmea.pipeline import PipelineSession, load_recording_with_eimage_sta

logger = logging.getLogger(__name__)

STEP_NAME = "load_recording_with_eimage_sta"


def load_recording_step(
    cmcr_path: Union[str, Path],
    cmtr_path: Union[str, Path],
    *,
    duration_s: float = 120.0,
    spike_limit: int = 10000,
    window_range: Tuple[int, int] = (-10, 40),
    session: PipelineSession,
) -> PipelineSession:
    """
    Load CMCR/CMTR recording and compute eimage_sta.
    
    This is Step 1 of the pipeline.
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
        duration_s: Duration to load in seconds
        spike_limit: Maximum spikes per unit
        window_range: STA window range in frames
        session: Pipeline session (required)
    
    Returns:
        Updated session with loaded recording data
    """
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME} - already completed")
        return session
    
    logger.info(f"Step 1: Loading recording from CMCR/CMTR...")
    
    # Call the existing implementation
    session = load_recording_with_eimage_sta(
        cmcr_path=str(cmcr_path),
        cmtr_path=str(cmtr_path),
        duration_s=duration_s,
        spike_limit=spike_limit,
        window_range=window_range,
        session=session,
    )
    
    logger.info(f"  Loaded {session.unit_count} units")
    session.mark_step_complete(STEP_NAME)
    
    return session

