"""
Step Wrappers: Section Time, Section Spike Times, Compute STA

These steps handle stimulus timing and spike time sectioning.
"""

import logging
from typing import Optional, Tuple

from hdmea.pipeline import PipelineSession, add_section_time
from hdmea.io import section_spike_times
from hdmea.features import compute_sta

logger = logging.getLogger(__name__)


def add_section_time_step(
    *,
    playlist_name: str = "play_optimization_set6_ipRGC_manual",
    session: PipelineSession,
) -> PipelineSession:
    """
    Add section timing from playlist.
    
    This is Step 2 of the pipeline.
    
    Args:
        playlist_name: Name of the playlist to use
        session: Pipeline session (required)
    
    Returns:
        Updated session with section timing
    """
    step_name = "add_section_time"
    
    if step_name in session.completed_steps:
        logger.info(f"Skipping {step_name} - already completed")
        return session
    
    logger.info(f"Step 2: Adding section time using playlist '{playlist_name}'...")
    
    session = add_section_time(
        playlist_name=playlist_name,
        session=session,
    )
    
    session.mark_step_complete(step_name)
    logger.info(f"  Section time added")
    
    return session


def section_spike_times_step(
    *,
    pad_margin: Tuple[float, float] = (0.0, 0.0),
    session: PipelineSession,
) -> PipelineSession:
    """
    Section spike times based on stimulus timing.
    
    This is Step 3 of the pipeline.
    
    Args:
        pad_margin: Padding margins (before, after) in seconds
        session: Pipeline session (required)
    
    Returns:
        Updated session with sectioned spike times
    """
    step_name = "section_spike_times"
    
    if step_name in session.completed_steps:
        logger.info(f"Skipping {step_name} - already completed")
        return session
    
    logger.info(f"Step 3: Sectioning spike times...")
    
    session = section_spike_times(
        pad_margin=pad_margin,
        session=session,
    )
    
    session.mark_step_complete(step_name)
    logger.info(f"  Spike times sectioned")
    
    return session


def compute_sta_step(
    *,
    cover_range: Tuple[int, int] = (-60, 0),
    session: PipelineSession,
) -> PipelineSession:
    """
    Compute Spike-Triggered Average (STA).
    
    This is Step 4 of the pipeline.
    
    Args:
        cover_range: Range of frames for STA computation
        session: Pipeline session (required)
    
    Returns:
        Updated session with computed STA
    """
    step_name = "compute_sta"
    
    if step_name in session.completed_steps:
        logger.info(f"Skipping {step_name} - already completed")
        return session
    
    logger.info(f"Step 4: Computing STA...")
    
    session = compute_sta(
        cover_range=cover_range,
        session=session,
    )
    
    session.mark_step_complete(step_name)
    logger.info(f"  STA computed")
    
    return session

