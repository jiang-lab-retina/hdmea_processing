"""
Pipeline Step Wrappers

This module provides thin wrapper functions around existing implementations
for each of the 11 pipeline steps. Each wrapper:
    - Accepts a PipelineSession object
    - Calls the underlying implementation
    - Marks the step as complete
    - Returns the updated session

Step Pattern:
    Each step function follows this signature:
    
    def step_name(
        *,
        param: type,
        session: PipelineSession,
    ) -> PipelineSession:
        '''Step description.'''
        logger.info("Starting step_name...")
        # Call existing implementation
        session.mark_step_complete("step_name")
        return session

Adding New Steps:
    1. Create a new file in this directory (e.g., my_step.py)
    2. Follow the pattern in template.py
    3. Import and add to __all__ in this file
    4. The step can be debugged independently by importing and calling directly

Available Steps:
    - load_recording_step: Load CMCR/CMTR with eimage_sta
    - add_section_time_step: Add section timing from playlist
    - add_section_time_analog_step: Add section timing from analog signal (raw_ch1)
    - section_spike_times_step: Section spike times
    - section_spike_times_analog_step: Section spike times using sample indices (no frame conversion)
    - compute_sta_step: Compute STA
    - add_metadata_step: Add CMTR/CMCR metadata
    - extract_soma_geometry_step: Extract soma geometry
    - extract_rf_geometry_step: Extract RF-STA geometry
    - add_gsheet_step: Load Google Sheet metadata
    - add_cell_type_step: Add manual cell type labels
    - compute_ap_tracking_step: Compute AP tracking
    - electrode_alignment_step: Align electrode coords to stimulus space
    - section_by_direction_step: Section by direction (DSGC)
    - set_6_compatibility_step: Create set6-compatible data from set6a recordings
"""

# Step imports
from .load_recording import load_recording_step
from .section_time import add_section_time_step, section_spike_times_step, compute_sta_step
from .section_time_analog import add_section_time_analog_step
from .section_spike_times_analog import section_spike_times_analog_step
from .metadata import add_metadata_step
from .geometry import extract_soma_geometry_step, extract_rf_geometry_step
from .gsheet import add_gsheet_step
from .cell_type import add_cell_type_step
from .ap_tracking import compute_ap_tracking_step
from .electrode_alignment import electrode_alignment_step
from .dsgc import section_by_direction_step
from .set_6_compatibility import set_6_compatibility_step

__all__ = [
    # Step 1
    "load_recording_step",
    # Steps 2-4
    "add_section_time_step",
    "section_spike_times_step",
    "compute_sta_step",
    # Analog section time (standalone steps)
    "add_section_time_analog_step",
    "section_spike_times_analog_step",
    # Step 5
    "add_metadata_step",
    # Steps 6-7
    "extract_soma_geometry_step",
    "extract_rf_geometry_step",
    # Step 8
    "add_gsheet_step",
    # Step 9
    "add_cell_type_step",
    # Step 10
    "compute_ap_tracking_step",
    # Step 11 - Electrode alignment
    "electrode_alignment_step",
    # Step 12 - DSGC
    "section_by_direction_step",
    # Set6 compatibility
    "set_6_compatibility_step",
    # Utilities
    "skip_if_completed",
]


def skip_if_completed(step_name: str):
    """
    Decorator to skip step if already completed in session.
    
    Usage:
        @skip_if_completed("my_step")
        def my_step(*, session: PipelineSession, ...) -> PipelineSession:
            ...
    """
    def decorator(func):
        def wrapper(*args, session=None, **kwargs):
            if session is None:
                raise ValueError("session parameter is required")
            
            if step_name in session.completed_steps:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Skipping {step_name} - already completed")
                return session
            
            return func(*args, session=session, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator

