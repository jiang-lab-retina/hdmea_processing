"""
Unified Pipeline for HD-MEA Data Processing

This module provides a unified pipeline that processes HD-MEA recordings
from CMCR/CMTR source files or existing HDF5 files through all analysis steps.

Entry Points:
    - run_single_from_cmcr.py: Process recording from raw CMCR/CMTR files
    - run_single_from_hdf5.py: Resume processing from existing HDF5 file

Pipeline Steps (11 total):
    1. Load recording with eimage_sta
    2. Add section time using playlist
    3. Section spike times
    4. Compute STA
    5. Add CMTR/CMCR metadata (unit_meta, sys_meta)
    6. Extract soma geometry
    7. Extract RF-STA geometry (Gaussian, DoG, ON/OFF fits)
    8. Load Google Sheet metadata
    9. Add manual cell type labels
    10. Compute AP tracking
    11. Section by direction (DSGC)

Usage:
    from Projects.unified_pipeline.steps import (
        load_recording_step,
        add_section_time_step,
        compute_sta_step,
        ...
    )
    
    session = create_session(dataset_id="2024.08.08-10.40.20-Rec")
    session = load_recording_step(cmcr_path, cmtr_path, session=session)
    session = add_section_time_step(playlist_name, session=session)
    ...
    session.save(output_path)

See Also:
    - specs/012-unified-pipeline-session/quickstart.md for detailed examples
    - Projects/pipeline_principle.md for design principles
"""

__version__ = "0.1.0"

