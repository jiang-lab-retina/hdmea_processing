"""
Step Wrapper: Add CMTR/CMCR Metadata

Adds extended metadata from source CMTR/CMCR files to the session.
Works in deferred mode - modifies session in memory.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from hdmea.pipeline import PipelineSession
from hdmea.io.cmtr import add_cmtr_unit_info
from hdmea.io.cmcr import add_sys_meta_info

logger = logging.getLogger(__name__)

STEP_NAME = "add_cmtr_cmcr_metadata"


def add_metadata_step(
    *,
    cmcr_path: Optional[Union[str, Path]] = None,
    cmtr_path: Optional[Union[str, Path]] = None,
    session: PipelineSession,
) -> PipelineSession:
    """
    Add CMTR/CMCR metadata to the session (deferred mode).
    
    This is Step 5 of the pipeline.
    
    Adds:
        - Extended CMTR unit metadata (row, column, SNR, etc.) -> units/*/unit_meta
        - CMCR file metadata -> metadata/cmcr_meta
        - CMTR file metadata -> metadata/cmtr_meta
    
    Works entirely in session/deferred mode - no HDF5 file required.
    
    Args:
        cmcr_path: Path to CMCR file (uses session.source_files if not provided)
        cmtr_path: Path to CMTR file (uses session.source_files if not provided)
        session: Pipeline session (required)
    
    Returns:
        Updated session with metadata added
    """
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME} - already completed")
        return session
    
    logger.info(f"Step 5: Adding CMTR/CMCR metadata (deferred)...")
    
    # Get paths from session if not provided
    if cmtr_path is None:
        cmtr_path = session.source_files.get("cmtr_path")
    if cmcr_path is None:
        cmcr_path = session.source_files.get("cmcr_path")
    
    try:
        # Add unit metadata from CMTR (session mode)
        if cmtr_path:
            session = add_cmtr_unit_info(
                session=session,
                cmtr_path=cmtr_path,
            )
            logger.info(f"  Added unit_meta from CMTR")
        else:
            logger.warning(f"  No CMTR path available, skipping unit_meta")
        
        # Add system metadata from CMCR/CMTR (session mode)
        if cmcr_path or cmtr_path:
            session = add_sys_meta_info(
                session=session,
                cmcr_path=cmcr_path,
                cmtr_path=cmtr_path,
            )
            logger.info(f"  Added system metadata from CMCR/CMTR")
        
        session.mark_step_complete(STEP_NAME)
        
    except FileNotFoundError as e:
        logger.warning(f"  Source file not found: {e}")
        session.warnings.append(f"{STEP_NAME}: {e}")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
    
    except Exception as e:
        logger.error(f"  Error adding metadata: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        session.warnings.append(f"{STEP_NAME}: {e}")
        session.completed_steps.add(f"{STEP_NAME}:failed")
    
    return session
