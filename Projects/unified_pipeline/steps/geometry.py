"""
Step Wrappers: Extract Soma Geometry, Extract RF Geometry

These steps extract geometric features from STA data.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

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
    session: PipelineSession,
) -> PipelineSession:
    """
    Extract RF-STA geometry (Gaussian, DoG, ON/OFF fits).
    
    This is Step 7 of the pipeline.
    
    Args:
        frame_range: Frames to use for analysis
        threshold_fraction: Threshold fraction for fitting
        sta_feature_name: Name of the STA feature to analyze
        session: Pipeline session (required)
    
    Returns:
        Updated session with RF geometry
    """
    step_name = "extract_rf_geometry"
    
    if step_name in session.completed_steps:
        logger.info(f"Skipping {step_name} - already completed")
        return session
    
    logger.info(f"Step 7: Extracting RF-STA geometry...")
    
    try:
        from rf_session import extract_rf_geometry_session
        
        session = extract_rf_geometry_session(
            session,
            frame_range=frame_range,
            threshold_fraction=threshold_fraction,
        )
        
        # Add rf_sta_geometry metadata (matches reference file structure)
        session.metadata['rf_sta_geometry'] = {
            'sta_feature_name': sta_feature_name,
            'frame_range': list(frame_range),
            'threshold_fraction': threshold_fraction,
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

