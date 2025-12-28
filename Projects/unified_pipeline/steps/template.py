"""
Step Template: Use this as a template for creating new pipeline steps.

Copy this file and modify it to add new analysis steps to the pipeline.

INSTRUCTIONS:
1. Copy this file to a new file (e.g., my_analysis.py)
2. Update STEP_NAME constant
3. Modify the function signature and parameters
4. Implement the step logic (call existing implementation)
5. Add the new step to __init__.py imports and __all__
6. Use the step in run_single_from_cmcr.py or run_single_from_hdf5.py

PATTERN:
- Each step wrapper takes a PipelineSession and returns a PipelineSession
- Use session.completed_steps to check/mark step completion
- Use session.warnings to record non-fatal issues
- Log progress using logger.info()
- Use red_warning() for visible warnings when external dependencies fail

DEBUGGING:
- Import the step function directly and call with a minimal session
- Use load_session_from_hdf5() to load test data
- Run steps in isolation before adding to full pipeline

Example standalone debugging:
    >>> from hdmea.pipeline import load_session_from_hdf5
    >>> from Projects.unified_pipeline.steps.my_analysis import my_analysis_step
    >>> session = load_session_from_hdf5("test_data.h5")
    >>> session = my_analysis_step(param1="value", session=session)
"""

import logging
from pathlib import Path
from typing import Optional

from hdmea.pipeline import PipelineSession
from Projects.unified_pipeline.config import red_warning

logger = logging.getLogger(__name__)

# Unique identifier for this step (used for skip detection and logging)
STEP_NAME = "my_step_name"


def my_step_template(
    *,
    param1: str = "default_value",
    param2: Optional[int] = None,
    session: PipelineSession,
) -> PipelineSession:
    """
    Brief description of what this step does.
    
    This is Step N of the pipeline.
    
    More detailed description of the step's purpose, inputs, outputs,
    and any side effects.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (optional)
        session: Pipeline session (required, keyword-only)
    
    Returns:
        Updated session with [description of what was added/modified]
    
    Raises:
        ImportError: If required dependencies are not available
        ValueError: If input data is invalid
    
    Example:
        >>> session = my_step_template(
        ...     param1="custom_value",
        ...     session=session,
        ... )
    """
    # =========================================================================
    # 1. Check if step already completed (skip if so)
    # =========================================================================
    
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME} - already completed")
        return session
    
    logger.info(f"Step N: Description of step...")
    
    # =========================================================================
    # 2. Validate prerequisites (return early with warning if missing)
    # =========================================================================
    
    # Example: Check if HDF5 file is needed
    # if session.hdf5_path is None or not session.hdf5_path.exists():
    #     logger.warning(red_warning(f"Cannot run {STEP_NAME} - session not saved"))
    #     session.warnings.append(f"{STEP_NAME}: No HDF5 path")
    #     session.completed_steps.add(f"{STEP_NAME}:skipped")
    #     return session
    
    # =========================================================================
    # 3. Import and call the underlying implementation
    # =========================================================================
    
    try:
        # Import the actual implementation
        # from my_module import my_implementation
        
        # Call the implementation
        # result = my_implementation(
        #     input_data=session.units,
        #     param1=param1,
        #     param2=param2,
        # )
        
        # Store results in session
        # Option A: Store per-unit features
        # for unit_id, unit_result in result.items():
        #     if unit_id in session.units:
        #         if 'features' not in session.units[unit_id]:
        #             session.units[unit_id]['features'] = {}
        #         session.units[unit_id]['features']['my_feature'] = unit_result
        
        # Option B: Store metadata
        # session.metadata['my_step_result'] = result
        
        # Mark step as complete
        session.mark_step_complete(STEP_NAME)
        logger.info(f"  Step complete")
        
    except ImportError as e:
        # Handle missing dependencies gracefully
        logger.warning(red_warning(f"  Cannot import required module: {e}"))
        session.warnings.append(f"{STEP_NAME}: Import error - {e}")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
    
    except FileNotFoundError as e:
        # Handle missing files gracefully
        logger.warning(red_warning(f"  Required file not found: {e}"))
        session.warnings.append(f"{STEP_NAME}: {e}")
        session.completed_steps.add(f"{STEP_NAME}:skipped")
    
    except Exception as e:
        # Handle other errors - log but don't crash the pipeline
        logger.error(red_warning(f"  Error in {STEP_NAME}: {e}"))
        session.warnings.append(f"{STEP_NAME}: {e}")
        session.completed_steps.add(f"{STEP_NAME}:failed")
    
    return session


# =============================================================================
# CHECKLIST for New Steps
# =============================================================================
# 
# Before considering your step complete, verify:
#
# [ ] Step has unique STEP_NAME constant
# [ ] Step function uses keyword-only session parameter
# [ ] Step checks session.completed_steps and skips if done
# [ ] Step marks itself complete with session.mark_step_complete()
# [ ] Step handles ImportError gracefully (for optional dependencies)
# [ ] Step handles FileNotFoundError gracefully (for optional files)
# [ ] Step logs progress with logger.info()
# [ ] Step uses red_warning() for visible warnings
# [ ] Step is added to __init__.py imports and __all__
# [ ] Step can be tested independently (debug instructions in docstring)
# [ ] Step is documented in README.md
#

