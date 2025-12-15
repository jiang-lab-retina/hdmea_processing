"""
API Contract: Metadata Fields for acquisition_rate and frame_time

This module defines the expected interface for metadata extraction and storage.
It serves as a contract for implementation - actual implementation may differ
but must satisfy this interface.

Feature Branch: 003-add-metadata-fields
"""

from typing import Any, Dict, Optional
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

DEFAULT_ACQUISITION_RATE: float = 20000.0  # Hz
MIN_VALID_ACQUISITION_RATE: float = 0.0  # Must be > 0
WARN_MIN_ACQUISITION_RATE: float = 1000.0  # Hz - warn if below
WARN_MAX_ACQUISITION_RATE: float = 100000.0  # Hz - warn if above


# =============================================================================
# CMTR Extraction (New Function)
# =============================================================================

def extract_acquisition_rate_from_cmtr(cmtr_path: Path) -> Optional[float]:
    """
    Extract acquisition_rate from CMTR file metadata.
    
    This function is the fallback when CMCR extraction fails or CMCR is unavailable.
    
    Args:
        cmtr_path: Path to .cmtr file
    
    Returns:
        Acquisition rate in Hz if found in metadata, None otherwise.
        
    Contract:
        - MUST return None if acquisition_rate cannot be determined
        - MUST NOT raise exceptions for missing metadata
        - MUST return positive float if value is found
    """
    ...


# =============================================================================
# Validation (New Function)
# =============================================================================

def validate_acquisition_rate(rate: float) -> bool:
    """
    Validate that acquisition_rate is acceptable.
    
    Args:
        rate: Proposed acquisition rate in Hz
    
    Returns:
        True if rate is valid (> 0), False otherwise.
        
    Contract:
        - MUST return False if rate <= 0
        - MUST return True if rate > 0
        - SHOULD log warning if rate outside typical range (1000-100000 Hz)
    """
    ...


# =============================================================================
# frame_time Computation (New Function)
# =============================================================================

def compute_frame_time(acquisition_rate: float) -> float:
    """
    Compute frame_time from acquisition_rate.
    
    Args:
        acquisition_rate: Sampling rate in Hz (must be > 0)
    
    Returns:
        Frame time in seconds (1 / acquisition_rate)
    
    Raises:
        ValueError: If acquisition_rate <= 0
        
    Contract:
        - MUST return 1.0 / acquisition_rate
        - MUST raise ValueError if acquisition_rate <= 0
    """
    ...


# =============================================================================
# Metadata Dict Structure
# =============================================================================

def get_timing_metadata(
    cmcr_result: Optional[Dict[str, Any]],
    cmtr_result: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Get timing metadata using priority chain.
    
    Priority: CMCR → CMTR → default
    
    Args:
        cmcr_result: Result dict from load_cmcr_data() or None
        cmtr_result: Result dict from load_cmtr_data() or None
    
    Returns:
        Dict with keys:
            - acquisition_rate: float (Hz)
            - frame_time: float (seconds)
            
    Contract:
        - MUST always return both keys
        - MUST use CMCR acquisition_rate if valid
        - MUST fall back to CMTR if CMCR unavailable or invalid
        - MUST fall back to DEFAULT_ACQUISITION_RATE as last resort
        - MUST log warning when using default
    """
    ...

