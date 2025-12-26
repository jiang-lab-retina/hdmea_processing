"""
Soma detection from 3D Spike-Triggered Average (STA) data.

This module provides functions to detect the soma (cell body) location
from 3D STA data by finding electrodes with high temporal variability.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def find_soma_from_3d_sta(
    sta: np.ndarray,
    std_threshold: float = 3.0,
    sta_temporal_range: Tuple[int, int] = (5, 27),
) -> Tuple[int, int]:
    """
    Detect soma center from 3D STA data.

    The algorithm finds electrodes with high temporal variability (std > threshold * overall_std)
    and returns the centroid of these responsive electrodes. If no electrodes pass the threshold,
    falls back to the electrode with maximum standard deviation.

    Args:
        sta: 3D STA array with shape (time, row, col)
        std_threshold: Number of standard deviations above overall std for detection
        sta_temporal_range: Time range (start, end) to analyze for soma detection

    Returns:
        Tuple of (row, col) for detected soma center

    Example:
        >>> sta = np.random.randn(50, 65, 65)
        >>> row, col = find_soma_from_3d_sta(sta)
    """
    # Extract temporal window for soma detection
    t_start, t_end = sta_temporal_range
    soma_sta = sta[t_start:t_end]

    # Calculate overall standard deviation
    overall_std = soma_sta.std()

    # Get the standard deviation along the time axis for each spatial location
    stds = soma_sta.std(axis=0)

    # Find electrodes with high temporal variability
    mask = stds > (std_threshold * overall_std)

    # Use np.argwhere to find the indices where the condition is true
    responsive_electrode = np.argwhere(mask)

    if len(responsive_electrode) > 0:
        # Calculate mean of row and col coordinates
        center = np.mean(responsive_electrode, axis=0)
        # Round to nearest integer
        row, col = np.round(center).astype(int)
        logger.debug(
            f"Soma detected at ({row}, {col}) from {len(responsive_electrode)} responsive electrodes"
        )
    else:
        # Fallback: use electrode with maximum std
        row, col = np.unravel_index(np.argmax(stds), stds.shape)
        logger.debug(f"Soma fallback to max std electrode at ({row}, {col})")

    return int(row), int(col)

