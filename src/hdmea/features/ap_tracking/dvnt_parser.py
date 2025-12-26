"""
DVNT (Dorsal-Ventral, Nasal-Temporal) position parsing from HDF5 metadata.

This module provides functions to parse anatomical position information
from the Center_xy metadata field in HDF5 recordings.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DVNTPosition:
    """Anatomical position derived from recording metadata."""

    dv_position: Optional[float]  # Dorsal-ventral (positive = dorsal)
    nt_position: Optional[float]  # Nasal-temporal (positive = nasal)
    lr_position: Optional[str]  # Left/Right eye ("L" or "R")


def parse_dvnt_from_center_xy(center_xy: Optional[str]) -> DVNTPosition:
    """
    Parse DVNT position from Center_xy metadata string.

    The Center_xy format is: "L/R, VD_coord, NT_coord"
    where:
    - L/R: Left or Right eye
    - VD_coord: Ventral-Dorsal coordinate (ventral is positive as labeled by Yan Zhu)
    - NT_coord: Nasal-Temporal coordinate (nasal is positive)

    The conversion applies:
    - DV_position = -VD_coord (so positive = dorsal, negative = ventral)
    - NT_position = NT_coord (positive = nasal, negative = temporal)
    - LR_position = L/R string

    Args:
        center_xy: String in format "L/R, VD_coord, NT_coord" (e.g., "L, 1.5, -0.8")
                   or None if metadata is missing

    Returns:
        DVNTPosition with parsed values (None for unparseable fields)

    Example:
        >>> result = parse_dvnt_from_center_xy("L, 1.5, -0.8")
        >>> result.dv_position  # -1.5 (ventral)
        >>> result.nt_position  # -0.8 (temporal)
        >>> result.lr_position  # "L"
    """
    if center_xy is None or center_xy == "":
        logger.debug("Center_xy is empty or None, returning empty DVNTPosition")
        return DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

    try:
        # Remove whitespace
        position_string = center_xy.replace(" ", "")

        if not position_string:
            return DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

        parts = position_string.split(",")

        if len(parts) < 3:
            logger.warning(
                f"Center_xy has unexpected format (expected 3 parts, got {len(parts)}): {center_xy}"
            )
            return DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

        # Parse LR position
        lr_position = parts[0].upper()
        if lr_position not in ("L", "R"):
            logger.warning(f"Invalid LR position '{lr_position}', expected 'L' or 'R'")

        # Parse VD coordinate (ventral is positive as labeled)
        try:
            vd_coordinate = float(parts[1])
            # Reverse so positive = dorsal, negative = ventral
            dv_position = -vd_coordinate
        except ValueError:
            logger.warning(f"Could not parse VD coordinate: {parts[1]}")
            dv_position = None

        # Parse NT coordinate (nasal is positive)
        try:
            nt_position = float(parts[2])
        except ValueError:
            logger.warning(f"Could not parse NT coordinate: {parts[2]}")
            nt_position = None

        logger.debug(
            f"Parsed DVNT: LR={lr_position}, DV={dv_position}, NT={nt_position}"
        )

        return DVNTPosition(
            dv_position=dv_position,
            nt_position=nt_position,
            lr_position=lr_position,
        )

    except Exception as e:
        logger.error(f"Error parsing Center_xy '{center_xy}': {e}")
        return DVNTPosition(dv_position=None, nt_position=None, lr_position=None)

