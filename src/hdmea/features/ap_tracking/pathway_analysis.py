"""
AP pathway analysis and soma polar coordinate calculation.

This module provides functions to fit lines to axon projections,
calculate optimal intersection points, and compute soma polar coordinates.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class APPathway:
    """Fitted line to axon projection."""

    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float
    num_points: int


@dataclass
class APIntersection:
    """Optimal intersection point of AP pathways."""

    x: float
    y: float
    mse: float  # Mean squared error


@dataclass
class SomaPolarCoordinates:
    """Soma position in polar coordinates (legacy method with angle correction)."""

    # Basic polar coordinates
    radius: float
    angle: float  # Final angle in radians (after correction)

    # Original Cartesian displacement from intersection
    cartesian_x: float  # dx = soma_col - intersection_x
    cartesian_y: float  # dy = soma_row - intersection_y

    # Quadrant labels
    quadrant: str  # "Q1", "Q2", "Q3", "Q4"
    anatomical_quadrant: Optional[str]  # e.g., "dorsal-nasal"

    # Legacy angle fields (degrees)
    theta_deg: Optional[float] = None  # Final angle in degrees
    theta_deg_raw: Optional[float] = None  # Raw angle before correction
    theta_deg_corrected: Optional[float] = None  # After angle correction

    # Transformed coordinates (after angle correction)
    transformed_x: Optional[float] = None
    transformed_y: Optional[float] = None

    # Original soma position
    original_x: Optional[int] = None  # soma row
    original_y: Optional[int] = None  # soma col

    # Angle correction info
    angle_correction_applied: Optional[float] = None  # Correction in degrees


def calculate_projections(post_processed_data: Dict) -> np.ndarray:
    """
    Calculate projections from post-processed axon data.

    Creates a 2D projection by taking max-min along the time axis
    of the filtered prediction data.

    Args:
        post_processed_data: Dictionary with 'filtered_prediction' key

    Returns:
        2D projection array
    """
    if "filtered_prediction" not in post_processed_data:
        logger.warning("No filtered_prediction in post_processed_data")
        return np.zeros((65, 65), dtype=np.float32)

    sta_data = post_processed_data["filtered_prediction"]

    if sta_data.ndim != 3:
        logger.warning(f"Expected 3D data, got {sta_data.ndim}D")
        return np.zeros((65, 65), dtype=np.float32)

    # Max - min along time axis gives projection
    projections = np.max(sta_data, axis=0) - np.min(sta_data, axis=0)

    return projections


def fit_line_to_projections(
    projections: np.ndarray,
    min_points: int = 10,
) -> Optional[APPathway]:
    """
    Fit line to non-zero points in the projections array.

    Uses linear regression to fit a line to the coordinates of
    non-zero pixels in the projection.

    Args:
        projections: 2D array with projection values
        min_points: Minimum points required for fitting

    Returns:
        APPathway with fit parameters, or None if insufficient points
    """
    # Get coordinates of non-zero points
    non_zero_coords = np.where(projections > 0)
    y_coords = non_zero_coords[0]  # Row indices
    x_coords = non_zero_coords[1]  # Column indices

    if len(x_coords) < min_points:
        logger.debug(
            f"Not enough points for fitting: {len(x_coords)} < {min_points}"
        )
        return None

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x_coords, y_coords
        )

        return APPathway(
            slope=float(slope),
            intercept=float(intercept),
            r_value=float(r_value),
            p_value=float(p_value),
            std_err=float(std_err),
            num_points=len(x_coords),
        )
    except Exception as e:
        logger.error(f"Error fitting line to projections: {e}")
        return None


def calculate_optimal_intersection(
    pathways: Dict[str, APPathway],
) -> Optional[APIntersection]:
    """
    Calculate optimal intersection point from multiple pathway fits.

    Uses weighted least squares to find the point that minimizes
    distance to all fitted lines, with R² as weights.

    Args:
        pathways: Dictionary mapping unit_id to APPathway

    Returns:
        APIntersection with (x, y, mse), or None if < 2 pathways
    """
    # Filter out None values
    valid_fits = {k: v for k, v in pathways.items() if v is not None}

    if len(valid_fits) < 2:
        logger.warning(
            f"Need at least 2 valid fits, got {len(valid_fits)}"
        )
        return None

    # Set up linear system: for each line m*x - y + b = 0
    A = []
    b = []
    weights = []

    for unit_id, fit in valid_fits.items():
        m = fit.slope
        c = fit.intercept
        r2 = fit.r_value ** 2

        A.append([m, -1])
        b.append(-c)
        weights.append(r2)

    A = np.array(A)
    b = np.array(b)
    weights = np.array(weights)

    # Apply weights
    W = np.diag(weights)
    A_weighted = W @ A
    b_weighted = W @ b

    try:
        # Solve weighted least squares
        x_opt, residuals, rank, s = np.linalg.lstsq(
            A_weighted, b_weighted, rcond=None
        )

        # Calculate mean squared error
        errors = []
        for unit_id, fit in valid_fits.items():
            m = fit.slope
            c = fit.intercept
            # Distance from point to line
            dist = abs(m * x_opt[0] - x_opt[1] + c) / np.sqrt(1 + m ** 2)
            errors.append(dist ** 2)

        mse = float(np.mean(errors))

        logger.debug(
            f"Optimal intersection at ({x_opt[0]:.2f}, {x_opt[1]:.2f}), MSE={mse:.4f}"
        )

        return APIntersection(x=float(x_opt[0]), y=float(x_opt[1]), mse=mse)

    except Exception as e:
        logger.error(f"Error calculating optimal intersection: {e}")
        return None


def _get_quadrant(x: float, y: float) -> str:
    """Get geometric quadrant from Cartesian coordinates."""
    if x >= 0 and y >= 0:
        return "Q1"
    elif x < 0 and y >= 0:
        return "Q2"
    elif x < 0 and y < 0:
        return "Q3"
    else:
        return "Q4"


def _get_anatomical_quadrant(
    dv_position: Optional[float],
    nt_position: Optional[float],
) -> Optional[str]:
    """
    Get anatomical quadrant label based on DVNT position values.

    Args:
        dv_position: Dorsal-ventral position (positive = dorsal)
        nt_position: Nasal-temporal position (positive = nasal)

    Returns:
        Anatomical quadrant string or None if DVNT not available
    """
    if dv_position is None or nt_position is None:
        return None

    if math.isnan(dv_position) or math.isnan(nt_position):
        return None

    # Determine labels based on DVNT sign
    dv_label = "dorsal" if dv_position > 0 else "ventral"
    nt_label = "nasal" if nt_position > 0 else "temporal"

    return f"{dv_label}-{nt_label}"


def _normalize_angle_deg(angle_deg: float) -> float:
    """Normalize angle to 0-360 range."""
    while angle_deg < 0:
        angle_deg += 360
    while angle_deg >= 360:
        angle_deg -= 360
    return angle_deg


def _calculate_angle_correction(
    intersection: APIntersection,
    dv_position: Optional[float],
    nt_position: Optional[float],
    reference_point: Tuple[float, float] = (33.0, 33.0),
) -> Optional[float]:
    """
    Calculate angle correction using reference point and DVNT position.

    Legacy method: Uses (33, 33) as reference point and DVNT to determine
    the expected angle, then calculates the correction needed.

    Args:
        intersection: The AP pathway intersection point
        dv_position: Dorsal-ventral position (positive = dorsal)
        nt_position: Nasal-temporal position (positive = nasal)
        reference_point: Reference point for angle correction (default: center of 65x65)

    Returns:
        Angle correction in degrees, or None if cannot calculate
    """
    if dv_position is None or nt_position is None:
        return None

    if math.isnan(dv_position) or math.isnan(nt_position):
        return None

    ref_x, ref_y = reference_point

    # Calculate reference point angle relative to intersection
    ref_dx = ref_x - intersection.x
    ref_dy = ref_y - intersection.y
    ref_theta_rad = math.atan2(ref_dy, ref_dx)
    ref_theta_deg = math.degrees(ref_theta_rad)
    ref_theta_deg = _normalize_angle_deg(ref_theta_deg)

    # Calculate expected angle based on DVNT position
    # This gives the anatomical direction of the reference point
    expected_ref_angle = math.degrees(math.atan2(dv_position, nt_position))
    expected_ref_angle = _normalize_angle_deg(expected_ref_angle)

    # Angle correction = expected - actual
    angle_correction = expected_ref_angle - ref_theta_deg

    logger.debug(
        f"Angle correction: ref_theta={ref_theta_deg:.1f}°, "
        f"expected={expected_ref_angle:.1f}°, correction={angle_correction:.1f}°"
    )

    return angle_correction


def calculate_soma_polar_coordinates(
    soma_xy: Tuple[int, int],
    intersection: APIntersection,
    dv_position: Optional[float] = None,
    nt_position: Optional[float] = None,
    angle_correction: Optional[float] = None,
) -> SomaPolarCoordinates:
    """
    Convert soma position to polar coordinates using legacy method.

    Uses angle correction based on reference point (33, 33) and DVNT position.
    This matches the legacy PKL processing method.

    Args:
        soma_xy: Soma position (row, col)
        intersection: AP pathway intersection point
        dv_position: Dorsal-ventral position for angle correction
        nt_position: Nasal-temporal position for angle correction
        angle_correction: Pre-calculated angle correction in degrees (if None, will calculate)

    Returns:
        SomaPolarCoordinates with radius, angle, quadrant, and correction info
    """
    soma_x, soma_y = soma_xy  # soma_x = row, soma_y = col

    # Calculate Cartesian displacement from intersection
    dx = soma_y - intersection.x  # Column displacement
    dy = soma_x - intersection.y  # Row displacement

    # Calculate basic polar coordinates
    radius = math.sqrt(dx ** 2 + dy ** 2)
    theta_raw_rad = math.atan2(dy, dx)
    theta_deg_raw = math.degrees(theta_raw_rad)
    theta_deg_raw = _normalize_angle_deg(theta_deg_raw)

    # Calculate angle correction if not provided
    if angle_correction is None:
        angle_correction = _calculate_angle_correction(
            intersection, dv_position, nt_position
        )

    # Apply angle correction
    if angle_correction is not None:
        theta_deg_corrected = _normalize_angle_deg(theta_deg_raw + angle_correction)
    else:
        theta_deg_corrected = theta_deg_raw

    # Final angle
    theta_deg_final = theta_deg_corrected
    theta_final_rad = math.radians(theta_deg_final)

    # Calculate transformed coordinates (after angle correction)
    transformed_x = radius * math.cos(theta_final_rad)
    transformed_y = radius * math.sin(theta_final_rad)

    # Get quadrant from transformed coordinates
    quadrant = _get_quadrant(transformed_x, transformed_y)

    # Get anatomical quadrant from DVNT
    anatomical_quadrant = _get_anatomical_quadrant(dv_position, nt_position)

    return SomaPolarCoordinates(
        radius=radius,
        angle=theta_final_rad,
        cartesian_x=dx,
        cartesian_y=dy,
        quadrant=quadrant,
        anatomical_quadrant=anatomical_quadrant,
        theta_deg=theta_deg_final,
        theta_deg_raw=theta_deg_raw,
        theta_deg_corrected=theta_deg_corrected,
        transformed_x=transformed_x,
        transformed_y=transformed_y,
        original_x=soma_x,
        original_y=soma_y,
        angle_correction_applied=angle_correction,
    )

