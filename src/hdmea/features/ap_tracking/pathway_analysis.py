"""
AP pathway analysis and soma polar coordinate calculation.

This module provides functions to fit lines to axon projections,
calculate optimal intersection points, and compute soma polar coordinates.

Enhanced with direction-based filtering and DBSCAN clustering for
robust optic nerve head (ONH) localization.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# =============================================================================
# Default Algorithm Parameters for ONH Detection
# =============================================================================

DEFAULT_R2_THRESHOLD = 0.8
DEFAULT_DIRECTION_TOLERANCE = 45.0  # degrees
DEFAULT_MAX_DISTANCE_FROM_CENTER = 98.0  # pixels 65 + 33 = 98
DEFAULT_CENTER_POINT = (33.0, 33.0)  # Center of 65x65 MEA grid
DEFAULT_CLUSTER_EPS = 15.0  # DBSCAN epsilon
DEFAULT_CLUSTER_MIN_SAMPLES = 3


@dataclass
class APPathway:
    """Fitted line to axon projection with direction information."""

    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float
    num_points: int
    # Direction from centroid temporal order (degrees, 0-360)
    direction_angle: Optional[float] = None
    # Ray origin (row, col) - typically first centroid position
    start_point: Optional[Tuple[float, float]] = None
    # Whether direction agrees with consensus (within tolerance)
    direction_valid: bool = True

    @property
    def r_squared(self) -> float:
        """R² value (coefficient of determination)."""
        return self.r_value ** 2


@dataclass
class APIntersection:
    """Optimal intersection point of AP pathways."""

    x: float
    y: float
    mse: float  # Mean squared error


@dataclass
class PairwiseIntersection:
    """A single pairwise intersection point between two pathway lines."""

    x: float
    y: float
    unit1: str
    unit2: str
    combined_r2: float  # Product of R² values
    mean_r2: float  # Mean of R² values
    is_valid: bool = True  # Passes all filters (direction, distance, cluster)


@dataclass
class ONHResult:
    """
    Enhanced optic nerve head (ONH) detection result.
    
    Contains the optimal intersection point along with metadata
    about the detection algorithm and filtering.
    """

    # Optimal intersection coordinates
    x: float
    y: float

    # Error metrics
    mse: float  # Mean squared error from optimal to all lines
    rmse: float  # Root mean squared error

    # Clustering results (legacy method)
    n_cluster_points: int  # Points in main cluster used for calculation
    n_total_intersections: int  # Total pairwise intersections before clustering
    n_valid_after_direction: int  # Intersections passing direction filter
    n_valid_after_distance: int  # Intersections passing distance filter

    # Direction information
    consensus_direction: Optional[float]  # Weighted mean direction in degrees

    # Algorithm parameters used
    r2_threshold: float
    direction_tolerance: float
    max_distance_from_center: float
    cluster_eps: float
    cluster_min_samples: int

    # Final valid intersection points (from main cluster) - legacy method
    # Shape: (N, 2) array with [x, y] per point
    cluster_points: Optional[np.ndarray] = None

    # Method identifier
    method: str = "clustered_weighted_mean"
    
    # New algorithm fields (global optimization)
    centroid_exclude_fraction: float = 0.1  # Fraction of centroids excluded
    n_cells_used: int = 0  # Number of cells used in optimization
    n_centroids_used: int = 0  # Total centroids used in optimization
    total_squared_error: float = 0.0  # Total squared error from optimization
    
    # Outlier removal fields (two-stage)
    outlier_unit_ids_stage1: Optional[List[str]] = None  # Units removed before intersection filter
    outlier_unit_ids_stage2: Optional[List[str]] = None  # Units removed during final optimization
    outlier_unit_ids: Optional[List[str]] = None  # All outliers combined (stage1 + stage2)
    kept_unit_ids: Optional[List[str]] = None  # Units kept after all outlier removal
    max_outlier_fraction: float = 0.2  # Max fraction of units to remove per stage


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


# =============================================================================
# Direction Calculation Functions
# =============================================================================


def calculate_direction_from_centroids(
    centroids: np.ndarray,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Calculate AP propagation direction from temporal order of axon centroids.
    
    The direction is from early frames to late frames, representing
    the direction of action potential propagation toward the optic nerve head.
    
    Args:
        centroids: Array of shape (N, 3) where columns are [t, row, col]
                   or (N, 2) where columns are [row, col] with implicit time order
    
    Returns:
        Tuple of (direction_angle in degrees 0-360, start_point (row, col))
        Returns (None, None) if insufficient data
    """
    if centroids is None or len(centroids) < 2:
        return None, None
    
    # Handle different centroid formats
    if centroids.ndim == 1:
        return None, None
    
    if centroids.shape[1] == 3:
        # [t, row, col] format - sort by time
        sorted_idx = np.argsort(centroids[:, 0])
        sorted_centroids = centroids[sorted_idx]
        row_coords = sorted_centroids[:, 1]
        col_coords = sorted_centroids[:, 2]
    elif centroids.shape[1] == 2:
        # [row, col] format - assume already time-ordered
        row_coords = centroids[:, 0]
        col_coords = centroids[:, 1]
    else:
        return None, None
    
    # Remove NaN values
    valid_mask = ~(np.isnan(row_coords) | np.isnan(col_coords))
    row_coords = row_coords[valid_mask]
    col_coords = col_coords[valid_mask]
    
    if len(row_coords) < 2:
        return None, None
    
    # Start point is the first centroid (earliest in time)
    start_point = (float(row_coords[0]), float(col_coords[0]))
    
    # Direction vector: weighted by time (later centroids have more weight)
    n = len(row_coords)
    weights = np.arange(1, n + 1)  # Later points weighted more
    
    # Weighted mean
    row_mean = np.average(row_coords, weights=weights)
    col_mean = np.average(col_coords, weights=weights)
    
    # Direction from start to weighted mean
    d_row = row_mean - row_coords[0]
    d_col = col_mean - col_coords[0]
    
    if abs(d_row) < 1e-10 and abs(d_col) < 1e-10:
        # No movement - use last point
        d_row = row_coords[-1] - row_coords[0]
        d_col = col_coords[-1] - col_coords[0]
    
    if abs(d_row) < 1e-10 and abs(d_col) < 1e-10:
        return None, start_point
    
    # Calculate angle in degrees (0-360)
    # Using atan2(d_row, d_col) to get angle where 0 = +col direction
    angle = np.degrees(np.arctan2(d_row, d_col))
    if angle < 0:
        angle += 360
    
    return float(angle), start_point


def weighted_circular_mean(angles: List[float], weights: List[float]) -> float:
    """
    Calculate weighted circular mean of angles.
    
    Handles angular wraparound correctly using circular statistics.
    
    Args:
        angles: List of angles in degrees (0-360)
        weights: Corresponding weights (e.g., R² values)
    
    Returns:
        Weighted mean angle in degrees (0-360)
    """
    if not angles or not weights:
        return 0.0
    
    angles_arr = np.array(angles)
    weights_arr = np.array(weights)
    
    # Normalize weights
    weights_arr = weights_arr / np.sum(weights_arr)
    
    # Convert to radians
    radians = np.radians(angles_arr)
    
    # Weighted sum of unit vectors
    x = np.sum(weights_arr * np.cos(radians))
    y = np.sum(weights_arr * np.sin(radians))
    
    # Mean angle
    mean_rad = np.arctan2(y, x)
    mean_deg = np.degrees(mean_rad)
    
    if mean_deg < 0:
        mean_deg += 360
    
    return float(mean_deg)


def angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest difference between two angles.
    
    Returns value in range [0, 180].
    """
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def is_within_angle_tolerance(
    angle: float,
    reference: float,
    tolerance: float = 90.0,
) -> bool:
    """
    Check if angle is within tolerance of reference angle.
    
    Args:
        angle: Test angle in degrees
        reference: Reference angle in degrees
        tolerance: Maximum allowed difference in degrees
    
    Returns:
        True if angle is within tolerance of reference
    """
    diff = angle_difference(angle, reference)
    return diff <= tolerance


def calculate_consensus_direction(
    pathways: Dict[str, APPathway],
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
) -> Optional[float]:
    """
    Calculate consensus direction using weighted circular mean.
    
    Uses R² as weights for each unit's direction.
    
    Args:
        pathways: Dictionary of unit_id to APPathway
        r2_threshold: Minimum R² for including pathway in consensus
    
    Returns:
        Consensus direction in degrees (0-360), or None if no valid directions
    """
    angles = []
    weights = []
    
    for uid, pw in pathways.items():
        if pw.r_squared >= r2_threshold and pw.direction_angle is not None:
            angles.append(pw.direction_angle)
            weights.append(pw.r_squared)
    
    if not angles:
        logger.debug("No pathways with direction data above R² threshold")
        return None
    
    consensus = weighted_circular_mean(angles, weights)
    logger.debug(f"Consensus direction: {consensus:.1f} deg from {len(angles)} pathways")
    return consensus


def filter_pathways_by_direction(
    pathways: Dict[str, APPathway],
    consensus_direction: float,
    tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
) -> Dict[str, APPathway]:
    """
    Mark pathways as valid/invalid based on direction agreement.
    
    Updates the direction_valid field of each pathway.
    
    Args:
        pathways: Dictionary of unit_id to APPathway
        consensus_direction: Reference direction in degrees
        tolerance: Maximum allowed difference from consensus
    
    Returns:
        Same pathways dict with updated direction_valid fields
    """
    valid_count = 0
    for uid, pw in pathways.items():
        if pw.direction_angle is not None:
            pw.direction_valid = is_within_angle_tolerance(
                pw.direction_angle, consensus_direction, tolerance
            )
            if pw.direction_valid:
                valid_count += 1
        else:
            # No direction info - default to valid (will be filtered by R² anyway)
            pw.direction_valid = True
    
    logger.debug(
        f"Direction filter: {valid_count} of {len(pathways)} pathways "
        f"within +/-{tolerance} deg of consensus"
    )
    return pathways


# =============================================================================
# ONH Global Optimization Functions
# =============================================================================


def perpendicular_distance_to_line(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """
    Calculate perpendicular distance from a point to a line defined by two points.
    
    Args:
        point: (row, col) coordinates of the point
        line_start: (row, col) coordinates of line start (e.g., ONH)
        line_end: (row, col) coordinates of line end (e.g., cell start point)
    
    Returns:
        Perpendicular distance from point to line
    """
    # Line defined by points A (line_start) and B (line_end)
    # Point P (point)
    ax, ay = line_start
    bx, by = line_end
    px, py = point
    
    # Vector AB
    abx = bx - ax
    aby = by - ay
    
    # If line has zero length, return distance to point A
    ab_length_sq = abx * abx + aby * aby
    if ab_length_sq < 1e-10:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    
    # Perpendicular distance formula: |AB x AP| / |AB|
    # Cross product in 2D gives signed area: (bx-ax)*(py-ay) - (by-ay)*(px-ax)
    cross = abx * (py - ay) - aby * (px - ax)
    
    return abs(cross) / np.sqrt(ab_length_sq)


def calculate_total_projection_error(
    onh_candidate: Tuple[float, float],
    cell_data: List[Tuple[Tuple[float, float], np.ndarray]],
) -> float:
    """
    Calculate total squared projection error for a candidate ONH point.
    
    For each cell:
    - Line from ONH candidate to cell's start point
    - Sum squared perpendicular distances from all centroids to this line
    
    Args:
        onh_candidate: (col, row) candidate ONH position
        cell_data: List of (start_point, centroids) tuples where:
                   - start_point: (row, col) of first valid centroid
                   - centroids: Array of remaining centroids [row, col]
    
    Returns:
        Total sum of squared perpendicular distances
    """
    onh_row, onh_col = onh_candidate[1], onh_candidate[0]  # Convert (col, row) to (row, col)
    total_error = 0.0
    
    for start_point, centroids in cell_data:
        if centroids is None or len(centroids) == 0:
            continue
        
        start_row, start_col = start_point
        
        for centroid in centroids:
            c_row, c_col = centroid[0], centroid[1]
            
            dist = perpendicular_distance_to_line(
                point=(c_row, c_col),
                line_start=(onh_row, onh_col),
                line_end=(start_row, start_col),
            )
            total_error += dist * dist
    
    return total_error


def calculate_unit_projection_error(
    onh: Tuple[float, float],
    start_point: Tuple[float, float],
    centroids: np.ndarray,
) -> float:
    """
    Calculate mean squared perpendicular distance for a single unit.
    
    Args:
        onh: (col, row) ONH position
        start_point: (row, col) of the unit's start point
        centroids: Array of centroids [row, col] for this unit
    
    Returns:
        Mean squared perpendicular distance (MSE) for this unit
    """
    if centroids is None or len(centroids) == 0:
        return float('inf')
    
    onh_col, onh_row = onh
    start_row, start_col = start_point
    
    total_sq_dist = 0.0
    for centroid in centroids:
        c_row, c_col = centroid[0], centroid[1]
        dist = perpendicular_distance_to_line(
            point=(c_row, c_col),
            line_start=(onh_row, onh_col),
            line_end=(start_row, start_col),
        )
        total_sq_dist += dist * dist
    
    return total_sq_dist / len(centroids)


def find_onh_with_outlier_removal(
    cell_data: List[Tuple[Tuple[float, float], np.ndarray]],
    unit_ids: List[str],
    max_outlier_fraction: float = 0.2,
    convergence_threshold: float = 0.1,
    max_iterations: int = 10,
    consensus_direction: Optional[float] = None,
    max_distance_from_center: float = DEFAULT_MAX_DISTANCE_FROM_CENTER,
    center_point: Tuple[float, float] = DEFAULT_CENTER_POINT,
) -> Tuple[Optional[Tuple[float, float]], float, int, List[str], List[str]]:
    """
    Find optimal ONH iteratively, removing worst-fitting units each round.
    
    Algorithm:
    1. Calculate ONH with all units
    2. Compute per-unit error (MSE of perpendicular distances)
    3. Remove the worst-fitting unit
    4. Repeat until (1 - max_outlier_fraction) of units remain or convergence
    
    Args:
        cell_data: List of (start_point, centroids) tuples from prepare_cell_data_for_onh
        unit_ids: List of unit IDs corresponding to cell_data entries
        max_outlier_fraction: Maximum fraction of units to remove (default 0.2 = 20%)
        convergence_threshold: Stop when ONH moves < this many pixels between iterations
        max_iterations: Maximum number of trimming iterations
        consensus_direction: Consensus AP direction in degrees (for optimization constraint)
        max_distance_from_center: Maximum distance from center_point
        center_point: Center of valid region (col, row)
    
    Returns:
        Tuple of:
        - Optimal (col, row) or None if optimization failed
        - Total squared error at optimal point
        - Number of centroids used in final optimization
        - List of kept unit IDs
        - List of outlier unit IDs
    """
    if not cell_data or len(cell_data) < 2:
        logger.warning("Not enough cell data for ONH optimization with outlier removal")
        return None, float('inf'), 0, [], []
    
    # Keep track of current data and unit IDs
    current_cell_data = list(cell_data)
    current_unit_ids = list(unit_ids)
    n_original = len(current_cell_data)
    n_to_keep = max(2, int(n_original * (1 - max_outlier_fraction)))  # Keep at least 2
    
    outlier_unit_ids = []
    prev_onh = None
    
    logger.debug(
        f"Starting outlier removal: {n_original} units, "
        f"will keep at least {n_to_keep} ({100*(1-max_outlier_fraction):.0f}%)"
    )
    
    for iteration in range(max_iterations):
        # Find ONH with current data
        onh, total_error, n_centroids = find_optimal_onh(
            current_cell_data,
            consensus_direction=consensus_direction,
            max_distance_from_center=max_distance_from_center,
            center_point=center_point,
        )
        
        if onh is None:
            logger.warning(f"ONH optimization failed at iteration {iteration}")
            break
        
        # Check convergence
        if prev_onh is not None:
            movement = np.sqrt((onh[0] - prev_onh[0])**2 + (onh[1] - prev_onh[1])**2)
            if movement < convergence_threshold:
                logger.debug(f"Converged at iteration {iteration}: ONH moved {movement:.3f} px")
                break
        prev_onh = onh
        
        # If we've trimmed enough, stop
        if len(current_cell_data) <= n_to_keep:
            logger.debug(f"Reached target: {len(current_cell_data)} units remaining")
            break
        
        # Calculate per-unit error
        unit_errors = []
        for i, (start_point, centroids) in enumerate(current_cell_data):
            unit_mse = calculate_unit_projection_error(onh, start_point, centroids)
            unit_errors.append((i, current_unit_ids[i], unit_mse))
        
        # Sort by error (highest first)
        unit_errors.sort(key=lambda x: x[2], reverse=True)
        
        # Remove worst unit
        worst_idx, worst_uid, worst_mse = unit_errors[0]
        logger.debug(
            f"Iteration {iteration}: removing {worst_uid} with MSE={worst_mse:.4f}"
        )
        
        outlier_unit_ids.append(worst_uid)
        current_cell_data = [cd for j, cd in enumerate(current_cell_data) if j != worst_idx]
        current_unit_ids = [uid for j, uid in enumerate(current_unit_ids) if j != worst_idx]
    
    # Final optimization with remaining units
    final_onh, final_error, final_n_centroids = find_optimal_onh(
        current_cell_data,
        consensus_direction=consensus_direction,
        max_distance_from_center=max_distance_from_center,
        center_point=center_point,
    )
    
    if final_onh is not None:
        logger.info(
            f"Outlier removal complete: {len(current_unit_ids)} kept, "
            f"{len(outlier_unit_ids)} removed, ONH=({final_onh[0]:.2f}, {final_onh[1]:.2f})"
        )
    
    return final_onh, final_error, final_n_centroids, current_unit_ids, outlier_unit_ids


def prepare_cell_data_for_onh(
    pathways: Dict[str, APPathway],
    all_centroids: Dict[str, np.ndarray],
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    exclude_fraction: float = 0.1,
) -> Tuple[List[Tuple[Tuple[float, float], np.ndarray]], List[str]]:
    """
    Prepare cell data for ONH optimization.
    
    For each valid cell:
    - Exclude first 10% of centroids
    - Extract start point (first remaining centroid)
    - Extract remaining centroids for error calculation
    
    Args:
        pathways: Dictionary of unit_id to APPathway
        all_centroids: Dictionary of unit_id to centroids array [t, row, col]
        r2_threshold: Minimum R² for including cell
        exclude_fraction: Fraction of centroids to exclude (default 0.1 = 10%)
    
    Returns:
        Tuple of:
        - List of (start_point, centroids) tuples for each valid cell
        - List of corresponding unit IDs
    """
    cell_data = []
    unit_ids = []
    
    for unit_id, pw in pathways.items():
        # Check R² threshold and direction validity
        if pw.r_squared < r2_threshold or not pw.direction_valid:
            continue
        
        # Get centroids for this unit
        if unit_id not in all_centroids:
            continue
        
        centroids = all_centroids[unit_id]
        if centroids is None or len(centroids) < 3:
            continue
        
        # Sort by time if 3-column format
        if centroids.shape[1] == 3:
            sorted_idx = np.argsort(centroids[:, 0])
            centroids = centroids[sorted_idx]
            coords = centroids[:, 1:3]  # [row, col]
        else:
            coords = centroids  # Assume [row, col]
        
        # Exclude first 10% of centroids
        n_total = len(coords)
        n_exclude = max(1, int(n_total * exclude_fraction))  # At least 1
        
        if n_total - n_exclude < 2:
            # Not enough centroids after exclusion
            continue
        
        # Start point is first centroid after exclusion
        start_point = (float(coords[n_exclude, 0]), float(coords[n_exclude, 1]))
        
        # Remaining centroids (including start point for error calculation)
        remaining_centroids = coords[n_exclude:]
        
        cell_data.append((start_point, remaining_centroids))
        unit_ids.append(unit_id)
    
    return cell_data, unit_ids


def find_optimal_onh(
    cell_data: List[Tuple[Tuple[float, float], np.ndarray]],
    consensus_direction: Optional[float] = None,
    max_distance_from_center: float = DEFAULT_MAX_DISTANCE_FROM_CENTER,
    center_point: Tuple[float, float] = DEFAULT_CENTER_POINT,
    initial_guess: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[Tuple[float, float]], float, int]:
    """
    Find optimal ONH location using constrained optimization.
    
    Loss Function:
        L(ONH) = sum of squared perpendicular distances from all centroids
                 to their respective projection lines (ONH -> cell start point)
    
    Constraints:
        1. ONH must be within max_distance_from_center of center_point
        2. ONH must be in the forward direction (consensus_direction) from cell start points
    
    Args:
        cell_data: List of (start_point, centroids) tuples from prepare_cell_data_for_onh
        consensus_direction: Consensus AP direction in degrees. If provided, ONH must be
                            in this direction from cell start points.
        max_distance_from_center: Maximum distance from center_point (default 98)
        center_point: Center of valid region (col, row) (default (33, 33))
        initial_guess: Initial (col, row) guess for optimization
    
    Returns:
        Tuple of:
        - Optimal (col, row) - always returns best solution within bounds
        - Total squared error at optimal point
        - Number of centroids used in optimization
    """
    if not cell_data:
        logger.warning("No cell data for ONH optimization")
        return None, float('inf'), 0
    
    # Count total centroids
    n_centroids = sum(len(c) for _, c in cell_data)
    
    if n_centroids < 5:
        logger.warning(f"Not enough centroids for optimization: {n_centroids}")
        return None, float('inf'), n_centroids
    
    center_col, center_row = center_point
    
    # Calculate mean start point for initial guess and direction constraint
    start_rows = [sp[0] for sp, _ in cell_data]
    start_cols = [sp[1] for sp, _ in cell_data]
    mean_start_row = np.mean(start_rows)
    mean_start_col = np.mean(start_cols)
    
    # Loss function: sum of squared perpendicular distances
    def loss_function(x):
        onh_col, onh_row = x[0], x[1]
        return calculate_total_projection_error((onh_col, onh_row), cell_data)
    
    # Circular constraint: distance from center <= max_distance
    # For scipy, inequality constraints must be >= 0
    def circular_constraint(x):
        dist_sq = (x[0] - center_col)**2 + (x[1] - center_row)**2
        return max_distance_from_center**2 - dist_sq  # >= 0 means inside circle
    
    constraints = [{'type': 'ineq', 'fun': circular_constraint}]
    
    # Direction constraint: ONH must be in the forward direction from mean start point
    # This ensures ONH is "ahead" of the cells, not "behind" them
    if consensus_direction is not None:
        dir_rad = np.radians(consensus_direction)
        # Direction unit vector: 0° = +col, 90° = +row
        dir_col = np.cos(dir_rad)
        dir_row = np.sin(dir_rad)
        
        def direction_constraint(x):
            onh_col, onh_row = x[0], x[1]
            # Vector from mean start point to ONH
            to_onh_col = onh_col - mean_start_col
            to_onh_row = onh_row - mean_start_row
            # Dot product with consensus direction should be positive
            dot = to_onh_col * dir_col + to_onh_row * dir_row
            return dot  # >= 0 means ONH is in forward direction
        
        constraints.append({'type': 'ineq', 'fun': direction_constraint})
    
    # Default initial guess: in the consensus direction from mean start point
    if initial_guess is None:
        if consensus_direction is not None:
            dir_rad = np.radians(consensus_direction)
            # Start 30 pixels ahead in consensus direction
            initial_guess = (
                mean_start_col + 30 * np.cos(dir_rad),
                mean_start_row + 30 * np.sin(dir_rad)
            )
        else:
            initial_guess = (center_col, center_row)
    
    # Rectangular bounds (containing the circular constraint)
    bounds = [
        (center_col - max_distance_from_center, center_col + max_distance_from_center),
        (center_row - max_distance_from_center, center_row + max_distance_from_center),
    ]
    
    # Try multiple starting points for robustness
    best_result = None
    best_error = float('inf')
    
    # Generate diverse starting points within the constraint region
    starting_points = [initial_guess]
    
    # Add center point
    starting_points.append((center_col, center_row))
    
    # Add points on the edge of the circle in different directions
    for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        angle_rad = np.radians(angle_deg)
        edge_col = center_col + max_distance_from_center * 0.8 * np.cos(angle_rad)
        edge_row = center_row + max_distance_from_center * 0.8 * np.sin(angle_rad)
        starting_points.append((edge_col, edge_row))
    
    # Add mean start point if within bounds
    if circular_constraint([mean_start_col, mean_start_row]) >= 0:
        starting_points.append((mean_start_col, mean_start_row))
    
    for sp in starting_points:
        # Ensure starting point is within bounds
        sp_col = np.clip(sp[0], bounds[0][0], bounds[0][1])
        sp_row = np.clip(sp[1], bounds[1][0], bounds[1][1])
        
        # Also ensure within circular constraint
        dist = np.sqrt((sp_col - center_col)**2 + (sp_row - center_row)**2)
        if dist > max_distance_from_center:
            # Project onto the circle boundary
            scale = max_distance_from_center / dist
            sp_col = center_col + (sp_col - center_col) * scale
            sp_row = center_row + (sp_row - center_row) * scale
        
        try:
            # Use SLSQP which supports constraints
            result = minimize(
                loss_function,
                x0=[sp_col, sp_row],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            # Accept the result regardless of success flag - SLSQP may report
            # "success=False" for numerical reasons even when finding a valid minimum
            if result.fun < best_error:
                # Verify the result is within the circular constraint
                opt_col, opt_row = result.x
                dist_from_center = np.sqrt((opt_col - center_col)**2 + (opt_row - center_row)**2)
                
                if dist_from_center <= max_distance_from_center * 1.01:  # Allow 1% tolerance
                    best_error = result.fun
                    best_result = result
        except Exception as e:
            logger.debug(f"Optimization from ({sp_col:.1f}, {sp_row:.1f}) failed: {e}")
            continue
    
    # Always return the best result found (never fail)
    if best_result is not None:
        optimal_col, optimal_row = best_result.x
        total_error = best_result.fun
        
        logger.debug(
            f"ONH optimization: ({optimal_col:.2f}, {optimal_row:.2f}), "
            f"total squared error = {total_error:.4f}, n_centroids = {n_centroids}"
        )
        
        return (optimal_col, optimal_row), total_error, n_centroids
    else:
        # Fallback: return center point with infinite error
        logger.warning("All optimization attempts failed, returning center point")
        total_error = loss_function([center_col, center_row])
        return (center_col, center_row), total_error, n_centroids


# =============================================================================
# Iterative ONH Refinement
# =============================================================================


def calculate_boundary_pull_contribution(
    onh: Tuple[float, float],
    start_point: Tuple[float, float],
    centroids: np.ndarray,
    center_point: Tuple[float, float],
) -> float:
    """
    Calculate how much a pathway contributes to pulling ONH toward the boundary.
    
    Higher values indicate the pathway is pulling the ONH away from center.
    
    Args:
        onh: Current ONH location (col, row)
        start_point: Pathway start point (row, col)
        centroids: Pathway centroids array [N, 3] with [t, row, col]
        center_point: Center of valid region (col, row)
    
    Returns:
        Boundary pull contribution score (higher = more pull toward boundary)
    """
    center_col, center_row = center_point
    onh_col, onh_row = onh
    
    # Direction from center to ONH (the "pull direction" we want to minimize)
    pull_dir_col = onh_col - center_col
    pull_dir_row = onh_row - center_row
    pull_magnitude = np.sqrt(pull_dir_col**2 + pull_dir_row**2)
    
    if pull_magnitude < 1e-6:
        return 0.0  # ONH is at center, no pull
    
    # Normalize pull direction
    pull_dir_col /= pull_magnitude
    pull_dir_row /= pull_magnitude
    
    # For this pathway, calculate how much its centroids contribute to the pull
    # We compute the perpendicular error and also how aligned the error is with pull direction
    sp_row, sp_col = start_point
    
    total_contribution = 0.0
    for centroid in centroids:
        # Handle both 2-column [row, col] and 3-column [t, row, col] formats
        if len(centroid) == 3:
            _, c_row, c_col = centroid
        else:
            c_row, c_col = centroid
        
        # Vector from ONH to centroid
        to_centroid_col = c_col - onh_col
        to_centroid_row = c_row - onh_row
        
        # Vector from start to centroid (the pathway direction)
        pathway_col = c_col - sp_col
        pathway_row = c_row - sp_row
        
        # The "pull" is how much moving ONH toward this centroid aligns with pull direction
        # If the centroid is in the pull direction, removing this pathway would reduce pull
        alignment = to_centroid_col * pull_dir_col + to_centroid_row * pull_dir_row
        
        # Weight by alignment (positive = contributes to pull)
        if alignment > 0:
            total_contribution += alignment
    
    return total_contribution


def identify_boundary_pull_pathway(
    onh: Tuple[float, float],
    cell_data: List[Tuple[Tuple[float, float], np.ndarray]],
    center_point: Tuple[float, float],
) -> int:
    """
    Identify which pathway contributes most to pulling ONH toward the boundary.
    
    Args:
        onh: Current ONH location (col, row)
        cell_data: List of (start_point, centroids) tuples
        center_point: Center of valid region (col, row)
    
    Returns:
        Index of the pathway with highest boundary pull contribution
    """
    contributions = []
    
    for start_point, centroids in cell_data:
        contribution = calculate_boundary_pull_contribution(
            onh, start_point, centroids, center_point
        )
        contributions.append(contribution)
    
    # Return index of pathway with highest contribution
    return int(np.argmax(contributions))


def find_onh_with_iterative_refinement(
    cell_data: List[Tuple[Tuple[float, float], np.ndarray]],
    unit_ids: List[str],
    consensus_direction: Optional[float] = None,
    max_distance_from_center: float = DEFAULT_MAX_DISTANCE_FROM_CENTER,
    center_point: Tuple[float, float] = DEFAULT_CENTER_POINT,
    boundary_tolerance: float = 0.95,  # Consider "at boundary" if > 95% of max
    min_pathways: int = 5,  # Stop if fewer than this remain
    max_iterations: int = 50,  # Safety limit
    removal_fraction: float = 0.1,  # Starting removal fraction (10%)
    min_remaining_fraction: float = 0.2,  # Stop if < 20% of original pathways remain
) -> Tuple[Optional[Tuple[float, float]], float, int, List[str], List[str]]:
    """
    Iteratively find ONH, removing worst-fitting pathways in batches.
    
    This uses a combined approach:
    1. Optimize with all pathways
    2. If ONH is comfortably within boundary (< 95% of max distance), done
    3. Otherwise, calculate per-unit projection error for each pathway
    4. Remove the worst 10% of pathways (by error, minimum 1)
    5. Repeat until ONH is within boundary or too few pathways remain
    
    This is faster than single-pathway removal and more effective because
    it removes pathways that fit poorly (high perpendicular error), which
    are likely outliers pushing the ONH to extreme positions.
    
    Args:
        cell_data: List of (start_point, centroids) tuples
        unit_ids: List of unit IDs corresponding to cell_data entries
        consensus_direction: Consensus AP direction for direction constraint
        max_distance_from_center: Maximum allowed distance from center
        center_point: Center of valid region (col, row)
        boundary_tolerance: Consider "at boundary" if distance > this * max_distance
        min_pathways: Minimum pathways required to continue
        max_iterations: Maximum refinement iterations (safety limit)
        removal_fraction: Starting fraction of pathways to remove per iteration (default 10%)
        min_remaining_fraction: Stop if fewer than this fraction of original pathways remain (default 20%)
    
    Returns:
        Tuple of:
        - Optimal (col, row) or None if optimization failed
        - Total squared error at optimal point
        - Number of centroids used in final optimization
        - List of kept unit IDs
        - List of removed unit IDs
    """
    if not cell_data or len(cell_data) < min_pathways:
        logger.warning(f"Not enough cell data for iterative refinement: {len(cell_data) if cell_data else 0}")
        return None, float('inf'), 0, [], []
    
    # Work with copies to avoid modifying input
    remaining_data = list(cell_data)
    remaining_ids = list(unit_ids)
    removed_ids: List[str] = []
    
    center_col, center_row = center_point
    boundary_threshold = max_distance_from_center * boundary_tolerance
    
    best_onh = None
    best_error = float('inf')
    best_n_centroids = 0
    best_remaining_ids = list(remaining_ids)
    best_removed_ids: List[str] = []
    
    # Track for escalating removal when stuck
    previous_dist = float('inf')
    no_progress_count = 0
    initial_pathway_count = len(remaining_data)
    current_removal_fraction = removal_fraction  # Start at 10%
    
    logger.debug(
        f"Starting iterative refinement with {len(remaining_data)} pathways, "
        f"boundary threshold = {boundary_threshold:.1f}px, "
        f"initial removal fraction = {removal_fraction*100:.0f}%"
    )
    
    for iteration in range(max_iterations):
        if len(remaining_data) < min_pathways:
            logger.debug(f"Too few pathways remaining ({len(remaining_data)}), stopping")
            break
        
        # Optimize with current pathways
        onh, total_error, n_centroids = find_optimal_onh(
            remaining_data,
            consensus_direction=consensus_direction,
            max_distance_from_center=max_distance_from_center,
            center_point=center_point,
        )
        
        if onh is None:
            logger.warning(f"Optimization failed at iteration {iteration}")
            break
        
        # Check distance from center
        dist_from_center = np.sqrt((onh[0] - center_col)**2 + (onh[1] - center_row)**2)
        
        # Track best result (prefer lower distance, then lower error)
        if dist_from_center < np.sqrt((best_onh[0] - center_col)**2 + (best_onh[1] - center_row)**2) if best_onh else True:
            best_onh = onh
            best_error = total_error
            best_n_centroids = n_centroids
            best_remaining_ids = list(remaining_ids)
            best_removed_ids = list(removed_ids)
        
        logger.debug(
            f"Iteration {iteration}: ONH at ({onh[0]:.1f}, {onh[1]:.1f}), "
            f"distance = {dist_from_center:.1f}px, pathways = {len(remaining_data)}"
        )
        
        if dist_from_center < boundary_threshold:
            # Success! ONH is comfortably within bounds
            logger.info(
                f"ONH converged within boundary after {iteration} iterations: "
                f"({onh[0]:.2f}, {onh[1]:.2f}), distance = {dist_from_center:.1f}px, "
                f"removed {len(removed_ids)} pathways"
            )
            return onh, total_error, n_centroids, remaining_ids, removed_ids
        
        # Check for progress and escalate removal if stuck
        if dist_from_center >= previous_dist - 0.5:  # Less than 0.5px improvement
            no_progress_count += 1
        else:
            no_progress_count = 0
            current_removal_fraction = removal_fraction  # Reset on progress
        
        # Escalate removal rate if no progress for 3 iterations
        if no_progress_count >= 3:
            if current_removal_fraction < 0.2:
                # First escalation: jump to 20%
                current_removal_fraction = 0.2
                logger.debug(f"Escalating removal to 20%")
            else:
                # Subsequent escalations: +10% steps, max 50%
                current_removal_fraction = min(current_removal_fraction + 0.1, 0.5)
                logger.debug(f"Escalating removal to {current_removal_fraction*100:.0f}%")
            no_progress_count = 0  # Reset counter after escalation
        
        previous_dist = dist_from_center
        
        # Calculate per-unit projection error for each pathway
        unit_errors = []
        for start_point, centroids in remaining_data:
            err = calculate_unit_projection_error(onh, start_point, centroids)
            unit_errors.append(err)
        
        # Determine how many pathways to remove (batch removal with current rate)
        n_to_remove = max(1, int(len(remaining_data) * current_removal_fraction))
        
        # Don't remove below min_pathways OR below min_remaining_fraction of original
        min_remaining = max(min_pathways, int(initial_pathway_count * min_remaining_fraction))
        n_to_remove = min(n_to_remove, len(remaining_data) - min_remaining)
        
        if n_to_remove <= 0:
            logger.debug(f"Cannot remove more pathways (at minimum limit: {min_remaining})")
            break
        
        # Get indices of worst-fitting pathways (highest error)
        worst_indices = np.argsort(unit_errors)[-n_to_remove:]
        
        # Remove in reverse order to preserve indices
        for idx in sorted(worst_indices, reverse=True):
            removed_id = remaining_ids.pop(idx)
            remaining_data.pop(idx)
            removed_ids.append(removed_id)
        
        logger.debug(f"Removed {n_to_remove} worst-fitting pathways (rate: {current_removal_fraction*100:.0f}%)")
    
    # Didn't converge within boundary, return best result found
    if best_onh is not None:
        dist = np.sqrt((best_onh[0] - center_col)**2 + (best_onh[1] - center_row)**2)
        logger.warning(
            f"Could not converge within boundary after {iteration + 1} iterations. "
            f"Best ONH: ({best_onh[0]:.2f}, {best_onh[1]:.2f}), distance = {dist:.1f}px, "
            f"removed {len(best_removed_ids)} pathways"
        )
        return best_onh, best_error, best_n_centroids, best_remaining_ids, best_removed_ids
    else:
        return None, float('inf'), 0, remaining_ids, removed_ids


# =============================================================================
# Projection and Line Fitting Functions
# =============================================================================


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


def fit_line_to_centroids(
    axon_centroids: np.ndarray,
    min_points: int = 10,
) -> Optional[APPathway]:
    """
    Fit line to axon centroids.

    Uses linear regression to fit a line to the spatial coordinates
    of axon centroids extracted from post-processed prediction data.

    Args:
        axon_centroids: Array of shape (N, 3) with (t, row, col) per point
        min_points: Minimum points required for fitting

    Returns:
        APPathway with fit parameters, or None if insufficient points
    """
    if axon_centroids is None or len(axon_centroids) < min_points:
        logger.debug(
            f"Not enough centroids for fitting: {len(axon_centroids) if axon_centroids is not None else 0} < {min_points}"
        )
        return None

    # Extract coordinates: x=column (index 2), y=row (index 1)
    # This matches the convention in fit_line_to_projections
    x_coords = axon_centroids[:, 2]  # column
    y_coords = axon_centroids[:, 1]  # row

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
        logger.error(f"Error fitting line to centroids: {e}")
        return None


def calculate_optimal_intersection(
    pathways: Dict[str, APPathway],
) -> Optional[APIntersection]:
    """
    Calculate optimal intersection point from multiple pathway fits.

    Uses pairwise intersection approach: calculates intersection point for
    each pair of lines, then computes weighted average using the mean R²
    of the two intersecting lines as weight.

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

    # Calculate all pairwise intersections
    unit_ids = list(valid_fits.keys())
    pairwise_intersections = []
    pairwise_weights = []

    for i in range(len(unit_ids)):
        for j in range(i + 1, len(unit_ids)):
            fit1 = valid_fits[unit_ids[i]]
            fit2 = valid_fits[unit_ids[j]]

            m1, c1 = fit1.slope, fit1.intercept
            m2, c2 = fit2.slope, fit2.intercept

            # Check if lines are parallel (same slope)
            if abs(m1 - m2) < 1e-10:
                continue

            # Calculate intersection point
            # Line 1: y = m1*x + c1
            # Line 2: y = m2*x + c2
            # m1*x + c1 = m2*x + c2
            # x * (m1 - m2) = c2 - c1
            # x = (c2 - c1) / (m1 - m2)
            x_int = (c2 - c1) / (m1 - m2)
            y_int = m1 * x_int + c1

            # Weight by mean R² of the two lines
            r2_1 = fit1.r_value ** 2
            r2_2 = fit2.r_value ** 2
            mean_r2 = (r2_1 + r2_2) / 2.0

            pairwise_intersections.append((x_int, y_int))
            pairwise_weights.append(mean_r2)

    if len(pairwise_intersections) == 0:
        logger.warning("No valid pairwise intersections found (all lines parallel)")
        return None

    # Convert to numpy arrays
    intersections = np.array(pairwise_intersections)
    weights = np.array(pairwise_weights)

    # Normalize weights to sum to 1
    weights_normalized = weights / weights.sum()

    # Calculate weighted average of intersection points
    x_opt = np.sum(intersections[:, 0] * weights_normalized)
    y_opt = np.sum(intersections[:, 1] * weights_normalized)

    # Calculate mean squared error (distance from optimal point to all lines)
    errors = []
    for unit_id, fit in valid_fits.items():
        m = fit.slope
        c = fit.intercept
        # Distance from point to line: |m*x - y + c| / sqrt(1 + m²)
        dist = abs(m * x_opt - y_opt + c) / np.sqrt(1 + m ** 2)
        errors.append(dist ** 2)

    mse = float(np.mean(errors))

    logger.debug(
        f"Optimal intersection at ({x_opt:.2f}, {y_opt:.2f}) from {len(pairwise_intersections)} pairs, MSE={mse:.4f}"
    )

    return APIntersection(x=float(x_opt), y=float(y_opt), mse=mse)


# =============================================================================
# Enhanced ONH Detection with Clustering
# =============================================================================


def filter_pathways_by_pairwise_intersection(
    pathways: Dict[str, APPathway],
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    max_distance_from_center: float = DEFAULT_MAX_DISTANCE_FROM_CENTER,
    center_point: Tuple[float, float] = DEFAULT_CENTER_POINT,
) -> Dict[str, APPathway]:
    """
    Filter pathways to keep only those with at least one valid pairwise intersection.
    
    A pathway is valid if its line intersects with at least one other valid pathway's
    line within max_distance_from_center of center_point.
    
    Args:
        pathways: Dictionary of unit_id to APPathway
        r2_threshold: Minimum R² for including pathway
        max_distance_from_center: Maximum distance from center for valid intersection
        center_point: Center of visible area (col, row)
    
    Returns:
        Dictionary of pathways that have at least one valid intersection
    """
    # Get candidate pathways (R² and direction valid)
    candidates = {
        k: v for k, v in pathways.items()
        if v is not None and v.r_squared >= r2_threshold and v.direction_valid
    }
    
    if len(candidates) < 2:
        logger.debug(f"Not enough candidates for pairwise intersection filter: {len(candidates)}")
        return candidates
    
    center_col, center_row = center_point
    
    # Track which units have at least one valid intersection
    units_with_valid_intersection: set = set()
    
    unit_ids = list(candidates.keys())
    
    for i in range(len(unit_ids)):
        for j in range(i + 1, len(unit_ids)):
            uid1, uid2 = unit_ids[i], unit_ids[j]
            p1, p2 = candidates[uid1], candidates[uid2]
            
            # Check if lines are parallel (no intersection)
            if abs(p1.slope - p2.slope) < 1e-10:
                continue
            
            # Calculate line intersection: y = m1*x + b1 = m2*x + b2
            # x = (b2 - b1) / (m1 - m2)
            x = (p2.intercept - p1.intercept) / (p1.slope - p2.slope)
            y = p1.slope * x + p1.intercept
            
            # Check distance from center
            dist_from_center = np.sqrt((x - center_col)**2 + (y - center_row)**2)
            
            if dist_from_center <= max_distance_from_center:
                # Both units have a valid intersection
                units_with_valid_intersection.add(uid1)
                units_with_valid_intersection.add(uid2)
    
    # Return only pathways that have at least one valid intersection
    filtered = {k: v for k, v in pathways.items() if k in units_with_valid_intersection}
    
    n_removed = len(candidates) - len(filtered)
    if n_removed > 0:
        logger.debug(
            f"Pairwise intersection filter: removed {n_removed} of {len(candidates)} "
            f"pathways (no valid intersections within {max_distance_from_center:.0f}px)"
        )
    
    return filtered


def ray_points_toward_intersection(
    start: Tuple[float, float],
    direction_angle: float,
    intersection: Tuple[float, float],
) -> bool:
    """
    Check if a ray points toward the intersection point.
    
    A ray points toward the intersection if the intersection is in 
    the forward direction (within 90°) of the ray.
    
    Args:
        start: Ray origin (row, col)
        direction_angle: Ray direction in degrees
        intersection: Point to check (col, row) - note convention difference
    
    Returns:
        True if ray points toward intersection
    """
    # start is (row, col), intersection is (col, row) = (x, y)
    dx = intersection[0] - start[1]  # col difference
    dy = intersection[1] - start[0]  # row difference
    
    angle_to_int = np.degrees(np.arctan2(dy, dx))
    if angle_to_int < 0:
        angle_to_int += 360
    
    # Check if intersection is in forward direction for this ray
    return angle_difference(direction_angle, angle_to_int) <= 90


def calculate_pairwise_intersections(
    pathways: Dict[str, APPathway],
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    max_distance_from_center: float = DEFAULT_MAX_DISTANCE_FROM_CENTER,
    center_point: Tuple[float, float] = DEFAULT_CENTER_POINT,
) -> List[PairwiseIntersection]:
    """
    Calculate all pairwise intersection points between pathway lines.
    
    Applies filters for R², direction validity, ray intersection validity,
    and distance from center.
    
    Args:
        pathways: Dictionary of unit_id to APPathway
        r2_threshold: Minimum R² for including pathway
        max_distance_from_center: Maximum distance from center for valid intersection
        center_point: Center of visible area (col, row)
    
    Returns:
        List of PairwiseIntersection objects
    """
    intersections = []
    
    # Filter to valid pathways (R² threshold and direction valid)
    valid_pathways = {
        k: v for k, v in pathways.items()
        if v is not None and v.r_squared >= r2_threshold and v.direction_valid
    }
    
    if len(valid_pathways) < 2:
        logger.debug(f"Not enough valid pathways for intersection: {len(valid_pathways)}")
        return intersections
    
    unit_ids = list(valid_pathways.keys())
    
    for i in range(len(unit_ids)):
        for j in range(i + 1, len(unit_ids)):
            uid1, uid2 = unit_ids[i], unit_ids[j]
            p1, p2 = valid_pathways[uid1], valid_pathways[uid2]
            
            # Check if lines are parallel
            if abs(p1.slope - p2.slope) < 1e-10:
                continue
            
            # Calculate line intersection
            x = (p2.intercept - p1.intercept) / (p1.slope - p2.slope)
            y = p1.slope * x + p1.intercept
            
            # Calculate weights
            combined_r2 = p1.r_squared * p2.r_squared
            mean_r2 = (p1.r_squared + p2.r_squared) / 2.0
            
            # Check distance from center
            dist_from_center = np.sqrt(
                (x - center_point[0])**2 + (y - center_point[1])**2
            )
            
            if dist_from_center > max_distance_from_center:
                # Mark as invalid but still record
                intersections.append(PairwiseIntersection(
                    x=x, y=y, unit1=uid1, unit2=uid2,
                    combined_r2=combined_r2, mean_r2=mean_r2,
                    is_valid=False,
                ))
                continue
            
            # Check if this is a valid ray intersection
            is_valid_ray = True
            if (p1.start_point is not None and p1.direction_angle is not None and
                p2.start_point is not None and p2.direction_angle is not None):
                # Both rays should point toward the intersection
                valid1 = ray_points_toward_intersection(
                    p1.start_point, p1.direction_angle, (x, y)
                )
                valid2 = ray_points_toward_intersection(
                    p2.start_point, p2.direction_angle, (x, y)
                )
                is_valid_ray = valid1 and valid2
            
            intersections.append(PairwiseIntersection(
                x=x, y=y, unit1=uid1, unit2=uid2,
                combined_r2=combined_r2, mean_r2=mean_r2,
                is_valid=is_valid_ray,
            ))
    
    valid_count = sum(1 for i in intersections if i.is_valid)
    logger.debug(
        f"Pairwise intersections: {valid_count} valid of {len(intersections)} total"
    )
    
    return intersections


def cluster_intersections(
    intersections: List[PairwiseIntersection],
    eps: float = DEFAULT_CLUSTER_EPS,
    min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
) -> Tuple[List[PairwiseIntersection], List[PairwiseIntersection]]:
    """
    Cluster intersection points using DBSCAN and return the major cluster.
    
    Args:
        intersections: List of pairwise intersections
        eps: Maximum distance between points in same cluster
        min_samples: Minimum points to form a cluster
    
    Returns:
        Tuple of (main_cluster_intersections, outlier_intersections)
    """
    # Import sklearn here to avoid import overhead if not used
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        logger.warning("sklearn not available, skipping clustering")
        valid_ints = [i for i in intersections if i.is_valid]
        return valid_ints, []
    
    # Filter to valid intersections only
    valid_ints = [i for i in intersections if i.is_valid]
    
    if len(valid_ints) < min_samples:
        logger.debug(f"Not enough valid intersections for clustering: {len(valid_ints)}")
        return valid_ints, []
    
    # Extract coordinates for clustering
    coords = np.array([[i.x, i.y] for i in valid_ints])
    
    # Run DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    
    # Find the largest cluster (excluding noise points labeled -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    
    if not unique_labels:
        logger.warning(f"No clusters found with eps={eps}, min_samples={min_samples}")
        return [], valid_ints
    
    # Count points in each cluster
    cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
    largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    
    # Separate into main cluster and outliers
    main_cluster = []
    outliers = []
    
    for i, ri in enumerate(valid_ints):
        if labels[i] == largest_cluster_label:
            main_cluster.append(ri)
        else:
            # Create new object with is_valid=False for outliers
            outliers.append(PairwiseIntersection(
                x=ri.x, y=ri.y, unit1=ri.unit1, unit2=ri.unit2,
                combined_r2=ri.combined_r2, mean_r2=ri.mean_r2,
                is_valid=False,
            ))
    
    logger.debug(
        f"Clustering: {len(main_cluster)} points in main cluster, "
        f"{len(outliers)} outliers ({len(unique_labels)} total clusters)"
    )
    
    return main_cluster, outliers


def calculate_optimal_from_cluster(
    main_cluster: List[PairwiseIntersection],
) -> Tuple[Optional[Tuple[float, float]], float]:
    """
    Calculate optimal intersection from clustered points using weighted mean.
    
    Uses mean R² of the two lines that formed each intersection as weight.
    
    Args:
        main_cluster: List of intersection points in the main cluster
    
    Returns:
        Tuple of (optimal (x, y), mean squared error)
    """
    if not main_cluster:
        return None, 0.0
    
    # Weighted average using mean R² as weights
    x_vals = np.array([i.x for i in main_cluster])
    y_vals = np.array([i.y for i in main_cluster])
    weights = np.array([i.mean_r2 for i in main_cluster])
    
    # Normalize weights
    weights_norm = weights / weights.sum()
    
    x_opt = np.sum(x_vals * weights_norm)
    y_opt = np.sum(y_vals * weights_norm)
    
    # Calculate MSE (variance from optimal point)
    distances = np.sqrt((x_vals - x_opt)**2 + (y_vals - y_opt)**2)
    mse = np.mean(distances**2)
    
    return (float(x_opt), float(y_opt)), float(mse)


def calculate_enhanced_intersection(
    pathways: Dict[str, APPathway],
    all_centroids: Optional[Dict[str, np.ndarray]] = None,
    r2_threshold: float = DEFAULT_R2_THRESHOLD,
    direction_tolerance: float = DEFAULT_DIRECTION_TOLERANCE,
    max_distance_from_center: float = DEFAULT_MAX_DISTANCE_FROM_CENTER,
    center_point: Tuple[float, float] = DEFAULT_CENTER_POINT,
    cluster_eps: float = DEFAULT_CLUSTER_EPS,
    cluster_min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
    centroid_exclude_fraction: float = 0.1,
    min_remaining_fraction: float = 0.2,
) -> Optional[ONHResult]:
    """
    Calculate optic nerve head location using global optimization.
    
    New algorithm (when all_centroids provided):
    1. Filter cells by R² threshold and direction validity
    2. Exclude first 10% of centroids for each cell (beginning part often not straight)
    3. Find ONH point minimizing sum of squared perpendicular distances from
       all centroids to their respective lines (ONH -> cell start point)
    
    Falls back to legacy pairwise intersection clustering if all_centroids not provided.
    
    Args:
        pathways: Dictionary of unit_id to APPathway (with direction info populated)
        all_centroids: Dictionary of unit_id to centroids array [t, row, col]
        r2_threshold: Minimum R² for including pathway
        direction_tolerance: Maximum angle difference from consensus
        max_distance_from_center: Maximum distance from center for valid intersection
        center_point: Center of visible area (col, row)
        cluster_eps: DBSCAN epsilon parameter (legacy method)
        cluster_min_samples: DBSCAN min_samples parameter (legacy method)
        centroid_exclude_fraction: Fraction of centroids to exclude (default 0.1 = 10%)
        min_remaining_fraction: Stop iterative refinement if < this fraction remains (default 0.2 = 20%)
    
    Returns:
        ONHResult with optimal intersection and metadata, or None if cannot calculate
    """
    # Filter out None pathways
    valid_pathways = {k: v for k, v in pathways.items() if v is not None}
    
    if len(valid_pathways) < 2:
        logger.warning(f"Need at least 2 pathways, got {len(valid_pathways)}")
        return None
    
    # Step 1: Calculate consensus direction
    consensus_direction = calculate_consensus_direction(valid_pathways, r2_threshold)
    
    # Step 2: Filter by direction
    if consensus_direction is not None:
        valid_pathways = filter_pathways_by_direction(
            valid_pathways, consensus_direction, direction_tolerance
        )
    
    # Count pathways passing R² and direction filters
    n_passing_filters = sum(
        1 for pw in valid_pathways.values()
        if pw.r_squared >= r2_threshold and pw.direction_valid
    )
    
    if n_passing_filters < 2:
        logger.warning(
            f"Not enough pathways passing filters: {n_passing_filters} "
            f"(need at least 2)"
        )
        return None
    
    # Use iterative refinement method if centroids are provided
    if all_centroids is not None:
        # Prepare cell data: exclude first 10% of centroids for each cell
        cell_data, initial_unit_ids = prepare_cell_data_for_onh(
            valid_pathways, all_centroids, r2_threshold, centroid_exclude_fraction
        )
        
        if len(cell_data) < 2:
            logger.warning(f"Not enough cells with valid centroids: {len(cell_data)}")
            return None
        
        # Find optimal ONH using iterative refinement
        # This removes pathways that push the ONH toward the boundary
        optimal, total_error, n_centroids, kept_unit_ids, removed_unit_ids = find_onh_with_iterative_refinement(
            cell_data,
            initial_unit_ids,
            consensus_direction=consensus_direction,  # Use direction constraint
            max_distance_from_center=max_distance_from_center,
            center_point=center_point,
            boundary_tolerance=0.95,  # Consider "at boundary" if > 95% of max
            min_pathways=5,  # Need at least 5 pathways
            max_iterations=50,  # Safety limit
            min_remaining_fraction=min_remaining_fraction,  # Stop if < 20% of pathways remain
        )
        
        if optimal is None:
            logger.warning("Iterative refinement failed")
            return None
        
        n_cells = len(kept_unit_ids)
        mse = total_error / n_centroids if n_centroids > 0 else 0.0
        rmse = np.sqrt(mse)
        
        logger.info(
            f"ONH detected at ({optimal[0]:.2f}, {optimal[1]:.2f}), "
            f"RMSE={rmse:.2f}, cells={n_cells}, "
            f"removed {len(removed_unit_ids)} pathways for boundary compliance, "
            f"centroids={n_centroids}"
        )
        
        return ONHResult(
            x=optimal[0],
            y=optimal[1],
            mse=mse,
            rmse=rmse,
            n_cluster_points=0,  # Not used in new method
            n_total_intersections=len(initial_unit_ids),  # Initial unit count
            n_valid_after_direction=n_passing_filters,
            n_valid_after_distance=n_passing_filters,
            consensus_direction=consensus_direction,
            r2_threshold=r2_threshold,
            direction_tolerance=direction_tolerance,
            max_distance_from_center=max_distance_from_center,
            cluster_eps=cluster_eps,
            cluster_min_samples=cluster_min_samples,
            cluster_points=None,  # Not used in new method
            method="iterative_refinement",
            centroid_exclude_fraction=centroid_exclude_fraction,
            n_cells_used=n_cells,
            n_centroids_used=n_centroids,
            total_squared_error=total_error,
            outlier_unit_ids_stage1=None,  # Not used in iterative refinement
            outlier_unit_ids_stage2=None,  # Not used in iterative refinement
            outlier_unit_ids=removed_unit_ids,  # Pathways removed for boundary compliance
            kept_unit_ids=kept_unit_ids,
            max_outlier_fraction=None,  # Not applicable for iterative refinement
        )
    
    # Fallback to legacy pairwise intersection clustering method
    logger.info("Using legacy pairwise intersection clustering method")
    
    # Step 3: Calculate pairwise intersections with distance filtering
    all_intersections = calculate_pairwise_intersections(
        valid_pathways, r2_threshold, max_distance_from_center, center_point
    )
    
    n_total = len(all_intersections)
    n_valid_direction = sum(1 for i in all_intersections if i.is_valid)
    
    # Step 4: Cluster intersections
    main_cluster, outliers = cluster_intersections(
        all_intersections, cluster_eps, cluster_min_samples
    )
    
    n_cluster_points = len(main_cluster)
    
    if n_cluster_points == 0:
        logger.warning("No points in main cluster after clustering")
        return None
    
    # Step 5: Calculate optimal from main cluster
    optimal, mse = calculate_optimal_from_cluster(main_cluster)
    
    if optimal is None:
        logger.warning("Could not calculate optimal intersection from cluster")
        return None
    
    rmse = np.sqrt(mse)
    
    # Extract cluster points as numpy array
    cluster_points = np.array([[i.x, i.y] for i in main_cluster], dtype=np.float32)
    
    logger.info(
        f"ONH detected at ({optimal[0]:.2f}, {optimal[1]:.2f}), "
        f"RMSE={rmse:.2f}, cluster={n_cluster_points}/{n_total} points"
    )
    
    return ONHResult(
        x=optimal[0],
        y=optimal[1],
        mse=mse,
        rmse=rmse,
        n_cluster_points=n_cluster_points,
        n_total_intersections=n_total,
        n_valid_after_direction=n_valid_direction,
        n_valid_after_distance=n_valid_direction,  # Same for now
        consensus_direction=consensus_direction,
        r2_threshold=r2_threshold,
        direction_tolerance=direction_tolerance,
        max_distance_from_center=max_distance_from_center,
        cluster_eps=cluster_eps,
        cluster_min_samples=cluster_min_samples,
        cluster_points=cluster_points,
        method="clustered_weighted_mean",
    )


# =============================================================================
# Polar Coordinate Utilities
# =============================================================================


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

