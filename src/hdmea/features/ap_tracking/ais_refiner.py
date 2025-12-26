"""
Soma and Axon Initial Segment (AIS) refinement algorithms.

This module provides functions to refine soma position and detect the
axon initial segment from 3D STA data.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RefinedSoma:
    """Refined soma position in 3D STA space."""

    t: int  # Time index
    x: int  # Row index
    y: int  # Column index


@dataclass
class AxonInitialSegment:
    """Axon initial segment position."""

    t: Optional[int]  # Time index (None if not found)
    x: int  # Row index
    y: int  # Column index


def _clip_coordinate_by_edge(
    coordinate: int,
    edge_range: Tuple[int, int] = (0, 65),
    span_radius: int = 5,
) -> Tuple[int, int]:
    """
    Clip coordinate search range to stay within valid bounds.

    Args:
        coordinate: Center coordinate
        edge_range: Valid range (min, max)
        span_radius: Search radius around coordinate

    Returns:
        Tuple of (start_idx, end_idx) clipped to valid range
    """
    start_idx = max(edge_range[0], coordinate - span_radius)
    end_idx = min(edge_range[1], coordinate + span_radius)
    return int(start_idx), int(end_idx)


def soma_refiner(
    sta: np.ndarray,
    soma_xy: Tuple[int, int],
    refine_radius: int = 5,
) -> Optional[RefinedSoma]:
    """
    Refine soma position by finding minimum value in local neighborhood.

    The algorithm searches a local neighborhood around the initial soma position
    and finds the voxel with the minimum value (strongest negative deflection).

    Args:
        sta: 3D STA array with shape (time, row, col)
        soma_xy: Initial soma position (row, col) from find_soma_from_3d_sta
        refine_radius: Search radius around initial position

    Returns:
        RefinedSoma with (t, x, y) coordinates, or None if failed

    Example:
        >>> sta = np.random.randn(50, 65, 65)
        >>> refined = soma_refiner(sta, (32, 32))
        >>> if refined:
        ...     print(f"Refined soma at t={refined.t}, x={refined.x}, y={refined.y}")
    """
    row, col = soma_xy

    # Clip search ranges to valid bounds
    x_start, x_end = _clip_coordinate_by_edge(
        row, edge_range=(0, sta.shape[1]), span_radius=refine_radius
    )
    y_start, y_end = _clip_coordinate_by_edge(
        col, edge_range=(0, sta.shape[2]), span_radius=refine_radius
    )

    if x_start is None or x_end is None or y_start is None or y_end is None:
        logger.warning(f"Invalid soma coordinates: ({row}, {col})")
        return None

    # Extract local STA data
    sta_local = sta[:, x_start:x_end, y_start:y_end]

    if sta_local.size == 0:
        logger.warning(f"Empty local STA slice for soma at ({row}, {col})")
        return None

    # Find minimum value (strongest negative deflection)
    flat_idx = np.argmin(sta_local)
    local_txy = np.unravel_index(flat_idx, sta_local.shape)

    # Convert back to global coordinates
    refined_t = int(local_txy[0])
    refined_x = int(local_txy[1] + x_start)
    refined_y = int(local_txy[2] + y_start)

    logger.debug(f"Soma refined from ({row}, {col}) to ({refined_x}, {refined_y}) at t={refined_t}")

    return RefinedSoma(t=refined_t, x=refined_x, y=refined_y)


def ais_refiner(
    sta: np.ndarray,
    soma_txy: RefinedSoma,
    search_xy_radius: int = 5,
    search_t_radius: int = 5,
) -> Optional[AxonInitialSegment]:
    """
    Detect axon initial segment near refined soma.

    The algorithm searches a local neighborhood around the soma and finds the
    electrode with the earliest peak (minimum time index), which likely corresponds
    to the axon initial segment where action potentials originate.

    Args:
        sta: 3D STA array with shape (time, row, col)
        soma_txy: Refined soma position from soma_refiner
        search_xy_radius: Spatial search radius around soma
        search_t_radius: Temporal search radius around soma time

    Returns:
        AxonInitialSegment with (t, x, y) coordinates, or None if not found

    Example:
        >>> sta = np.random.randn(50, 65, 65)
        >>> soma = soma_refiner(sta, (32, 32))
        >>> if soma:
        ...     ais = ais_refiner(sta, soma)
    """
    if soma_txy is None:
        return None

    sta_copy = sta.copy()
    time_dim, x_dim, y_dim = sta_copy.shape

    center_t_idx = soma_txy.t
    center_x_idx = soma_txy.x
    center_y_idx = soma_txy.y

    # Define search window boundaries
    t_start = max(0, center_t_idx - search_t_radius)
    t_end = min(time_dim, center_t_idx + search_t_radius + 1)

    y_search_start = max(0, center_y_idx - search_xy_radius)
    y_search_end = min(y_dim, center_y_idx + search_xy_radius + 1)

    x_search_start = max(0, center_x_idx - search_xy_radius)
    x_search_end = min(x_dim, center_x_idx + search_xy_radius + 1)

    # Extract local STA data
    sta_local = sta_copy[t_start:t_end, x_search_start:x_search_end, y_search_start:y_search_end]

    if sta_local.size == 0:
        logger.warning(
            f"Empty local search window for soma ({center_x_idx}, {center_y_idx})"
        )
        return None

    local_time_dim, local_x_dim, local_y_dim = sta_local.shape

    # Find the time index of the minimum for each local spatial location
    min_time_idx_array = np.full((local_x_dim, local_y_dim), np.inf)

    # Get reference trace at central electrode
    central_trace = sta_copy[:, center_x_idx, center_y_idx]
    central_min = np.min(central_trace)

    for x_local in range(local_x_dim):
        for y_local in range(local_y_dim):
            time_trace = sta_local[:, x_local, y_local]
            # Check if there's significant signal (30% of central electrode's min)
            if np.min(time_trace) < 0.3 * central_min:
                min_time_idx_array[x_local, y_local] = np.argmin(time_trace)

    # Check if any peaks were found
    if np.all(min_time_idx_array == np.inf):
        logger.debug(f"No significant peaks found near soma ({center_x_idx}, {center_y_idx})")
        return None

    # Find the local spatial index where the peak time is earliest (minimum)
    flat_idx = np.argmin(min_time_idx_array)
    x_local_ais, y_local_ais = np.unravel_index(flat_idx, min_time_idx_array.shape)

    # Convert local coordinates to global
    y_global_ais = y_search_start + y_local_ais
    x_global_ais = x_search_start + x_local_ais

    # Get the time index for this AIS location
    t_local_ais = min_time_idx_array[x_local_ais, y_local_ais]
    if t_local_ais == np.inf:
        t_global_ais = None
    else:
        t_global_ais = int(t_start + int(t_local_ais))

    logger.debug(
        f"AIS detected at t={t_global_ais}, x={x_global_ais}, y={y_global_ais}"
    )

    return AxonInitialSegment(t=t_global_ais, x=int(x_global_ais), y=int(y_global_ais))

