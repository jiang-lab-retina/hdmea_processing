"""
Post-processing of CNN predictions for axon detection.

This module provides functions to filter noise from prediction data
and extract axon centroids from the processed predictions.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import label, center_of_mass

logger = logging.getLogger(__name__)


def filter_noise_slice(
    image_slice: np.ndarray,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """
    Filter noise from a single 2D image slice using connected components.

    Args:
        image_slice: 2D numpy array (row, col)
        min_cluster_size: Minimum number of connected pixels to keep

    Returns:
        Filtered 2D image slice
    """
    if not isinstance(image_slice, np.ndarray) or image_slice.ndim != 2:
        logger.warning("filter_noise_slice received invalid input")
        return image_slice

    # Create a binary mask of non-zero pixels
    binary_mask = image_slice > 0
    total_valid_pixels = np.sum(binary_mask)

    # If there are very few points overall, keep them
    if 0 < total_valid_pixels < min_cluster_size:
        return image_slice

    if total_valid_pixels == 0:
        return image_slice

    # Find connected components in the binary mask
    labeled_array, num_features = label(binary_mask)

    filtered_slice = np.zeros_like(image_slice)

    # Keep only components larger than min_cluster_size
    for i in range(1, num_features + 1):
        component_mask = labeled_array == i
        component_size = np.sum(component_mask)

        if component_size >= min_cluster_size:
            filtered_slice[component_mask] = image_slice[component_mask]

    return filtered_slice


def improved_filter_noise_3d(
    unit_data: np.ndarray,
    min_cluster_size: int = 3,
    threshold: float = 0.1,
    smooth_radius: int = 1,
) -> np.ndarray:
    """
    Enhanced 3D noise filtering with threshold and temporal consistency.

    Args:
        unit_data: 3D numpy array (time, row, col)
        min_cluster_size: Minimum cluster size for filtering
        threshold: Signal intensity threshold (below = noise)
        smooth_radius: Smoothing radius for temporal consistency check

    Returns:
        Filtered 3D data
    """
    if not isinstance(unit_data, np.ndarray) or unit_data.ndim != 3:
        logger.warning("improved_filter_noise_3d received invalid input")
        return unit_data

    t_dim, x_dim, y_dim = unit_data.shape

    # Step 1: Threshold preprocessing
    thresholded_data = unit_data.copy()
    thresholded_data[thresholded_data < threshold] = 0

    # Step 2: Spatial filtering per time slice
    filtered_data = np.zeros_like(unit_data)
    for t in range(t_dim):
        filtered_data[t, :, :] = filter_noise_slice(
            thresholded_data[t, :, :], min_cluster_size
        )

    # Step 3: Temporal consistency check
    if t_dim > 2:
        temp_result = filtered_data.copy()
        for t in range(1, t_dim - 1):
            current_slice = temp_result[t, :, :].copy()
            prev_slice = temp_result[t - 1, :, :]
            next_slice = temp_result[t + 1, :, :]

            # Find active points in current frame
            active_points = np.where(current_slice > 0)
            for i in range(len(active_points[0])):
                x, y = active_points[0][i], active_points[1][i]

                # Define search area
                x_min = max(0, x - smooth_radius)
                x_max = min(x_dim, x + smooth_radius + 1)
                y_min = max(0, y - smooth_radius)
                y_max = min(y_dim, y + smooth_radius + 1)

                # Check if adjacent frames have active points nearby
                prev_active = np.any(prev_slice[x_min:x_max, y_min:y_max] > 0)
                next_active = np.any(next_slice[x_min:x_max, y_min:y_max] > 0)

                # If isolated in time, remove
                if not prev_active and not next_active:
                    temp_result[t, x, y] = 0

        filtered_data = temp_result

    return filtered_data


def extract_axon_centroids(
    filtered_prediction: np.ndarray,
    exclude_center: Optional[Tuple[int, int]] = None,
    exclude_radius: int = 5,
    max_displacement: int = 5,
) -> np.ndarray:
    """
    Extract axon centroids from filtered prediction data.

    Args:
        filtered_prediction: 3D filtered prediction array (time, row, col)
        exclude_center: Optional (row, col) center to exclude (e.g., soma)
        exclude_radius: Radius around center to exclude
        max_displacement: Maximum allowed displacement between frames

    Returns:
        Array of centroids with shape (N, 3) for (t, x, y)
    """
    t_dim = filtered_prediction.shape[0]
    centroids = []

    prev_centroid = None

    for t in range(t_dim):
        frame = filtered_prediction[t].copy()

        # Exclude soma region if specified
        if exclude_center is not None:
            cx, cy = exclude_center
            for dx in range(-exclude_radius, exclude_radius + 1):
                for dy in range(-exclude_radius, exclude_radius + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < frame.shape[0] and 0 <= ny < frame.shape[1]:
                        frame[nx, ny] = 0

        # Find connected components
        binary = frame > 0
        if not np.any(binary):
            continue

        labeled_array, num_features = label(binary)

        # Get centroid of largest component
        component_sizes = []
        for i in range(1, num_features + 1):
            component_sizes.append((i, np.sum(labeled_array == i)))

        if not component_sizes:
            continue

        # Sort by size, get largest
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        largest_label = component_sizes[0][0]

        # Calculate centroid of largest component
        com = center_of_mass(frame, labeled_array, largest_label)

        if np.any(np.isnan(com)):
            continue

        # Check displacement constraint
        if prev_centroid is not None and max_displacement > 0:
            dist = np.sqrt(
                (com[0] - prev_centroid[0]) ** 2 + (com[1] - prev_centroid[1]) ** 2
            )
            if dist > max_displacement:
                continue

        centroids.append([t, com[0], com[1]])
        prev_centroid = com

    if centroids:
        return np.array(centroids, dtype=np.float32)
    else:
        return np.array([], dtype=np.float32).reshape(0, 3)


def process_predictions(
    prediction: np.ndarray,
    soma_xy: Optional[Tuple[int, int]] = None,
    temporal_window_size: int = 5,
    exclude_radius: int = 5,
    centroid_threshold: float = 0.05,
    max_displacement: int = 5,
    min_cluster_size: int = 3,
) -> Dict:
    """
    Process CNN predictions to extract axon tracking data.

    Args:
        prediction: 3D prediction array (time, row, col)
        soma_xy: Optional soma position (row, col) to exclude
        temporal_window_size: Window size for temporal filtering
        exclude_radius: Radius around soma to exclude
        centroid_threshold: Threshold for centroid extraction
        max_displacement: Maximum frame-to-frame displacement
        min_cluster_size: Minimum cluster size for noise filtering

    Returns:
        Dictionary with 'filtered_prediction' and 'axon_centroids'
    """
    # Apply noise filtering
    filtered = improved_filter_noise_3d(
        prediction,
        min_cluster_size=min_cluster_size,
        threshold=centroid_threshold,
        smooth_radius=temporal_window_size // 2,
    )

    # Extract centroids
    centroids = extract_axon_centroids(
        filtered,
        exclude_center=soma_xy,
        exclude_radius=exclude_radius,
        max_displacement=max_displacement,
    )

    logger.debug(f"Extracted {len(centroids)} axon centroids")

    return {
        "filtered_prediction": filtered,
        "axon_centroids": centroids,
    }

