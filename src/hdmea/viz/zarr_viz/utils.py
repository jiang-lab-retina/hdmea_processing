"""
Shared utilities for zarr_viz module (now supports HDF5).

Provides functions for memory management, array sampling, and size estimation.
Supports both HDF5 files (.h5) and legacy Zarr archives (.zarr).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)

__all__ = [
    "ZarrVizError",
    "InvalidZarrPathError",
    "InvalidHDF5PathError",
    "UnsupportedArrayError",
    "sample_array",
    "should_warn_large",
    "estimate_memory",
    "validate_zarr_path",
    "validate_hdf5_path",
]


# =============================================================================
# Custom Exceptions
# =============================================================================


class ZarrVizError(Exception):
    """Base exception for zarr_viz module."""

    pass


class InvalidZarrPathError(ZarrVizError):
    """Raised when zarr path is invalid or inaccessible."""

    pass


class InvalidHDF5PathError(ZarrVizError):
    """Raised when HDF5 path is invalid or inaccessible."""

    pass


class UnsupportedArrayError(ZarrVizError):
    """Raised when array cannot be visualized (e.g., non-numeric)."""

    pass


# =============================================================================
# Array Utilities
# =============================================================================


def sample_array(
    array: Union["h5py.Dataset", np.ndarray],
    max_elements: int = 10_000_000,
) -> np.ndarray:
    """Sample large array to fit in memory.

    For arrays exceeding max_elements, uniformly samples along the first
    axis to reduce the total number of elements.

    Args:
        array: HDF5 dataset or NumPy array to sample from.
        max_elements: Maximum number of elements to load (default 10M).

    Returns:
        NumPy array with at most max_elements elements.
    """
    total_elements = int(np.prod(array.shape))

    if total_elements <= max_elements:
        logger.debug(f"Loading full array: {array.shape} ({total_elements:,} elements)")
        return np.asarray(array[:])

    # Calculate sampling rate along first axis
    other_dims_size = int(np.prod(array.shape[1:])) if len(array.shape) > 1 else 1
    target_first_dim = max(1, int(max_elements / other_dims_size))
    target_first_dim = min(target_first_dim, array.shape[0])

    # Create uniform indices
    indices = np.linspace(0, array.shape[0] - 1, target_first_dim, dtype=int)

    logger.info(
        f"Sampling array: {array.shape} -> ({target_first_dim}, ...) "
        f"({target_first_dim * other_dims_size:,} elements)"
    )

    return np.asarray(array[indices])


def should_warn_large(
    array: Union["h5py.Dataset", np.ndarray],
    threshold_mb: int = 100,
) -> bool:
    """Check if array exceeds size warning threshold.

    Args:
        array: HDF5 dataset or array to check.
        threshold_mb: Size threshold in megabytes (default 100MB).

    Returns:
        True if array size exceeds threshold, False otherwise.
    """
    size_mb = estimate_memory(array) / (1024 * 1024)
    return size_mb > threshold_mb


def estimate_memory(array: Union["h5py.Dataset", np.ndarray]) -> int:
    """Estimate memory required to load array.

    Args:
        array: HDF5 dataset or array to estimate.

    Returns:
        Estimated size in bytes.
    """
    # Get itemsize from dtype
    itemsize = np.dtype(array.dtype).itemsize
    total_elements = int(np.prod(array.shape))
    return total_elements * itemsize


# =============================================================================
# Path Validation
# =============================================================================


def validate_hdf5_path(path: str | Path) -> Path:
    """Validate that path points to a valid HDF5 file.

    Args:
        path: Path to validate.

    Returns:
        Validated Path object.

    Raises:
        InvalidHDF5PathError: If path is invalid or not an HDF5 file.
    """
    path = Path(path)

    if not path.exists():
        raise InvalidHDF5PathError(f"Path does not exist: {path}")

    if not path.is_file():
        raise InvalidHDF5PathError(f"Path is not a file: {path}")

    # Check file extension
    if path.suffix.lower() not in ('.h5', '.hdf5', '.hdf'):
        raise InvalidHDF5PathError(
            f"Path does not have HDF5 extension (.h5, .hdf5): {path}"
        )

    return path


def validate_zarr_path(path: str | Path) -> Path:
    """Validate that path points to a valid zarr archive (legacy support).

    Args:
        path: Path to validate.

    Returns:
        Validated Path object.

    Raises:
        InvalidZarrPathError: If path is invalid or not a zarr archive.
    """
    path = Path(path)

    if not path.exists():
        raise InvalidZarrPathError(f"Path does not exist: {path}")

    if not path.is_dir():
        raise InvalidZarrPathError(f"Path is not a directory: {path}")

    # Check for zarr metadata files (v2 or v3)
    has_v2_metadata = (path / ".zgroup").exists() or (path / ".zarray").exists()
    has_v3_metadata = (path / "zarr.json").exists()

    if not (has_v2_metadata or has_v3_metadata):
        raise InvalidZarrPathError(
            f"Path is not a valid zarr archive (no metadata found): {path}"
        )

    return path
