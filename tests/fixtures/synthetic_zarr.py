"""
Synthetic zarr archive generator for testing zarr_viz module.

Provides functions to create small, controlled zarr archives
for unit and integration testing.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import zarr
import pytest


def _create_array(group: zarr.Group, name: str, data: np.ndarray, **kwargs) -> zarr.Array:
    """Helper to create array compatible with zarr v2 and v3.

    Args:
        group: Parent zarr group.
        name: Array name.
        data: Array data.
        **kwargs: Additional arguments.

    Returns:
        Created zarr array.
    """
    # zarr v3 requires explicit shape
    return group.create_dataset(
        name,
        data=data,
        shape=data.shape,
        dtype=data.dtype,
        **kwargs,
    )


def create_synthetic_zarr(
    path: Path,
    include_1d: bool = True,
    include_2d: bool = True,
    include_nd: bool = True,
    include_nested: bool = True,
    include_attributes: bool = True,
) -> zarr.Group:
    """Create a synthetic zarr archive for testing.

    Args:
        path: Directory path for the zarr archive.
        include_1d: Include 1D array examples.
        include_2d: Include 2D array examples.
        include_nd: Include ND (>2) array examples.
        include_nested: Include nested group structure.
        include_attributes: Include zarr attributes.

    Returns:
        Root zarr group of the created archive.
    """
    root = zarr.open(str(path), mode="w")

    # Add root attributes
    if include_attributes:
        root.attrs["name"] = "Synthetic Test Archive"
        root.attrs["version"] = "1.0"
        root.attrs["created_by"] = "zarr_viz tests"

    # 1D arrays
    if include_1d:
        data_1d = root.create_group("data_1d")

        # Simple 1D array
        simple_data = np.linspace(0, 10, 100).astype("float64")
        arr_simple = _create_array(data_1d, "simple", simple_data)
        if include_attributes:
            arr_simple.attrs["description"] = "Simple linear ramp"
            arr_simple.attrs["unit"] = "arbitrary"

        # Sinusoid
        t = np.linspace(0, 4 * np.pi, 500)
        sin_data = np.sin(t).astype("float32")
        _create_array(data_1d, "sinusoid", sin_data)

        # Noisy data
        noisy_data = np.random.randn(1000).astype("float64")
        _create_array(data_1d, "noisy", noisy_data)

    # 2D arrays
    if include_2d:
        data_2d = root.create_group("data_2d")

        # Small image
        img = np.random.rand(64, 64).astype("float32")
        _create_array(data_2d, "image_small", img)

        # Larger image
        img_large = np.random.rand(256, 256).astype("float64")
        _create_array(data_2d, "image_large", img_large)

        # Integer array
        int_arr = np.arange(100).reshape(10, 10).astype("int32")
        _create_array(data_2d, "integers", int_arr)

    # ND arrays
    if include_nd:
        data_nd = root.create_group("data_nd")

        # 3D array
        arr_3d = np.random.rand(10, 20, 30).astype("float32")
        _create_array(data_nd, "cube", arr_3d)

        # 4D array
        arr_4d = np.random.rand(5, 10, 15, 20).astype("float64")
        _create_array(data_nd, "hypercube", arr_4d)

    # Nested structure
    if include_nested:
        units = root.create_group("units")

        for i in range(3):
            unit = units.create_group(f"unit_{i:03d}")

            # Spike times
            n_spikes = np.random.randint(50, 200)
            spike_times = np.sort(np.random.rand(n_spikes) * 100).astype("float64")
            _create_array(unit, "spike_times", spike_times)

            # Waveform
            waveform = np.random.randn(30).astype("float32")
            _create_array(unit, "waveform", waveform)

            if include_attributes:
                unit.attrs["cell_type"] = "RGC"
                unit.attrs["quality"] = np.random.choice(["good", "ok", "poor"])

    return root


def create_empty_zarr(path: Path) -> zarr.Group:
    """Create an empty zarr archive for testing edge cases.

    Args:
        path: Directory path for the zarr archive.

    Returns:
        Root zarr group (empty).
    """
    return zarr.open(str(path), mode="w")


def create_large_zarr(path: Path, size_mb: int = 10) -> zarr.Group:
    """Create a zarr archive with large arrays for testing warnings.

    Args:
        path: Directory path for the zarr archive.
        size_mb: Approximate size in megabytes.

    Returns:
        Root zarr group.
    """
    root = zarr.open(str(path), mode="w")

    # Calculate array size for target MB (float64 = 8 bytes)
    n_elements = (size_mb * 1024 * 1024) // 8
    side = int(np.sqrt(n_elements))

    root.create_dataset(
        "large_array",
        data=np.random.rand(side, side),
        dtype="float64",
    )

    return root


@pytest.fixture
def synthetic_zarr_path(tmp_path: Path) -> Iterator[Path]:
    """Pytest fixture providing a synthetic zarr archive.

    Yields:
        Path to a temporary zarr archive.
    """
    zarr_path = tmp_path / "test.zarr"
    create_synthetic_zarr(zarr_path)
    yield zarr_path


@pytest.fixture
def empty_zarr_path(tmp_path: Path) -> Iterator[Path]:
    """Pytest fixture providing an empty zarr archive.

    Yields:
        Path to an empty zarr archive.
    """
    zarr_path = tmp_path / "empty.zarr"
    create_empty_zarr(zarr_path)
    yield zarr_path
