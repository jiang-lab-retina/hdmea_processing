"""
Pytest configuration and shared fixtures for HD-MEA pipeline tests.

This module provides:
    - Synthetic data generators
    - Temporary Zarr store fixtures
    - Common test utilities
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_zarr_path(temp_dir):
    """Provide a path for a temporary Zarr store."""
    return temp_dir / "test_recording.zarr"


@pytest.fixture
def sample_dataset_id():
    """Provide a sample dataset ID for testing."""
    return "TEST001_2025-01-01"

