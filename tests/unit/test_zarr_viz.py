"""
Unit tests for zarr_viz module.

Tests the core functionality of tree parsing, plotting, and metadata formatting.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import shutil

import numpy as np
import pytest
import zarr

from hdmea.viz.zarr_viz.tree import (
    TreeNode,
    parse_zarr_tree,
    get_node_by_path,
)
from hdmea.viz.zarr_viz.utils import (
    sample_array,
    should_warn_large,
    estimate_memory,
    validate_zarr_path,
    InvalidZarrPathError,
)
from hdmea.viz.zarr_viz.metadata import (
    format_array_info,
    format_attributes,
)

# Import fixture generator
from tests.fixtures.synthetic_zarr import create_synthetic_zarr, create_empty_zarr


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_zarr_path(tmp_path):
    """Create a temporary zarr archive for testing."""
    zarr_path = tmp_path / "test.zarr"
    create_synthetic_zarr(zarr_path)
    yield zarr_path


@pytest.fixture
def empty_zarr_path(tmp_path):
    """Create an empty zarr archive for testing."""
    zarr_path = tmp_path / "empty.zarr"
    create_empty_zarr(zarr_path)
    yield zarr_path


# =============================================================================
# TreeNode Tests
# =============================================================================


class TestTreeNode:
    """Tests for TreeNode dataclass."""

    def test_array_node_properties(self):
        """Test TreeNode properties for array nodes."""
        node = TreeNode(
            path="/data/array1",
            name="array1",
            node_type="array",
            shape=(100,),
            dtype="float64",
        )

        assert node.is_array
        assert not node.is_group
        assert node.icon == "📊"
        assert "array1" in node.display_name
        assert "(100,)" in node.display_name

    def test_group_node_properties(self):
        """Test TreeNode properties for group nodes."""
        node = TreeNode(
            path="/data",
            name="data",
            node_type="group",
            children=[],
        )

        assert node.is_group
        assert not node.is_array
        assert node.icon == "📁"


# =============================================================================
# Tree Parsing Tests
# =============================================================================


class TestParseZarrTree:
    """Tests for parse_zarr_tree function."""

    def test_parse_basic_structure(self, temp_zarr_path):
        """Test parsing a basic zarr archive."""
        tree = parse_zarr_tree(temp_zarr_path)

        assert tree is not None
        assert tree.is_group
        assert tree.name == temp_zarr_path.name
        assert len(tree.children) > 0

    def test_parse_finds_arrays(self, temp_zarr_path):
        """Test that parsing finds array nodes."""
        tree = parse_zarr_tree(temp_zarr_path)

        # Recursively find all arrays
        arrays = []

        def find_arrays(node):
            if node.is_array:
                arrays.append(node)
            for child in node.children:
                find_arrays(child)

        find_arrays(tree)

        assert len(arrays) > 0
        # All arrays should have shape and dtype
        for arr in arrays:
            assert arr.shape is not None
            assert arr.dtype is not None

    def test_parse_empty_archive(self, empty_zarr_path):
        """Test parsing an empty zarr archive."""
        tree = parse_zarr_tree(empty_zarr_path)

        assert tree is not None
        assert tree.is_group
        assert len(tree.children) == 0

    def test_parse_invalid_path(self, tmp_path):
        """Test parsing with invalid path."""
        with pytest.raises(InvalidZarrPathError):
            parse_zarr_tree(tmp_path / "nonexistent")


class TestGetNodeByPath:
    """Tests for get_node_by_path function."""

    def test_find_root(self, temp_zarr_path):
        """Test finding root node."""
        tree = parse_zarr_tree(temp_zarr_path)
        node = get_node_by_path(tree, "/")

        assert node is not None
        assert node.path == "/"

    def test_find_nested_group(self, temp_zarr_path):
        """Test finding nested group."""
        tree = parse_zarr_tree(temp_zarr_path)
        node = get_node_by_path(tree, "/data_1d")

        assert node is not None
        assert node.is_group
        assert node.name == "data_1d"

    def test_find_array(self, temp_zarr_path):
        """Test finding array node."""
        tree = parse_zarr_tree(temp_zarr_path)
        node = get_node_by_path(tree, "/data_1d/simple")

        assert node is not None
        assert node.is_array
        assert node.name == "simple"

    def test_find_nonexistent(self, temp_zarr_path):
        """Test finding non-existent path."""
        tree = parse_zarr_tree(temp_zarr_path)
        node = get_node_by_path(tree, "/does/not/exist")

        assert node is None


# =============================================================================
# Utils Tests
# =============================================================================


class TestSampleArray:
    """Tests for sample_array function."""

    def test_small_array_unchanged(self, temp_zarr_path):
        """Test that small arrays are not sampled."""
        root = zarr.open(str(temp_zarr_path), mode="r")
        array = root["data_1d/simple"]

        result = sample_array(array)

        assert len(result) == array.shape[0]

    def test_large_array_sampled(self):
        """Test that large arrays are sampled."""
        # Create a large temporary array
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.zarr"
            root = zarr.open(str(path), mode="w")
            data = np.random.rand(100000).astype("float64")
            large_array = root.create_dataset(
                "large",
                data=data,
                shape=data.shape,
                dtype=data.dtype,
            )

            result = sample_array(large_array, max_elements=1000)

            assert len(result) <= 1000


class TestShouldWarnLarge:
    """Tests for should_warn_large function."""

    def test_small_array_no_warning(self, temp_zarr_path):
        """Test small arrays don't trigger warning."""
        root = zarr.open(str(temp_zarr_path), mode="r")
        array = root["data_1d/simple"]

        assert not should_warn_large(array)

    def test_large_array_warning(self):
        """Test large arrays trigger warning."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.zarr"
            root = zarr.open(str(path), mode="w")
            # Create 200MB array (threshold is 100MB)
            size = (5000, 5000)  # 25M elements * 8 bytes = 200MB
            data = np.zeros(size, dtype="float64")
            large_array = root.create_dataset(
                "large",
                data=data,
                shape=data.shape,
                dtype=data.dtype,
            )

            assert should_warn_large(large_array, threshold_mb=100)


class TestValidateZarrPath:
    """Tests for validate_zarr_path function."""

    def test_valid_path(self, temp_zarr_path):
        """Test validation of valid zarr path."""
        result = validate_zarr_path(temp_zarr_path)
        assert result == temp_zarr_path

    def test_nonexistent_path(self, tmp_path):
        """Test validation of non-existent path."""
        with pytest.raises(InvalidZarrPathError, match="does not exist"):
            validate_zarr_path(tmp_path / "nonexistent")

    def test_file_not_directory(self, tmp_path):
        """Test validation of file (not directory)."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(InvalidZarrPathError, match="not a directory"):
            validate_zarr_path(file_path)

    def test_not_zarr_archive(self, tmp_path):
        """Test validation of directory that's not a zarr archive."""
        dir_path = tmp_path / "not_zarr"
        dir_path.mkdir()

        with pytest.raises(InvalidZarrPathError, match="not a valid zarr"):
            validate_zarr_path(dir_path)


# =============================================================================
# Metadata Tests
# =============================================================================


class TestFormatArrayInfo:
    """Tests for format_array_info function."""

    def test_format_basic_info(self, temp_zarr_path):
        """Test formatting basic array info."""
        root = zarr.open(str(temp_zarr_path), mode="r")
        array = root["data_1d/simple"]

        info = format_array_info(array)

        assert "Shape" in info
        assert "Data Type" in info
        assert "(100,)" in info["Shape"]


class TestFormatAttributes:
    """Tests for format_attributes function."""

    def test_format_string_attrs(self, temp_zarr_path):
        """Test formatting string attributes."""
        root = zarr.open(str(temp_zarr_path), mode="r")

        attrs = format_attributes(root.attrs)

        assert "name" in attrs
        assert attrs["name"] == "Synthetic Test Archive"

    def test_format_empty_attrs(self, empty_zarr_path):
        """Test formatting empty attributes."""
        root = zarr.open(str(empty_zarr_path), mode="r")

        attrs = format_attributes(root.attrs)

        assert isinstance(attrs, dict)
