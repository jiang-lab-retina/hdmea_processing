"""
Metadata display formatting for zarr_viz module (HDF5 compatible).

Provides functions for formatting HDF5/zarr attributes and array properties
for display in the UI.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)

__all__ = [
    "format_array_info",
    "format_group_info",
    "format_attributes",
]


def format_array_info(array) -> dict[str, Any]:
    """Format array/dataset metadata for display.

    Args:
        array: HDF5 dataset or Zarr array object.

    Returns:
        Dict with formatted metadata fields.
    """
    info = {
        "Shape": str(array.shape),
        "Data Type": str(array.dtype),
    }
    
    # Chunks - different attribute access for HDF5 vs Zarr
    try:
        if hasattr(array, "chunks") and array.chunks:
            info["Chunks"] = str(array.chunks)
        else:
            info["Chunks"] = "Not chunked"
    except Exception:
        info["Chunks"] = "Unknown"

    # Add size info - HDF5 uses id.get_storage_size(), Zarr uses nbytes
    try:
        size_bytes = None
        if hasattr(array, 'id') and hasattr(array.id, 'get_storage_size'):
            # HDF5 dataset
            size_bytes = array.id.get_storage_size()
        elif hasattr(array, 'nbytes'):
            # Zarr array
            size_bytes = array.nbytes
            
        if size_bytes:
            if size_bytes < 1024:
                info["Size"] = f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                info["Size"] = f"{size_bytes / 1024:.2f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                info["Size"] = f"{size_bytes / (1024 * 1024):.2f} MB"
            else:
                info["Size"] = f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    except Exception:
        pass

    # Add compression info if available
    try:
        if hasattr(array, "compression") and array.compression:
            info["Compression"] = str(array.compression)
            if hasattr(array, "compression_opts") and array.compression_opts:
                info["Compression Level"] = str(array.compression_opts)
        elif hasattr(array, "compressor") and array.compressor:
            # Zarr uses compressor attribute
            info["Compression"] = str(array.compressor)
    except Exception:
        pass

    # Add fill value if set
    try:
        if hasattr(array, "fillvalue") and array.fillvalue is not None:
            info["Fill Value"] = str(array.fillvalue)
        elif hasattr(array, "fill_value") and array.fill_value is not None:
            # Zarr uses fill_value
            info["Fill Value"] = str(array.fill_value)
    except Exception:
        pass

    return info


def format_group_info(group) -> dict[str, Any]:
    """Format group metadata for display.

    Args:
        group: HDF5 group/file or Zarr group object.

    Returns:
        Dict with formatted metadata fields.
    """
    info = {}

    # Count children - works for both HDF5 and Zarr
    try:
        num_groups = 0
        num_arrays = 0
        
        for name in group.keys():
            child = group[name]
            # Check if it's an array/dataset
            if hasattr(child, 'shape') and hasattr(child, 'dtype'):
                # Has array-like properties
                if not hasattr(child, 'keys'):
                    # It's a leaf node (array/dataset)
                    num_arrays += 1
                else:
                    num_groups += 1
            else:
                num_groups += 1

        info["Groups"] = str(num_groups)
        info["Datasets"] = str(num_arrays)
        info["Total Children"] = str(num_groups + num_arrays)
    except Exception as e:
        logger.warning(f"Failed to count children: {e}")

    return info


def format_attributes(attrs: Mapping) -> dict[str, str]:
    """Format zarr attributes, handling non-serializable values.

    Args:
        attrs: Zarr attributes mapping (from .attrs).

    Returns:
        Dict with string representations of all attributes.
    """
    result = {}

    try:
        # Convert to dict if it's a special attrs object
        if hasattr(attrs, "asdict"):
            attrs_dict = attrs.asdict()
        else:
            attrs_dict = dict(attrs)
    except Exception:
        attrs_dict = {}

    for key, value in attrs_dict.items():
        try:
            result[str(key)] = _format_value(value)
        except Exception as e:
            result[str(key)] = f"<Error: {e}>"

    return result


def _format_value(value: Any) -> str:
    """Format a single attribute value.

    Args:
        value: Attribute value to format.

    Returns:
        String representation.
    """
    if value is None:
        return "None"

    if isinstance(value, str):
        # Truncate long strings
        if len(value) > 200:
            return value[:200] + "..."
        return value

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, (list, tuple)):
        if len(value) > 10:
            formatted = ", ".join(str(v) for v in value[:10])
            return f"[{formatted}, ... ({len(value)} items)]"
        return str(value)

    if isinstance(value, dict):
        try:
            formatted = json.dumps(value, indent=2, default=str)
            if len(formatted) > 500:
                return formatted[:500] + "\n..."
            return formatted
        except Exception:
            return str(value)

    # Try JSON serialization for complex objects
    try:
        return json.dumps(value, default=str)
    except Exception:
        return f"<{type(value).__name__}>"
