"""
Metadata display formatting for zarr_viz module.

Provides functions for formatting zarr attributes and array properties
for display in the UI.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    import zarr

logger = logging.getLogger(__name__)

__all__ = [
    "format_array_info",
    "format_group_info",
    "format_attributes",
]


def format_array_info(array: "zarr.Array") -> dict[str, Any]:
    """Format array metadata for display.

    Args:
        array: Zarr array object.

    Returns:
        Dict with formatted metadata fields.
    """
    info = {
        "Shape": str(array.shape),
        "Data Type": str(array.dtype),
        "Chunks": str(array.chunks) if array.chunks else "Not chunked",
    }

    # Add size info
    if hasattr(array, "nbytes") and array.nbytes:
        size_bytes = array.nbytes
        if size_bytes < 1024:
            info["Size"] = f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            info["Size"] = f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            info["Size"] = f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            info["Size"] = f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    # Add compression info if available (handle Zarr v2 vs v3 API)
    try:
        # Zarr v3 uses compressors (plural)
        if hasattr(array, "compressors") and array.compressors:
            info["Compressor"] = str(array.compressors)
        # Zarr v2 uses compressor (singular)
        elif hasattr(array, "compressor") and array.compressor:
            info["Compressor"] = str(array.compressor)
    except (TypeError, AttributeError):
        # Zarr v3 raises TypeError when accessing compressor
        pass

    # Add fill value if set
    if hasattr(array, "fill_value") and array.fill_value is not None:
        info["Fill Value"] = str(array.fill_value)

    return info


def format_group_info(group: "zarr.Group") -> dict[str, Any]:
    """Format group metadata for display.

    Args:
        group: Zarr group object.

    Returns:
        Dict with formatted metadata fields.
    """
    info = {}

    # Count children
    try:
        num_groups = 0
        num_arrays = 0
        for name in group.keys():
            child = group[name]
            if hasattr(child, "shape"):
                num_arrays += 1
            else:
                num_groups += 1

        info["Groups"] = str(num_groups)
        info["Arrays"] = str(num_arrays)
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
