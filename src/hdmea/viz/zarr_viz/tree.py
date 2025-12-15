"""
Zarr tree structure and traversal.

Provides data structures and functions for parsing and navigating
zarr archive hierarchies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import zarr

from hdmea.viz.zarr_viz.utils import InvalidZarrPathError, validate_zarr_path

logger = logging.getLogger(__name__)

__all__ = [
    "TreeNode",
    "parse_zarr_tree",
    "get_node_by_path",
    "render_tree",
]


@dataclass
class TreeNode:
    """Represents a node in the zarr hierarchy.

    Attributes:
        path: Full path within zarr archive (e.g., "/units/unit_000/spike_times")
        name: Node name (last component of path)
        node_type: Whether this is a "group" (container) or "array" (data)
        children: Child nodes (empty for arrays)
        shape: Array dimensions (None for groups)
        dtype: Data type string (None for groups)
        chunks: Chunk configuration (None for groups)
        nbytes: Total size in bytes (None for groups)
    """

    path: str
    name: str
    node_type: Literal["group", "array"]
    children: list["TreeNode"] = field(default_factory=list)
    shape: Optional[tuple[int, ...]] = None
    dtype: Optional[str] = None
    chunks: Optional[tuple[int, ...]] = None
    nbytes: Optional[int] = None

    @property
    def is_array(self) -> bool:
        """Check if this node is an array (leaf node)."""
        return self.node_type == "array"

    @property
    def is_group(self) -> bool:
        """Check if this node is a group (container)."""
        return self.node_type == "group"

    @property
    def icon(self) -> str:
        """Get display icon for this node type."""
        return "📊" if self.is_array else "📁"

    @property
    def display_name(self) -> str:
        """Get formatted display name with icon."""
        suffix = ""
        if self.is_array and self.shape:
            suffix = f" {self.shape}"
        return f"{self.icon} {self.name}{suffix}"


def parse_zarr_tree(zarr_path: str | Path) -> TreeNode:
    """Parse zarr archive into tree structure.

    Args:
        zarr_path: Path to zarr archive directory.

    Returns:
        Root TreeNode containing the complete hierarchy.

    Raises:
        InvalidZarrPathError: If path is invalid or not a zarr archive.
    """
    validated_path = validate_zarr_path(zarr_path)

    try:
        root = zarr.open(str(validated_path), mode="r")
    except Exception as e:
        raise InvalidZarrPathError(f"Failed to open zarr archive: {e}") from e

    logger.info(f"Parsing zarr tree: {validated_path}")

    return _parse_node(root, "/", validated_path.name)


def _parse_node(node: zarr.Group | zarr.Array, path: str, name: str) -> TreeNode:
    """Recursively parse a zarr node into TreeNode.

    Args:
        node: Zarr group or array object.
        path: Current path in the hierarchy.
        name: Name of this node.

    Returns:
        TreeNode representing this node and its children.
    """
    if isinstance(node, zarr.Array):
        # Leaf node - array
        return TreeNode(
            path=path,
            name=name,
            node_type="array",
            children=[],
            shape=tuple(node.shape),
            dtype=str(node.dtype),
            chunks=tuple(node.chunks) if node.chunks else None,
            nbytes=node.nbytes if hasattr(node, "nbytes") else None,
        )
    else:
        # Container node - group
        children = []
        try:
            # Get all items in the group
            for child_name in sorted(node.keys()):
                child_path = f"{path.rstrip('/')}/{child_name}"
                try:
                    child_node = node[child_name]
                    children.append(_parse_node(child_node, child_path, child_name))
                except Exception as e:
                    logger.warning(f"Failed to parse child {child_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to iterate group {path}: {e}")

        return TreeNode(
            path=path,
            name=name,
            node_type="group",
            children=children,
        )


def get_node_by_path(root: TreeNode, path: str) -> Optional[TreeNode]:
    """Find node by its path in the tree.

    Args:
        root: Root TreeNode to search from.
        path: Path to find (e.g., "/units/unit_000/spike_times").

    Returns:
        TreeNode if found, None otherwise.
    """
    if root.path == path:
        return root

    # Normalize path for comparison
    search_path = path.strip("/")
    root_path = root.path.strip("/")

    # If this is the root, start searching children
    if not search_path:
        return root

    # Check if the search path starts with root path
    if root_path and not search_path.startswith(root_path):
        # Path doesn't match this subtree
        return None

    # Get the remaining path after root
    if root_path:
        remaining = search_path[len(root_path):].strip("/")
    else:
        remaining = search_path

    if not remaining:
        return root

    # Find the next component
    parts = remaining.split("/", 1)
    next_name = parts[0]

    # Search children
    for child in root.children:
        if child.name == next_name:
            if len(parts) == 1:
                return child
            return get_node_by_path(child, path)

    return None


def render_tree(
    root: TreeNode,
    on_select: Callable[[str], None],
    expanded_nodes: set[str] | None = None,
    indent: int = 0,
) -> None:
    """Render tree structure using Streamlit widgets.

    This function should be called within a Streamlit app context.

    Args:
        root: Root TreeNode to render.
        on_select: Callback function when a node is selected.
        expanded_nodes: Set of node paths that should be expanded.
        indent: Current indentation level (for recursion).
    """
    import streamlit as st

    if expanded_nodes is None:
        expanded_nodes = set()

    # Create a unique key for this node
    key = f"node_{root.path.replace('/', '_')}"

    if root.is_array:
        # Leaf node - clickable button
        col1, col2 = st.columns([0.1 * indent + 1 if indent else 1, 4])
        with col2:
            if st.button(root.display_name, key=key, use_container_width=True):
                on_select(root.path)
    else:
        # Group node - expander with children
        if root.children:
            is_expanded = root.path in expanded_nodes
            with st.expander(root.display_name, expanded=is_expanded):
                for child in root.children:
                    render_tree(child, on_select, expanded_nodes, indent + 1)
        else:
            # Empty group
            st.text(f"{root.display_name} (empty)")
