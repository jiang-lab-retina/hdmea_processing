# Contracts: Zarr Visualization GUI

This directory contains the internal API contracts for the zarr_viz module.

## Module APIs

### Public API (zarr_viz/__init__.py)

```python
from hdmea.viz.zarr_viz import launch

# Launch visualization tool
launch(zarr_path: str | Path | None = None) -> None
```

### tree.py

```python
@dataclass
class TreeNode:
    path: str
    name: str
    node_type: Literal["group", "array"]
    children: list["TreeNode"]
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    chunks: tuple[int, ...] | None = None
    nbytes: int | None = None

def parse_zarr_tree(zarr_path: Path) -> TreeNode:
    """Parse zarr archive into tree structure."""
    ...

def get_node_by_path(root: TreeNode, path: str) -> TreeNode | None:
    """Find node by its path in the tree."""
    ...
```

### plots.py

```python
def create_plot(
    array: zarr.Array,
    title: str | None = None,
    slice_indices: dict[int, int] | None = None,
) -> go.Figure:
    """Create appropriate plot based on array dimensions."""
    ...

def export_figure(
    fig: go.Figure,
    path: Path,
    format: Literal["png", "svg"] = "png",
    width: int = 1200,
    height: int = 800,
) -> None:
    """Export figure to file."""
    ...
```

### metadata.py

```python
def format_array_info(array: zarr.Array) -> dict[str, Any]:
    """Format array metadata for display."""
    ...

def format_attributes(attrs: Mapping) -> dict[str, str]:
    """Format zarr attributes, handling non-serializable values."""
    ...
```

### utils.py

```python
def sample_array(
    array: zarr.Array,
    max_elements: int = 10_000_000,
) -> np.ndarray:
    """Sample large array to fit in memory."""
    ...

def should_warn_large(
    array: zarr.Array,
    threshold_mb: int = 100,
) -> bool:
    """Check if array exceeds size warning threshold."""
    ...
```

## Error Types

```python
class ZarrVizError(Exception):
    """Base exception for zarr_viz module."""
    pass

class InvalidZarrPathError(ZarrVizError):
    """Raised when zarr path is invalid or inaccessible."""
    pass

class UnsupportedArrayError(ZarrVizError):
    """Raised when array cannot be visualized (e.g., non-numeric)."""
    pass
```
