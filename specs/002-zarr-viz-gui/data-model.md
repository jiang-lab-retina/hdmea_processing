# Data Model: Zarr Visualization GUI

**Feature**: 002-zarr-viz-gui  
**Date**: 2025-01-14

## Entities

### TreeNode

Represents a node in the zarr hierarchy (group or array).

| Field | Type | Description |
|-------|------|-------------|
| path | str | Full path within zarr archive (e.g., "/units/unit_000/spike_times") |
| name | str | Node name (last component of path) |
| node_type | Literal["group", "array"] | Whether this is a container or data |
| children | list[TreeNode] | Child nodes (empty for arrays) |
| is_expanded | bool | UI state - whether node is expanded in tree view |
| is_selected | bool | UI state - whether node is currently selected |

**For arrays only**:

| Field | Type | Description |
|-------|------|-------------|
| shape | tuple[int, ...] | Array dimensions |
| dtype | str | Data type (e.g., "float32", "int64") |
| chunks | tuple[int, ...] | Chunk configuration |
| nbytes | int | Total size in bytes |

### ZarrArchive

Represents an opened zarr archive.

| Field | Type | Description |
|-------|------|-------------|
| path | Path | Filesystem path to zarr directory |
| root | zarr.Group | Root group of the archive |
| tree | TreeNode | Parsed tree structure |
| attributes | dict | Root-level attributes |

### PlotConfig

Configuration for plot generation.

| Field | Type | Description |
|-------|------|-------------|
| plot_type | Literal["line", "heatmap", "image"] | Type of visualization |
| title | str | Plot title (default: array path) |
| slice_indices | dict[int, int] | For ND arrays, which index for each extra dimension |
| colormap | str | Colormap for 2D plots (default: "viridis") |
| show_colorbar | bool | Whether to show colorbar (default: True) |

### ExportConfig

Configuration for plot export.

| Field | Type | Description |
|-------|------|-------------|
| format | Literal["png", "svg"] | Export format |
| width | int | Image width in pixels (default: 1200) |
| height | int | Image height in pixels (default: 800) |
| scale | float | Resolution scale factor (default: 2.0) |

## State Management

### Session State Keys

Streamlit session state keys used by the application:

| Key | Type | Description |
|-----|------|-------------|
| zarr_path | str or None | Currently loaded zarr archive path |
| archive | ZarrArchive or None | Parsed archive object |
| selected_path | str or None | Path of currently selected node |
| expanded_nodes | set[str] | Set of expanded node paths |
| slice_config | dict[str, int] | Current slice indices for ND arrays |
| last_error | str or None | Most recent error message |

## Data Flow

```
User Input (path)
      │
      ▼
┌─────────────────┐
│ parse_zarr_tree │ → TreeNode (root)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  render_tree    │ → UI with expand/collapse
└─────────────────┘
      │
      ▼ (on node click)
┌─────────────────┐
│ load_array_data │ → np.ndarray (possibly sampled)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  create_plot    │ → plotly.Figure
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ display + export│ → Browser / PNG / SVG
└─────────────────┘
```

## Validation Rules

### Path Validation

- Path must exist on filesystem
- Path must be a directory (zarr archives are directories)
- Directory must contain zarr.json (v3) or .zarray/.zgroup (v2)

### Array Validation

- Array must have numeric dtype for plotting
- For 2D plots, at least 2 dimensions required
- Shape must have at least 1 element

### Slice Validation

- Slice indices must be within array bounds
- For each extra dimension beyond 2, a valid index must be provided
