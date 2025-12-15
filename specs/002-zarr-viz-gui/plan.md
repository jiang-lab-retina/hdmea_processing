# Implementation Plan: Zarr Visualization GUI

**Branch**: `002-zarr-viz-gui` | **Date**: 2025-01-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-zarr-viz-gui/spec.md`

## Summary

Build a standalone Streamlit-based visualization tool for exploring and plotting Zarr archive data. The tool displays an interactive tree view of the zarr hierarchy and generates interactive plots (with zoom, pan, hover) when users click on array nodes. Located at `src/hdmea/viz/zarr_viz/` as an independent sub-module.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: Streamlit (GUI framework), Plotly (interactive plots), zarr (data access)  
**Storage**: Zarr archives (read-only access)  
**Testing**: pytest with synthetic zarr fixtures  
**Target Platform**: Local desktop (browser-based via Streamlit)  
**Project Type**: Single project extension (new sub-module in existing package)  
**Performance Goals**: Tree load <3s for 1000 nodes, plot render <2s for 10M elements  
**Constraints**: Memory-efficient for large arrays (lazy loading), single-user local access  
**Scale/Scope**: HD-MEA pipeline zarr outputs, but compatible with any valid zarr archive

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Code in `src/hdmea/viz/zarr_viz/` |
| II. Modular Subpackage Layout | ✅ PASS | `viz/` is designated for visualization code |
| III. Explicit I/O | ✅ PASS | Zarr path passed explicitly, no global state |
| IV. Single Zarr Per Recording | ✅ PASS | Tool reads existing Zarr, doesn't create new ones |
| V. Data Format Standards | ✅ PASS | Consumes Zarr format |
| VI. No Hidden Global State | ✅ PASS | Streamlit handles session state explicitly |
| VII. Independence from Legacy | ✅ PASS | No legacy imports |
| Visualization Standards | ✅ PASS | Follows viz/ location, returns figure objects |
| Testing Requirements | ✅ PASS | Unit tests with synthetic zarr fixtures |

**Gate Result**: ✅ PASS - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/002-zarr-viz-gui/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (internal module API)
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
src/hdmea/viz/
├── __init__.py              # Existing
└── zarr_viz/                # NEW: Independent sub-module
    ├── __init__.py          # Public API exports
    ├── app.py               # Streamlit application entry point
    ├── tree.py              # Zarr tree structure and traversal
    ├── plots.py             # Plot generation (1D, 2D, ND)
    ├── metadata.py          # Metadata display formatting
    ├── utils.py             # Shared utilities (lazy loading, sampling)
    └── __main__.py          # CLI entry: python -m hdmea.viz.zarr_viz

tests/
├── unit/
│   └── test_zarr_viz.py     # Unit tests for zarr_viz components
└── fixtures/
    └── synthetic_zarr.py    # Synthetic zarr archive generator
```

**Structure Decision**: Single project extension - new sub-module `zarr_viz/` under existing `viz/` package. This follows the constitution's modular layout and keeps all visualization code together.

## Complexity Tracking

> No violations requiring justification. Design follows constitution principles.

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| GUI Framework | Streamlit | User specified; rapid development, browser-based |
| Plotting Library | Plotly | Interactive (zoom/pan/hover) out of the box |
| Tree Component | Custom with st.expander | Streamlit-native, no external dependencies |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit App (app.py)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Sidebar    │  │  Main Area  │  │  Right Panel        │  │
│  │  - Path     │  │  - Plot     │  │  - Metadata         │  │
│  │  - Tree     │  │  - Export   │  │  - Attributes       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
   ┌──────────┐      ┌──────────┐         ┌──────────┐
   │ tree.py  │      │ plots.py │         │metadata.py│
   │ - parse  │      │ - 1D     │         │ - format  │
   │ - render │      │ - 2D     │         │ - display │
   └──────────┘      │ - ND     │         └──────────┘
         │           └──────────┘
         ▼                │
   ┌──────────┐           ▼
   │ utils.py │    ┌──────────────┐
   │ - lazy   │    │ zarr library │
   │ - sample │    └──────────────┘
   └──────────┘
```

## Module Specifications

### tree.py - Zarr Tree Handling

**Responsibilities**:
- Parse zarr archive into tree structure
- Identify groups vs arrays
- Provide node selection interface

**Key Functions**:
```python
def parse_zarr_tree(zarr_path: Path) -> TreeNode
def render_tree(root: TreeNode, on_select: Callable) -> None
def get_node_by_path(root: TreeNode, path: str) -> TreeNode | None
```

### plots.py - Visualization Generation

**Responsibilities**:
- Generate appropriate plot based on array dimensions
- Handle large arrays with sampling
- Support interactive features (zoom, pan, hover)
- Export to PNG/SVG

**Key Functions**:
```python
def create_plot(array: zarr.Array, slice_indices: dict | None = None) -> go.Figure
def plot_1d(data: np.ndarray, title: str) -> go.Figure
def plot_2d(data: np.ndarray, title: str) -> go.Figure
def plot_nd(array: zarr.Array, selected_dims: dict) -> go.Figure
def export_figure(fig: go.Figure, path: Path, format: str) -> None
```

### metadata.py - Metadata Display

**Responsibilities**:
- Format zarr attributes for display
- Show array properties (shape, dtype, chunks)
- Handle non-serializable attributes

**Key Functions**:
```python
def format_attributes(attrs: dict) -> str
def format_array_info(array: zarr.Array) -> dict
def format_group_info(group: zarr.Group) -> dict
```

### utils.py - Shared Utilities

**Responsibilities**:
- Lazy loading of large arrays
- Intelligent sampling for oversized data
- Memory management

**Key Functions**:
```python
def sample_array(array: zarr.Array, max_elements: int = 10_000_000) -> np.ndarray
def estimate_memory(array: zarr.Array) -> int
def should_warn_large(array: zarr.Array, threshold_mb: int = 100) -> bool
```

## Dependencies

### Required (add to pyproject.toml)

```toml
[project.optional-dependencies]
viz = [
    "streamlit>=1.28.0",
    "plotly>=5.18.0",
]
```

### Already Available
- `zarr` - Core data access
- `numpy` - Array operations

## Launch Interface

### CLI Entry Point

```bash
# Launch with zarr path
python -m hdmea.viz.zarr_viz /path/to/archive.zarr

# Launch with file picker
python -m hdmea.viz.zarr_viz
```

### Programmatic Launch

```python
from hdmea.viz.zarr_viz import launch

# With path
launch("/path/to/archive.zarr")

# With file picker
launch()
```

## Error Handling Strategy

| Error Condition | User Experience |
|-----------------|-----------------|
| Invalid zarr path | Red error banner with message |
| Corrupted zarr | Error message, tree shows "Error loading" |
| Large array (>100MB) | Warning dialog with sample option |
| Non-numeric array | Info message "Cannot plot non-numeric data" |
| Export failure | Toast notification with error |

## Performance Considerations

1. **Lazy Tree Loading**: Only expand nodes when clicked
2. **Array Sampling**: For arrays >10M elements, sample uniformly
3. **Chunk-aware Loading**: Read only required chunks for slicing
4. **Caching**: Use Streamlit's `@st.cache_data` for parsed tree structure
