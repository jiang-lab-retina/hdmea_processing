# Quickstart: Zarr Visualization GUI

## Installation

The zarr visualization tool requires optional dependencies. Install with:

```bash
pip install -e ".[viz]"
```

Or install dependencies directly:

```bash
pip install streamlit plotly
```

## Usage

### Command Line

```bash
# Launch with a specific zarr archive
python -m hdmea.viz.zarr_viz /path/to/archive.zarr

# Launch with file picker (no argument)
python -m hdmea.viz.zarr_viz
```

### From Python

```python
from hdmea.viz.zarr_viz import launch

# With specific path
launch("/path/to/archive.zarr")

# With file picker
launch()
```

## Interface Overview

```
┌────────────────────────────────────────────────────────────────┐
│  Zarr Visualization Tool                                       │
├────────────────┬───────────────────────────┬───────────────────┤
│ SIDEBAR        │ MAIN AREA                 │ METADATA PANEL    │
│                │                           │                   │
│ Path: [____]   │  ┌─────────────────────┐  │ Shape: (1000,)    │
│                │  │                     │  │ Dtype: float32    │
│ 📁 root        │  │   Interactive Plot  │  │ Chunks: (100,)    │
│  ├─📁 units    │  │                     │  │                   │
│  │  ├─📊 u_001 │  │   (zoom, pan, hover)│  │ Attributes:       │
│  │  └─📊 u_002 │  │                     │  │  - created: ...   │
│  └─📁 stimulus │  └─────────────────────┘  │  - version: 1.0   │
│     └─📊 light │                           │                   │
│                │  [Save PNG] [Save SVG]    │                   │
└────────────────┴───────────────────────────┴───────────────────┘
```

## Features

### Tree Navigation

- Click folder icons to expand/collapse groups
- Click array icons to visualize data
- Arrays show small preview of shape/type

### Interactive Plots

- **Zoom**: Scroll or use box select
- **Pan**: Click and drag
- **Hover**: See exact values at cursor
- **Reset**: Double-click to reset view

### Multi-dimensional Arrays

For arrays with more than 2 dimensions, sliders appear to select the slice:

```
Dimension 2: [slider 0-99] = 45
Dimension 3: [slider 0-49] = 20

[2D heatmap of data[:, :, 45, 20]]
```

### Export

- Click "Save PNG" or "Save SVG" to download the current plot
- Files are saved to your browser's download folder

## Example Session

1. Launch the tool:
   ```bash
   python -m hdmea.viz.zarr_viz artifacts/my_recording.zarr
   ```

2. The tree view shows all groups and arrays in the archive

3. Click on an array (e.g., `units/unit_000/spike_times`)

4. The plot appears in the main area with full interactivity

5. Use zoom to focus on a region of interest

6. Click "Save PNG" to export the current view

## Troubleshooting

### "Invalid zarr path"
- Check that the path exists and is a directory
- Ensure the directory contains zarr metadata files

### "Large array warning"
- Arrays over 100MB trigger a warning
- Choose to sample the data or cancel
- For very large arrays, consider using dimension slicers

### Plot not appearing
- Check that the array contains numeric data
- String or object arrays cannot be plotted

### Slow tree loading
- For archives with thousands of nodes, initial loading may take a few seconds
- Tree structure is cached after first load
