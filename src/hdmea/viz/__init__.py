"""
Visualization module for HD-MEA pipeline.

Provides:
    - Raster plots
    - Receptive field visualizations
    - Tuning curve plots
    - Common plotting styles
    - zarr_viz: Interactive HDF5/Zarr archive explorer (requires viz extras)

Sub-modules:
    zarr_viz: Streamlit-based visualization tool for HDF5 and Zarr files
              Install with: pip install hdmea[viz]
              Launch with: python -m hdmea.viz.zarr_viz [path]
              Supports: .h5, .hdf5 files (HDF5) and .zarr directories (legacy)
"""

# Optional: expose zarr_viz launch for convenience
try:
    from hdmea.viz.zarr_viz import launch as launch_zarr_viz
except ImportError:
    # viz extras not installed
    launch_zarr_viz = None

__all__ = ["launch_zarr_viz"]
