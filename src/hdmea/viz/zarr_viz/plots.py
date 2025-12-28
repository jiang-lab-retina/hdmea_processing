"""
Plot generation for zarr_viz module (HDF5 compatible).

Provides functions for creating interactive Plotly visualizations
of HDF5 dataset and zarr array data.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING, Union

import numpy as np
import plotly.graph_objects as go

from hdmea.viz.zarr_viz.utils import sample_array, should_warn_large, UnsupportedArrayError

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)

__all__ = [
    "create_plot",
    "plot_1d",
    "plot_1d_histogram",
    "plot_2d",
    "plot_nd",
    "export_figure",
    "get_plot_bytes",
]

# Type alias for array-like objects
ArrayLike = Union["h5py.Dataset", np.ndarray]


# =============================================================================
# Main Plot Dispatcher
# =============================================================================


def create_plot(
    array: ArrayLike,
    title: Optional[str] = None,
    slice_indices: Optional[dict[int, int]] = None,
    sampled: bool = False,
    plot_type: Literal["line", "histogram"] = "line",
    acquisition_rate: Optional[float] = None,
    x_limits: Optional[tuple[float, float]] = None,
    slide_dim: int = 0,
    blur_sigma: Optional[float] = None,
    color_range: Optional[tuple[float, float]] = None,
    bin_size: Optional[float] = None,
) -> go.Figure:
    """Create appropriate plot based on array dimensions.

    Args:
        array: HDF5 dataset or numpy array to plot.
        title: Plot title. If None, uses array info.
        slice_indices: For ND arrays, indices for dimensions > 2.
        sampled: Whether data was sampled (for title annotation).
        plot_type: For 1D arrays, "line" for line plot or "histogram" for histogram.
        acquisition_rate: Data acquisition rate in Hz (used for histogram bin calculation).
        x_limits: Optional (min, max) tuple to set x-axis range (for histograms).
        slide_dim: For 3D arrays, the dimension to slice through (default: 0).
        blur_sigma: Gaussian blur sigma (None = no blur).
        color_range: (min, max) tuple for consistent color scale.
        bin_size: Override histogram bin width. If provided, used instead of acquisition_rate.

    Returns:
        Plotly Figure object.

    Raises:
        UnsupportedArrayError: If array cannot be visualized.
    """
    # Validate array is numeric
    if not np.issubdtype(array.dtype, np.number):
        raise UnsupportedArrayError(
            f"Cannot plot non-numeric array with dtype: {array.dtype}"
        )

    # Generate default title
    if title is None:
        title = f"Shape: {array.shape}, Type: {array.dtype}"

    ndim = len(array.shape)

    if ndim == 0:
        # Scalar - show as text
        raise UnsupportedArrayError("Cannot plot scalar (0-dimensional) array")

    elif ndim == 1:
        # 1D array - line plot or histogram
        data = sample_array(array)
        if plot_type == "histogram":
            fig = plot_1d_histogram(data, title, acquisition_rate=acquisition_rate, x_limits=x_limits, bin_size=bin_size)
        else:
            fig = plot_1d(data, title)
        if sampled or len(data) < array.shape[0]:
            _add_sampled_annotation(fig, array.shape[0], len(data))
        return fig

    elif ndim == 2:
        # 2D array - heatmap
        data = sample_array(array)
        fig = plot_2d(data, title, color_range=color_range)
        if sampled:
            _add_sampled_annotation(fig, np.prod(array.shape), data.size)
        return fig

    else:
        # ND array - slice to 2D and plot
        return plot_nd(array, title, slice_indices, slide_dim=slide_dim, 
                       blur_sigma=blur_sigma, color_range=color_range)


# =============================================================================
# 1D Plotting
# =============================================================================


def plot_1d(data: np.ndarray, title: str = "") -> go.Figure:
    """Create line plot for 1D data.

    Args:
        data: 1D numpy array.
        title: Plot title.

    Returns:
        Plotly Figure with line plot.
    """
    # Convert to float and handle NaN/Inf for JSON serialization
    data = np.asarray(data, dtype=np.float64)
    data = np.where(np.isfinite(data), data, np.nan)  # Replace Inf with NaN
    
    # Convert to list for Plotly (handles NaN as null in JSON)
    y_data = [None if np.isnan(v) else float(v) for v in data]
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(data))),
            y=y_data,
            mode="lines",
            name="Data",
            line=dict(color="#1f77b4", width=1),
            hovertemplate="Index: %{x}<br>Value: %{y:.6g}<extra></extra>",
            connectgaps=False,  # Don't connect across NaN gaps
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Index",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    # Enable zoom, pan, hover by default
    fig.update_xaxes(rangeslider_visible=False)

    return fig


def plot_1d_histogram(
    data: np.ndarray, 
    title: str = "", 
    acquisition_rate: Optional[float] = None,
    x_limits: Optional[tuple[float, float]] = None,
    bin_size: Optional[float] = None,
) -> go.Figure:
    """Create histogram for 1D data with 100 ms bins.

    Args:
        data: 1D numpy array.
        title: Plot title.
        acquisition_rate: Data acquisition rate in Hz. Used to calculate 
                         bin width as acquisition_rate × 0.1 (100 ms worth of samples).
        x_limits: Optional (min, max) tuple to set x-axis range.
        bin_size: Override bin width. If provided, this value is used directly
                 instead of calculating from acquisition_rate.

    Returns:
        Plotly Figure with histogram.
    """
    # Convert to float and handle NaN/Inf
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]  # Remove NaN and Inf values
    
    # Use explicit bin_size if provided, otherwise calculate from acquisition_rate
    if bin_size is not None and bin_size > 0:
        bin_width = bin_size
    elif acquisition_rate is not None and acquisition_rate > 0:
        bin_width = acquisition_rate * 0.1  # 100 ms in samples
    else:
        # Fallback: use 50 bins if no acquisition rate available
        bin_width = None
    
    # If no acquisition rate, calculate bin width from data range (fallback to ~50 bins)
    if bin_width is None and len(data) > 0:
        data_range = np.max(data) - np.min(data)
        if data_range > 0:
            bin_width = data_range / 50  # Approximate 50 bins as fallback
        else:
            bin_width = 1.0  # Default for constant data
    
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data.tolist(),
            xbins=dict(size=bin_width),
            name="Data",
            marker=dict(color="#1f77b4", line=dict(color="#0d47a1", width=1)),
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Value",
        yaxis_title="Count",
        template="plotly_white",
        margin=dict(l=60, r=40, t=80, b=60),  # Extra top margin for annotation
        bargap=0.05,
    )

    # Add bin width annotation
    if bin_size is not None and bin_size > 0:
        # Explicit bin size was provided
        fig.add_annotation(
            text=f"Bin width: {bin_width:.1f} (fixed)",
            xref="paper",
            yref="paper",
            x=1,
            y=1.02,
            showarrow=False,
            font=dict(size=11, color="#666666"),
            xanchor="right",
        )
    elif acquisition_rate is not None and acquisition_rate > 0:
        fig.add_annotation(
            text=f"Bin width: {bin_width:.1f} (100 ms)",
            xref="paper",
            yref="paper",
            x=1,
            y=1.02,
            showarrow=False,
            font=dict(size=11, color="#666666"),
            xanchor="right",
        )
    else:
        fig.add_annotation(
            text=f"Bin width: {bin_width:.4g} (auto, no acquisition rate)",
            xref="paper",
            yref="paper",
            x=1,
            y=1.02,
            showarrow=False,
            font=dict(size=11, color="#999999"),
            xanchor="right",
        )

    # Apply x-axis limits if provided
    if x_limits is not None:
        fig.update_xaxes(range=[x_limits[0], x_limits[1]])

    return fig


# =============================================================================
# 2D Plotting
# =============================================================================


def plot_2d(
    data: np.ndarray, 
    title: str = "",
    color_range: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """Create heatmap for 2D data.

    Args:
        data: 2D numpy array.
        title: Plot title.
        color_range: Optional (min, max) tuple for consistent color scale.

    Returns:
        Plotly Figure with heatmap.
    """
    # Convert to float and handle NaN/Inf for JSON serialization
    data = np.asarray(data, dtype=np.float64)
    # Replace Inf with NaN (Plotly handles NaN in heatmaps)
    data = np.where(np.isfinite(data), data, np.nan)
    
    # Convert to nested list for JSON serialization (NaN becomes null)
    z_data = [[None if np.isnan(v) else float(v) for v in row] for row in data]
    
    fig = go.Figure()

    # Build heatmap kwargs with optional color range
    heatmap_kwargs = {
        "z": z_data,
        "colorscale": "Jet",
        "hovertemplate": "Row: %{y}<br>Col: %{x}<br>Value: %{z:.6g}<extra></extra>",
    }
    if color_range is not None:
        heatmap_kwargs["zmin"] = color_range[0]
        heatmap_kwargs["zmax"] = color_range[1]

    fig.add_trace(go.Heatmap(**heatmap_kwargs))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Column",
        yaxis_title="Row",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )

    # Maintain aspect ratio for square-ish data
    if data.shape[0] > 0 and data.shape[1] > 0:
        aspect_ratio = data.shape[1] / data.shape[0]
        if 0.5 < aspect_ratio < 2:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


# =============================================================================
# ND Plotting
# =============================================================================


def plot_nd(
    array: ArrayLike,
    title: str = "",
    slice_indices: Optional[dict[int, int]] = None,
    slide_dim: int = 0,
    blur_sigma: Optional[float] = None,
    color_range: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """Create 2D heatmap from ND array using slicing.

    For arrays with more than 2 dimensions, this function slices the
    array to 2D using the provided indices for extra dimensions.

    Args:
        array: ND HDF5 dataset or numpy array (ndim > 2).
        title: Plot title.
        slice_indices: Dict mapping dimension index to slice position.
        slide_dim: For 3D arrays, the dimension to slice through (default: 0).
                  The other two dimensions form the 2D plot.
                  For 4D+ arrays, dimensions 0 and 1 are used for the plot,
                  and other dimensions are sliced.
        blur_sigma: Gaussian blur sigma (None = no blur).
        color_range: (min, max) tuple for consistent color scale.

    Returns:
        Plotly Figure with 2D heatmap of the sliced data.
    """
    ndim = len(array.shape)

    if slice_indices is None:
        slice_indices = {}

    # Build slice tuple
    slices = []
    
    if ndim == 3:
        # For 3D arrays: slide_dim gets indexed, other two dims form the 2D plot
        for dim in range(ndim):
            if dim == slide_dim:
                # This dimension is sliced through
                idx = slice_indices.get(dim, 0)
                idx = max(0, min(idx, array.shape[dim] - 1))
                slices.append(idx)
            else:
                # This dimension is part of the 2D plot
                slices.append(slice(None))
    else:
        # For 4D+ arrays: keep original behavior (dims 0,1 plotted, others sliced)
        for dim in range(ndim):
            if dim < 2:
                # First two dimensions are plotted
                slices.append(slice(None))
            else:
                # Other dimensions need an index
                idx = slice_indices.get(dim, 0)
                idx = max(0, min(idx, array.shape[dim] - 1))
                slices.append(idx)

    # Extract 2D slice
    data_2d = np.asarray(array[tuple(slices)])

    # Sample if needed
    data_2d = sample_array_2d(data_2d)
    
    # Apply Gaussian blur if requested
    if blur_sigma is not None and blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        data_2d = gaussian_filter(data_2d, sigma=blur_sigma)

    # Update title with slice info
    if ndim == 3:
        slice_info = f"dim{slide_dim}={slice_indices.get(slide_dim, 0)}"
    else:
        slice_info = ", ".join(
            f"dim{d}={slice_indices.get(d, 0)}"
            for d in range(2, ndim)
        )
    full_title = f"{title}\nSlice: [{slice_info}]" if slice_info else title

    return plot_2d(data_2d, full_title, color_range=color_range)


def sample_array_2d(data: np.ndarray, max_size: int = 2000) -> np.ndarray:
    """Sample 2D array if too large for display.

    Args:
        data: 2D numpy array.
        max_size: Maximum dimension size.

    Returns:
        Sampled array with dimensions <= max_size.
    """
    if data.ndim != 2:
        return data

    rows, cols = data.shape
    new_rows = min(rows, max_size)
    new_cols = min(cols, max_size)

    if rows <= max_size and cols <= max_size:
        return data

    row_indices = np.linspace(0, rows - 1, new_rows, dtype=int)
    col_indices = np.linspace(0, cols - 1, new_cols, dtype=int)

    return data[np.ix_(row_indices, col_indices)]


# =============================================================================
# Export Functions
# =============================================================================


def export_figure(
    fig: go.Figure,
    path: Path,
    format: Literal["png", "svg"] = "png",
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> None:
    """Export figure to file.

    Args:
        fig: Plotly figure to export.
        path: Output file path.
        format: Export format ("png" or "svg").
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Resolution scale factor (for PNG).
    """
    path = Path(path)

    if format == "png":
        fig.write_image(
            str(path),
            format="png",
            width=width,
            height=height,
            scale=scale,
        )
    elif format == "svg":
        fig.write_image(
            str(path),
            format="svg",
            width=width,
            height=height,
        )
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Exported figure to {path}")


def get_plot_bytes(
    fig: go.Figure,
    format: Literal["png", "svg"] = "png",
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> bytes:
    """Get figure as bytes for download.

    Args:
        fig: Plotly figure to export.
        format: Export format ("png" or "svg").
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Resolution scale factor (for PNG).

    Returns:
        Image data as bytes.
    """
    if format == "png":
        return fig.to_image(
            format="png",
            width=width,
            height=height,
            scale=scale,
        )
    elif format == "svg":
        return fig.to_image(
            format="svg",
            width=width,
            height=height,
        )
    else:
        raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# Helper Functions
# =============================================================================


def _add_sampled_annotation(fig: go.Figure, original: int, sampled: int) -> None:
    """Add annotation indicating data was sampled.

    Args:
        fig: Figure to annotate.
        original: Original element count.
        sampled: Sampled element count.
    """
    fig.add_annotation(
        text=f"⚠️ Sampled: {sampled:,} of {original:,} elements",
        xref="paper",
        yref="paper",
        x=1,
        y=1.02,
        showarrow=False,
        font=dict(size=10, color="orange"),
        xanchor="right",
    )
