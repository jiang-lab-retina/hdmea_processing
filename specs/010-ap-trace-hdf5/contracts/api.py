"""
API Contracts for AP Tracking Feature

This file defines the public function signatures and data structures
for the AP tracking (axon trace) feature module.

These are contracts only - implementation is in src/hdmea/features/ap_tracking/

HDF5 Storage Note:
    All values are stored as explicit HDF5 datasets (not attributes).
    Scalar values use shape () datasets. Strings use h5py.special_dtype(vlen=str).
    This ensures consistent access patterns and better tooling compatibility.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class RefinedSoma:
    """Refined soma position in 3D STA space."""

    t: int  # Time index
    x: int  # Row index
    y: int  # Column index


@dataclass
class AxonInitialSegment:
    """Axon initial segment position."""

    t: Optional[int]  # Time index (None if not found)
    x: int  # Row index
    y: int  # Column index


@dataclass
class APPathway:
    """Fitted line to axon projection."""

    slope: float
    intercept: float
    r_value: float
    p_value: float
    std_err: float


@dataclass
class APIntersection:
    """Optimal intersection point of AP pathways."""

    x: float
    y: float
    mse: float  # Mean squared error of fit


@dataclass
class SomaPolarCoordinates:
    """Soma position in polar coordinates."""

    radius: float
    angle: float  # Radians
    cartesian_x: float
    cartesian_y: float
    quadrant: str  # "Q1", "Q2", "Q3", "Q4"
    anatomical_quadrant: Optional[str]  # e.g., "dorsal-nasal"


@dataclass
class DVNTPosition:
    """Anatomical position from recording metadata."""

    dv_position: Optional[float]  # Dorsal-ventral (positive = dorsal)
    nt_position: Optional[float]  # Nasal-temporal (positive = nasal)
    lr_position: Optional[str]  # "L" or "R"


@dataclass
class APTrackingResult:
    """Complete AP tracking result for a single unit."""

    unit_id: str
    dvnt: DVNTPosition
    refined_soma: Optional[RefinedSoma]
    axon_initial_segment: Optional[AxonInitialSegment]
    prediction_sta_data: Optional[np.ndarray]  # Shape (T, H, W)
    axon_centroids: Optional[np.ndarray]  # Shape (N, 3)
    ap_pathway: Optional[APPathway]
    all_ap_intersection: Optional[APIntersection]
    soma_polar_coordinates: Optional[SomaPolarCoordinates]
    processing_status: str  # "complete", "skipped", "partial"
    skip_reason: Optional[str]


# =============================================================================
# Public API Functions
# =============================================================================


def compute_ap_tracking(
    hdf5_path: Path,
    model_path: Path,
    *,
    session: Optional[Any] = None,  # PipelineSession
    force_cpu: bool = False,
    max_units: Optional[int] = None,
    # Soma/AIS detection parameters
    soma_std_threshold: float = 3.0,
    soma_temporal_range: Tuple[int, int] = (5, 27),
    soma_refine_radius: int = 5,
    ais_search_xy_radius: int = 5,
    ais_search_t_radius: int = 5,
    # Model inference parameters
    batch_size: Optional[int] = None,  # Auto-determined from GPU memory
    # Post-processing parameters
    temporal_window_size: int = 5,
    exclude_radius: int = 5,
    centroid_threshold: float = 0.05,
    max_displacement: int = 5,
    # Pathway fitting parameters
    min_points_for_fit: int = 10,
    r2_threshold: float = 0.8,
) -> Optional[Any]:  # Returns PipelineSession if session provided
    """
    Compute AP tracking features for all units in an HDF5 file.

    This is the main entry point for AP tracking analysis. It:
    1. Reads eimage_sta data from each unit
    2. Detects soma and axon initial segment
    3. Applies CNN model to predict axon signal probability
    4. Post-processes predictions to extract axon centroids
    5. Fits AP pathway lines and calculates intersection
    6. Computes soma polar coordinates
    7. Parses DVNT positions from metadata
    8. Writes all results to units/{unit_id}/features/ap_tracking/

    Args:
        hdf5_path: Path to input HDF5 file with eimage_sta computed
        model_path: Path to trained CNN model (.pth file)
        session: Optional PipelineSession for deferred save mode
        force_cpu: Force CPU inference even if GPU available
        max_units: Maximum number of units to process (None = all)
        soma_std_threshold: Threshold for soma detection (default: 3.0)
        soma_temporal_range: Time range for soma detection (default: (5, 27))
        soma_refine_radius: Search radius for soma refinement (default: 5)
        ais_search_xy_radius: Spatial search radius for AIS (default: 5)
        ais_search_t_radius: Temporal search radius for AIS (default: 5)
        batch_size: Batch size for GPU inference (None = auto)
        temporal_window_size: Window for temporal filtering (default: 5)
        exclude_radius: Radius around soma to exclude (default: 5)
        centroid_threshold: Min prediction value for centroid (default: 0.05)
        max_displacement: Max frame-to-frame movement (default: 5)
        min_points_for_fit: Min points for pathway fitting (default: 10)
        r2_threshold: R² threshold for valid fits (default: 0.8)

    Returns:
        If session provided: Updated PipelineSession with ap_tracking results
        If no session: None (results written directly to HDF5)

    Raises:
        FileNotFoundError: If HDF5 or model file not found
        ValueError: If HDF5 has no units with eimage_sta data
    """
    ...


def compute_ap_tracking_batch(
    hdf5_paths: List[Path],
    model_path: Path,
    *,
    force_cpu: bool = False,
    skip_existing: bool = True,
    progress_callback: Optional[Any] = None,  # Callable[[int, int, str], None]
    **kwargs,
) -> Dict[str, str]:
    """
    Process multiple HDF5 files for AP tracking.

    Args:
        hdf5_paths: List of HDF5 file paths to process
        model_path: Path to trained CNN model
        force_cpu: Force CPU inference
        skip_existing: Skip files that already have ap_tracking features
        progress_callback: Optional callback(current, total, filename)
        **kwargs: Additional parameters passed to compute_ap_tracking

    Returns:
        Dictionary mapping file paths to status strings:
        - "complete": Successfully processed
        - "skipped": Already has ap_tracking
        - "error: {message}": Processing failed
    """
    ...


# =============================================================================
# Internal API Functions (exposed for testing)
# =============================================================================


def find_soma_from_3d_sta(
    sta: np.ndarray,
    std_threshold: float = 3.0,
    sta_temporal_range: Tuple[int, int] = (5, 27),
) -> Tuple[int, int]:
    """
    Detect soma center from 3D STA data.

    Args:
        sta: 3D STA array (time, row, col)
        std_threshold: Number of std deviations above mean for detection
        sta_temporal_range: Time range to analyze

    Returns:
        Tuple of (row, col) for detected soma center
    """
    ...


def soma_refiner(
    sta: np.ndarray,
    soma_xy: Tuple[int, int],
    refine_radius: int = 5,
) -> Optional[RefinedSoma]:
    """
    Refine soma position by finding minimum value in local neighborhood.

    Args:
        sta: 3D STA array (time, row, col)
        soma_xy: Initial soma position (row, col)
        refine_radius: Search radius around initial position

    Returns:
        RefinedSoma with (t, x, y) coordinates, or None if failed
    """
    ...


def ais_refiner(
    sta: np.ndarray,
    soma_txy: RefinedSoma,
    search_xy_radius: int = 5,
    search_t_radius: int = 5,
) -> Optional[AxonInitialSegment]:
    """
    Detect axon initial segment near refined soma.

    Args:
        sta: 3D STA array (time, row, col)
        soma_txy: Refined soma position
        search_xy_radius: Spatial search radius
        search_t_radius: Temporal search radius

    Returns:
        AxonInitialSegment with (t, x, y) coordinates, or None if not found
    """
    ...


def parse_dvnt_from_center_xy(center_xy: Optional[str]) -> DVNTPosition:
    """
    Parse DVNT position from Center_xy metadata string.

    Args:
        center_xy: String in format "L/R, VD_coord, NT_coord" or None

    Returns:
        DVNTPosition with parsed values (None for unparseable fields)

    Example:
        >>> parse_dvnt_from_center_xy("L, 1.5, -0.8")
        DVNTPosition(dv_position=-1.5, nt_position=-0.8, lr_position="L")
    """
    ...


def load_cnn_model(
    model_path: Path,
    device: str = "auto",
) -> Any:  # torch.nn.Module
    """
    Load trained CNN model for inference.

    Args:
        model_path: Path to .pth model file
        device: "auto", "cuda", "cpu", or "mps"

    Returns:
        Loaded PyTorch model in eval mode on specified device
    """
    ...


def run_model_inference(
    sta_data: np.ndarray,
    model: Any,  # torch.nn.Module
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Run CNN inference on STA data.

    Args:
        sta_data: 3D STA array (time, row, col)
        model: Loaded CNN model
        device: Device string ("cuda", "cpu", "mps")
        batch_size: Batch size for inference

    Returns:
        Prediction array with same shape as sta_data
    """
    ...


def fit_line_to_projections(
    axon_centroids: np.ndarray,
    min_points: int = 10,
) -> Optional[APPathway]:
    """
    Fit line to axon centroid projections.

    Args:
        axon_centroids: Array of (t, x, y) points
        min_points: Minimum points required for fitting

    Returns:
        APPathway with fit parameters, or None if insufficient points
    """
    ...


def calculate_optimal_intersection(
    pathways: Dict[str, APPathway],
) -> Optional[APIntersection]:
    """
    Calculate optimal intersection point from multiple pathway fits.

    Args:
        pathways: Dictionary mapping unit_id to APPathway

    Returns:
        APIntersection with (x, y, mse), or None if < 2 pathways
    """
    ...


def calculate_soma_polar_coordinates(
    soma_xy: Tuple[int, int],
    intersection: APIntersection,
    dvnt: Optional[DVNTPosition] = None,
) -> SomaPolarCoordinates:
    """
    Convert soma position to polar coordinates.

    Args:
        soma_xy: Soma position (row, col)
        intersection: AP pathway intersection point
        dvnt: Optional DVNT position for anatomical labeling

    Returns:
        SomaPolarCoordinates with radius, angle, and quadrant info
    """
    ...

