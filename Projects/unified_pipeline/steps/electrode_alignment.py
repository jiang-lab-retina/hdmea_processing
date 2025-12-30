"""
Step Wrapper: Electrode-to-Stimulus Coordinate Alignment

Aligns electrode array coordinates (65×65 grid from eimage_sta) to light
stimulation coordinates (300×300 grid) by fitting a linear model between
eimage_sta geometry and sta_geometry centers.

Uses a two-pass approach:
1. Initial linear fit on all paired data points
2. Remove outliers based on residual threshold
3. Refit on clean data (final fit saved to metadata)
4. Apply correction to each unit's eimage center

Based on legacy code: align_sta_center_to_electrode_center_helper.py
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from hdmea.pipeline import PipelineSession
from Projects.unified_pipeline.config import ElectrodeAlignmentConfig, red_warning

logger = logging.getLogger(__name__)

STEP_NAME = "electrode_alignment"

# Coordinate scale factor from 15×15 STA grid to 300×300 stimulus space
COORDINATE_SCALE_FACTOR = 20


# =============================================================================
# Linear Fitting Functions
# =============================================================================

def constrained_linear_fit(
    x_data: np.ndarray,
    y_data: np.ndarray,
    slope_bounds: Tuple[float, float] = (-np.inf, np.inf),
    intercept_bounds: Tuple[float, float] = (-np.inf, np.inf),
    initial_guess: Tuple[float, float] = (-0.14, 0),
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fit a linear model (y = mx + b) with optional constraints on slope and intercept.
    
    Args:
        x_data: Independent variable array
        y_data: Dependent variable array
        slope_bounds: (min_slope, max_slope) constraints
        intercept_bounds: (min_intercept, max_intercept) constraints
        initial_guess: Initial guess for (slope, intercept)
    
    Returns:
        Tuple of (slope, intercept, r_squared) or (None, None, None) if fit fails
    """
    if x_data is None or y_data is None or len(x_data) == 0 or len(y_data) == 0:
        return None, None, None
    
    if len(x_data) != len(y_data):
        return None, None, None
    
    def linear_function(x, m, b):
        return m * np.array(x) + b
    
    bounds = (
        [slope_bounds[0], intercept_bounds[0]],
        [slope_bounds[1], intercept_bounds[1]]
    )
    
    try:
        popt, _ = curve_fit(
            linear_function,
            x_data,
            y_data,
            bounds=bounds,
            p0=initial_guess,
            maxfev=5000,
        )
        slope, intercept = popt
        
        # Calculate R²
        y_pred = linear_function(x_data, slope, intercept)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        
        if ss_tot == 0:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        return float(slope), float(intercept), float(r_squared)
        
    except Exception as e:
        logger.debug(f"Linear fit failed: {e}")
        return None, None, None


# =============================================================================
# Signal Quality Filter Functions
# =============================================================================

def check_signal_quality_3d(
    data_3d: np.ndarray,
    center_row: int,
    center_col: int,
    snr_threshold: float = 3.0,
    exclusion_fraction: float = 0.3,
) -> bool:
    """
    Check if the center pixel has sufficient signal variation along the time axis.
    
    Compares the inter-percentile range (P90 - P10) of the center pixel's time 
    series to the inter-percentile range of the background (excluding 30% of 
    pixels closest to the center).
    
    Args:
        data_3d: 3D array of shape (time, row, col)
        center_row: Row index of center pixel
        center_col: Column index of center pixel
        snr_threshold: Minimum ratio of center_range to background_range
        exclusion_fraction: Fraction of pixels closest to center to exclude (default: 0.3)
    
    Returns:
        True if center pixel has sufficient signal, False otherwise
    """
    if data_3d is None or data_3d.ndim != 3:
        return False
    
    n_time, n_rows, n_cols = data_3d.shape
    
    # Need at least 2 time points
    if n_time < 2:
        return False
    
    # Validate center coordinates
    center_row = int(np.clip(center_row, 0, n_rows - 1))
    center_col = int(np.clip(center_col, 0, n_cols - 1))
    
    # Get the time series at the center pixel and compute its inter-percentile range
    center_time_series = data_3d[:, center_row, center_col]
    center_range = np.percentile(center_time_series, 90) - np.percentile(center_time_series, 10)
    
    # Create distance map from center for all pixels
    row_coords, col_coords = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')
    distances = np.sqrt((row_coords - center_row)**2 + (col_coords - center_col)**2)
    
    # Find distance threshold to exclude closest 30% of pixels
    n_pixels = n_rows * n_cols
    n_exclude = int(n_pixels * exclusion_fraction)
    
    # Flatten distances and find the threshold
    distances_flat = distances.flatten()
    sorted_distances = np.sort(distances_flat)
    
    if n_exclude >= n_pixels:
        # If we'd exclude everything, just use all pixels
        distance_threshold = -1
    else:
        distance_threshold = sorted_distances[n_exclude]
    
    # Create mask for background pixels (those beyond the exclusion zone)
    background_mask = distances > distance_threshold
    
    # If no background pixels remain, fall back to using all non-center pixels
    if not np.any(background_mask):
        background_mask = np.ones((n_rows, n_cols), dtype=bool)
        background_mask[center_row, center_col] = False
    
    # Extract background data: all time points for background pixels
    # Shape: (n_time, n_background_pixels)
    background_data = data_3d[:, background_mask]
    
    # Compute inter-percentile range of the entire background
    background_range = np.percentile(background_data, 90) - np.percentile(background_data, 10)
    
    if background_range == 0:
        return False
    
    # Compute SNR as ratio of center range to background range
    snr = center_range / background_range
    
    return snr >= snr_threshold


def get_eimage_data(features: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extract eimage_sta 3D data from features dict.
    
    Args:
        features: Unit features dictionary
    
    Returns:
        3D numpy array of shape (time, row, col) or None
    """
    eimage_sta = features.get('eimage_sta', {})
    
    # Try different possible data locations
    data = eimage_sta.get('data')
    if data is None:
        data = eimage_sta.get('eimage')
    if data is None:
        # Check if there's a 'frames' key
        data = eimage_sta.get('frames')
    
    if isinstance(data, np.ndarray) and data.ndim == 3:
        return data
    
    return None


def get_sta_data(features: Dict[str, Any], sta_feature_name: str) -> Optional[np.ndarray]:
    """
    Extract STA 3D data from features dict.
    
    Args:
        features: Unit features dictionary
        sta_feature_name: Name of the STA feature
    
    Returns:
        3D numpy array of shape (time, row, col) or None
    """
    sta_feature = features.get(sta_feature_name, {})
    
    # Try 'data' at the root level first (most common case)
    data = sta_feature.get('data')
    if isinstance(data, np.ndarray) and data.ndim == 3:
        return data
    
    # Try nested 'sta' key (alternative structure)
    sta = sta_feature.get('sta')
    if sta is not None:
        if isinstance(sta, np.ndarray) and sta.ndim == 3:
            return sta
        if isinstance(sta, dict):
            nested_data = sta.get('data')
            if isinstance(nested_data, np.ndarray) and nested_data.ndim == 3:
                return nested_data
    
    return None


# =============================================================================
# Data Collection Functions
# =============================================================================

@dataclass
class AlignmentDataPoint:
    """Single data point for alignment fitting."""
    unit_id: str
    eimage_row: float
    eimage_col: float
    sta_row: float
    sta_col: float


@dataclass
class CollectionResult:
    """Result of data collection with statistics."""
    data_points: List["AlignmentDataPoint"]
    n_units_filtered_snr: int


def collect_alignment_data(
    session: PipelineSession,
    sta_feature_name: str = "sta_perfect_dense_noise_15x15_15hz_r42_3min",
    eimage_snr_threshold: float = 40.0,
    sta_snr_threshold: float = 1.0,
) -> CollectionResult:
    """
    Collect paired (eimage, sta) center coordinates from all units.
    
    Uses gaussian_fit center for STA coordinates (more accurate than sta_geometry):
    - gaussian_fit/center_y -> sta_row
    - gaussian_fit/center_x -> sta_col
    
    Applies signal quality filter with separate thresholds for eimage and STA:
    - eimage: High threshold (default 40) to ensure strong electrode signal
    - STA: Low threshold (default 1.0) since RF may span significant grid area
    
    Args:
        session: Pipeline session containing unit data
        sta_feature_name: Name of the STA feature containing sta_geometry
        eimage_snr_threshold: Minimum SNR for eimage center (default: 40.0)
        sta_snr_threshold: Minimum SNR for STA center (default: 1.0)
    
    Returns:
        CollectionResult with data points and filter statistics
    """
    data_points: List[AlignmentDataPoint] = []
    units_filtered_snr = 0
    
    for unit_id, unit_data in session.units.items():
        features = unit_data.get('features', {})
        
        # Get eimage_sta geometry
        eimage_sta = features.get('eimage_sta', {})
        eimage_geometry = eimage_sta.get('geometry', {})
        eimage_row = eimage_geometry.get('center_row')
        eimage_col = eimage_geometry.get('center_col')
        
        # Get gaussian_fit center from sta_geometry (more accurate than center_row/col)
        # gaussian_fit/center_y -> sta_row, gaussian_fit/center_x -> sta_col
        sta_feature = features.get(sta_feature_name, {})
        sta_geometry = sta_feature.get('sta_geometry', {})
        gaussian_fit = sta_geometry.get('gaussian_fit', {})
        
        # Use gaussian_fit center_y for row, center_x for col
        sta_row = gaussian_fit.get('center_y')
        sta_col = gaussian_fit.get('center_x')
        
        # Handle numpy arrays (extract scalar value)
        if isinstance(eimage_row, np.ndarray):
            eimage_row = float(eimage_row.flat[0]) if eimage_row.size > 0 else None
        if isinstance(eimage_col, np.ndarray):
            eimage_col = float(eimage_col.flat[0]) if eimage_col.size > 0 else None
        if isinstance(sta_row, np.ndarray):
            sta_row = float(sta_row.flat[0]) if sta_row.size > 0 else None
        if isinstance(sta_col, np.ndarray):
            sta_col = float(sta_col.flat[0]) if sta_col.size > 0 else None
        
        # Skip if any coordinate is missing
        if any(v is None for v in [eimage_row, eimage_col, sta_row, sta_col]):
            continue
        
        # Skip if any coordinate is NaN
        if any(np.isnan(v) for v in [eimage_row, eimage_col, sta_row, sta_col]):
            continue
        
        # Apply eimage signal quality filter
        if eimage_snr_threshold > 0:
            eimage_data = get_eimage_data(features)
            if eimage_data is not None:
                eimage_has_signal = check_signal_quality_3d(
                    eimage_data,
                    center_row=int(round(eimage_row)),
                    center_col=int(round(eimage_col)),
                    snr_threshold=eimage_snr_threshold,
                )
                if not eimage_has_signal:
                    units_filtered_snr += 1
                    continue
        
        # Apply STA signal quality filter (lower threshold)
        if sta_snr_threshold > 0:
            sta_data = get_sta_data(features, sta_feature_name)
            if sta_data is not None:
                sta_has_signal = check_signal_quality_3d(
                    sta_data,
                    center_row=int(round(sta_row)),
                    center_col=int(round(sta_col)),
                    snr_threshold=sta_snr_threshold,
                )
                if not sta_has_signal:
                    units_filtered_snr += 1
                    continue
        
        data_points.append(AlignmentDataPoint(
            unit_id=unit_id,
            eimage_row=float(eimage_row),
            eimage_col=float(eimage_col),
            sta_row=float(sta_row),
            sta_col=float(sta_col),
        ))
    
    if units_filtered_snr > 0:
        logger.info(f"  Filtered {units_filtered_snr} units due to low SNR at center")
    
    return CollectionResult(
        data_points=data_points,
        n_units_filtered_snr=units_filtered_snr,
    )


# =============================================================================
# Alignment Fitting
# =============================================================================

@dataclass
class AlignmentFitResult:
    """Result of alignment fitting."""
    row_slope: Optional[float]
    row_intercept: Optional[float]
    row_r_squared: Optional[float]
    col_slope: Optional[float]
    col_intercept: Optional[float]
    col_r_squared: Optional[float]
    is_valid: bool
    n_units_used: int
    n_outliers_removed: int


def fit_alignment_model(
    data_points: List[AlignmentDataPoint],
    min_units_threshold: int = 3,
    r_square_threshold: float = 0.7,
    outlier_threshold: float = 3.0,
) -> AlignmentFitResult:
    """
    Fit linear alignment model with outlier removal.
    
    Based on legacy code, the axes are swapped in the fit:
    - sta_row = m_row * eimage_col + b_row
    - sta_col = m_col * eimage_row + b_col
    
    Args:
        data_points: List of paired coordinate data
        min_units_threshold: Minimum units required for valid fit
        r_square_threshold: Minimum R² for valid fit
        outlier_threshold: Maximum residual before marking as outlier
    
    Returns:
        AlignmentFitResult with fit parameters and validation
    """
    if len(data_points) < min_units_threshold:
        logger.warning(f"Not enough data points for alignment: {len(data_points)} < {min_units_threshold}")
        return AlignmentFitResult(
            row_slope=None, row_intercept=None, row_r_squared=None,
            col_slope=None, col_intercept=None, col_r_squared=None,
            is_valid=False, n_units_used=0, n_outliers_removed=0,
        )
    
    # Extract arrays (note: axes are swapped based on legacy code)
    eimage_rows = np.array([dp.eimage_row for dp in data_points])
    eimage_cols = np.array([dp.eimage_col for dp in data_points])
    sta_rows = np.array([dp.sta_row for dp in data_points])
    sta_cols = np.array([dp.sta_col for dp in data_points])
    
    # First pass: Initial fit
    # sta_row = m_row * eimage_col + b_row (note: using eimage_col for sta_row)
    # sta_col = m_col * eimage_row + b_col (note: using eimage_row for sta_col)
    row_slope_init, row_intercept_init, row_r2_init = constrained_linear_fit(
        eimage_cols, sta_rows
    )
    col_slope_init, col_intercept_init, col_r2_init = constrained_linear_fit(
        eimage_rows, sta_cols
    )
    
    if row_slope_init is None or col_slope_init is None:
        logger.warning("Initial fit failed")
        return AlignmentFitResult(
            row_slope=None, row_intercept=None, row_r_squared=None,
            col_slope=None, col_intercept=None, col_r_squared=None,
            is_valid=False, n_units_used=0, n_outliers_removed=0,
        )
    
    # Second pass: Remove outliers and refit
    # Compute residuals
    sta_row_pred = row_slope_init * eimage_cols + row_intercept_init
    sta_col_pred = col_slope_init * eimage_rows + col_intercept_init
    
    row_residuals = np.abs(sta_rows - sta_row_pred)
    col_residuals = np.abs(sta_cols - sta_col_pred)
    
    # Identify inliers (both row and col residuals must be within threshold)
    inlier_mask = (row_residuals <= outlier_threshold) & (col_residuals <= outlier_threshold)
    n_outliers = np.sum(~inlier_mask)
    
    # Filter to inliers
    eimage_rows_clean = eimage_rows[inlier_mask]
    eimage_cols_clean = eimage_cols[inlier_mask]
    sta_rows_clean = sta_rows[inlier_mask]
    sta_cols_clean = sta_cols[inlier_mask]
    
    if len(eimage_rows_clean) < min_units_threshold:
        logger.warning(f"Not enough inliers after outlier removal: {len(eimage_rows_clean)} < {min_units_threshold}")
        return AlignmentFitResult(
            row_slope=None, row_intercept=None, row_r_squared=None,
            col_slope=None, col_intercept=None, col_r_squared=None,
            is_valid=False, n_units_used=len(eimage_rows_clean), n_outliers_removed=int(n_outliers),
        )
    
    # Final fit on clean data
    row_slope, row_intercept, row_r2 = constrained_linear_fit(
        eimage_cols_clean, sta_rows_clean
    )
    col_slope, col_intercept, col_r2 = constrained_linear_fit(
        eimage_rows_clean, sta_cols_clean
    )
    
    if row_slope is None or col_slope is None:
        logger.warning("Final fit failed after outlier removal")
        return AlignmentFitResult(
            row_slope=None, row_intercept=None, row_r_squared=None,
            col_slope=None, col_intercept=None, col_r_squared=None,
            is_valid=False, n_units_used=len(eimage_rows_clean), n_outliers_removed=int(n_outliers),
        )
    
    # Check R² threshold
    is_valid = (row_r2 >= r_square_threshold) and (col_r2 >= r_square_threshold)
    
    if not is_valid:
        logger.info(f"Fit R² below threshold: row={row_r2:.3f}, col={col_r2:.3f} (threshold={r_square_threshold})")
    
    return AlignmentFitResult(
        row_slope=row_slope,
        row_intercept=row_intercept,
        row_r_squared=row_r2,
        col_slope=col_slope,
        col_intercept=col_intercept,
        col_r_squared=col_r2,
        is_valid=is_valid,
        n_units_used=len(eimage_rows_clean),
        n_outliers_removed=int(n_outliers),
    )


def convert_eimage_to_300(
    eimage_row: float,
    eimage_col: float,
    fit_result: AlignmentFitResult,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert eimage center coordinates to 300×300 stimulus space.
    
    Uses the fitted linear model to first predict STA coordinates (15×15),
    then scales to 300×300 space.
    
    Args:
        eimage_row: Row coordinate in 65×65 electrode space
        eimage_col: Column coordinate in 65×65 electrode space
        fit_result: Fitted alignment model
    
    Returns:
        (corrected_row, corrected_col) in 300×300 space, or (None, None) if invalid
    """
    if not fit_result.is_valid:
        return None, None
    
    if fit_result.row_slope is None or fit_result.col_slope is None:
        return None, None
    
    # Predict STA center (15×15 space) using fitted model
    # Note: axes are swapped in the model
    sta_row_pred = fit_result.row_slope * eimage_col + fit_result.row_intercept
    sta_col_pred = fit_result.col_slope * eimage_row + fit_result.col_intercept
    
    # Convert to 300×300 space
    corrected_row = sta_row_pred * COORDINATE_SCALE_FACTOR + COORDINATE_SCALE_FACTOR // 2
    corrected_col = sta_col_pred * COORDINATE_SCALE_FACTOR + COORDINATE_SCALE_FACTOR // 2
    
    # Clip to valid range
    corrected_row = max(0, min(299, corrected_row))
    corrected_col = max(0, min(299, corrected_col))
    
    return float(corrected_row), float(corrected_col)


# =============================================================================
# Main Step Function
# =============================================================================

def electrode_alignment_step(
    *,
    config: Optional[ElectrodeAlignmentConfig] = None,
    session: PipelineSession,
) -> PipelineSession:
    """
    Align electrode coordinates to stimulus space.
    
    This step:
    1. Collects paired (eimage, sta) center coordinates from all units
    2. Fits a linear model with outlier removal
    3. Stores fit parameters in session.metadata['electrode_alignment']
    4. Applies correction to each unit's eimage center
    
    Args:
        config: Configuration for alignment parameters
        session: Pipeline session (required)
    
    Returns:
        Updated session with alignment metadata and corrected coordinates
    """
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME} - already completed")
        return session
    
    logger.info("Step: Electrode-to-stimulus coordinate alignment...")
    
    # Use default config if not provided
    if config is None:
        config = ElectrodeAlignmentConfig()
    
    # Collect paired data points (with SNR filtering)
    collection = collect_alignment_data(
        session=session,
        sta_feature_name=config.sta_feature_name,
        eimage_snr_threshold=config.eimage_snr_threshold,
        sta_snr_threshold=config.sta_snr_threshold,
    )
    data_points = collection.data_points
    n_filtered_snr = collection.n_units_filtered_snr
    
    logger.info(f"  Collected {len(data_points)} units (eimage_SNR≥{config.eimage_snr_threshold}, sta_SNR≥{config.sta_snr_threshold})")
    
    if len(data_points) < config.min_units_threshold:
        logger.warning(red_warning(
            f"  Not enough units for alignment: {len(data_points)} < {config.min_units_threshold}"
        ))
        session.warnings.append(f"{STEP_NAME}: Not enough units ({len(data_points)})")
        
        # Store empty alignment metadata
        session.metadata['electrode_alignment'] = {
            'row_slope': None,
            'row_intercept': None,
            'row_r_squared': None,
            'col_slope': None,
            'col_intercept': None,
            'col_r_squared': None,
            'is_valid': False,
            'n_units_used': 0,
            'n_outliers_removed': 0,
            'n_units_filtered_snr': n_filtered_snr,
            'eimage_snr_threshold': config.eimage_snr_threshold,
            'sta_snr_threshold': config.sta_snr_threshold,
        }
        
        session.completed_steps.add(f"{STEP_NAME}:skipped")
        return session
    
    # Fit alignment model
    fit_result = fit_alignment_model(
        data_points=data_points,
        min_units_threshold=config.min_units_threshold,
        r_square_threshold=config.r_square_threshold,
        outlier_threshold=config.outlier_threshold,
    )
    
    # Store alignment metadata
    session.metadata['electrode_alignment'] = {
        'row_slope': fit_result.row_slope,
        'row_intercept': fit_result.row_intercept,
        'row_r_squared': fit_result.row_r_squared,
        'col_slope': fit_result.col_slope,
        'col_intercept': fit_result.col_intercept,
        'col_r_squared': fit_result.col_r_squared,
        'is_valid': fit_result.is_valid,
        'n_units_used': fit_result.n_units_used,
        'n_outliers_removed': fit_result.n_outliers_removed,
        'n_units_filtered_snr': n_filtered_snr,
        'eimage_snr_threshold': config.eimage_snr_threshold,
        'sta_snr_threshold': config.sta_snr_threshold,
    }
    
    if not fit_result.is_valid:
        logger.warning(red_warning(f"  Alignment fit not valid (R² too low)"))
        session.warnings.append(f"{STEP_NAME}: Fit not valid")
        session.completed_steps.add(f"{STEP_NAME}:invalid")
        return session
    
    logger.info(f"  Fit result: row R²={fit_result.row_r_squared:.3f}, col R²={fit_result.col_r_squared:.3f}")
    logger.info(f"  Units used: {fit_result.n_units_used}, outliers removed: {fit_result.n_outliers_removed}")
    
    # Apply correction to each unit's eimage center
    units_corrected = 0
    units_skipped = 0
    
    for unit_id, unit_data in session.units.items():
        features = unit_data.get('features', {})
        eimage_sta = features.get('eimage_sta', {})
        geometry = eimage_sta.get('geometry', {})
        
        eimage_row = geometry.get('center_row')
        eimage_col = geometry.get('center_col')
        
        # Handle numpy arrays
        if isinstance(eimage_row, np.ndarray):
            eimage_row = float(eimage_row.flat[0]) if eimage_row.size > 0 else None
        if isinstance(eimage_col, np.ndarray):
            eimage_col = float(eimage_col.flat[0]) if eimage_col.size > 0 else None
        
        if eimage_row is None or eimage_col is None:
            units_skipped += 1
            continue
        
        if np.isnan(eimage_row) or np.isnan(eimage_col):
            units_skipped += 1
            continue
        
        # Convert to 300×300 space
        corrected_row, corrected_col = convert_eimage_to_300(
            eimage_row=float(eimage_row),
            eimage_col=float(eimage_col),
            fit_result=fit_result,
        )
        
        if corrected_row is not None and corrected_col is not None:
            # Store corrected coordinates
            if 'eimage_sta' not in session.units[unit_id]['features']:
                session.units[unit_id]['features']['eimage_sta'] = {}
            if 'geometry' not in session.units[unit_id]['features']['eimage_sta']:
                session.units[unit_id]['features']['eimage_sta']['geometry'] = {}
            
            session.units[unit_id]['features']['eimage_sta']['geometry']['center_corrected_row'] = corrected_row
            session.units[unit_id]['features']['eimage_sta']['geometry']['center_corrected_col'] = corrected_col
            units_corrected += 1
        else:
            units_skipped += 1
    
    logger.info(f"  Corrected: {units_corrected} units, Skipped: {units_skipped}")
    
    session.mark_step_complete(STEP_NAME)
    return session

