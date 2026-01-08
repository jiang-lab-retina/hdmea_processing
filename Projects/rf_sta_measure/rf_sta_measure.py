"""
RF-STA Receptive Field Measurement with Gaussian and DoG Fitting

Analyzes receptive field structure from STA (spike-triggered average) data 
computed with dense noise stimulus. Uses:
- Baseline subtraction FIRST using frames 0-10 (before padding)
- 5-pixel padding with zero value (data is already baseline-subtracted)
- Gaussian blur (sigma=1.5) for preprocessing
- Extreme value method for center detection
- 2D Gaussian fitting with center radius constraint (5 pixels)
- DoG fitting with maximum sigma constraint (15 pixel diameter = 7.5 sigma)
- ON/OFF separate Gaussian fits

Data source: features/sta_perfect_dense_noise_15x15_15hz_r42_3min/data

Author: Generated for experimental analysis
Date: 2024-12
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.colors import Normalize
import warnings

# Enable logging
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Input HDF5 file (same as sta_quantification)
HDF5_PATH = Path(__file__).parent.parent / "pipeline_test" / "data" / "2024.03.01-14.40.14-Rec.h5"

# Output directory for plots
OUTPUT_DIR = Path(__file__).parent / "results"

# Feature data path (within each unit)
STA_FEATURE_NAME = "sta_perfect_dense_noise_15x15_15hz_r42_3min"
STA_DATA_PATH = f"features/{STA_FEATURE_NAME}/data"

# Analysis parameters
FRAME_RANGE = (40, 60)  # Frames to use for RF size estimation
THRESHOLD_FRACTION = 0.5  # Threshold for RF mask


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TemporalConfig:
    """Configuration for temporal preprocessing."""
    sigma_t: float = 2.0  # Temporal Gaussian smoothing sigma (frames)
    baseline_frames: int = 10  # Number of initial frames for baseline (0-10)
    savgol_window: int = 7  # Savitzky-Golay window size
    savgol_order: int = 3   # Polynomial order


@dataclass
class SpatialConfig:
    """Configuration for spatial preprocessing."""
    padding: int = 5  # Padding in pixels (uses baseline mean value)
    gaussian_sigma: float = 1.5  # 2D Gaussian blur sigma
    center_fit_radius: float = 5.0  # Radius for center fitting constraint (pixels)
    max_rf_diameter: float = 15.0  # Maximum RF diameter for sigma bounds (pixels)
    
    @property
    def max_sigma(self) -> float:
        """Maximum sigma based on max RF diameter (diameter = 2*sigma for ~68% of Gaussian)."""
        return self.max_rf_diameter / 2.0


TEMPORAL_CONFIG = TemporalConfig()
SPATIAL_CONFIG = SpatialConfig()


@dataclass
class GaussianFit:
    """2D Gaussian fit results."""
    center_x: float  # In original coordinates
    center_y: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    theta: float  # Rotation angle in radians
    offset: float
    r_squared: float  # Fit quality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'center_x': self.center_x,
            'center_y': self.center_y,
            'sigma_x': self.sigma_x,
            'sigma_y': self.sigma_y,
            'amplitude': self.amplitude,
            'theta': self.theta,
            'offset': self.offset,
            'r_squared': self.r_squared,
        }


@dataclass
class DoGFit:
    """Difference of Gaussians (center-surround) fit results."""
    center_x: float  # In original coordinates
    center_y: float
    amp_exc: float      # Excitatory (center) amplitude
    sigma_exc: float    # Excitatory sigma (smaller)
    amp_inh: float      # Inhibitory (surround) amplitude
    sigma_inh: float    # Inhibitory sigma (larger)
    offset: float
    r_squared: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'center_x': self.center_x,
            'center_y': self.center_y,
            'amp_exc': self.amp_exc,
            'sigma_exc': self.sigma_exc,
            'amp_inh': self.amp_inh,
            'sigma_inh': self.sigma_inh,
            'offset': self.offset,
            'r_squared': self.r_squared,
        }


@dataclass
class OnOffFit:
    """Separate ON/OFF Gaussian fits."""
    # ON component (positive values)
    on_center_x: float
    on_center_y: float
    on_sigma_x: float
    on_sigma_y: float
    on_amplitude: float
    on_r_squared: float
    # OFF component (negative values)
    off_center_x: float
    off_center_y: float
    off_sigma_x: float
    off_sigma_y: float
    off_amplitude: float
    off_r_squared: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'on_center_x': self.on_center_x,
            'on_center_y': self.on_center_y,
            'on_sigma_x': self.on_sigma_x,
            'on_sigma_y': self.on_sigma_y,
            'on_amplitude': self.on_amplitude,
            'on_r_squared': self.on_r_squared,
            'off_center_x': self.off_center_x,
            'off_center_y': self.off_center_y,
            'off_sigma_x': self.off_sigma_x,
            'off_sigma_y': self.off_sigma_y,
            'off_amplitude': self.off_amplitude,
            'off_r_squared': self.off_r_squared,
        }


@dataclass
class LNLFit:
    """
    Linear-Nonlinear model fit results.
    
    The LNL model maps stimulus through a linear filter (STA) to produce
    a generator signal g_t, then applies a static nonlinearity f(g) to
    predict firing rate.
    
    Attributes:
        a: Gain parameter in f(g) = exp(b + a*g) in ORIGINAL generator scale
        b: Baseline parameter in f(g) = exp(b + a*g)
        a_norm: Gain in NORMALIZED generator scale (effect of 1 std change on log-rate)
        log_likelihood: Poisson log-likelihood of the fitted model
        null_log_likelihood: Log-likelihood of constant-rate null model
        deviance_explained: Bits per spike (information gain over null model)
        r_squared: Correlation between predicted and observed rates
        n_frames: Number of frames used for fitting
        n_spikes: Total number of spikes used
        g_bin_centers: Generator signal bin centers for histogram nonlinearity
        rate_vs_g: Estimated firing rate at each bin (histogram nonlinearity)
        rectification_index: Asymmetry of response (0=symmetric, 1=fully rectified)
        nonlinearity_index: Deviation from linear fit (0=linear, higher=more curved)
        threshold_g: Generator signal value at which rate crosses mean (in std units)
    """
    # Parametric LNP parameters
    a: float          # Original scale (very small due to large g_std)
    b: float
    a_norm: float     # Normalized scale (interpretable: effect per std of g)
    # Fit quality metrics
    log_likelihood: float
    null_log_likelihood: float
    deviance_explained: float
    r_squared: float
    # Nonlinearity metrics
    rectification_index: float   # Asymmetry: 0=symmetric, 1=fully ON or OFF rectified
    nonlinearity_index: float    # Curvature: deviation from linear (R² of linear fit)
    threshold_g: float           # Threshold in std units where rate = mean_rate
    # Data info
    n_frames: int
    n_spikes: int
    # Histogram nonlinearity (for visualization)
    g_bin_centers: np.ndarray
    rate_vs_g: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'a': self.a,
            'b': self.b,
            'a_norm': self.a_norm,
            'log_likelihood': self.log_likelihood,
            'null_log_likelihood': self.null_log_likelihood,
            'bits_per_spike': self.deviance_explained,  # More descriptive name
            'r_squared': self.r_squared,
            'rectification_index': self.rectification_index,
            'nonlinearity_index': self.nonlinearity_index,
            'threshold_g': self.threshold_g,
            'n_frames': self.n_frames,
            'n_spikes': self.n_spikes,
            'g_bin_centers': self.g_bin_centers,
            'rate_vs_g': self.rate_vs_g,
        }


@dataclass
class RFGeometry:
    """
    Container for receptive field geometry results.
    
    Attributes:
        center_row: Row coordinate of RF center (y) in original coordinates
        center_col: Column coordinate of RF center (x) in original coordinates
        size_x: Width in pixels
        size_y: Height in pixels
        area: Area in pixels^2
        equivalent_diameter: Equivalent circular diameter
        diff_map: 2D max-min difference map (for visualization)
        peak_frame: Frame index with maximum activity
        gaussian_fit: 2D Gaussian fit results
        dog_fit: DoG center-surround fit results
        on_off_fit: Separate ON/OFF Gaussian fits
        sta_time_course: Time course at RF center (1D array, from preprocessed data)
        lnl_fit: Linear-Nonlinear model fit results
    """
    center_row: float
    center_col: float
    size_x: float
    size_y: float
    area: float
    equivalent_diameter: float
    diff_map: Optional[np.ndarray] = None
    peak_frame: Optional[int] = None
    gaussian_fit: Optional[GaussianFit] = None
    dog_fit: Optional[DoGFit] = None
    on_off_fit: Optional[OnOffFit] = None
    sta_time_course: Optional[np.ndarray] = None
    lnl_fit: Optional[LNLFit] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            'center_row': self.center_row,
            'center_col': self.center_col,
            'size_x': self.size_x,
            'size_y': self.size_y,
            'area': self.area,
            'equivalent_diameter': self.equivalent_diameter,
            'peak_frame': self.peak_frame,
        }
        if self.gaussian_fit:
            result['gaussian_fit'] = self.gaussian_fit.to_dict()
        if self.dog_fit:
            result['dog_fit'] = self.dog_fit.to_dict()
        if self.on_off_fit:
            result['on_off_fit'] = self.on_off_fit.to_dict()
        return result


# =============================================================================
# Spatial Preprocessing Utilities
# =============================================================================

def add_padding(data: np.ndarray, pad: int = None, pad_value: float = 0.0) -> np.ndarray:
    """
    Add padding to spatial dimensions with specified value.
    
    Args:
        data: 3D array (time, rows, cols)
        pad: Padding size in pixels (default from config)
        pad_value: Value to use for padding (default 0.0)
        
    Returns:
        Padded array (time, rows+2*pad, cols+2*pad)
    """
    if pad is None:
        pad = SPATIAL_CONFIG.padding
    
    if pad <= 0:
        return data.copy()
    
    n_frames, rows, cols = data.shape
    padded = np.full((n_frames, rows + 2*pad, cols + 2*pad), pad_value, dtype=data.dtype)
    padded[:, pad:pad+rows, pad:pad+cols] = data
    return padded


def remove_padding_coords(x: float, y: float, pad: int = None) -> Tuple[float, float]:
    """
    Convert padded coordinates back to original frame coordinates.
    
    Args:
        x: X coordinate in padded frame
        y: Y coordinate in padded frame
        pad: Padding size used
        
    Returns:
        Tuple of (x_original, y_original)
    """
    if pad is None:
        pad = SPATIAL_CONFIG.padding
    return x - pad, y - pad


def apply_gaussian_blur_2d(data: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    Apply 2D Gaussian blur to each frame.
    
    Args:
        data: 3D array (time, rows, cols)
        sigma: Gaussian sigma (default from config)
        
    Returns:
        Blurred array
    """
    if sigma is None:
        sigma = SPATIAL_CONFIG.gaussian_sigma
    
    if sigma <= 0:
        return data.copy()
    
    # Apply 2D Gaussian filter to each frame
    blurred = np.zeros_like(data)
    for t in range(data.shape[0]):
        blurred[t] = gaussian_filter(data[t], sigma=sigma)
    return blurred


# =============================================================================
# Temporal Preprocessing Utilities
# =============================================================================

def baseline_subtract(data: np.ndarray, n_baseline_frames: int = None) -> np.ndarray:
    """
    Subtract temporal baseline using mean of initial frames (0 to n).
    
    Args:
        data: 3D array (time, rows, cols)
        n_baseline_frames: Number of frames for baseline (uses 0:n)
        
    Returns:
        Baseline-subtracted array
    """
    if n_baseline_frames is None:
        n_baseline_frames = TEMPORAL_CONFIG.baseline_frames
    
    n_baseline_frames = min(n_baseline_frames, data.shape[0])
    # Use frames 0 to n_baseline_frames for baseline
    baseline = np.nanmean(data[:n_baseline_frames], axis=0, keepdims=True)
    return data - baseline


def temporal_smooth(data: np.ndarray, sigma_t: float = None) -> np.ndarray:
    """Apply 1D Gaussian smoothing along the temporal axis."""
    if sigma_t is None:
        sigma_t = TEMPORAL_CONFIG.sigma_t
    if sigma_t <= 0:
        return data.copy()
    return gaussian_filter1d(data, sigma=sigma_t, axis=0, mode='nearest')


def compute_baseline(data: np.ndarray, n_baseline_frames: int = None) -> np.ndarray:
    """
    Compute baseline using mean of initial frames.
    
    Args:
        data: 3D array (time, rows, cols)
        n_baseline_frames: Number of frames for baseline (uses 0:n)
        
    Returns:
        2D baseline array (rows, cols)
    """
    if n_baseline_frames is None:
        n_baseline_frames = TEMPORAL_CONFIG.baseline_frames
    
    n_baseline_frames = min(n_baseline_frames, data.shape[0])
    return np.nanmean(data[:n_baseline_frames], axis=0)


def preprocess_sta(
    sta_data: np.ndarray,
    apply_padding: bool = True,
    apply_blur: bool = True,
    subtract_baseline: bool = True,
    smooth_temporal: bool = True,
) -> Tuple[np.ndarray, int, float]:
    """
    Full preprocessing pipeline for STA data.
    
    Order of operations:
    1. Compute and subtract baseline FIRST (before padding)
    2. Add padding using baseline mean value
    3. Apply 2D Gaussian blur
    4. Apply temporal smoothing
    
    Args:
        sta_data: 3D array (time, rows, cols)
        apply_padding: Add padding with baseline mean value
        apply_blur: Apply 2D Gaussian blur
        subtract_baseline: Subtract baseline from frames 0-10
        smooth_temporal: Apply temporal smoothing
        
    Returns:
        Tuple of (preprocessed_data, padding_used, baseline_mean)
    """
    # Handle NaN values
    data = np.nan_to_num(sta_data, nan=0.0).astype(np.float64)
    
    # Step 1: Compute baseline and subtract FIRST (before padding)
    baseline_mean = 0.0
    if subtract_baseline:
        baseline = compute_baseline(data, TEMPORAL_CONFIG.baseline_frames)
        baseline_mean = float(np.mean(baseline))  # Mean of baseline for padding
        data = data - baseline[np.newaxis, :, :]  # Subtract baseline
    
    # Step 2: Add padding with baseline mean value (after baseline subtraction, pad value is 0)
    pad = SPATIAL_CONFIG.padding if apply_padding else 0
    if apply_padding:
        # After baseline subtraction, the data is zero-centered, so pad with 0
        data = add_padding(data, pad, pad_value=0.0)
    
    # Step 3: Apply 2D Gaussian blur
    if apply_blur:
        data = apply_gaussian_blur_2d(data, SPATIAL_CONFIG.gaussian_sigma)
    
    # Step 4: Temporal smoothing
    if smooth_temporal:
        data = temporal_smooth(data, TEMPORAL_CONFIG.sigma_t)
    
    return data, pad, baseline_mean


def robust_peak_trough_detection(
    signal: np.ndarray,
    window: int = None,
    order: int = None,
) -> Tuple[float, float, int, int]:
    """
    Detect peak and trough in a temporal signal using Savitzky-Golay filtering.
    
    Args:
        signal: 1D temporal signal
        window: Savitzky-Golay window size
        order: Savitzky-Golay polynomial order
        
    Returns:
        Tuple of (peak_value, trough_value, peak_idx, trough_idx)
    """
    if window is None:
        window = TEMPORAL_CONFIG.savgol_window
    if order is None:
        order = TEMPORAL_CONFIG.savgol_order
    
    # Ensure window is valid
    if len(signal) < window:
        window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    if window < order + 2:
        window = order + 2 if (order + 2) % 2 == 1 else order + 3
    
    # Apply Savitzky-Golay filter for smoothing
    smoothed = savgol_filter(signal, window_length=window, polyorder=order)
    
    peak_idx = int(np.argmax(smoothed))
    trough_idx = int(np.argmin(smoothed))
    
    return smoothed[peak_idx], smoothed[trough_idx], peak_idx, trough_idx


# =============================================================================
# Center Detection Functions
# =============================================================================

def find_center_extreme(data: np.ndarray, frame_range: Tuple[int, int] = None) -> Tuple[int, int, np.ndarray]:
    """
    Find RF center using extreme absolute value method.
    
    For each electrode (x, y), finds the maximum absolute value across time,
    then identifies the electrode with the largest absolute value as the center.
    
    Args:
        data: 3D array (time, rows, cols) - preprocessed (baseline subtracted)
        frame_range: Optional range of frames to analyze
        
    Returns:
        Tuple of (center_row, center_col, extreme_map)
    """
    if frame_range is not None:
        start, end = frame_range
        start = max(0, start)
        end = min(data.shape[0], end)
        data_subset = data[start:end+1]
    else:
        data_subset = data
    
    # For each electrode, find the extreme value (max absolute)
    # Take the value with largest absolute magnitude at each position
    max_pos = np.nanmax(data_subset, axis=0)  # Maximum positive
    min_neg = np.nanmin(data_subset, axis=0)  # Minimum (most negative)
    
    # Create extreme map: use the value with larger absolute magnitude
    extreme_map = np.where(np.abs(max_pos) > np.abs(min_neg), max_pos, min_neg)
    
    # Find center at maximum absolute value
    abs_extreme = np.abs(extreme_map)
    center_idx = np.nanargmax(abs_extreme)
    center_row, center_col = np.unravel_index(center_idx, extreme_map.shape)
    
    return int(center_row), int(center_col), extreme_map


def find_center_maxmin(sta_data: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Find RF center using max-min difference along time axis.
    
    Uses Savitzky-Golay filtering for robust peak/trough detection,
    considering temporal waveform shape rather than simple max/min.
    
    Args:
        sta_data: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (center_row, center_col, diff_map)
    """
    n_frames, rows, cols = sta_data.shape
    
    # Compute robust peak-to-peak for each pixel
    diff_map = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            signal = sta_data[:, r, c]
            peak_val, trough_val, _, _ = robust_peak_trough_detection(signal)
            diff_map[r, c] = peak_val - trough_val
    
    # Find location of maximum difference
    center_idx = np.nanargmax(diff_map)
    center_row, center_col = np.unravel_index(center_idx, diff_map.shape)
    
    return int(center_row), int(center_col), diff_map


def compute_robust_extreme_map(sta_data: np.ndarray) -> np.ndarray:
    """
    Compute a robust extreme map using Savitzky-Golay filtered peak/trough detection.
    
    For each pixel, finds the robust peak and trough values using temporal filtering,
    then takes the value with larger absolute magnitude (preserving sign).
    This is more robust to noise than simple max/min, suitable for ON/OFF fitting.
    
    Args:
        sta_data: 3D array (time, rows, cols) - preprocessed
        
    Returns:
        2D array (rows, cols) with signed extreme values
    """
    n_frames, rows, cols = sta_data.shape
    
    # Compute robust signed extreme for each pixel
    extreme_map = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            signal = sta_data[:, r, c]
            peak_val, trough_val, _, _ = robust_peak_trough_detection(signal)
            # Use the value with larger absolute magnitude, preserving sign
            if abs(peak_val) > abs(trough_val):
                extreme_map[r, c] = peak_val
            else:
                extreme_map[r, c] = trough_val
    
    return extreme_map


# =============================================================================
# 2D Gaussian Fitting Functions
# =============================================================================

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function with rotation.
    
    Args:
        xy: Tuple of (x, y) meshgrid arrays
        amplitude: Peak amplitude
        x0, y0: Center coordinates
        sigma_x, sigma_y: Standard deviations along principal axes
        theta: Rotation angle in radians
        offset: Baseline offset
        
    Returns:
        Flattened 1D array of Gaussian values
    """
    x, y = xy
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    
    g = offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
    return g.ravel()


def fit_2d_gaussian(
    diff_map: np.ndarray,
    initial_center: Tuple[int, int],
    center_radius: float = None,
    max_sigma: float = None,
) -> Optional[GaussianFit]:
    """
    Fit a 2D Gaussian to the difference map.
    
    Args:
        diff_map: 2D array of values to fit
        initial_center: Initial guess for center (row, col)
        center_radius: Maximum distance from initial center for fitted center (default from config)
        max_sigma: Maximum sigma value (default from config, 15 pixel diameter = 7.5 sigma)
        
    Returns:
        GaussianFit object or None if fitting fails
    """
    if center_radius is None:
        center_radius = SPATIAL_CONFIG.center_fit_radius
    if max_sigma is None:
        max_sigma = SPATIAL_CONFIG.max_sigma
    
    rows, cols = diff_map.shape
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    
    # Initial guesses
    y0, x0 = initial_center
    amplitude = diff_map[y0, x0] - np.median(diff_map)
    offset = np.median(diff_map)
    sigma_init = min(max(1.0, min(rows, cols) / 6), max_sigma)
    
    # Bounds with center radius constraint and max sigma
    # Center must stay within center_radius of initial center
    x_min = max(0, x0 - center_radius)
    x_max = min(cols, x0 + center_radius)
    y_min = max(0, y0 - center_radius)
    y_max = min(rows, y0 + center_radius)
    
    bounds = (
        [-np.inf, x_min, y_min, 0.5, 0.5, -np.pi, -np.inf],  # Lower
        [np.inf, x_max, y_max, max_sigma, max_sigma, np.pi, np.inf],  # Upper
    )
    
    p0 = [amplitude, x0, y0, sigma_init, sigma_init, 0, offset]
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                gaussian_2d,
                (x_grid, y_grid),
                diff_map.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
        
        # Calculate R-squared
        fitted = gaussian_2d((x_grid, y_grid), *popt).reshape(rows, cols)
        ss_res = np.sum((diff_map - fitted)**2)
        ss_tot = np.sum((diff_map - np.mean(diff_map))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return GaussianFit(
            center_x=float(popt[1]),
            center_y=float(popt[2]),
            sigma_x=float(abs(popt[3])),
            sigma_y=float(abs(popt[4])),
            amplitude=float(popt[0]),
            theta=float(popt[5]),
            offset=float(popt[6]),
            r_squared=float(r_squared),
        )
    except Exception as e:
        logger.warning(f"Gaussian fit failed: {e}")
        return None


# =============================================================================
# DoG (Difference of Gaussians) Fitting Functions
# =============================================================================

def dog_2d(xy, amp_exc, amp_inh, x0, y0, sigma_exc, sigma_inh, offset):
    """
    Difference of Gaussians (center-surround) function.
    
    DoG = A_exc * G(sigma_exc) - A_inh * G(sigma_inh)
    
    Args:
        xy: Tuple of (x, y) meshgrid arrays
        amp_exc: Excitatory (center) amplitude
        amp_inh: Inhibitory (surround) amplitude
        x0, y0: Center coordinates
        sigma_exc: Excitatory sigma (typically smaller)
        sigma_inh: Inhibitory sigma (typically larger)
        offset: Baseline offset
        
    Returns:
        Flattened 1D array of DoG values
    """
    x, y = xy
    r_sq = (x - x0)**2 + (y - y0)**2
    
    exc = amp_exc * np.exp(-r_sq / (2 * sigma_exc**2))
    inh = amp_inh * np.exp(-r_sq / (2 * sigma_inh**2))
    
    dog = offset + exc - inh
    return dog.ravel()


def fit_dog_center_surround(
    diff_map: np.ndarray,
    initial_center: Tuple[int, int],
    center_radius: float = None,
    max_sigma: float = None,
) -> Optional[DoGFit]:
    """
    Fit a Difference of Gaussians (center-surround) model.
    
    Args:
        diff_map: 2D array of values to fit
        initial_center: Initial guess for center (row, col)
        center_radius: Maximum distance from initial center for fitted center (default from config)
        max_sigma: Maximum sigma value (default from config, 15 pixel diameter = 7.5 sigma)
        
    Returns:
        DoGFit object or None if fitting fails
    """
    if center_radius is None:
        center_radius = SPATIAL_CONFIG.center_fit_radius
    if max_sigma is None:
        max_sigma = SPATIAL_CONFIG.max_sigma
    
    rows, cols = diff_map.shape
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    
    # Initial guesses
    y0, x0 = initial_center
    center_val = diff_map[y0, x0]
    offset = np.median(diff_map)
    
    # Determine if center is positive (ON) or negative (OFF)
    if center_val > offset:
        amp_exc_init = center_val - offset
        amp_inh_init = amp_exc_init * 0.5
    else:
        amp_exc_init = abs(center_val - offset)
        amp_inh_init = amp_exc_init * 0.5
    
    sigma_exc_init = min(max(0.5, min(rows, cols) / 8), max_sigma * 0.5)
    sigma_inh_init = min(sigma_exc_init * 2, max_sigma)
    
    # Center radius constraint
    x_min = max(0, x0 - center_radius)
    x_max = min(cols, x0 + center_radius)
    y_min = max(0, y0 - center_radius)
    y_max = min(rows, y0 + center_radius)
    
    # Bounds with center radius and max sigma constraints
    bounds = (
        [0, 0, x_min, y_min, 0.3, 0.5, -np.inf],  # Lower
        [np.inf, np.inf, x_max, y_max, max_sigma, max_sigma, np.inf],  # Upper
    )
    
    p0 = [amp_exc_init, amp_inh_init, x0, y0, sigma_exc_init, sigma_inh_init, offset]
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                dog_2d,
                (x_grid, y_grid),
                diff_map.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=5000,
            )
        
        # Calculate R-squared
        fitted = dog_2d((x_grid, y_grid), *popt).reshape(rows, cols)
        ss_res = np.sum((diff_map - fitted)**2)
        ss_tot = np.sum((diff_map - np.mean(diff_map))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return DoGFit(
            center_x=float(popt[2]),
            center_y=float(popt[3]),
            amp_exc=float(popt[0]),
            sigma_exc=float(abs(popt[4])),
            amp_inh=float(popt[1]),
            sigma_inh=float(abs(popt[5])),
            offset=float(popt[6]),
            r_squared=float(r_squared),
        )
    except Exception as e:
        logger.warning(f"DoG fit failed: {e}")
        return None


def gaussian_2d_simple(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """Simple 2D Gaussian without rotation."""
    x, y = xy
    g = offset + amplitude * np.exp(
        -((x-x0)**2 / (2*sigma_x**2) + (y-y0)**2 / (2*sigma_y**2))
    )
    return g.ravel()


def fit_dog_on_off(
    diff_map: np.ndarray,
    initial_center: Tuple[int, int],
    center_radius: float = None,
    max_sigma: float = None,
) -> Optional[OnOffFit]:
    """
    Fit separate Gaussians to ON (positive) and OFF (negative) components.
    
    Args:
        diff_map: 2D array of values to fit (baseline-subtracted)
        initial_center: Initial guess for center (row, col)
        center_radius: Maximum distance from initial center for fitted center (default from config)
        max_sigma: Maximum sigma value (default from config, 15 pixel diameter = 7.5 sigma)
        
    Returns:
        OnOffFit object or None if fitting fails
    """
    if center_radius is None:
        center_radius = SPATIAL_CONFIG.center_fit_radius
    if max_sigma is None:
        max_sigma = SPATIAL_CONFIG.max_sigma
    
    rows, cols = diff_map.shape
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    
    y0, x0 = initial_center
    
    # Separate ON and OFF components
    on_map = np.clip(diff_map, 0, None)  # Positive values only
    off_map = np.clip(-diff_map, 0, None)  # Absolute negative values
    
    # Fit ON component
    on_result = None
    if np.max(on_map) > 0:
        on_center_idx = np.argmax(on_map)
        on_y, on_x = np.unravel_index(on_center_idx, on_map.shape)
        amplitude = on_map[on_y, on_x]
        sigma_init = min(max(0.5, min(rows, cols) / 6), max_sigma)
        
        # Center radius constraint around ON peak
        on_x_min = max(0, on_x - center_radius)
        on_x_max = min(cols, on_x + center_radius)
        on_y_min = max(0, on_y - center_radius)
        on_y_max = min(rows, on_y + center_radius)
        
        bounds = (
            [0, on_x_min, on_y_min, 0.3, 0.3, -np.inf],
            [np.inf, on_x_max, on_y_max, max_sigma, max_sigma, np.inf],
        )
        p0 = [amplitude, on_x, on_y, sigma_init, sigma_init, 0]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    gaussian_2d_simple,
                    (x_grid, y_grid),
                    on_map.ravel(),
                    p0=p0,
                    bounds=bounds,
                    maxfev=3000,
                )
            
            fitted = gaussian_2d_simple((x_grid, y_grid), *popt).reshape(rows, cols)
            ss_res = np.sum((on_map - fitted)**2)
            ss_tot = np.sum((on_map - np.mean(on_map))**2)
            r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            on_result = {
                'center_x': popt[1], 'center_y': popt[2],
                'sigma_x': abs(popt[3]), 'sigma_y': abs(popt[4]),
                'amplitude': popt[0], 'r_squared': r_sq
            }
        except Exception:
            pass
    
    # Fit OFF component
    off_result = None
    if np.max(off_map) > 0:
        off_center_idx = np.argmax(off_map)
        off_y, off_x = np.unravel_index(off_center_idx, off_map.shape)
        amplitude = off_map[off_y, off_x]
        sigma_init = min(max(0.5, min(rows, cols) / 6), max_sigma)
        
        # Center radius constraint around OFF peak
        off_x_min = max(0, off_x - center_radius)
        off_x_max = min(cols, off_x + center_radius)
        off_y_min = max(0, off_y - center_radius)
        off_y_max = min(rows, off_y + center_radius)
        
        bounds = (
            [0, off_x_min, off_y_min, 0.3, 0.3, -np.inf],
            [np.inf, off_x_max, off_y_max, max_sigma, max_sigma, np.inf],
        )
        p0 = [amplitude, off_x, off_y, sigma_init, sigma_init, 0]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    gaussian_2d_simple,
                    (x_grid, y_grid),
                    off_map.ravel(),
                    p0=p0,
                    bounds=bounds,
                    maxfev=3000,
                )
            
            fitted = gaussian_2d_simple((x_grid, y_grid), *popt).reshape(rows, cols)
            ss_res = np.sum((off_map - fitted)**2)
            ss_tot = np.sum((off_map - np.mean(off_map))**2)
            r_sq = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            off_result = {
                'center_x': popt[1], 'center_y': popt[2],
                'sigma_x': abs(popt[3]), 'sigma_y': abs(popt[4]),
                'amplitude': popt[0], 'r_squared': r_sq
            }
        except Exception:
            pass
    
    # Only return if at least one component was fit
    if on_result is None and off_result is None:
        return None
    
    # Fill in defaults for missing components
    if on_result is None:
        on_result = {
            'center_x': x0, 'center_y': y0,
            'sigma_x': 0, 'sigma_y': 0,
            'amplitude': 0, 'r_squared': 0
        }
    if off_result is None:
        off_result = {
            'center_x': x0, 'center_y': y0,
            'sigma_x': 0, 'sigma_y': 0,
            'amplitude': 0, 'r_squared': 0
        }
    
    return OnOffFit(
        on_center_x=float(on_result['center_x']),
        on_center_y=float(on_result['center_y']),
        on_sigma_x=float(on_result['sigma_x']),
        on_sigma_y=float(on_result['sigma_y']),
        on_amplitude=float(on_result['amplitude']),
        on_r_squared=float(on_result['r_squared']),
        off_center_x=float(off_result['center_x']),
        off_center_y=float(off_result['center_y']),
        off_sigma_x=float(off_result['sigma_x']),
        off_sigma_y=float(off_result['sigma_y']),
        off_amplitude=float(off_result['amplitude']),
        off_r_squared=float(off_result['r_squared']),
    )


# =============================================================================
# Core Geometry Extraction Functions
# =============================================================================

def estimate_rf_size_maxmin(
    sta_data: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (40, 60),
    threshold_fraction: float = 0.5,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Estimate RF size using max-min difference map.
    
    Args:
        sta_data: 3D array (time, rows, cols)
        center: Center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        threshold_fraction: Fraction of max for threshold
        
    Returns:
        Tuple of (size_x, size_y, area, equivalent_diameter, diff_map)
    """
    n_frames, rows, cols = sta_data.shape
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    data_subset = sta_data[start_frame:end_frame + 1]
    
    # Compute max - min along time axis
    max_vals = np.nanmax(data_subset, axis=0)
    min_vals = np.nanmin(data_subset, axis=0)
    diff_map = max_vals - min_vals
    diff_map = np.nan_to_num(diff_map, nan=0.0)
    
    # Smooth
    diff_smooth = gaussian_filter(diff_map, sigma=0.5)
    
    # Threshold
    max_diff = np.max(diff_smooth)
    threshold = max_diff * threshold_fraction
    high_activity_mask = diff_smooth > threshold
    
    # Connected component containing center
    labeled, n_labels = label(high_activity_mask)
    
    center_row = int(np.clip(center[0], 0, rows - 1))
    center_col = int(np.clip(center[1], 0, cols - 1))
    center_label = labeled[center_row, center_col]
    
    if center_label > 0:
        rf_mask = labeled == center_label
    else:
        rf_mask = high_activity_mask
    
    area = float(np.sum(rf_mask))
    
    if area > 0:
        y_coords, x_coords = np.where(rf_mask)
        size_y = float(np.max(y_coords) - np.min(y_coords) + 1)
        size_x = float(np.max(x_coords) - np.min(x_coords) + 1)
    else:
        size_x = size_y = 0.0
    
    equivalent_diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0.0
    
    return size_x, size_y, area, equivalent_diameter, diff_map


def extract_rf_geometry(
    sta_data: np.ndarray,
    frame_range: Tuple[int, int] = (40, 60),
    threshold_fraction: float = 0.5,
) -> RFGeometry:
    """
    Extract RF geometry with full preprocessing and fitting pipeline.
    
    Pipeline:
    1. Add 5-pixel zero padding
    2. Apply Gaussian blur (sigma=1.5)
    3. Baseline subtraction (frames 0-10)
    4. Find center using diff_map (max-min with temporal filtering, more robust to noise)
    5. Fit 2D Gaussian
    6. Fit DoG center-surround model
    7. Fit ON/OFF model (uses robust extreme_map with temporal filtering)
    8. Convert coordinates back to original frame
    
    Args:
        sta_data: 3D array (time, rows, cols)
        frame_range: Range of frames to use for analysis
        threshold_fraction: Fraction of max for threshold in size estimation
        
    Returns:
        RFGeometry object with all geometry attributes and fit results
    """
    original_shape = sta_data.shape
    
    # Step 1-4: Full preprocessing (baseline subtraction first, then padding)
    preprocessed, pad, baseline_mean = preprocess_sta(
        sta_data,
        apply_padding=True,
        apply_blur=True,
        subtract_baseline=True,
        smooth_temporal=True,
    )
    
    # Step 5: Find center using diff_map (max-min with temporal filtering)
    # This is more robust to noise than extreme value method
    center_row_pad, center_col_pad, diff_map = find_center_maxmin(preprocessed)
    
    # Compute robust extreme_map for ON/OFF model fitting (uses temporal filtering)
    extreme_map = compute_robust_extreme_map(preprocessed)
    
    # Step 6: Estimate size
    size_x, size_y, area, equiv_diam, diff_map_size = estimate_rf_size_maxmin(
        preprocessed,
        center=(center_row_pad, center_col_pad),
        frame_range=frame_range,
        threshold_fraction=threshold_fraction,
    )
    
    # Find peak frame
    frame_activity = np.nanmax(np.abs(preprocessed), axis=(1, 2))
    peak_frame = int(np.argmax(frame_activity))
    
    # Step 7: Fit 2D Gaussian
    gaussian_fit = fit_2d_gaussian(diff_map, (center_row_pad, center_col_pad))
    
    # Step 8: Fit DoG center-surround
    dog_fit = fit_dog_center_surround(diff_map, (center_row_pad, center_col_pad))
    
    # Step 9: Fit ON/OFF model
    on_off_fit = fit_dog_on_off(extreme_map, (center_row_pad, center_col_pad))
    
    # Step 10: Extract STA time course at gaussian center (before coordinate conversion)
    # Uses preprocessed (Gaussian filtered) data at rounded gaussian center
    sta_time_course = None
    if gaussian_fit is not None:
        # Round gaussian center to nearest pixel (still in padded coords at this point)
        tc_row = int(round(gaussian_fit.center_y))  # center_y = row
        tc_col = int(round(gaussian_fit.center_x))  # center_x = col
        # Ensure within bounds of preprocessed array
        tc_row = max(0, min(tc_row, preprocessed.shape[1] - 1))
        tc_col = max(0, min(tc_col, preprocessed.shape[2] - 1))
        sta_time_course = preprocessed[:, tc_row, tc_col].copy()
    
    # Step 11: Convert coordinates back to original frame
    center_col_orig, center_row_orig = remove_padding_coords(center_col_pad, center_row_pad, pad)
    
    # Also adjust fit coordinates
    if gaussian_fit is not None:
        gf_x, gf_y = remove_padding_coords(gaussian_fit.center_x, gaussian_fit.center_y, pad)
        gaussian_fit = GaussianFit(
            center_x=gf_x,
            center_y=gf_y,
            sigma_x=gaussian_fit.sigma_x,
            sigma_y=gaussian_fit.sigma_y,
            amplitude=gaussian_fit.amplitude,
            theta=gaussian_fit.theta,
            offset=gaussian_fit.offset,
            r_squared=gaussian_fit.r_squared,
        )
    
    if dog_fit is not None:
        df_x, df_y = remove_padding_coords(dog_fit.center_x, dog_fit.center_y, pad)
        dog_fit = DoGFit(
            center_x=df_x,
            center_y=df_y,
            amp_exc=dog_fit.amp_exc,
            sigma_exc=dog_fit.sigma_exc,
            amp_inh=dog_fit.amp_inh,
            sigma_inh=dog_fit.sigma_inh,
            offset=dog_fit.offset,
            r_squared=dog_fit.r_squared,
        )
    
    if on_off_fit is not None:
        on_x, on_y = remove_padding_coords(on_off_fit.on_center_x, on_off_fit.on_center_y, pad)
        off_x, off_y = remove_padding_coords(on_off_fit.off_center_x, on_off_fit.off_center_y, pad)
        on_off_fit = OnOffFit(
            on_center_x=on_x,
            on_center_y=on_y,
            on_sigma_x=on_off_fit.on_sigma_x,
            on_sigma_y=on_off_fit.on_sigma_y,
            on_amplitude=on_off_fit.on_amplitude,
            on_r_squared=on_off_fit.on_r_squared,
            off_center_x=off_x,
            off_center_y=off_y,
            off_sigma_x=on_off_fit.off_sigma_x,
            off_sigma_y=on_off_fit.off_sigma_y,
            off_amplitude=on_off_fit.off_amplitude,
            off_r_squared=on_off_fit.off_r_squared,
        )
    
    # Remove padding from diff_map for visualization (crop to original size)
    if diff_map is not None and pad > 0:
        diff_map = diff_map[pad:-pad, pad:-pad] if diff_map.shape[0] > 2*pad else diff_map
    
    return RFGeometry(
        center_row=float(center_row_orig),
        center_col=float(center_col_orig),
        size_x=size_x,
        size_y=size_y,
        area=area,
        equivalent_diameter=equiv_diam,
        diff_map=diff_map,
        peak_frame=peak_frame,
        gaussian_fit=gaussian_fit,
        dog_fit=dog_fit,
        on_off_fit=on_off_fit,
        sta_time_course=sta_time_course,
    )


# =============================================================================
# LNL (Linear-Nonlinear) Model Fitting Functions
# =============================================================================

def compute_generator_signal(
    sta: np.ndarray,
    movie_array: np.ndarray,
    cover_range: Tuple[int, int] = (-60, 0),
) -> Tuple[np.ndarray, int, int]:
    """
    Compute generator signal g_t by convolving STA with stimulus movie.
    
    The generator signal is the dot product of the (mean-subtracted) STA filter 
    with the (mean-subtracted) time-lagged stimulus at each frame:
    
        g_t = sum_{tau,x,y} (STA[tau,y,x] - STA_mean) * (s_{t-tau}[y,x] - s_mean)
    
    Mean-subtraction is critical: for binary stimuli (0/255), the raw STA has
    mean ~127.5. Without centering, the generator signal would be dominated by
    a DC component with no discriminative power.
    
    Args:
        sta: STA array with shape (L, H, W) where L is temporal window length
        movie_array: Stimulus movie with shape (T, H, W)
        cover_range: Frame window relative to spike (start, end), e.g., (-60, 0)
    
    Returns:
        Tuple of:
        - generator_signal: Array of shape (n_valid_frames,) with g_t values
        - first_valid_frame: First frame index where g_t is valid
        - last_valid_frame: Last frame index (exclusive) where g_t is valid
    """
    L = sta.shape[0]  # Temporal window length
    T = movie_array.shape[0]  # Total movie frames
    
    # Determine valid frame range
    # For cover_range=(-60, 0), we need 60 frames of history
    # So first valid frame is at index 60 (0-indexed: frame 60)
    first_valid_frame = abs(cover_range[0])  # e.g., 60
    last_valid_frame = T + cover_range[1]  # e.g., T if cover_range[1]=0
    
    if first_valid_frame >= last_valid_frame:
        logger.warning(f"No valid frames for generator signal: first={first_valid_frame}, last={last_valid_frame}")
        return np.array([]), first_valid_frame, last_valid_frame
    
    n_valid_frames = last_valid_frame - first_valid_frame
    generator_signal = np.zeros(n_valid_frames, dtype=np.float32)
    
    # CRITICAL: Mean-subtract the STA to remove DC component
    # For binary noise (0/255), raw STA mean is ~127.5
    # The centered STA captures deviations that predict spikes
    sta_centered = sta.astype(np.float32) - sta.mean()
    sta_flat = sta_centered.ravel()
    
    # Also compute stimulus mean for centering (use global mean for efficiency)
    stim_mean = float(movie_array.mean())
    
    # Compute g_t for each valid frame
    for i, t in enumerate(range(first_valid_frame, last_valid_frame)):
        # Extract time-lagged stimulus window: s_{t+cover_range[0]} to s_{t+cover_range[1]-1}
        # For cover_range=(-60, 0), this is s_{t-60} to s_{t-1}
        stim_window = movie_array[t + cover_range[0]:t + cover_range[1]]  # Shape: (L, H, W)
        
        # Center stimulus and compute dot product
        stim_centered = stim_window.astype(np.float32).ravel() - stim_mean
        generator_signal[i] = np.dot(sta_flat, stim_centered)
    
    return generator_signal, first_valid_frame, last_valid_frame


def estimate_histogram_nonlinearity(
    g_all: np.ndarray,
    spike_frames: np.ndarray,
    first_valid_frame: int,
    dt: float,
    n_bins: int = 50,
    smooth_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate nonlinearity using histogram / Bayes rule method.
    
    Uses Bayes' rule:
        p(spike|g) = p(g|spike) * p(spike) / p(g)
    
    Then converts to firing rate:
        lambda(g) = p(spike|g) / dt
    
    Args:
        g_all: Generator signal for all valid frames
        spike_frames: Spike frame indices (absolute, not relative to g_all)
        first_valid_frame: First valid frame index (to convert spike_frames to g_all indices)
        dt: Time bin width in seconds (frame duration)
        n_bins: Number of bins for histogram
        smooth_sigma: Gaussian smoothing sigma for output curve
    
    Returns:
        Tuple of:
        - g_bin_centers: Bin centers for generator signal
        - rate_vs_g: Estimated firing rate at each bin
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Convert spike frames to indices in g_all
    spike_indices = spike_frames - first_valid_frame
    # Keep only spikes that fall within valid range
    valid_spike_mask = (spike_indices >= 0) & (spike_indices < len(g_all))
    spike_indices = spike_indices[valid_spike_mask].astype(int)
    
    if len(spike_indices) == 0:
        logger.warning("No valid spikes for histogram nonlinearity estimation")
        g_bin_centers = np.linspace(g_all.min(), g_all.max(), n_bins)
        rate_vs_g = np.zeros(n_bins)
        return g_bin_centers, rate_vs_g
    
    # Get generator values at spike times
    g_at_spikes = g_all[spike_indices]
    
    # Compute histograms
    g_min, g_max = g_all.min(), g_all.max()
    # Add small margin to avoid edge effects
    margin = (g_max - g_min) * 0.01
    bin_edges = np.linspace(g_min - margin, g_max + margin, n_bins + 1)
    g_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # p(g): histogram of all generator values
    hist_all, _ = np.histogram(g_all, bins=bin_edges)
    # p(g|spike): histogram of generator values at spike times
    hist_spike, _ = np.histogram(g_at_spikes, bins=bin_edges)
    
    # Avoid division by zero
    hist_all_safe = np.maximum(hist_all, 1)
    
    # p(spike) = total spikes / total frames
    p_spike = len(spike_indices) / len(g_all)
    
    # p(spike|g) via Bayes rule
    # p(spike|g) = p(g|spike) * p(spike) / p(g)
    # Using histogram counts: p(spike|g) ∝ hist_spike / hist_all
    p_spike_given_g = hist_spike / hist_all_safe
    
    # Convert to rate: lambda(g) = p(spike|g) / dt
    rate_vs_g = p_spike_given_g / dt
    
    # Smooth the curve
    if smooth_sigma > 0:
        rate_vs_g = gaussian_filter1d(rate_vs_g, sigma=smooth_sigma)
    
    return g_bin_centers.astype(np.float32), rate_vs_g.astype(np.float32)


def fit_lnp_parametric(
    g_all: np.ndarray,
    spike_frames: np.ndarray,
    first_valid_frame: int,
    dt: float,
) -> Tuple[float, float, float, float]:
    """
    Fit parametric LNP nonlinearity f(g) = exp(b + a*g) via Poisson MLE.
    
    Maximizes Poisson log-likelihood:
        log L = sum_t [y_t * log(lambda_t) - lambda_t * dt]
    
    where lambda_t = exp(b + a * g_t)
    
    Args:
        g_all: Generator signal for all valid frames
        spike_frames: Spike frame indices (absolute)
        first_valid_frame: First valid frame index
        dt: Time bin width in seconds
    
    Returns:
        Tuple of (a, b, log_likelihood, null_log_likelihood)
    """
    from scipy.optimize import minimize
    
    # Convert spike frames to per-frame spike counts
    n_frames = len(g_all)
    spike_indices = spike_frames - first_valid_frame
    valid_spike_mask = (spike_indices >= 0) & (spike_indices < n_frames)
    spike_indices = spike_indices[valid_spike_mask].astype(int)
    
    # Create spike count array
    spike_counts = np.bincount(spike_indices, minlength=n_frames).astype(np.float32)
    
    total_spikes = spike_counts.sum()
    
    if total_spikes == 0:
        logger.warning("No spikes for LNP fitting")
        return 0.0, 0.0, -np.inf, -np.inf
    
    # Normalize generator signal for numerical stability
    g_mean = g_all.mean()
    g_std = g_all.std()
    
    logger.debug(f"Generator signal stats: mean={g_mean:.2f}, std={g_std:.2f}, "
                 f"min={g_all.min():.2f}, max={g_all.max():.2f}")
    
    if g_std < 1e-10:
        logger.warning("Generator signal has near-zero variance - STA may not be predictive")
        g_std = 1.0
    g_norm = (g_all - g_mean) / g_std
    
    # Initial parameters
    mean_rate = total_spikes / (n_frames * dt)
    b_init = np.log(max(mean_rate, 1e-10))
    a_init = 1.0
    
    logger.debug(f"LNP fitting: {int(total_spikes)} spikes, {n_frames} frames, "
                 f"mean_rate={mean_rate:.2f} Hz")
    
    def neg_log_likelihood(params):
        a, b = params
        # lambda_t = exp(b + a * g)
        log_lambda = b + a * g_norm
        # Clip for numerical stability
        log_lambda = np.clip(log_lambda, -20, 20)
        lambda_t = np.exp(log_lambda)
        
        # Poisson log-likelihood: y*log(lambda) - lambda*dt
        # Add small epsilon to avoid log(0)
        ll = np.sum(spike_counts * log_lambda - lambda_t * dt)
        return -ll  # Minimize negative log-likelihood
    
    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0=[a_init, b_init],
        method='L-BFGS-B',
        bounds=[(-10, 10), (-20, 20)],
    )
    
    a_opt, b_opt = result.x
    log_likelihood = -result.fun
    
    # Transform a back to original scale (b already includes offset from normalization)
    a_original = a_opt / g_std
    b_original = b_opt - a_opt * g_mean / g_std
    
    # Null model log-likelihood (constant rate)
    null_rate = total_spikes / (n_frames * dt)
    null_ll = np.sum(spike_counts * np.log(max(null_rate, 1e-10)) - null_rate * dt)
    
    # Also return the normalized-space a for interpretability
    # a_norm is the effect of 1 std change in generator signal on log-rate
    return a_original, b_original, log_likelihood, null_ll, a_opt


def compute_nonlinearity_metrics(
    g_bin_centers: np.ndarray,
    rate_vs_g: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute metrics that quantify the shape of the nonlinearity.
    
    Args:
        g_bin_centers: Generator signal bin centers (original scale)
        rate_vs_g: Firing rate at each bin (Hz)
    
    Returns:
        Tuple of (rectification_index, nonlinearity_index, threshold_g)
        
        - rectification_index: Asymmetry of response (0=symmetric, 1=fully rectified)
          Measures whether the neuron responds more to positive or negative g
        
        - nonlinearity_index: Deviation from linear fit (1 - R² of linear fit)
          0 = perfectly linear, 1 = highly nonlinear
        
        - threshold_g: Generator signal value (in std units) where rate crosses mean
          Indicates the threshold for response
    """
    # Handle edge cases
    if len(g_bin_centers) < 5 or len(rate_vs_g) < 5:
        return 0.0, 0.0, 0.0
    
    # Filter out invalid values
    valid_mask = np.isfinite(rate_vs_g) & (rate_vs_g >= 0)
    if valid_mask.sum() < 5:
        return 0.0, 0.0, 0.0
    
    g = g_bin_centers[valid_mask]
    rate = rate_vs_g[valid_mask]
    
    # Normalize g to std units for interpretability
    g_mean = g.mean()
    g_std = g.std()
    if g_std > 0:
        g_norm = (g - g_mean) / g_std
    else:
        g_norm = g - g_mean
    
    mean_rate = rate.mean()
    
    # 1. Rectification Index
    # Compare response to positive vs negative generator signal
    # RI = (rate_positive - rate_negative) / (rate_positive + rate_negative)
    # Ranges from -1 (OFF cell) to +1 (ON cell), 0 = symmetric
    pos_mask = g_norm > 0
    neg_mask = g_norm < 0
    
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        rate_pos = rate[pos_mask].mean()
        rate_neg = rate[neg_mask].mean()
        if (rate_pos + rate_neg) > 0:
            rectification_index = (rate_pos - rate_neg) / (rate_pos + rate_neg)
        else:
            rectification_index = 0.0
    else:
        rectification_index = 0.0
    
    # 2. Nonlinearity Index
    # Fit a linear model and compute 1 - R²
    # Higher values indicate more nonlinear (curved) relationship
    try:
        # Linear fit: rate = slope * g_norm + intercept
        coeffs = np.polyfit(g_norm, rate, 1)
        rate_linear_pred = np.polyval(coeffs, g_norm)
        
        ss_res = np.sum((rate - rate_linear_pred) ** 2)
        ss_tot = np.sum((rate - mean_rate) ** 2)
        
        if ss_tot > 0:
            r2_linear = 1 - ss_res / ss_tot
            nonlinearity_index = 1 - max(0, r2_linear)  # Clamp R² to [0, 1]
        else:
            nonlinearity_index = 0.0
    except Exception:
        nonlinearity_index = 0.0
    
    # 3. Threshold
    # Find g value (in std units) where rate crosses mean_rate
    # Use linear interpolation to find the crossing point
    try:
        # Look for where rate crosses mean_rate from below
        above_mean = rate >= mean_rate
        crossings = np.where(np.diff(above_mean.astype(int)) == 1)[0]
        
        if len(crossings) > 0:
            # Take the crossing closest to g=0
            idx = crossings[np.argmin(np.abs(g_norm[crossings]))]
            # Linear interpolation between idx and idx+1
            g1, g2 = g_norm[idx], g_norm[idx + 1]
            r1, r2 = rate[idx], rate[idx + 1]
            if r2 != r1:
                threshold_g = g1 + (mean_rate - r1) * (g2 - g1) / (r2 - r1)
            else:
                threshold_g = (g1 + g2) / 2
        else:
            # No crossing found, use median g where rate > mean
            high_rate_mask = rate > mean_rate
            if high_rate_mask.sum() > 0:
                threshold_g = np.median(g_norm[high_rate_mask])
            else:
                threshold_g = 0.0
    except Exception:
        threshold_g = 0.0
    
    return float(rectification_index), float(nonlinearity_index), float(threshold_g)


def fit_lnl_model(
    sta: np.ndarray,
    movie_array: np.ndarray,
    spike_frames: np.ndarray,
    cover_range: Tuple[int, int] = (-60, 0),
    frame_rate: float = 15.0,
    n_histogram_bins: int = 50,
) -> Optional[LNLFit]:
    """
    Fit complete LNL model: compute generator signal and estimate nonlinearity.
    
    This is the main entry point for LNL fitting. It:
    1. Computes the generator signal by convolving STA with stimulus
    2. Estimates histogram-based nonlinearity via Bayes rule
    3. Fits parametric LNP nonlinearity via Poisson MLE
    4. Computes fit quality metrics
    5. Computes nonlinearity shape metrics
    
    Args:
        sta: STA array with shape (L, H, W)
        movie_array: Stimulus movie with shape (T, H, W)
        spike_frames: Spike frame indices
        cover_range: Frame window relative to spike (start, end)
        frame_rate: Stimulus frame rate in Hz
        n_histogram_bins: Number of bins for histogram nonlinearity
    
    Returns:
        LNLFit object with all fit results, or None if fitting fails
    """
    dt = 1.0 / frame_rate
    
    # Step 1: Compute generator signal
    g_all, first_valid, last_valid = compute_generator_signal(sta, movie_array, cover_range)
    
    if len(g_all) == 0:
        logger.warning("Cannot compute generator signal - no valid frames")
        return None
    
    n_frames = len(g_all)
    
    # Count valid spikes
    spike_indices = spike_frames - first_valid
    valid_spike_mask = (spike_indices >= 0) & (spike_indices < n_frames)
    n_spikes = int(valid_spike_mask.sum())
    
    if n_spikes < 10:
        logger.warning(f"Too few spikes ({n_spikes}) for reliable LNL fitting")
        return None
    
    # Step 2: Estimate histogram nonlinearity
    g_bin_centers, rate_vs_g = estimate_histogram_nonlinearity(
        g_all, spike_frames, first_valid, dt, n_bins=n_histogram_bins
    )
    
    # Step 3: Fit parametric LNP
    a, b, log_likelihood, null_ll, a_norm = fit_lnp_parametric(
        g_all, spike_frames, first_valid, dt
    )
    
    # Step 4: Compute fit quality metrics
    # 
    # Since we omit the constant -log(y!) term, our log-likelihoods can be positive.
    # Use simpler metrics that are robust to this.
    
    spike_indices_valid = (spike_frames - first_valid)[valid_spike_mask].astype(int)
    spike_counts = np.bincount(spike_indices_valid, minlength=n_frames).astype(np.float32)
    
    # Log-likelihood improvement (positive = model is better than null)
    ll_improvement = log_likelihood - null_ll
    
    # Bits per spike: information gain over null model, normalized by spike count
    # This is a standard metric for neural encoding models
    if n_spikes > 0:
        bits_per_spike = (log_likelihood - null_ll) / (n_spikes * np.log(2))
    else:
        bits_per_spike = 0.0
    
    # Use bits_per_spike as deviance_explained (more interpretable for Poisson models)
    # Values > 0 indicate the model predicts better than the null
    # Typical values for good LN models: 0.5 - 2.0 bits/spike
    deviance_explained = float(bits_per_spike)
    
    # Predicted rate using original-scale parameters
    predicted_rate = np.exp(b + a * g_all)
    
    # R-squared: correlation between predicted rate and spike counts
    # Note: This will be low for sparse spike trains, which is expected
    if np.std(spike_counts) > 0 and np.std(predicted_rate) > 0:
        r_squared = float(np.corrcoef(spike_counts, predicted_rate)[0, 1] ** 2)
    else:
        r_squared = 0.0
    
    # Step 5: Compute nonlinearity metrics from histogram
    rect_idx, nl_idx, thresh_g = compute_nonlinearity_metrics(g_bin_centers, rate_vs_g)
    
    return LNLFit(
        a=float(a),
        b=float(b),
        a_norm=float(a_norm),
        log_likelihood=float(log_likelihood),
        null_log_likelihood=float(null_ll),
        deviance_explained=float(deviance_explained),  # Actually bits_per_spike
        r_squared=float(r_squared),
        rectification_index=float(rect_idx),
        nonlinearity_index=float(nl_idx),
        threshold_g=float(thresh_g),
        n_frames=n_frames,
        n_spikes=n_spikes,
        g_bin_centers=g_bin_centers,
        rate_vs_g=rate_vs_g,
    )


# =============================================================================
# Data Loading
# =============================================================================

def load_sta_data(hdf5_path: Path) -> Dict[str, np.ndarray]:
    """
    Load STA data for all units from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        Dictionary mapping unit_id to STA array (time, rows, cols)
    """
    data = {}
    
    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            raise ValueError(f"No 'units' group found in {hdf5_path}")
        
        unit_ids = list(f["units"].keys())
        logger.info(f"Found {len(unit_ids)} units in HDF5 file")
        
        for unit_id in unit_ids:
            sta_path = f"units/{unit_id}/{STA_DATA_PATH}"
            if sta_path in f:
                data[unit_id] = f[sta_path][:]
                logger.info(f"Loaded {unit_id}: shape={data[unit_id].shape}")
            else:
                logger.warning(f"No STA data found for {unit_id} at {STA_DATA_PATH}")
    
    return data


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_unit_rf_geometry(
    unit_id: str,
    sta_data: np.ndarray,
    geometry: RFGeometry,
    output_path: Path,
) -> None:
    """
    Plot RF geometry visualization for a single unit.
    
    Creates a 3x3 subplot with:
    - Row 1: Peak frame, Diff map, Extreme map
    - Row 2: Gaussian fit, DoG fit, ON/OFF comparison
    - Row 3: Temporal profile, Fit statistics
    """
    fig = plt.figure(figsize=(16, 14))
    
    n_frames, rows, cols = sta_data.shape
    center_row = int(np.clip(geometry.center_row, 0, rows-1))
    center_col = int(np.clip(geometry.center_col, 0, cols-1))
    
    # Preprocess for visualization
    preprocessed, pad, _ = preprocess_sta(sta_data, apply_padding=True, apply_blur=True, 
                                           subtract_baseline=True, smooth_temporal=True)
    
    # Get extreme map (using robust temporal filtering)
    extreme_map = compute_robust_extreme_map(preprocessed)
    if pad > 0:
        extreme_map_crop = extreme_map[pad:-pad, pad:-pad] if extreme_map.shape[0] > 2*pad else extreme_map
    else:
        extreme_map_crop = extreme_map
    
    # Peak frame
    peak_frame_idx = geometry.peak_frame if geometry.peak_frame is not None else n_frames // 2
    peak_frame = sta_data[peak_frame_idx]
    
    vmin = np.nanpercentile(peak_frame, 1)
    vmax = np.nanpercentile(peak_frame, 99)
    vmax_sym = max(abs(vmin), abs(vmax))
    
    # --- Row 1, Col 1: Peak frame ---
    ax = fig.add_subplot(3, 3, 1)
    im = ax.imshow(peak_frame, cmap='RdBu_r', vmin=-vmax_sym, vmax=vmax_sym, origin='lower')
    ax.plot(geometry.center_col, geometry.center_row, 'g+', markersize=15, markeredgewidth=3)
    if geometry.size_x > 0 and geometry.size_y > 0:
        ellipse = Ellipse(
            (geometry.center_col, geometry.center_row),
            width=geometry.size_x, height=geometry.size_y,
            fill=False, edgecolor='lime', linewidth=2, linestyle='--',
        )
        ax.add_patch(ellipse)
    ax.set_title(f'Peak Frame (t={peak_frame_idx})', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # --- Row 1, Col 2: Diff map ---
    ax = fig.add_subplot(3, 3, 2)
    if geometry.diff_map is not None:
        im = ax.imshow(geometry.diff_map, cmap='hot', origin='lower')
        ax.plot(geometry.center_col, geometry.center_row, 'c+', markersize=15, markeredgewidth=3)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Max-Min Diff')
    ax.set_title('Difference Map', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # --- Row 1, Col 3: Extreme map ---
    ax = fig.add_subplot(3, 3, 3)
    ext_max = np.max(np.abs(extreme_map_crop))
    im = ax.imshow(extreme_map_crop, cmap='RdBu_r', vmin=-ext_max, vmax=ext_max, origin='lower')
    ax.plot(geometry.center_col, geometry.center_row, 'g+', markersize=15, markeredgewidth=3)
    ax.set_title('Extreme Value Map', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Extreme Value')
    
    # --- Row 2, Col 1: Gaussian fit contours ---
    ax = fig.add_subplot(3, 3, 4)
    if geometry.diff_map is not None:
        im = ax.imshow(geometry.diff_map, cmap='hot', origin='lower', alpha=0.7)
    if geometry.gaussian_fit is not None:
        gf = geometry.gaussian_fit
        # Draw Gaussian contours
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        fitted = gaussian_2d(
            (x_grid, y_grid), gf.amplitude, gf.center_x, gf.center_y,
            gf.sigma_x, gf.sigma_y, gf.theta, gf.offset
        ).reshape(rows, cols)
        # Compute contour levels and ensure they are sorted
        level1 = gf.offset + gf.amplitude * np.exp(-0.5)
        level2 = gf.offset + gf.amplitude * np.exp(-2)
        levels = sorted([level1, level2])
        try:
            ax.contour(fitted, levels=levels, colors=['cyan', 'lime'], linewidths=2)
        except ValueError:
            pass  # Skip if contour fails
        ax.plot(gf.center_x, gf.center_y, 'g*', markersize=12)
        ax.set_title(f'Gaussian Fit (R2={gf.r_squared:.3f})', fontsize=11, fontweight='bold')
    else:
        ax.set_title('Gaussian Fit (Failed)', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # --- Row 2, Col 2: DoG fit ---
    ax = fig.add_subplot(3, 3, 5)
    if geometry.dog_fit is not None:
        df = geometry.dog_fit
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        fitted_dog = dog_2d(
            (x_grid, y_grid), df.amp_exc, df.amp_inh, df.center_x, df.center_y,
            df.sigma_exc, df.sigma_inh, df.offset
        ).reshape(rows, cols)
        im = ax.imshow(fitted_dog, cmap='RdBu_r', origin='lower')
        ax.plot(df.center_x, df.center_y, 'g*', markersize=12)
        # Draw sigma circles
        exc_circle = Circle((df.center_x, df.center_y), df.sigma_exc, 
                            fill=False, edgecolor='lime', linewidth=2, label='Excitatory')
        inh_circle = Circle((df.center_x, df.center_y), df.sigma_inh,
                            fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Inhibitory')
        ax.add_patch(exc_circle)
        ax.add_patch(inh_circle)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'DoG Fit (R²={df.r_squared:.3f})', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, 'DoG Fit Failed', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('DoG Fit (Failed)', fontsize=11, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # --- Row 2, Col 3: ON/OFF comparison ---
    ax = fig.add_subplot(3, 3, 6)
    if geometry.on_off_fit is not None:
        oof = geometry.on_off_fit
        # Plot ON and OFF centers
        if oof.on_amplitude > 0:
            on_circle = Circle((oof.on_center_x, oof.on_center_y), 
                              max(oof.on_sigma_x, oof.on_sigma_y),
                              fill=False, edgecolor='red', linewidth=2, label=f'ON (σ={oof.on_sigma_x:.1f})')
            ax.add_patch(on_circle)
            ax.plot(oof.on_center_x, oof.on_center_y, 'r+', markersize=12, markeredgewidth=2)
        if oof.off_amplitude > 0:
            off_circle = Circle((oof.off_center_x, oof.off_center_y),
                               max(oof.off_sigma_x, oof.off_sigma_y),
                               fill=False, edgecolor='blue', linewidth=2, label=f'OFF (σ={oof.off_sigma_x:.1f})')
            ax.add_patch(off_circle)
            ax.plot(oof.off_center_x, oof.off_center_y, 'b+', markersize=12, markeredgewidth=2)
        
        im = ax.imshow(extreme_map_crop, cmap='RdBu_r', vmin=-ext_max, vmax=ext_max, 
                      origin='lower', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'ON/OFF (ON R²={oof.on_r_squared:.2f}, OFF R²={oof.off_r_squared:.2f})', 
                    fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'ON/OFF Fit Failed', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('ON/OFF Fit (Failed)', fontsize=11, fontweight='bold')
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(-0.5, rows-0.5)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # --- Row 3, Col 1: Temporal profile ---
    ax = fig.add_subplot(3, 3, 7)
    if 0 <= center_row < rows and 0 <= center_col < cols:
        center_signal = sta_data[:, center_row, center_col]
        frames = np.arange(n_frames)
        ax.plot(frames, center_signal, 'b-', linewidth=2, label='Center pixel')
        ax.axvspan(FRAME_RANGE[0], FRAME_RANGE[1], alpha=0.2, color='yellow', label='Analysis window')
        ax.axvspan(0, TEMPORAL_CONFIG.baseline_frames, alpha=0.2, color='gray', label='Baseline (0-10)')
        ax.axvline(x=peak_frame_idx, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('STA Value')
    ax.set_title('Temporal Profile at RF Center', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Row 3, Col 2-3: Summary statistics ---
    ax = fig.add_subplot(3, 3, 8)
    ax.axis('off')
    
    info_lines = [
        "RF GEOMETRY SUMMARY",
        "=" * 35,
        "",
        f"Center (x, y): ({geometry.center_col:.2f}, {geometry.center_row:.2f})",
        f"Size: {geometry.size_x:.2f} × {geometry.size_y:.2f} pixels",
        f"Area: {geometry.area:.2f} pixels²",
        f"Equiv. Diameter: {geometry.equivalent_diameter:.2f} pixels",
        f"Peak Frame: {geometry.peak_frame}",
        "",
        "PREPROCESSING (order)",
        "-" * 35,
        f"1. Baseline: frames 0-{TEMPORAL_CONFIG.baseline_frames}",
        f"2. Padding: {SPATIAL_CONFIG.padding} pixels",
        f"3. Blur sigma: {SPATIAL_CONFIG.gaussian_sigma}",
        "",
        "FIT CONSTRAINTS",
        "-" * 35,
        f"Center radius: {SPATIAL_CONFIG.center_fit_radius} px",
        f"Max sigma: {SPATIAL_CONFIG.max_sigma} px",
    ]
    
    ax.text(0.05, 0.95, '\n'.join(info_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    
    fit_lines = ["FIT RESULTS", "=" * 35, ""]
    
    if geometry.gaussian_fit is not None:
        gf = geometry.gaussian_fit
        fit_lines.extend([
            "2D GAUSSIAN FIT",
            f"  Center: ({gf.center_x:.2f}, {gf.center_y:.2f})",
            f"  σx, σy: {gf.sigma_x:.2f}, {gf.sigma_y:.2f}",
            f"  R²: {gf.r_squared:.4f}",
            "",
        ])
    
    if geometry.dog_fit is not None:
        df = geometry.dog_fit
        fit_lines.extend([
            "DoG (CENTER-SURROUND)",
            f"  σ_exc: {df.sigma_exc:.2f}, σ_inh: {df.sigma_inh:.2f}",
            f"  A_exc: {df.amp_exc:.3f}, A_inh: {df.amp_inh:.3f}",
            f"  R²: {df.r_squared:.4f}",
            "",
        ])
    
    if geometry.on_off_fit is not None:
        oof = geometry.on_off_fit
        fit_lines.extend([
            "ON/OFF MODEL",
            f"  ON: ({oof.on_center_x:.1f}, {oof.on_center_y:.1f}), σ={oof.on_sigma_x:.2f}",
            f"  OFF: ({oof.off_center_x:.1f}, {oof.off_center_y:.1f}), σ={oof.off_sigma_x:.2f}",
        ])
    
    ax.text(0.05, 0.95, '\n'.join(fit_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    fig.suptitle(f'{unit_id} - RF Analysis (Gaussian + DoG Fitting)\n{STA_FEATURE_NAME}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {output_path.name}")


def plot_rf_summary(
    all_geometries: Dict[str, RFGeometry],
    output_path: Path,
) -> None:
    """
    Create summary plot comparing RF geometries across all units.
    """
    if not all_geometries:
        print("No geometries to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    unit_ids = list(all_geometries.keys())
    centers_row = [g.center_row for g in all_geometries.values()]
    centers_col = [g.center_col for g in all_geometries.values()]
    diameters = [g.equivalent_diameter for g in all_geometries.values()]
    
    # Gaussian fit data
    gauss_sigmas = []
    gauss_r2 = []
    for g in all_geometries.values():
        if g.gaussian_fit is not None:
            gauss_sigmas.append((g.gaussian_fit.sigma_x + g.gaussian_fit.sigma_y) / 2)
            gauss_r2.append(g.gaussian_fit.r_squared)
    
    # DoG fit data
    dog_exc_sigmas = []
    dog_inh_sigmas = []
    dog_r2 = []
    for g in all_geometries.values():
        if g.dog_fit is not None:
            dog_exc_sigmas.append(g.dog_fit.sigma_exc)
            dog_inh_sigmas.append(g.dog_fit.sigma_inh)
            dog_r2.append(g.dog_fit.r_squared)
    
    # --- Top-left: RF center locations ---
    ax = axes[0, 0]
    scatter = ax.scatter(centers_col, centers_row, c=diameters, cmap='viridis', 
                         s=80, alpha=0.7, edgecolors='black')
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.set_title('RF Center Locations', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Equivalent Diameter')
    ax.grid(True, alpha=0.3)
    
    # --- Top-center: Diameter distribution ---
    ax = axes[0, 1]
    valid_diameters = [d for d in diameters if d > 0]
    if valid_diameters:
        ax.hist(valid_diameters, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(valid_diameters), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(valid_diameters):.2f}')
    ax.set_xlabel('Equivalent Diameter (pixels)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('RF Size Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Top-right: Gaussian fit quality ---
    ax = axes[0, 2]
    if gauss_r2:
        ax.hist(gauss_r2, bins=15, color='green', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(gauss_r2), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean R²: {np.mean(gauss_r2):.3f}')
    ax.set_xlabel('R² (Gaussian Fit)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Gaussian Fit Quality', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Bottom-left: Gaussian sigma distribution ---
    ax = axes[1, 0]
    if gauss_sigmas:
        ax.hist(gauss_sigmas, bins=15, color='orange', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(gauss_sigmas), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean σ: {np.mean(gauss_sigmas):.2f}')
    ax.set_xlabel('Gaussian σ (pixels)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Gaussian Sigma Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- Bottom-center: DoG sigma comparison ---
    ax = axes[1, 1]
    if dog_exc_sigmas and dog_inh_sigmas:
        ax.scatter(dog_exc_sigmas, dog_inh_sigmas, c=dog_r2, cmap='RdYlGn', 
                   s=60, alpha=0.7, edgecolors='black')
        ax.plot([0, max(dog_inh_sigmas)], [0, max(dog_inh_sigmas)], 'k--', alpha=0.3)
        ax.set_xlabel('σ_excitatory (pixels)', fontsize=11)
        ax.set_ylabel('σ_inhibitory (pixels)', fontsize=11)
        ax.set_title('DoG Center-Surround Structure', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('R²')
    ax.grid(True, alpha=0.3)
    
    # --- Bottom-right: Summary statistics ---
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_lines = [
        "ANALYSIS SUMMARY",
        "=" * 40,
        "",
        f"Total units analyzed: {len(all_geometries)}",
        "",
        "RF Size (Equivalent Diameter):",
        f"  Mean: {np.mean(diameters):.2f} ± {np.std(diameters):.2f} pixels",
        f"  Range: {np.min(diameters):.2f} - {np.max(diameters):.2f} pixels",
        "",
    ]
    
    if gauss_r2:
        summary_lines.extend([
            f"Gaussian Fits: {len(gauss_r2)}/{len(all_geometries)} successful",
            f"  Mean R²: {np.mean(gauss_r2):.3f}",
            f"  Mean σ: {np.mean(gauss_sigmas):.2f} pixels",
            "",
        ])
    
    if dog_r2:
        summary_lines.extend([
            f"DoG Fits: {len(dog_r2)}/{len(all_geometries)} successful",
            f"  Mean R²: {np.mean(dog_r2):.3f}",
            f"  Mean σ_exc: {np.mean(dog_exc_sigmas):.2f} pixels",
            f"  Mean σ_inh: {np.mean(dog_inh_sigmas):.2f} pixels",
        ])
    
    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle(f'RF Geometry Summary (Gaussian + DoG Fitting)\n{STA_FEATURE_NAME}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved summary: {output_path.name}")


def plot_all_rf_montage(
    sta_data: Dict[str, np.ndarray],
    geometries: Dict[str, RFGeometry],
    output_path: Path,
    max_units: int = 20,
) -> None:
    """
    Create a montage showing all unit RF maps with fit overlays.
    """
    unit_ids = list(geometries.keys())[:max_units]
    n_units = len(unit_ids)
    
    if n_units == 0:
        print("No units to plot in montage")
        return
    
    n_cols = min(5, n_units)
    n_rows = (n_units + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, unit_id in enumerate(unit_ids):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if unit_id not in sta_data:
            ax.axis('off')
            continue
        
        sta = sta_data[unit_id]
        geom = geometries[unit_id]
        
        # Use peak frame
        peak_frame_idx = geom.peak_frame if geom.peak_frame is not None else sta.shape[0] // 2
        frame = sta[peak_frame_idx]
        
        vmin, vmax = np.nanpercentile(frame, [1, 99])
        vmax_sym = max(abs(vmin), abs(vmax))
        im = ax.imshow(frame, cmap='RdBu_r', vmin=-vmax_sym, vmax=vmax_sym, origin='lower')
        
        # Mark center
        ax.plot(geom.center_col, geom.center_row, 'g+', markersize=10, markeredgewidth=2)
        
        # Draw Gaussian fit if available
        if geom.gaussian_fit is not None and geom.gaussian_fit.r_squared > 0.3:
            gf = geom.gaussian_fit
            ellipse = Ellipse(
                (gf.center_x, gf.center_y),
                width=2*gf.sigma_x, height=2*gf.sigma_y,
                angle=np.degrees(gf.theta),
                fill=False, edgecolor='lime', linewidth=1.5,
            )
            ax.add_patch(ellipse)
        
        r2_str = ""
        if geom.gaussian_fit is not None:
            r2_str = f"R²={geom.gaussian_fit.r_squared:.2f}"
        
        ax.set_title(f'{unit_id}\nØ={geom.equivalent_diameter:.1f} {r2_str}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide empty subplots
    for idx in range(n_units, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'RF Maps Montage (Frame Range: {FRAME_RANGE}, Gaussian Fit Overlay)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved montage: {output_path.name}")


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    """
    Main workflow for RF geometry extraction with Gaussian and DoG fitting.
    """
    print("=" * 70)
    print("RF-STA Receptive Field Measurement")
    print("Gaussian + DoG Fitting Analysis")
    print("=" * 70)
    print(f"Input file: {HDF5_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Feature: {STA_FEATURE_NAME}")
    print(f"\nPreprocessing (order: baseline -> padding -> blur):")
    print(f"  1. Baseline subtraction: frames 0-{TEMPORAL_CONFIG.baseline_frames}")
    print(f"  2. Padding: {SPATIAL_CONFIG.padding} pixels (with baseline mean)")
    print(f"  3. Gaussian blur sigma: {SPATIAL_CONFIG.gaussian_sigma}")
    print(f"\nFit Constraints:")
    print(f"  Center fit radius: {SPATIAL_CONFIG.center_fit_radius} pixels")
    print(f"  Max RF diameter: {SPATIAL_CONFIG.max_rf_diameter} pixels (max sigma: {SPATIAL_CONFIG.max_sigma})")
    print(f"\nAnalysis:")
    print(f"  Frame range: {FRAME_RANGE}")
    print(f"  Threshold: {THRESHOLD_FRACTION}")
    print(f"  Center detection: Extreme absolute value method")
    print(f"  Fitting: 2D Gaussian, DoG (center-surround), ON/OFF")
    
    if not HDF5_PATH.exists():
        print(f"Error: HDF5 file not found: {HDF5_PATH}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load STA data
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Loading STA data")
    print("-" * 70)
    
    sta_data = load_sta_data(HDF5_PATH)
    
    if not sta_data:
        print("Error: No STA data found in HDF5 file")
        return
    
    print(f"\nLoaded STA data for {len(sta_data)} units")
    
    # =========================================================================
    # Step 2: Extract RF geometry for each unit
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Extracting RF geometry with Gaussian + DoG fitting")
    print("-" * 70)
    
    all_geometries = {}
    
    for unit_id, sta in sta_data.items():
        print(f"\nProcessing {unit_id}...")
        
        try:
            geometry = extract_rf_geometry(
                sta,
                frame_range=FRAME_RANGE,
                threshold_fraction=THRESHOLD_FRACTION,
            )
            all_geometries[unit_id] = geometry
            
            print(f"  Center: ({geometry.center_col:.1f}, {geometry.center_row:.1f})")
            print(f"  Size: {geometry.size_x:.1f} × {geometry.size_y:.1f} pixels")
            print(f"  Equivalent Diameter: {geometry.equivalent_diameter:.1f} pixels")
            
            if geometry.gaussian_fit is not None:
                gf = geometry.gaussian_fit
                print(f"  Gaussian: sigma=({gf.sigma_x:.2f}, {gf.sigma_y:.2f}), R2={gf.r_squared:.3f}")
            
            if geometry.dog_fit is not None:
                df = geometry.dog_fit
                print(f"  DoG: sigma_exc={df.sigma_exc:.2f}, sigma_inh={df.sigma_inh:.2f}, R2={df.r_squared:.3f}")
            
        except Exception as e:
            print(f"  Error processing {unit_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[Summary] Extracted geometry for {len(all_geometries)}/{len(sta_data)} units")
    
    # =========================================================================
    # Step 3: Generate visualization plots
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 3: Generating visualization plots")
    print("-" * 70)
    
    # Individual unit plots
    print("\nGenerating individual unit plots...")
    for unit_id, geometry in all_geometries.items():
        if unit_id in sta_data:
            output_path = OUTPUT_DIR / f'{unit_id}_rf_geometry.png'
            plot_unit_rf_geometry(unit_id, sta_data[unit_id], geometry, output_path)
    
    # Summary plot
    print("\nGenerating summary plot...")
    summary_path = OUTPUT_DIR / 'rf_geometry_summary.png'
    plot_rf_summary(all_geometries, summary_path)
    
    # Montage plot
    print("\nGenerating montage plot...")
    montage_path = OUTPUT_DIR / 'rf_montage.png'
    plot_all_rf_montage(sta_data, all_geometries, montage_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Units processed: {len(all_geometries)}")
    print(f"\nPreprocessing applied (order):")
    print(f"  1. Baseline subtraction: frames 0-{TEMPORAL_CONFIG.baseline_frames}")
    print(f"  2. Padding: {SPATIAL_CONFIG.padding} pixels")
    print(f"  3. Gaussian blur: sigma={SPATIAL_CONFIG.gaussian_sigma}")
    print(f"\nFit constraints:")
    print(f"  - Center radius: {SPATIAL_CONFIG.center_fit_radius} pixels")
    print(f"  - Max sigma: {SPATIAL_CONFIG.max_sigma} pixels (diameter: {SPATIAL_CONFIG.max_rf_diameter})")
    print(f"\nResults directory: {OUTPUT_DIR}")
    
    # Print summary statistics
    if all_geometries:
        diameters = [g.equivalent_diameter for g in all_geometries.values() if g.equivalent_diameter > 0]
        gauss_r2 = [g.gaussian_fit.r_squared for g in all_geometries.values() 
                    if g.gaussian_fit is not None]
        dog_r2 = [g.dog_fit.r_squared for g in all_geometries.values() 
                  if g.dog_fit is not None]
        
        print(f"\nRF Size Statistics:")
        print(f"  Mean Diameter: {np.mean(diameters):.2f} ± {np.std(diameters):.2f} pixels")
        print(f"  Range: {np.min(diameters):.2f} - {np.max(diameters):.2f} pixels")
        
        if gauss_r2:
            print(f"\nGaussian Fit Statistics:")
            print(f"  Successful fits: {len(gauss_r2)}/{len(all_geometries)}")
            print(f"  Mean R²: {np.mean(gauss_r2):.4f}")
        
        if dog_r2:
            print(f"\nDoG Fit Statistics:")
            print(f"  Successful fits: {len(dog_r2)}/{len(all_geometries)}")
            print(f"  Mean R²: {np.mean(dog_r2):.4f}")


if __name__ == "__main__":
    main()
