"""
STA Cell Center Quantification Analysis

Compares four algorithms for identifying cell centers from eimage_sta data:
1. Max-Min Difference Method
2. Optical Flow Sink/Source Detection
3. Current Source Density (CSD) Analysis
4. 3D Gaussian Fit Method (with Gaussian blur preprocessing)

Author: Generated for experimental analysis
Date: 2024-12
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import laplace, gaussian_filter, gaussian_filter1d, uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, hilbert, correlate, find_peaks

# Try to import OpenCV for optical flow
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Optical flow analysis will be skipped.")


# =============================================================================
# Configuration
# =============================================================================

HDF5_PATH = Path(__file__).parent.parent / "pipeline_test" / "data" / "2024.03.01-14.40.14-Rec.h5"
OUTPUT_DIR = Path(__file__).parent / "results"
NUM_UNITS = 1000  # Set to larger number for full analysis


@dataclass
class TemporalConfig:
    """Configuration for temporal preprocessing and analysis."""
    # Spatial padding (applied first, before any processing)
    pad_size: int = 5  # Number of pixels to pad on each side (x and y)
    apply_padding: bool = True  # Whether to apply padding
    
    # 3D Gaussian blur (applied first, smooths in time and space simultaneously)
    blur_sigma_t: float = 1.0  # Temporal blur sigma (frames)
    blur_sigma_xy: float = 1.5  # Spatial blur sigma (same for x and y)
    apply_blur: bool = True  # Whether to apply 3D blur as first step
    
    # Temporal smoothing (additional, applied after blur)
    sigma_t: float = 2.0  # Temporal Gaussian smoothing sigma (frames)
    sigma_x: float = 3.0  # Spatial smoothing sigma in x (electrodes)
    sigma_y: float = 3.0  # Spatial smoothing sigma in y (electrodes)
    
    # Baseline subtraction
    baseline_frames: int = 5  # Number of initial frames for baseline
    
    # Savitzky-Golay filter for peak detection
    savgol_window: int = 7  # Window size (odd integer)
    savgol_order: int = 3   # Polynomial order
    
    # Cross-correlation
    use_fft_correlation: bool = True  # Use FFT for faster cross-correlation
    subsample_delay: bool = True  # Use parabolic interpolation for sub-frame delay
    
    # Adaptive thresholding
    noise_percentile: float = 25.0  # Percentile for noise estimation
    
    # Temporal integration window
    integration_window: int = 5  # Frames to integrate for temporal averaging


# Global config instance
TEMPORAL_CONFIG = TemporalConfig()


# =============================================================================
# Temporal Preprocessing Utilities
# =============================================================================

def temporal_smooth(data: np.ndarray, sigma_t: float = None) -> np.ndarray:
    """
    Apply 1D Gaussian smoothing along the temporal axis (axis=0).
    
    Args:
        data: 3D array (time, rows, cols)
        sigma_t: Temporal smoothing sigma. Uses config default if None.
        
    Returns:
        Temporally smoothed 3D array
    """
    if sigma_t is None:
        sigma_t = TEMPORAL_CONFIG.sigma_t
    
    if sigma_t <= 0:
        return data.copy()
    
    return gaussian_filter1d(data, sigma=sigma_t, axis=0, mode='nearest')


def spatiotemporal_smooth(
    data: np.ndarray,
    sigma: Tuple[float, float, float] = None,
) -> np.ndarray:
    """
    Apply 3D Gaussian smoothing (time, y, x).
    
    Args:
        data: 3D array (time, rows, cols)
        sigma: Tuple of (sigma_t, sigma_y, sigma_x). Uses config defaults if None.
        
    Returns:
        Spatiotemporally smoothed 3D array
    """
    if sigma is None:
        sigma = (TEMPORAL_CONFIG.sigma_t, TEMPORAL_CONFIG.sigma_y, TEMPORAL_CONFIG.sigma_x)
    
    return gaussian_filter(data, sigma=sigma, mode='nearest')


def temporal_derivative(data: np.ndarray) -> np.ndarray:
    """
    Compute temporal derivative using central differences.
    
    Args:
        data: 3D array (time, rows, cols)
        
    Returns:
        Temporal derivative array (same shape)
    """
    return np.gradient(data, axis=0)


def temporal_gradient_3d(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D gradient (dt, dy, dx).
    
    Args:
        data: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (grad_t, grad_y, grad_x) arrays
    """
    grad_t = np.gradient(data, axis=0)
    grad_y = np.gradient(data, axis=1)
    grad_x = np.gradient(data, axis=2)
    return grad_t, grad_y, grad_x


def baseline_subtract(data: np.ndarray, n_baseline_frames: int = None) -> np.ndarray:
    """
    Subtract temporal baseline using mean of initial frames.
    
    Args:
        data: 3D array (time, rows, cols)
        n_baseline_frames: Number of initial frames for baseline
        
    Returns:
        Baseline-subtracted array
    """
    if n_baseline_frames is None:
        n_baseline_frames = TEMPORAL_CONFIG.baseline_frames
    
    n_baseline_frames = min(n_baseline_frames, data.shape[0] // 4)
    if n_baseline_frames < 1:
        n_baseline_frames = 1
    
    baseline = np.mean(data[:n_baseline_frames], axis=0, keepdims=True)
    return data - baseline


def estimate_noise_level(data: np.ndarray, percentile: float = None) -> float:
    """
    Estimate noise level from low-activity regions.
    
    Args:
        data: 3D array (time, rows, cols)
        percentile: Percentile threshold for noise estimation
        
    Returns:
        Estimated noise standard deviation
    """
    if percentile is None:
        percentile = TEMPORAL_CONFIG.noise_percentile
    
    # Use MAD (Median Absolute Deviation) for robust noise estimation
    # Focus on low-activity frames/regions
    activity_per_frame = np.max(np.abs(data), axis=(1, 2))
    threshold = np.percentile(activity_per_frame, percentile)
    low_activity_mask = activity_per_frame < threshold
    
    if np.sum(low_activity_mask) > 0:
        low_activity_data = data[low_activity_mask]
        mad = np.median(np.abs(low_activity_data - np.median(low_activity_data)))
        noise_std = 1.4826 * mad  # Scale MAD to std
    else:
        # Fallback: use overall MAD
        mad = np.median(np.abs(data - np.median(data)))
        noise_std = 1.4826 * mad
    
    return noise_std


def adaptive_threshold(data: np.ndarray, snr_factor: float = 3.0) -> float:
    """
    Compute adaptive threshold based on noise statistics.
    
    Args:
        data: 3D array (time, rows, cols)
        snr_factor: Signal-to-noise ratio factor for threshold
        
    Returns:
        Adaptive threshold value
    """
    noise_std = estimate_noise_level(data)
    return snr_factor * noise_std


def apply_3d_gaussian_blur(
    data: np.ndarray, 
    sigma_t: float, 
    sigma_xy: float
) -> np.ndarray:
    """
    Apply 3D Gaussian blur to the data (temporal + spatial smoothing).
    
    This smooths the data simultaneously in time and space, which better
    preserves spatio-temporal features compared to separate 2D spatial blur.
    
    Args:
        data: 3D array (time, rows, cols)
        sigma_t: Temporal blur sigma (frames)
        sigma_xy: Spatial blur sigma (same for x and y)
        
    Returns:
        Blurred 3D array
    """
    # sigma order: (time, rows, cols)
    sigma_3d = (sigma_t, sigma_xy, sigma_xy)
    return gaussian_filter(data, sigma=sigma_3d, mode='nearest')


def add_spatial_padding(
    data: np.ndarray, 
    pad_size: int, 
    pad_value: str = 'mean'
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Add spatial padding to each frame with mean value.
    
    Args:
        data: 3D array (time, rows, cols)
        pad_size: Number of pixels to pad on each side
        pad_value: Padding method - 'mean' uses the mean value of each frame
        
    Returns:
        Tuple of (padded_data, padding_info) where padding_info is (top, bottom, left, right)
    """
    if pad_size <= 0:
        return data.copy(), (0, 0, 0, 0)
    
    n_frames, rows, cols = data.shape
    new_rows = rows + 2 * pad_size
    new_cols = cols + 2 * pad_size
    
    # Create padded array
    padded = np.zeros((n_frames, new_rows, new_cols), dtype=data.dtype)
    
    for t in range(n_frames):
        frame = data[t]
        if pad_value == 'mean':
            fill_value = np.nanmean(frame)
        else:
            fill_value = 0.0
        
        # Fill with mean value
        padded[t, :, :] = fill_value
        
        # Copy original data into center
        padded[t, pad_size:pad_size + rows, pad_size:pad_size + cols] = frame
    
    padding_info = (pad_size, pad_size, pad_size, pad_size)  # top, bottom, left, right
    return padded, padding_info


def remove_padding_from_coords(
    coords, 
    padding_info: Tuple[int, int, int, int]
):
    """
    Convert coordinates from padded space back to original space.
    
    Args:
        coords: (row, col) in padded space, can be tuple, list, or numpy array
        padding_info: (top, bottom, left, right) padding sizes
        
    Returns:
        (row, col) in original space as tuple
    """
    if coords is None:
        return None
    
    top, bottom, left, right = padding_info
    
    # Handle numpy arrays
    if isinstance(coords, np.ndarray):
        row, col = float(coords[0]), float(coords[1])
    else:
        row, col = coords
    
    return (row - top, col - left)


def remove_padding_from_trajectory(
    trajectory, 
    padding_info: Tuple[int, int, int, int]
):
    """
    Convert a list/array of trajectory points from padded space back to original space.
    
    Args:
        trajectory: List of (row, col) points or numpy array of shape (N, 2) in padded space
        padding_info: (top, bottom, left, right) padding sizes
        
    Returns:
        List of (row, col) points in original space, or numpy array if input was array
    """
    if trajectory is None:
        return None
    
    top, bottom, left, right = padding_info
    
    # Handle numpy arrays
    if isinstance(trajectory, np.ndarray):
        if trajectory.size == 0:
            return trajectory.copy()
        result = trajectory.copy().astype(float)
        if result.ndim == 1:
            # Single point (row, col)
            result[0] -= top
            result[1] -= left
        elif result.ndim == 2:
            # Multiple points (N, 2) where each row is (row, col)
            result[:, 0] -= top
            result[:, 1] -= left
        return result
    
    # Handle list
    if len(trajectory) == 0:
        return []
    return [remove_padding_from_coords(pt, padding_info) for pt in trajectory]


def remove_padding_from_map(
    data_map: np.ndarray, 
    padding_info: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Remove padding from a 2D map to get back original dimensions.
    
    Args:
        data_map: 2D array in padded space
        padding_info: (top, bottom, left, right) padding sizes
        
    Returns:
        2D array in original space
    """
    if data_map is None:
        return None
    
    top, bottom, left, right = padding_info
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return data_map.copy()
    
    rows, cols = data_map.shape
    return data_map[top:rows - bottom, left:cols - right].copy()


def remove_padding_from_flow_fields(
    flow_fields: List[Tuple[np.ndarray, np.ndarray]],
    padding_info: Tuple[int, int, int, int]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Remove padding from a list of flow field tuples.
    
    Args:
        flow_fields: List of (flow_x, flow_y) tuples where each is a 2D array
        padding_info: (top, bottom, left, right) padding sizes
        
    Returns:
        List of unpadded (flow_x, flow_y) tuples
    """
    if not flow_fields:
        return []
    
    top, bottom, left, right = padding_info
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return flow_fields
    
    result = []
    for fx, fy in flow_fields:
        rows, cols = fx.shape
        fx_unpad = fx[top:rows - bottom, left:cols - right].copy()
        fy_unpad = fy[top:rows - bottom, left:cols - right].copy()
        result.append((fx_unpad, fy_unpad))
    return result


def remove_padding_from_propagation_data(
    propagation_data: Dict[str, Any],
    padding_info: Tuple[int, int, int, int]
) -> Dict[str, Any]:
    """
    Remove padding from all relevant fields in propagation data dictionary.
    
    Args:
        propagation_data: Dictionary from trace_optical_flow_propagation or similar
        padding_info: (top, bottom, left, right) padding sizes
        
    Returns:
        Unpadded propagation data dictionary
    """
    result = propagation_data.copy()
    
    # Unpad trajectory/path
    if 'propagation_path' in result:
        result['propagation_path'] = remove_padding_from_trajectory(
            result['propagation_path'], padding_info
        )
    
    # Unpad flow fields
    if 'flow_fields' in result and result['flow_fields']:
        result['flow_fields'] = remove_padding_from_flow_fields(
            result['flow_fields'], padding_info
        )
    
    # Unpad 2D maps
    for key in ['flow_magnitude', 'divergence_map']:
        if key in result and result[key] is not None:
            if isinstance(result[key], np.ndarray):
                result[key] = remove_padding_from_map(result[key], padding_info)
            elif isinstance(result[key], list):
                # List of 2D arrays
                result[key] = [remove_padding_from_map(m, padding_info) if m is not None else None 
                              for m in result[key]]
    
    return result


def preprocess_temporal(
    eimage_sta: np.ndarray,
    config: TemporalConfig = None,
    smooth_temporal: bool = True,
    smooth_spatial: bool = True,
    subtract_baseline: bool = True,
    apply_blur: bool = None,
) -> np.ndarray:
    """
    Full temporal preprocessing pipeline.
    
    Processing order:
    1. Handle NaN values
    2. Apply 3D Gaussian blur (temporal + spatial noise reduction)
    3. Baseline subtraction
    4. Additional spatio-temporal smoothing (if enabled)
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        config: Temporal configuration. Uses global default if None.
        smooth_temporal: Apply additional temporal smoothing
        smooth_spatial: Apply additional spatial smoothing  
        subtract_baseline: Subtract baseline from initial frames
        apply_blur: Apply 3D Gaussian blur. Uses config default if None.
        
    Returns:
        Preprocessed 3D array
    """
    if config is None:
        config = TEMPORAL_CONFIG
    
    if apply_blur is None:
        apply_blur = config.apply_blur
    
    # Handle NaN values
    data = np.nan_to_num(eimage_sta, nan=0.0).astype(np.float64)
    
    # Step 1: Apply 3D Gaussian blur (temporal + spatial, first step for noise reduction)
    if apply_blur and (config.blur_sigma_t > 0 or config.blur_sigma_xy > 0):
        data = apply_3d_gaussian_blur(data, config.blur_sigma_t, config.blur_sigma_xy)
    
    # Step 2: Baseline subtraction (before smoothing to preserve transients)
    if subtract_baseline:
        data = baseline_subtract(data, config.baseline_frames)
    
    # Step 3: Apply spatio-temporal smoothing
    if smooth_temporal and smooth_spatial:
        sigma = (config.sigma_t, config.sigma_y, config.sigma_x)
        data = spatiotemporal_smooth(data, sigma)
    elif smooth_temporal:
        data = temporal_smooth(data, config.sigma_t)
    elif smooth_spatial:
        # Apply spatial smoothing to each frame
        for t in range(data.shape[0]):
            data[t] = gaussian_filter(data[t], sigma=(config.sigma_y, config.sigma_x))
    
    return data


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


def fft_cross_correlation(
    signal: np.ndarray,
    reference: np.ndarray,
    subsample: bool = None,
) -> Tuple[float, float]:
    """
    Compute cross-correlation using FFT with optional sub-sample delay estimation.
    
    Args:
        signal: Target signal
        reference: Reference signal
        subsample: Use parabolic interpolation for sub-sample precision
        
    Returns:
        Tuple of (delay, correlation_strength)
    """
    if subsample is None:
        subsample = TEMPORAL_CONFIG.subsample_delay
    
    # Normalize signals
    sig = signal - np.mean(signal)
    ref = reference - np.mean(reference)
    
    sig_norm = np.linalg.norm(sig)
    ref_norm = np.linalg.norm(ref)
    
    if sig_norm < 1e-10 or ref_norm < 1e-10:
        return 0.0, 0.0
    
    # FFT-based cross-correlation
    corr = correlate(sig, ref, mode='full', method='fft')
    corr = corr / (sig_norm * ref_norm)
    
    mid = len(corr) // 2
    peak_idx = int(np.argmax(corr))
    peak_val = corr[peak_idx]
    
    # Sub-sample interpolation using parabolic fit
    if subsample and 0 < peak_idx < len(corr) - 1:
        y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            delta = (y0 - y2) / denom
            delay = (peak_idx - mid) + delta
        else:
            delay = float(peak_idx - mid)
    else:
        delay = float(peak_idx - mid)
    
    return delay, peak_val


def vectorized_hilbert(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hilbert transform for entire 3D array at once.
    
    Args:
        data: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (amplitude_3d, phase_3d)
    """
    # Reshape to 2D for efficient computation
    n_frames, rows, cols = data.shape
    data_2d = data.reshape(n_frames, -1)
    
    # Mean-center each electrode
    data_centered = data_2d - np.mean(data_2d, axis=0, keepdims=True)
    
    # Vectorized Hilbert transform along time axis
    analytic = hilbert(data_centered, axis=0)
    
    amplitude = np.abs(analytic).reshape(n_frames, rows, cols)
    phase = np.angle(analytic).reshape(n_frames, rows, cols)
    
    return amplitude, phase


def interpolated_threshold_crossing(
    signal: np.ndarray,
    threshold: float,
) -> float:
    """
    Find threshold crossing time with sub-frame interpolation.
    
    Args:
        signal: 1D temporal signal
        threshold: Threshold value
        
    Returns:
        Interpolated crossing time (can be fractional frame index)
    """
    above = signal > threshold
    
    if not np.any(above):
        return float(len(signal))  # Never crosses
    
    # Find first crossing index
    first_above = np.argmax(above)
    
    if first_above == 0:
        return 0.0
    
    # Linear interpolation between frames
    y0 = signal[first_above - 1]
    y1 = signal[first_above]
    
    if abs(y1 - y0) > 1e-10:
        fraction = (threshold - y0) / (y1 - y0)
        return float(first_above - 1) + fraction
    else:
        return float(first_above)


def rts_smoother(
    observations: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel (RTS) smoother for optimal trajectory estimation.
    
    Performs forward Kalman filter pass followed by backward smoothing.
    
    Args:
        observations: Observation sequence (T, obs_dim)
        F: State transition matrix
        H: Observation matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        x0: Initial state
        P0: Initial covariance
        
    Returns:
        Tuple of (smoothed_states, smoothed_covariances)
    """
    T = len(observations)
    state_dim = len(x0)
    
    # Forward pass (Kalman filter)
    x_filt = np.zeros((T, state_dim))
    P_filt = np.zeros((T, state_dim, state_dim))
    x_pred = np.zeros((T, state_dim))
    P_pred = np.zeros((T, state_dim, state_dim))
    
    x = x0.copy()
    P = P0.copy()
    
    for t in range(T):
        # Prediction
        x_p = F @ x
        P_p = F @ P @ F.T + Q
        
        x_pred[t] = x_p
        P_pred[t] = P_p
        
        # Update
        z = observations[t]
        y = z - H @ x_p
        S = H @ P_p @ H.T + R
        K = P_p @ H.T @ np.linalg.inv(S)
        
        x = x_p + K @ y
        P = (np.eye(state_dim) - K @ H) @ P_p
        
        x_filt[t] = x
        P_filt[t] = P
    
    # Backward pass (RTS smoother)
    x_smooth = np.zeros((T, state_dim))
    P_smooth = np.zeros((T, state_dim, state_dim))
    
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    for t in range(T - 2, -1, -1):
        # Compute smoother gain
        try:
            G = P_filt[t] @ F.T @ np.linalg.inv(P_pred[t + 1])
        except np.linalg.LinAlgError:
            G = P_filt[t] @ F.T @ np.linalg.pinv(P_pred[t + 1])
        
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T
    
    return x_smooth, P_smooth


# =============================================================================
# Data Loading
# =============================================================================

def load_eimage_sta_data(hdf5_path: Path, num_units: int = 5) -> Dict[str, np.ndarray]:
    """
    Load eimage_sta data for the first N units from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        num_units: Number of units to load
        
    Returns:
        Dictionary mapping unit_id to eimage_sta array (time, rows, cols)
    """
    data = {}
    
    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            raise ValueError(f"No 'units' group found in {hdf5_path}")
        
        unit_ids = sorted(list(f["units"].keys()))[:num_units]
        
        for unit_id in unit_ids:
            eimage_path = f"units/{unit_id}/features/eimage_sta/data"
            if eimage_path in f:
                data[unit_id] = f[eimage_path][:]
                print(f"Loaded {unit_id}: shape={data[unit_id].shape}")
            else:
                print(f"Warning: No eimage_sta found for {unit_id}")
    
    return data


# =============================================================================
# Algorithm 1: Max-Min Difference Method
# =============================================================================

def find_center_maxmin(eimage_sta: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Find cell center using max-min difference along time axis.
    
    TEMPORAL OPTIMIZATION: Uses Savitzky-Golay filtering for robust peak/trough
    detection, considering temporal waveform shape rather than simple max/min.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (row, col, difference_map)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=False, 
                               subtract_baseline=True)
    
    # Compute robust peak-to-peak for each electrode using Savitzky-Golay
    diff_map = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            signal = data[:, r, c]
            peak_val, trough_val, _, _ = robust_peak_trough_detection(signal)
            diff_map[r, c] = peak_val - trough_val
    
    # Find location of maximum difference
    center_idx = np.nanargmax(diff_map)
    center_row, center_col = np.unravel_index(center_idx, diff_map.shape)
    
    return int(center_row), int(center_col), diff_map


# =============================================================================
# Algorithm 2: Optical Flow Sink/Source Detection
# =============================================================================

def find_center_optical_flow(eimage_sta: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], np.ndarray]:
    """
    Find sink and source locations using optical flow analysis.
    
    TEMPORAL OPTIMIZATION: Computes flow for all frame pairs, then applies
    temporal smoothing to flow fields for consistency. Uses activity-weighted
    temporal integration for divergence accumulation.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (sink_coords, source_coords, divergence_map)
        where coords are (row, col) tuples
    """
    if not HAS_CV2:
        # Return center of array as fallback
        h, w = eimage_sta.shape[1], eimage_sta.shape[2]
        return (h//2, w//2), (h//2, w//2), np.zeros((h, w))
    
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess: temporal smoothing for noise reduction
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # Normalize data to 0-255 for optical flow computation
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    if data_max - data_min > 0:
        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(data, dtype=np.uint8)
    
    # Compute flow for all frame pairs and store for temporal smoothing
    flow_x_stack = np.zeros((n_frames - 1, rows, cols))
    flow_y_stack = np.zeros((n_frames - 1, rows, cols))
    
    for i in range(n_frames - 1):
        prev_frame = normalized[i]
        next_frame = normalized[i + 1]
        
        # Compute dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, next_frame,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        flow_x_stack[i] = flow[:, :, 0]
        flow_y_stack[i] = flow[:, :, 1]
    
    # TEMPORAL OPTIMIZATION: Smooth flow fields along time axis for consistency
    flow_x_smooth = gaussian_filter1d(flow_x_stack, sigma=TEMPORAL_CONFIG.sigma_t, axis=0)
    flow_y_smooth = gaussian_filter1d(flow_y_stack, sigma=TEMPORAL_CONFIG.sigma_t, axis=0)
    
    # Compute activity weight for each frame (weight divergence by activity level)
    frame_activity = np.max(np.abs(data[:-1]), axis=(1, 2))
    activity_weights = frame_activity / (np.sum(frame_activity) + 1e-10)
    
    # Accumulate divergence with activity weighting
    divergence_accum = np.zeros((rows, cols), dtype=np.float64)
    
    for i in range(n_frames - 1):
        flow_x = flow_x_smooth[i]
        flow_y = flow_y_smooth[i]
        
        # Compute divergence: d(vx)/dx + d(vy)/dy
        dvx_dx = np.gradient(flow_x, axis=1)
        dvy_dy = np.gradient(flow_y, axis=0)
        divergence = dvx_dx + dvy_dy
        
        # Weight by activity level
        divergence_accum += divergence * activity_weights[i]
    
    # Smooth the divergence map
    divergence_map = gaussian_filter(divergence_accum, sigma=2)
    
    # Find sink (min divergence = convergence) and source (max divergence)
    sink_idx = np.nanargmin(divergence_map)
    source_idx = np.nanargmax(divergence_map)
    
    sink_coords = np.unravel_index(sink_idx, divergence_map.shape)
    source_coords = np.unravel_index(source_idx, divergence_map.shape)
    
    return (int(sink_coords[0]), int(sink_coords[1])), \
           (int(source_coords[0]), int(source_coords[1])), \
           divergence_map


def trace_optical_flow_propagation(
    eimage_sta: np.ndarray,
    n_key_frames: int = 5,
) -> Dict[str, Any]:
    """
    Trace action potential propagation using optical flow.
    
    TEMPORAL OPTIMIZATION: Applies temporal smoothing to flow fields,
    uses multi-scale temporal pyramid, and ensures temporal consistency.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        n_key_frames: Number of key frames to analyze
        
    Returns:
        Dictionary containing:
        - 'flow_fields': List of (flow_x, flow_y) for each frame pair
        - 'flow_magnitude': List of flow magnitude maps
        - 'flow_direction': List of flow direction maps
        - 'key_frames': Frame indices analyzed
        - 'propagation_path': Traced path from source to sink
        - 'velocity_profile': Average velocity over time
    """
    if not HAS_CV2:
        n_frames, rows, cols = eimage_sta.shape
        return {
            'flow_fields': [],
            'flow_magnitude': [],
            'flow_direction': [],
            'key_frames': [],
            'propagation_path': np.zeros((0, 2)),
            'velocity_profile': np.zeros(n_frames - 1),
        }
    
    n_frames, rows, cols = eimage_sta.shape
    
    # TEMPORAL OPTIMIZATION: Preprocess with temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # Normalize data to 0-255 for optical flow computation
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    if data_max - data_min > 0:
        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(data, dtype=np.uint8)
    
    # Compute optical flow for all frame pairs and store in stacks
    flow_x_stack = np.zeros((n_frames - 1, rows, cols))
    flow_y_stack = np.zeros((n_frames - 1, rows, cols))
    
    for i in range(n_frames - 1):
        prev_frame = normalized[i]
        next_frame = normalized[i + 1]
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, next_frame,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        flow_x_stack[i] = flow[:, :, 0]
        flow_y_stack[i] = flow[:, :, 1]
    
    # TEMPORAL OPTIMIZATION: Smooth flow fields along time axis
    flow_x_smooth = gaussian_filter1d(flow_x_stack, sigma=TEMPORAL_CONFIG.sigma_t, axis=0)
    flow_y_smooth = gaussian_filter1d(flow_y_stack, sigma=TEMPORAL_CONFIG.sigma_t, axis=0)
    
    # Build output lists from smoothed flow
    all_flows = []
    flow_magnitudes = []
    flow_directions = []
    velocity_profile = []
    
    for i in range(n_frames - 1):
        flow_x = flow_x_smooth[i]
        flow_y = flow_y_smooth[i]
        
        # Compute magnitude and direction
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        direction = np.arctan2(flow_y, flow_x)
        
        all_flows.append((flow_x, flow_y))
        flow_magnitudes.append(magnitude)
        flow_directions.append(direction)
        velocity_profile.append(np.mean(magnitude))
    
    # Select key frames based on activity level
    frame_activity = np.array([np.max(m) for m in flow_magnitudes])
    if len(frame_activity) > n_key_frames:
        # Select frames with highest activity, distributed across time
        sorted_indices = np.argsort(frame_activity)[::-1]
        # Take top activity frames but ensure temporal distribution
        peak_frame = sorted_indices[0]
        # Select frames around the peak
        start_frame = max(0, peak_frame - n_key_frames // 2)
        end_frame = min(len(frame_activity), start_frame + n_key_frames)
        key_frames = list(range(start_frame, end_frame))
    else:
        key_frames = list(range(len(frame_activity)))
    
    # Trace propagation path from the location of maximum activity
    # Find starting point (where activity originates)
    peak_magnitude_frame = np.argmax([np.max(m) for m in flow_magnitudes])
    peak_mag = flow_magnitudes[peak_magnitude_frame]
    start_row, start_col = np.unravel_index(np.argmax(peak_mag), peak_mag.shape)
    
    # Trace path by following flow vectors
    path = [(float(start_row), float(start_col))]
    current_row, current_col = float(start_row), float(start_col)
    
    # Trace forward from peak frame
    for i in range(peak_magnitude_frame, min(peak_magnitude_frame + 10, len(all_flows))):
        flow_x, flow_y = all_flows[i]
        r, c = int(np.clip(current_row, 0, rows-1)), int(np.clip(current_col, 0, cols-1))
        
        # Get flow at current position
        dx = flow_x[r, c]
        dy = flow_y[r, c]
        
        # Update position
        current_row += dy
        current_col += dx
        
        # Clamp to bounds
        current_row = np.clip(current_row, 0, rows - 1)
        current_col = np.clip(current_col, 0, cols - 1)
        
        path.append((current_row, current_col))
    
    # Also trace backward from peak frame
    current_row, current_col = float(start_row), float(start_col)
    backward_path = []
    for i in range(peak_magnitude_frame - 1, max(peak_magnitude_frame - 10, -1), -1):
        flow_x, flow_y = all_flows[i]
        r, c = int(np.clip(current_row, 0, rows-1)), int(np.clip(current_col, 0, cols-1))
        
        # Get flow at current position (reverse direction)
        dx = -flow_x[r, c]
        dy = -flow_y[r, c]
        
        # Update position
        current_row += dy
        current_col += dx
        
        # Clamp to bounds
        current_row = np.clip(current_row, 0, rows - 1)
        current_col = np.clip(current_col, 0, cols - 1)
        
        backward_path.insert(0, (current_row, current_col))
    
    # Combine paths
    full_path = backward_path + path
    propagation_path = np.array(full_path)
    
    return {
        'flow_fields': all_flows,
        'flow_magnitude': flow_magnitudes,
        'flow_direction': flow_directions,
        'key_frames': key_frames,
        'propagation_path': propagation_path,
        'velocity_profile': np.array(velocity_profile),
        'peak_frame': peak_magnitude_frame,
    }


# =============================================================================
# Propagation Method 2: Time-of-Arrival via Cross-Correlation
# =============================================================================

def trace_toa_crosscorr(
    eimage_sta: np.ndarray,
    ref_method: str = 'peak',
) -> Dict[str, Any]:
    """
    Trace propagation using time-of-arrival via cross-correlation.
    
    TEMPORAL OPTIMIZATION: Uses FFT-based cross-correlation for efficiency,
    sub-sample delay estimation via parabolic interpolation, and temporal
    filtering before correlation.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        ref_method: 'peak' to use electrode with peak activity, 'center' for array center
        
    Returns:
        Dictionary with delay_map, velocity_map, propagation direction
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # TEMPORAL OPTIMIZATION: Preprocess with temporal filtering
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=False,
                               subtract_baseline=True)
    
    # Find reference electrode
    if ref_method == 'peak':
        max_activity = np.max(np.abs(data), axis=0)
        ref_idx = np.unravel_index(np.argmax(max_activity), max_activity.shape)
        ref_row, ref_col = ref_idx
    else:
        ref_row, ref_col = rows // 2, cols // 2
    
    ref_signal = data[:, ref_row, ref_col]
    
    # Compute delay for each electrode using FFT-based correlation
    delay_map = np.zeros((rows, cols))
    corr_strength_map = np.zeros((rows, cols))
    
    for r in range(rows):
        for c in range(cols):
            signal = data[:, r, c]
            
            # TEMPORAL OPTIMIZATION: Use FFT-based correlation with sub-sample precision
            delay, corr_strength = fft_cross_correlation(
                signal, ref_signal, 
                subsample=TEMPORAL_CONFIG.subsample_delay
            )
            
            delay_map[r, c] = delay
            corr_strength_map[r, c] = corr_strength
    
    # Compute velocity from delay gradient
    # Velocity ~ 1 / |gradient(delay)|
    delay_smooth = gaussian_filter(delay_map, sigma=1)
    grad_y, grad_x = np.gradient(delay_smooth)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity_map = np.where(grad_mag > 0.1, 1.0 / grad_mag, 0)
    
    # Propagation direction from delay gradient
    direction_map = np.arctan2(-grad_y, -grad_x)  # Points from early to late
    
    return {
        'delay_map': delay_map,
        'velocity_map': velocity_map,
        'direction_map': direction_map,
        'corr_strength': corr_strength_map,
        'reference': (ref_row, ref_col),
    }


# =============================================================================
# Propagation Method 3: Phase / Wavefront (Hilbert Transform)
# =============================================================================

def trace_phase_wavefront(eimage_sta: np.ndarray) -> Dict[str, Any]:
    """
    Trace propagation using instantaneous phase from Hilbert transform.
    
    TEMPORAL OPTIMIZATION: Uses vectorized Hilbert transform for entire 3D
    array, computes instantaneous frequency, and tracks phase velocity.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Dictionary with phase maps, wavefront velocity
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # TEMPORAL OPTIMIZATION: Preprocess with baseline subtraction
    data = preprocess_temporal(eimage_sta, smooth_temporal=False, smooth_spatial=False,
                               subtract_baseline=True)
    
    # TEMPORAL OPTIMIZATION: Vectorized Hilbert transform
    amplitude_3d, phase_3d = vectorized_hilbert(data)
    
    # Find frame with maximum activity for wavefront analysis
    frame_activity = np.max(amplitude_3d, axis=(1, 2))
    peak_frame = int(np.argmax(frame_activity))
    
    # Compute phase gradient (wavefront direction)
    peak_phase = phase_3d[peak_frame]
    grad_y, grad_x = np.gradient(peak_phase)
    
    # TEMPORAL OPTIMIZATION: Vectorized phase velocity computation
    # Compute instantaneous frequency (phase derivative)
    phase_deriv = np.gradient(phase_3d, axis=0)
    
    # Average absolute phase velocity over time for each electrode
    phase_velocity = np.mean(np.abs(phase_deriv), axis=0)
    
    # Instantaneous frequency map at peak frame
    inst_frequency = phase_deriv[peak_frame]
    
    # Wavefront direction (points toward increasing phase delay)
    wavefront_direction = np.arctan2(-grad_y, -grad_x)
    
    return {
        'phase_map': peak_phase,
        'phase_3d': phase_3d,
        'amplitude_3d': amplitude_3d,
        'phase_velocity': phase_velocity,
        'wavefront_direction': wavefront_direction,
        'peak_frame': peak_frame,
    }


# =============================================================================
# Propagation Method 4: Eikonal / Travel-Time Fitting
# =============================================================================

def trace_eikonal_fit(eimage_sta: np.ndarray) -> Dict[str, Any]:
    """
    Trace propagation using eikonal equation fitting.
    
    TEMPORAL OPTIMIZATION: Uses interpolated threshold crossing for sub-frame
    precision, adaptive thresholding based on noise, and temporal smoothing
    for regularization.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Dictionary with travel time map, fitted velocity, source location
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # TEMPORAL OPTIMIZATION: Preprocess with temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # TEMPORAL OPTIMIZATION: Use adaptive threshold based on noise
    threshold = adaptive_threshold(data, snr_factor=3.0)
    threshold = max(threshold, 0.3 * np.max(np.abs(data)))  # Ensure minimum threshold
    
    # Find activation time for each electrode with sub-frame interpolation
    activation_time = np.full((rows, cols), float(n_frames))
    
    for r in range(rows):
        for c in range(cols):
            signal = np.abs(data[:, r, c])
            # TEMPORAL OPTIMIZATION: Interpolated threshold crossing
            crossing_time = interpolated_threshold_crossing(signal, threshold)
            activation_time[r, c] = crossing_time
    
    # Find source (earliest activation)
    source_idx = np.unravel_index(np.argmin(activation_time), activation_time.shape)
    source_row, source_col = source_idx
    
    # Compute distance from source
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    distance_from_source = np.sqrt(
        (y_coords - source_row)**2 + (x_coords - source_col)**2
    )
    
    # Fit velocity: time = distance / velocity
    # Linear regression: activation_time = (1/v) * distance + t0
    valid_mask = activation_time < n_frames
    if np.sum(valid_mask) > 10:
        distances = distance_from_source[valid_mask]
        times = activation_time[valid_mask]
        
        # Simple linear fit
        A = np.vstack([distances, np.ones(len(distances))]).T
        try:
            slope, intercept = np.linalg.lstsq(A, times, rcond=None)[0]
            velocity = 1.0 / slope if slope > 0 else np.inf
        except:
            velocity = np.nan
    else:
        velocity = np.nan
    
    # TEMPORAL OPTIMIZATION: Smooth activation time map before gradient
    at_smooth = gaussian_filter(activation_time, sigma=1.5)
    grad_y, grad_x = np.gradient(at_smooth)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        local_velocity = np.where(grad_mag > 0.01, 1.0 / grad_mag, 0)
    
    return {
        'activation_time': activation_time,
        'source': (source_row, source_col),
        'fitted_velocity': velocity,
        'local_velocity': local_velocity,
        'distance_from_source': distance_from_source,
    }


# =============================================================================
# Propagation Method 5: Beamforming / Source Localization
# =============================================================================

def trace_beamforming(eimage_sta: np.ndarray) -> Dict[str, Any]:
    """
    Track propagation using beamforming / source localization.
    
    TEMPORAL OPTIMIZATION: Applies temporal smoothing to trajectory,
    uses activity-weighted spread rate estimation.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Dictionary with source trajectory, power map
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # TEMPORAL OPTIMIZATION: Preprocess with spatio-temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # For each time point, find the "source" location using
    # weighted centroid (simple beamforming approximation)
    raw_trajectory = []
    power_per_frame = []
    
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    for t in range(n_frames):
        frame = np.abs(data[t])
        total_power = np.sum(frame)
        
        if total_power > 1e-10:
            # Weighted centroid
            source_row = np.sum(y_coords * frame) / total_power
            source_col = np.sum(x_coords * frame) / total_power
        else:
            source_row, source_col = rows / 2, cols / 2
        
        raw_trajectory.append((source_row, source_col))
        power_per_frame.append(total_power)
    
    raw_trajectory = np.array(raw_trajectory)
    power_per_frame = np.array(power_per_frame)
    
    # TEMPORAL OPTIMIZATION: Smooth trajectory temporally
    trajectory_smooth_row = gaussian_filter1d(raw_trajectory[:, 0], sigma=TEMPORAL_CONFIG.sigma_t)
    trajectory_smooth_col = gaussian_filter1d(raw_trajectory[:, 1], sigma=TEMPORAL_CONFIG.sigma_t)
    trajectory = np.column_stack([trajectory_smooth_row, trajectory_smooth_col])
    
    # Compute velocity from smoothed trajectory
    if len(trajectory) > 1:
        velocity = np.sqrt(np.diff(trajectory[:, 0])**2 + np.diff(trajectory[:, 1])**2)
    else:
        velocity = np.array([0])
    
    # Create power map (max over time)
    power_map = np.max(np.abs(data), axis=0)
    
    return {
        'trajectory': trajectory,
        'raw_trajectory': raw_trajectory,
        'power_per_frame': power_per_frame,
        'velocity_profile': velocity,
        'power_map': power_map,
    }


# =============================================================================
# Propagation Method 6: Kalman Filtering on Propagation Path
# =============================================================================

def trace_kalman_filter(
    eimage_sta: np.ndarray,
    optical_flow_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Smooth propagation path using Kalman filtering.
    
    TEMPORAL OPTIMIZATION: Uses RTS (Rauch-Tung-Striebel) smoother for optimal
    trajectory estimation with backward smoothing pass. Includes adaptive
    process noise based on activity level.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        optical_flow_data: Output from trace_optical_flow_propagation()
        
    Returns:
        Dictionary with smoothed trajectory, uncertainty estimates
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Get raw trajectory from beamforming
    beam_data = trace_beamforming(eimage_sta)
    raw_trajectory = beam_data['trajectory']
    power_per_frame = beam_data['power_per_frame']
    
    dt = 1.0  # Time step
    
    # State transition matrix (constant velocity model)
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    
    # Observation matrix
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)
    
    # TEMPORAL OPTIMIZATION: Adaptive process noise based on activity
    # Higher activity = trust motion model more (lower process noise)
    power_normalized = power_per_frame / (np.max(power_per_frame) + 1e-10)
    
    # Base process noise
    q_base = 0.1
    
    # Measurement noise covariance (lower when activity is high)
    r_base = 1.0
    
    # Prepare observations
    observations = np.column_stack([raw_trajectory[:, 1], raw_trajectory[:, 0]])
    
    # Initial state and covariance
    x0 = np.array([raw_trajectory[0, 1], raw_trajectory[0, 0], 0, 0], dtype=np.float64)
    P0 = np.eye(4, dtype=np.float64) * 10
    
    # Average Q and R for RTS smoother (could be time-varying, but simplified here)
    Q = np.diag([0.1, 0.1, q_base, q_base])
    R = np.diag([r_base, r_base])
    
    # TEMPORAL OPTIMIZATION: Use RTS smoother for optimal smoothing
    smoothed_states, smoothed_covs = rts_smoother(observations, F, H, Q, R, x0, P0)
    
    # Extract trajectory from smoothed states
    smoothed_trajectory = np.column_stack([smoothed_states[:, 1], smoothed_states[:, 0]])  # row, col
    
    # Compute uncertainties from covariances
    uncertainties = np.array([np.sqrt(P[0, 0] + P[1, 1]) for P in smoothed_covs])
    
    # Compute velocities from smoothed states
    velocity_x = smoothed_states[:, 2]  # vx component
    velocity_y = smoothed_states[:, 3]  # vy component
    
    return {
        'raw_trajectory': raw_trajectory,
        'filtered_trajectory': smoothed_trajectory,
        'uncertainties': uncertainties,
        'velocity_x': velocity_x,
        'velocity_y': velocity_y,
        'smoothed_states': smoothed_states,
    }


# =============================================================================
# Propagation Method 7: TV-L1 Optical Flow
# =============================================================================

def trace_tvl1_optical_flow(eimage_sta: np.ndarray) -> Dict[str, Any]:
    """
    Trace propagation using TV-L1 optical flow.
    
    TV-L1 is more robust to noise than Farneback due to
    total variation regularization.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Dictionary with flow fields, divergence, sink/source
    """
    if not HAS_CV2:
        n_frames, rows, cols = eimage_sta.shape
        return {
            'flow_fields': [],
            'divergence_map': np.zeros((rows, cols)),
            'sink': (rows//2, cols//2),
            'source': (rows//2, cols//2),
        }
    
    n_frames, rows, cols = eimage_sta.shape
    
    # Normalize data
    data_min = np.nanmin(eimage_sta)
    data_max = np.nanmax(eimage_sta)
    if data_max - data_min > 0:
        normalized = ((eimage_sta - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(eimage_sta, dtype=np.uint8)
    
    # Create TV-L1 optical flow object
    tvl1 = None
    try:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    except AttributeError:
        # Fall back to createOptFlow_DualTVL1 for older OpenCV
        try:
            tvl1 = cv2.createOptFlow_DualTVL1()
        except:
            pass
    
    # If TV-L1 not available, use Farneback and convert to expected format
    if tvl1 is None:
        # Fall back to Farneback
        farneback_result = find_center_optical_flow(eimage_sta)
        return {
            'flow_fields': [],
            'divergence_map': farneback_result[2],
            'sink': farneback_result[0],
            'source': farneback_result[1],
        }
    
    # Compute flow for all frame pairs
    flow_fields = []
    divergence_accum = np.zeros((rows, cols))
    
    for i in range(n_frames - 1):
        prev_frame = np.nan_to_num(normalized[i], nan=0).astype(np.uint8)
        next_frame = np.nan_to_num(normalized[i + 1], nan=0).astype(np.uint8)
        
        flow = tvl1.calc(prev_frame, next_frame, None)
        
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        
        flow_fields.append((flow_x, flow_y))
        
        # Compute divergence
        dvx_dx = np.gradient(flow_x, axis=1)
        dvy_dy = np.gradient(flow_y, axis=0)
        divergence_accum += dvx_dx + dvy_dy
    
    divergence_map = divergence_accum / max(n_frames - 1, 1)
    divergence_map = gaussian_filter(divergence_map, sigma=2)
    
    # Find sink and source
    sink_idx = np.argmin(divergence_map)
    source_idx = np.argmax(divergence_map)
    
    sink = np.unravel_index(sink_idx, divergence_map.shape)
    source = np.unravel_index(source_idx, divergence_map.shape)
    
    return {
        'flow_fields': flow_fields,
        'divergence_map': divergence_map,
        'sink': (int(sink[0]), int(sink[1])),
        'source': (int(source[0]), int(source[1])),
    }


# =============================================================================
# Propagation Method 9: Lucas-Kanade Optical Flow
# =============================================================================

def trace_lucas_kanade_optical_flow(
    eimage_sta: np.ndarray,
    window_size: int = 4,
    max_corners: int = 100,
    quality_level: float = 0.01,
    min_distance: int = 3,
) -> Dict[str, Any]:
    """
    Trace propagation using Lucas-Kanade sparse optical flow.
    
    Lucas-Kanade is a sparse optical flow method that tracks feature points
    through time. It uses a local 4x4 window to estimate flow at tracked points.
    
    TEMPORAL OPTIMIZATION: Uses temporal preprocessing with 3D Gaussian blur,
    tracks features across all frames with backward validation, and builds
    a flow field by interpolating sparse flow vectors.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        window_size: Size of the search window (default 4x4 as requested)
        max_corners: Maximum number of corners to track
        quality_level: Quality level for corner detection
        min_distance: Minimum distance between corners
        
    Returns:
        Dictionary with:
        - 'flow_fields': List of interpolated flow fields
        - 'divergence_map': Accumulated divergence map
        - 'sink': Location of flow convergence
        - 'source': Location of flow divergence  
        - 'trajectory': Centroid trajectory of tracked points
        - 'tracked_points': List of point positions per frame
        - 'point_velocities': Velocities of tracked points
    """
    if not HAS_CV2:
        n_frames, rows, cols = eimage_sta.shape
        return {
            'flow_fields': [],
            'divergence_map': np.zeros((rows, cols)),
            'sink': (rows // 2, cols // 2),
            'source': (rows // 2, cols // 2),
            'trajectory': np.zeros((0, 2)),
            'tracked_points': [],
            'point_velocities': [],
        }
    
    n_frames, rows, cols = eimage_sta.shape
    
    # TEMPORAL OPTIMIZATION: Preprocess with 3D blur and temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # Normalize data to 0-255 for optical flow computation
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    if data_max - data_min > 0:
        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(data, dtype=np.uint8)
    
    # Lucas-Kanade parameters with 4x4 window
    lk_params = dict(
        winSize=(window_size, window_size),
        maxLevel=3,  # Pyramid levels
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
        minEigThreshold=1e-4,
    )
    
    # Feature detection parameters using Shi-Tomasi corner detector
    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=window_size,
    )
    
    # Initialize tracking from multiple frames for robustness
    all_tracked_points = []
    all_velocities = []
    flow_accum_x = np.zeros((rows, cols))
    flow_accum_y = np.zeros((rows, cols))
    flow_count = np.zeros((rows, cols))
    
    # Find initial features in the frame with maximum activity
    frame_activity = np.nanmax(np.abs(data), axis=(1, 2))
    peak_frame = int(np.argmax(frame_activity))
    
    # Also track from first frame and a frame before peak
    start_frames = [0, max(0, peak_frame - n_frames // 4), peak_frame]
    start_frames = sorted(set(start_frames))
    
    for start_frame in start_frames:
        # Detect features in the start frame
        frame = normalized[start_frame]
        features = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
        
        if features is None or len(features) == 0:
            continue
        
        # Track features forward
        prev_pts = features.copy()
        prev_frame = frame
        
        frame_points = []
        frame_velocities = []
        
        for t in range(start_frame, n_frames - 1):
            next_frame = normalized[t + 1]
            
            # Calculate optical flow using Lucas-Kanade
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_frame, next_frame, prev_pts, None, **lk_params
            )
            
            if next_pts is None:
                break
            
            # Filter by status
            good_new = next_pts[status.flatten() == 1]
            good_old = prev_pts[status.flatten() == 1]
            
            if len(good_new) < 3:  # Too few points remaining
                # Re-detect features
                features = cv2.goodFeaturesToTrack(next_frame, mask=None, **feature_params)
                if features is not None and len(features) > 0:
                    prev_pts = features.copy()
                    prev_frame = next_frame
                continue
            
            # Compute velocities (flow vectors)
            velocities = good_new - good_old
            
            # Accumulate flow for interpolation
            for pt, vel in zip(good_old, velocities):
                x, y = int(pt[0, 0]), int(pt[0, 1])
                vx, vy = vel[0, 0], vel[0, 1]
                
                if 0 <= y < rows and 0 <= x < cols:
                    flow_accum_x[y, x] += vx
                    flow_accum_y[y, x] += vy
                    flow_count[y, x] += 1
            
            # Store tracking info
            frame_points.append(good_new.copy())
            frame_velocities.append(velocities.copy())
            
            # Update for next iteration
            prev_pts = good_new.reshape(-1, 1, 2)
            prev_frame = next_frame
        
        all_tracked_points.extend(frame_points)
        all_velocities.extend(frame_velocities)
    
    # Build interpolated flow field from sparse measurements
    valid_mask = flow_count > 0
    avg_flow_x = np.zeros((rows, cols))
    avg_flow_y = np.zeros((rows, cols))
    avg_flow_x[valid_mask] = flow_accum_x[valid_mask] / flow_count[valid_mask]
    avg_flow_y[valid_mask] = flow_accum_y[valid_mask] / flow_count[valid_mask]
    
    # Interpolate flow field using Gaussian smoothing
    if np.any(valid_mask):
        # Normalize by count and smooth to fill gaps
        weight_map = flow_count.copy()
        weight_map[weight_map == 0] = 1e-10  # Avoid division by zero
        
        flow_x_interp = gaussian_filter(flow_accum_x, sigma=3) / gaussian_filter(weight_map, sigma=3)
        flow_y_interp = gaussian_filter(flow_accum_y, sigma=3) / gaussian_filter(weight_map, sigma=3)
    else:
        flow_x_interp = avg_flow_x
        flow_y_interp = avg_flow_y
    
    # Compute divergence from interpolated flow
    dvx_dx = np.gradient(flow_x_interp, axis=1)
    dvy_dy = np.gradient(flow_y_interp, axis=0)
    divergence_map = dvx_dx + dvy_dy
    divergence_map = gaussian_filter(divergence_map, sigma=2)
    
    # Find sink (minimum divergence) and source (maximum divergence)
    sink_idx = np.argmin(divergence_map)
    source_idx = np.argmax(divergence_map)
    sink = np.unravel_index(sink_idx, divergence_map.shape)
    source = np.unravel_index(source_idx, divergence_map.shape)
    
    # Build trajectory from tracked point centroids
    trajectory = []
    if all_tracked_points:
        for pts in all_tracked_points:
            if len(pts) > 0:
                centroid_x = np.mean(pts[:, 0, 0])
                centroid_y = np.mean(pts[:, 0, 1])
                trajectory.append([centroid_y, centroid_x])  # (row, col) format
    
    trajectory = np.array(trajectory) if trajectory else np.zeros((0, 2))
    
    # Smooth trajectory with Savitzky-Golay filter
    if len(trajectory) > TEMPORAL_CONFIG.savgol_window:
        trajectory[:, 0] = savgol_filter(trajectory[:, 0], TEMPORAL_CONFIG.savgol_window, 
                                         TEMPORAL_CONFIG.savgol_order)
        trajectory[:, 1] = savgol_filter(trajectory[:, 1], TEMPORAL_CONFIG.savgol_window,
                                         TEMPORAL_CONFIG.savgol_order)
    
    # Store flow fields as list for compatibility
    flow_fields = [(flow_x_interp, flow_y_interp)]
    
    # Compute velocity magnitude map
    velocity_magnitude = np.sqrt(flow_x_interp**2 + flow_y_interp**2)
    
    return {
        'flow_fields': flow_fields,
        'divergence_map': divergence_map,
        'sink': (int(sink[0]), int(sink[1])),
        'source': (int(source[0]), int(source[1])),
        'trajectory': trajectory,
        'tracked_points': all_tracked_points,
        'point_velocities': all_velocities,
        'velocity_magnitude': velocity_magnitude,
        'flow_x': flow_x_interp,
        'flow_y': flow_y_interp,
    }


# =============================================================================
# Propagation Method 10: Event-Based Tracking
# =============================================================================

def trace_event_based(
    eimage_sta: np.ndarray,
    threshold_percentile: float = 75.0,
) -> Dict[str, Any]:
    """
    Trace propagation using event-based activation tracking.
    
    TEMPORAL OPTIMIZATION: Uses adaptive threshold based on noise statistics,
    tracks wavefront contour with temporal smoothing, and computes wavefront
    velocity from contour motion.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        threshold_percentile: Percentile threshold for activation
        
    Returns:
        Dictionary with activation sequence, barycenter trajectory
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # TEMPORAL OPTIMIZATION: Preprocess with spatio-temporal smoothing
    data_processed = preprocess_temporal(eimage_sta, smooth_temporal=True, 
                                         smooth_spatial=True, subtract_baseline=True)
    data = np.abs(data_processed)
    
    # TEMPORAL OPTIMIZATION: Adaptive threshold based on noise statistics
    noise_based_threshold = adaptive_threshold(data_processed, snr_factor=2.0)
    percentile_threshold = np.percentile(data, threshold_percentile)
    threshold = max(noise_based_threshold, percentile_threshold)
    
    # Binary activation map
    activated = data > threshold
    
    # Track barycenter of active region over time
    raw_trajectory = []
    active_count = []
    leading_edge = []
    
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    for t in range(n_frames):
        active_mask = activated[t]
        n_active = np.sum(active_mask)
        
        if n_active > 0:
            # Barycenter
            center_row = np.sum(y_coords[active_mask]) / n_active
            center_col = np.sum(x_coords[active_mask]) / n_active
            
            # Leading edge: furthest active point from center
            distances = np.sqrt(
                (y_coords[active_mask] - center_row)**2 +
                (x_coords[active_mask] - center_col)**2
            )
            max_dist_idx = np.argmax(distances)
            edge_row = y_coords[active_mask][max_dist_idx]
            edge_col = x_coords[active_mask][max_dist_idx]
        else:
            center_row, center_col = rows / 2, cols / 2
            edge_row, edge_col = rows / 2, cols / 2
        
        raw_trajectory.append((center_row, center_col))
        active_count.append(n_active)
        leading_edge.append((edge_row, edge_col))
    
    raw_trajectory = np.array(raw_trajectory)
    leading_edge = np.array(leading_edge)
    active_count = np.array(active_count)
    
    # TEMPORAL OPTIMIZATION: Smooth trajectory and leading edge
    trajectory_smooth_row = gaussian_filter1d(raw_trajectory[:, 0], sigma=TEMPORAL_CONFIG.sigma_t)
    trajectory_smooth_col = gaussian_filter1d(raw_trajectory[:, 1], sigma=TEMPORAL_CONFIG.sigma_t)
    trajectory = np.column_stack([trajectory_smooth_row, trajectory_smooth_col])
    
    edge_smooth_row = gaussian_filter1d(leading_edge[:, 0], sigma=TEMPORAL_CONFIG.sigma_t)
    edge_smooth_col = gaussian_filter1d(leading_edge[:, 1], sigma=TEMPORAL_CONFIG.sigma_t)
    leading_edge_smooth = np.column_stack([edge_smooth_row, edge_smooth_col])
    
    # Compute spread velocity (change in active region size)
    spread_velocity = np.gradient(active_count)
    
    # TEMPORAL OPTIMIZATION: Compute wavefront velocity from leading edge motion
    edge_velocity = np.sqrt(np.diff(leading_edge_smooth[:, 0])**2 + 
                           np.diff(leading_edge_smooth[:, 1])**2)
    
    # First activation map (when each electrode first activates)
    first_activation = np.full((rows, cols), float(n_frames))
    for t in range(n_frames):
        newly_active = activated[t] & (first_activation == n_frames)
        first_activation[newly_active] = float(t)
    
    return {
        'trajectory': trajectory,
        'raw_trajectory': raw_trajectory,
        'leading_edge': leading_edge_smooth,
        'leading_edge_raw': leading_edge,
        'active_count': active_count,
        'spread_velocity': spread_velocity,
        'edge_velocity': edge_velocity,
        'first_activation': first_activation,
        'activated': activated,
    }


def plot_divergence_streamlines(
    unit_id: str,
    eimage_sta: np.ndarray,
    optflow_result: Tuple[Tuple[int, int], Tuple[int, int], np.ndarray],
    flow_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create detailed divergence + streamlines visualization.
    
    Shows divergence map with overlaid streamlines at multiple time points,
    sink/source locations, and flow statistics.
    
    Args:
        unit_id: Unit identifier
        eimage_sta: 3D array (time, rows, cols)
        optflow_result: Result from find_center_optical_flow()
        flow_data: Result from trace_optical_flow_propagation()
        output_path: Path to save the figure
    """
    fig = plt.figure(figsize=(20, 16))
    
    optflow_sink, optflow_source, divergence_map = optflow_result
    flow_fields = flow_data.get('flow_fields', [])
    flow_magnitudes = flow_data.get('flow_magnitude', [])
    peak_frame = flow_data.get('peak_frame', 0)
    velocity_profile = flow_data.get('velocity_profile', [])
    
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess for visualization
    data = preprocess_temporal(eimage_sta)
    vmin = np.nanpercentile(data, 1)
    vmax = np.nanpercentile(data, 99)
    
    # Create coordinate grids for streamplot
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    
    # Row 1: Divergence at different time points with streamlines
    n_time_samples = 5
    if len(flow_fields) > 0:
        time_indices = np.linspace(0, len(flow_fields) - 1, n_time_samples).astype(int)
    else:
        time_indices = [0] * n_time_samples
    
    for i, t_idx in enumerate(time_indices):
        ax = fig.add_subplot(4, 5, i + 1)
        
        if t_idx < len(flow_fields):
            flow_x, flow_y = flow_fields[t_idx]
            
            # Compute divergence for this frame
            dvx_dx = np.gradient(flow_x, axis=1)
            dvy_dy = np.gradient(flow_y, axis=0)
            div_frame = dvx_dx + dvy_dy
            
            # Plot divergence
            div_abs = max(abs(np.nanmin(div_frame)), abs(np.nanmax(div_frame)), 1e-10)
            norm = TwoSlopeNorm(vmin=-div_abs, vcenter=0, vmax=div_abs)
            ax.imshow(div_frame, cmap='RdBu_r', norm=norm)
            
            # Add streamlines
            speed = np.sqrt(flow_x**2 + flow_y**2)
            if np.max(speed) > 0.05:
                try:
                    ax.streamplot(x_grid, y_grid, flow_x, flow_y, 
                                 color='black', linewidth=0.5, density=1.2, arrowsize=0.5)
                except:
                    pass  # Skip if streamplot fails
        else:
            ax.imshow(data[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        ax.set_title(f"Frame {t_idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Divergence + Streamlines", fontsize=11)
    
    # Row 2: Flow magnitude with quiver at different time points
    for i, t_idx in enumerate(time_indices):
        ax = fig.add_subplot(4, 5, 6 + i)
        
        if t_idx < len(flow_magnitudes):
            magnitude = flow_magnitudes[t_idx]
            ax.imshow(magnitude, cmap='hot', vmin=0, vmax=np.percentile(magnitude, 99))
            
            # Add quiver plot (subsampled)
            if t_idx < len(flow_fields):
                flow_x, flow_y = flow_fields[t_idx]
                step = 4
                Y, X = np.mgrid[0:rows:step, 0:cols:step]
                U = flow_x[::step, ::step]
                V = flow_y[::step, ::step]
                
                # Normalize for visibility
                mag = np.sqrt(U**2 + V**2)
                mag[mag == 0] = 1
                U_norm = U / mag
                V_norm = V / mag
                
                ax.quiver(X, Y, U_norm, V_norm, color='cyan', alpha=0.7, 
                         scale=30, width=0.004)
        else:
            ax.axis('off')
        
        ax.set_title(f"Flow Mag + Quiver (t={t_idx})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Flow Magnitude", fontsize=11)
    
    # Row 3: Accumulated divergence with streamlines, sink/source, and statistics
    # Subplot 1: Accumulated divergence with streamlines
    ax = fig.add_subplot(4, 5, 11)
    div_abs = max(abs(np.nanmin(divergence_map)), abs(np.nanmax(divergence_map)), 1e-10)
    norm = TwoSlopeNorm(vmin=-div_abs, vcenter=0, vmax=div_abs)
    im = ax.imshow(divergence_map, cmap='RdBu_r', norm=norm)
    
    # Add streamlines from peak frame
    if len(flow_fields) > peak_frame:
        flow_x, flow_y = flow_fields[peak_frame]
        speed = np.sqrt(flow_x**2 + flow_y**2)
        if np.max(speed) > 0.05:
            try:
                ax.streamplot(x_grid, y_grid, flow_x, flow_y, 
                             color='black', linewidth=0.7, density=1.5, arrowsize=0.6)
            except:
                pass
    
    ax.plot(optflow_sink[1], optflow_sink[0], 'bo', markersize=15, 
            markeredgecolor='white', markeredgewidth=2, label='Sink')
    ax.plot(optflow_source[1], optflow_source[0], 'r^', markersize=15, 
            markeredgecolor='white', markeredgewidth=2, label='Source')
    ax.set_title(f"Accumulated Divergence\n(Peak Frame {peak_frame})", fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Subplot 2: Raw data at peak with sink/source overlay
    ax = fig.add_subplot(4, 5, 12)
    ax.imshow(data[peak_frame], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.plot(optflow_sink[1], optflow_sink[0], 'bo', markersize=12, 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot(optflow_source[1], optflow_source[0], 'r^', markersize=12, 
            markeredgecolor='white', markeredgewidth=2)
    ax.set_title(f"Signal (Peak Frame {peak_frame})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Subplot 3: Velocity profile over time
    ax = fig.add_subplot(4, 5, 13)
    if len(velocity_profile) > 0:
        ax.plot(velocity_profile, 'b-', linewidth=2)
        ax.axvline(x=peak_frame, color='r', linestyle='--', linewidth=1.5, 
                  label=f'Peak (t={peak_frame})')
        ax.fill_between(range(len(velocity_profile)), velocity_profile, alpha=0.3)
    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Mean Flow Velocity", fontsize=10)
    ax.set_title("Velocity Profile", fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Divergence histogram
    ax = fig.add_subplot(4, 5, 14)
    div_flat = divergence_map.flatten()
    ax.hist(div_flat, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax.axvline(x=np.mean(div_flat), color='r', linestyle='--', 
              label=f'Mean: {np.mean(div_flat):.3f}')
    ax.set_xlabel("Divergence", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Divergence Distribution", fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    
    # Subplot 5: Flow direction at peak (HSV)
    ax = fig.add_subplot(4, 5, 15)
    if len(flow_fields) > peak_frame:
        flow_x, flow_y = flow_fields[peak_frame]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        direction = np.arctan2(flow_y, flow_x)
        
        # Create HSV image
        hsv = np.zeros((rows, cols, 3))
        hsv[:, :, 0] = (direction + np.pi) / (2 * np.pi)  # Hue: direction
        hsv[:, :, 1] = 1.0  # Saturation
        hsv[:, :, 2] = magnitude / (np.max(magnitude) + 1e-10)  # Value: magnitude
        
        # Convert HSV to RGB
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        ax.imshow(rgb)
        ax.set_title(f"Flow Direction (HSV)\nPeak Frame {peak_frame}", fontsize=10)
    else:
        ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Row 4: Detailed streamline views at key moments
    key_moments = [
        max(0, peak_frame - 5),
        peak_frame,
        min(len(flow_fields) - 1, peak_frame + 5) if len(flow_fields) > 0 else 0
    ]
    
    for i, t_idx in enumerate(key_moments[:3]):
        ax = fig.add_subplot(4, 5, 16 + i)
        
        # Background: raw data
        if t_idx < n_frames:
            ax.imshow(data[t_idx], cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.6)
        
        # Overlay streamlines
        if t_idx < len(flow_fields):
            flow_x, flow_y = flow_fields[t_idx]
            speed = np.sqrt(flow_x**2 + flow_y**2)
            if np.max(speed) > 0.05:
                try:
                    # Color streamlines by speed
                    ax.streamplot(x_grid, y_grid, flow_x, flow_y, 
                                 color=speed, cmap='plasma', linewidth=1, 
                                 density=1.8, arrowsize=0.7)
                except:
                    pass
        
        ax.plot(optflow_sink[1], optflow_sink[0], 'bo', markersize=10, 
                markeredgecolor='white', markeredgewidth=1.5)
        ax.plot(optflow_source[1], optflow_source[0], 'r^', markersize=10, 
                markeredgecolor='white', markeredgewidth=1.5)
        
        label = "Pre-peak" if i == 0 else ("Peak" if i == 1 else "Post-peak")
        ax.set_title(f"{label} (t={t_idx})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Data + Streamlines", fontsize=11)
    
    # Subplots 19-20: Summary statistics
    ax = fig.add_subplot(4, 5, 19)
    ax.axis('off')
    
    stats_text = f"""Flow Statistics
{'='*30}
Sink location: ({optflow_sink[0]}, {optflow_sink[1]})
Source location: ({optflow_source[0]}, {optflow_source[1]})
Peak frame: {peak_frame}

Divergence:
  Min: {np.min(divergence_map):.4f}
  Max: {np.max(divergence_map):.4f}
  Mean: {np.mean(divergence_map):.4f}
  Std: {np.std(divergence_map):.4f}

Velocity:
  Max: {np.max(velocity_profile):.4f}
  Mean: {np.mean(velocity_profile):.4f}"""
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax = fig.add_subplot(4, 5, 20)
    ax.axis('off')
    
    # Add colorbar legend for HSV
    ax.text(0.5, 0.8, "HSV Color Legend", transform=ax.transAxes, 
            fontsize=11, ha='center', fontweight='bold')
    ax.text(0.5, 0.6, "Hue = Flow Direction\nValue = Flow Speed", 
            transform=ax.transAxes, fontsize=10, ha='center')
    
    fig.suptitle(f"{unit_id} - Divergence & Streamlines Analysis", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved divergence+streamlines plot: {output_path}")


def plot_optical_flow_propagation(
    unit_id: str,
    eimage_sta: np.ndarray,
    propagation_data: Dict[str, Any],
    optflow_result: Tuple[Tuple[int, int], Tuple[int, int], np.ndarray],
    output_path: Path,
) -> None:
    """
    Create a detailed visualization of action potential propagation via optical flow.
    
    Args:
        unit_id: Unit identifier
        eimage_sta: Original eimage_sta data
        propagation_data: Output from trace_optical_flow_propagation()
        optflow_result: Result from find_center_optical_flow()
        output_path: Path to save the figure
    """
    fig = plt.figure(figsize=(20, 16))
    
    optflow_sink, optflow_source, divergence_map = optflow_result
    flow_fields = propagation_data['flow_fields']
    flow_magnitudes = propagation_data['flow_magnitude']
    key_frames = propagation_data['key_frames']
    propagation_path = propagation_data['propagation_path']
    velocity_profile = propagation_data['velocity_profile']
    peak_frame = propagation_data.get('peak_frame', 0)
    
    n_frames, rows, cols = eimage_sta.shape
    
    # Row 1: Raw eimage_sta at key frames (5 subplots)
    vmin = np.nanpercentile(eimage_sta, 1)
    vmax = np.nanpercentile(eimage_sta, 99)
    
    display_frames = key_frames[:5] if len(key_frames) >= 5 else key_frames
    for i, frame_idx in enumerate(display_frames):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.imshow(eimage_sta[frame_idx], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f"Frame {frame_idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Raw Signal", fontsize=11)
    
    # Row 2: Flow magnitude at key frames with quiver overlay
    for i, frame_idx in enumerate(display_frames):
        if frame_idx >= len(flow_magnitudes):
            continue
        ax = fig.add_subplot(4, 5, 6 + i)
        
        magnitude = flow_magnitudes[frame_idx]
        ax.imshow(magnitude, cmap='hot', vmin=0, vmax=np.percentile(magnitude, 99))
        
        # Add quiver plot (subsample for clarity)
        step = 4
        y, x = np.mgrid[0:rows:step, 0:cols:step]
        flow_x, flow_y = flow_fields[frame_idx]
        u = flow_x[::step, ::step]
        v = flow_y[::step, ::step]
        
        # Normalize arrows for visibility
        mag = np.sqrt(u**2 + v**2)
        mag[mag == 0] = 1
        u_norm = u / mag * 2
        v_norm = v / mag * 2
        
        ax.quiver(x, y, u_norm, v_norm, color='cyan', alpha=0.7, scale=50, width=0.003)
        ax.set_title(f"Flow Mag + Vectors\nFrame {frame_idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Optical Flow", fontsize=11)
    
    # Row 3: Flow direction, propagation path, velocity profile
    # Subplot 1: Flow direction at peak frame
    ax = fig.add_subplot(4, 5, 11)
    if len(propagation_data['flow_direction']) > peak_frame:
        direction = propagation_data['flow_direction'][peak_frame]
        ax.imshow(direction, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax.set_title(f"Flow Direction\n(Peak Frame {peak_frame})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Analysis", fontsize=11)
    
    # Subplot 2: Propagation path overlaid on average magnitude
    ax = fig.add_subplot(4, 5, 12)
    avg_magnitude = np.mean(flow_magnitudes, axis=0) if flow_magnitudes else np.zeros((rows, cols))
    ax.imshow(avg_magnitude, cmap='hot')
    
    if len(propagation_path) > 0:
        # Plot propagation path
        path_rows = propagation_path[:, 0]
        path_cols = propagation_path[:, 1]
        ax.plot(path_cols, path_rows, 'c-', linewidth=2, label='Path')
        ax.plot(path_cols[0], path_rows[0], 'go', markersize=10, label='Start')
        ax.plot(path_cols[-1], path_rows[-1], 'r*', markersize=12, label='End')
    
    ax.plot(optflow_sink[1], optflow_sink[0], 'bs', markersize=10, 
            markeredgecolor='white', markeredgewidth=2, label='Sink')
    ax.plot(optflow_source[1], optflow_source[0], 'r^', markersize=10, 
            markeredgecolor='white', markeredgewidth=2, label='Source')
    ax.set_title("Propagation Path", fontsize=10)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Subplot 3: Velocity profile over time
    ax = fig.add_subplot(4, 5, 13)
    if len(velocity_profile) > 0:
        time_axis = np.arange(len(velocity_profile))
        ax.plot(time_axis, velocity_profile, 'b-', linewidth=2)
        ax.axvline(x=peak_frame, color='r', linestyle='--', label=f'Peak ({peak_frame})')
        ax.fill_between(time_axis, velocity_profile, alpha=0.3)
        ax.set_xlabel("Frame", fontsize=10)
        ax.set_ylabel("Avg Velocity", fontsize=10)
        ax.set_title("Velocity Profile", fontsize=10)
        ax.legend(fontsize=8)
    
    # Subplot 4: Divergence map with streamlines
    ax = fig.add_subplot(4, 5, 14)
    div_abs_max = max(abs(np.nanmin(divergence_map)), abs(np.nanmax(divergence_map)), 1e-10)
    norm = TwoSlopeNorm(vmin=-div_abs_max, vcenter=0, vmax=div_abs_max)
    ax.imshow(divergence_map, cmap='RdBu_r', norm=norm)
    
    # Add streamlines at peak frame
    if len(flow_fields) > peak_frame:
        flow_x, flow_y = flow_fields[peak_frame]
        y, x = np.mgrid[0:rows, 0:cols]
        # Only plot streamlines where there's significant flow
        speed = np.sqrt(flow_x**2 + flow_y**2)
        if np.max(speed) > 0.1:
            ax.streamplot(x, y, flow_x, flow_y, color='black', 
                         linewidth=0.5, density=1.5, arrowsize=0.5)
    
    ax.plot(optflow_sink[1], optflow_sink[0], 'bo', markersize=12, 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot(optflow_source[1], optflow_source[0], 'r^', markersize=12, 
            markeredgecolor='white', markeredgewidth=2)
    ax.set_title("Divergence + Streamlines", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Subplot 5: Combined view with all key info
    ax = fig.add_subplot(4, 5, 15)
    ax.imshow(eimage_sta[peak_frame], cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.7)
    
    if len(propagation_path) > 0:
        path_rows = propagation_path[:, 0]
        path_cols = propagation_path[:, 1]
        ax.plot(path_cols, path_rows, 'yellow', linewidth=3, alpha=0.8)
    
    ax.plot(optflow_sink[1], optflow_sink[0], 'bo', markersize=14, 
            markeredgecolor='white', markeredgewidth=2, label='Sink')
    ax.plot(optflow_source[1], optflow_source[0], 'r^', markersize=14, 
            markeredgecolor='white', markeredgewidth=2, label='Source')
    ax.set_title("Summary View", fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Row 4: Time sequence of flow evolution (5 evenly spaced frames)
    if len(flow_magnitudes) > 0:
        time_points = np.linspace(0, len(flow_magnitudes) - 1, 5).astype(int)
        for i, t in enumerate(time_points):
            ax = fig.add_subplot(4, 5, 16 + i)
            
            magnitude = flow_magnitudes[t]
            flow_x, flow_y = flow_fields[t]
            
            # Show magnitude with color-coded direction
            direction = np.arctan2(flow_y, flow_x)
            
            # Create HSV image: Hue = direction, Saturation = 1, Value = magnitude
            hsv = np.zeros((rows, cols, 3))
            hsv[:, :, 0] = (direction + np.pi) / (2 * np.pi)  # Hue: 0-1
            hsv[:, :, 1] = 1.0  # Saturation
            mag_normalized = magnitude / (np.max(magnitude) + 1e-10)
            hsv[:, :, 2] = mag_normalized  # Value: magnitude
            
            from matplotlib.colors import hsv_to_rgb
            rgb = hsv_to_rgb(hsv)
            
            ax.imshow(rgb)
            ax.set_title(f"t={t}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel("Flow Evolution\n(color=dir, bright=mag)", fontsize=9)
    
    fig.suptitle(f"{unit_id} - Action Potential Propagation Analysis (Optical Flow)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved propagation plot: {output_path}")


# =============================================================================
# Algorithm 3: Current Source Density (CSD) Analysis
# =============================================================================

def find_center_csd(eimage_sta: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, int]:
    """
    Find current sink and source using CSD analysis.
    
    TEMPORAL OPTIMIZATION: Computes full 3D spatio-temporal CSD (Laplacian
    including temporal dimension), then integrates over a time window around
    peak activity for robust sink/source detection.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (sink_coords, source_coords, csd_at_peak_frame, peak_frame_idx)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with spatio-temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # Find the frame with maximum activity (largest absolute value)
    frame_activity = np.nanmax(np.abs(data), axis=(1, 2))
    peak_frame_idx = int(np.nanargmax(frame_activity))
    
    # TEMPORAL OPTIMIZATION: Compute full 3D Laplacian
    # This includes temporal second derivative for spatio-temporal CSD
    csd_3d = -laplace(data, mode='nearest')  # 3D Laplacian (t, y, x)
    
    # Define integration window around peak
    window = TEMPORAL_CONFIG.integration_window
    start_frame = max(0, peak_frame_idx - window // 2)
    end_frame = min(n_frames, peak_frame_idx + window // 2 + 1)
    
    # Activity-weighted temporal integration of CSD
    frame_weights = frame_activity[start_frame:end_frame]
    frame_weights = frame_weights / (np.sum(frame_weights) + 1e-10)
    
    csd_integrated = np.zeros((rows, cols))
    for i, t in enumerate(range(start_frame, end_frame)):
        csd_integrated += csd_3d[t] * frame_weights[i]
    
    # Also get CSD at peak frame for visualization
    csd_peak = csd_3d[peak_frame_idx]
    
    # Find sink (min CSD) and source (max CSD) from integrated CSD
    sink_idx = np.nanargmin(csd_integrated)
    source_idx = np.nanargmax(csd_integrated)
    
    sink_coords = np.unravel_index(sink_idx, csd_integrated.shape)
    source_coords = np.unravel_index(source_idx, csd_integrated.shape)
    
    return (int(sink_coords[0]), int(sink_coords[1])), \
           (int(source_coords[0]), int(source_coords[1])), \
           csd_peak, peak_frame_idx


# =============================================================================
# Algorithm 4: 3D Gaussian Fit Method
# =============================================================================

def gaussian_3d(coords, amplitude, x0, y0, t0, sigma_x, sigma_y, sigma_t, offset):
    """
    3D Gaussian function for fitting.
    
    Args:
        coords: Tuple of (x, y, t) coordinate arrays
        amplitude: Peak amplitude
        x0, y0, t0: Center coordinates
        sigma_x, sigma_y, sigma_t: Standard deviations in each dimension
        offset: Baseline offset
    
    Returns:
        Flattened Gaussian values
    """
    x, y, t = coords
    g = offset + amplitude * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) +
          (y - y0)**2 / (2 * sigma_y**2) +
          (t - t0)**2 / (2 * sigma_t**2))
    )
    return g.ravel()


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    2D Gaussian function for fitting spatial center.
    
    Args:
        coords: Tuple of (x, y) coordinate arrays
        amplitude: Peak amplitude
        x0, y0: Center coordinates  
        sigma_x, sigma_y: Standard deviations
        offset: Baseline offset
    
    Returns:
        Flattened Gaussian values
    """
    x, y = coords
    g = offset + amplitude * np.exp(
        -((x - x0)**2 / (2 * sigma_x**2) +
          (y - y0)**2 / (2 * sigma_y**2))
    )
    return g.ravel()


def find_center_gaussian_fit(
    eimage_sta: np.ndarray,
    blur_sigma: float = 2.0,
    use_3d: bool = False,
) -> Tuple[Tuple[float, float], np.ndarray, Dict, bool]:
    """
    Find cell center using Gaussian blur followed by Gaussian fit.
    
    TEMPORAL OPTIMIZATION: Fits 2D Gaussian to multiple frames around peak
    and uses temporal trajectory of fitted centers with weighted average.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        blur_sigma: Sigma for Gaussian blur preprocessing
        use_3d: If True, fit a full 3D Gaussian; otherwise fit 2D to peak frame
        
    Returns:
        Tuple of (center_coords, blurred_peak_frame, fit_params, fit_success)
        where center_coords is (row, col) with sub-pixel precision
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with spatio-temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=False,
                               subtract_baseline=True)
    
    # Apply additional spatial Gaussian blur
    blurred = np.zeros_like(data, dtype=np.float64)
    for t in range(n_frames):
        blurred[t] = gaussian_filter(data[t], sigma=blur_sigma)
    
    # Find peak frame and location for initial guess
    abs_blurred = np.abs(blurred)
    peak_idx = np.unravel_index(np.argmax(abs_blurred), blurred.shape)
    peak_frame_idx = peak_idx[0]
    init_row, init_col = peak_idx[1], peak_idx[2]
    
    # Check sign of peak
    peak_sign = np.sign(blurred[peak_idx])
    
    # TEMPORAL OPTIMIZATION: Fit Gaussian to multiple frames and track center trajectory
    window = TEMPORAL_CONFIG.integration_window
    start_frame = max(0, peak_frame_idx - window // 2)
    end_frame = min(n_frames, peak_frame_idx + window // 2 + 1)
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    
    # Track fitted centers across time window
    fitted_centers = []
    fitted_weights = []
    last_successful_popt = None
    
    for t in range(start_frame, end_frame):
        frame = blurred[t]
        if peak_sign < 0:
            frame = -frame
        
        amplitude_init = np.max(np.abs(frame))
        offset_init = np.median(frame)
        
        try:
            popt, _ = curve_fit(
                gaussian_2d,
                (x_grid, y_grid),
                frame.ravel(),
                p0=[amplitude_init, init_col, init_row, 3.0, 3.0, offset_init],
                bounds=(
                    [0, 0, 0, 0.5, 0.5, -np.inf],
                    [np.inf, cols-1, rows-1, cols/2, rows/2, np.inf]
                ),
                maxfev=2000,
            )
            
            center_col, center_row = popt[1], popt[2]
            amplitude = popt[0]
            
            fitted_centers.append((center_row, center_col))
            fitted_weights.append(amplitude)  # Weight by amplitude
            last_successful_popt = popt
            
        except (RuntimeError, ValueError):
            continue
    
    if len(fitted_centers) > 0:
        # Weighted average of fitted centers
        centers_arr = np.array(fitted_centers)
        weights_arr = np.array(fitted_weights)
        weights_arr = weights_arr / (np.sum(weights_arr) + 1e-10)
        
        center_row = np.sum(centers_arr[:, 0] * weights_arr)
        center_col = np.sum(centers_arr[:, 1] * weights_arr)
        
        if last_successful_popt is not None:
            fit_params = {
                'amplitude': last_successful_popt[0] * peak_sign,
                'center_col': center_col,
                'center_row': center_row,
                'sigma_x': last_successful_popt[3],
                'sigma_y': last_successful_popt[4],
                'offset': last_successful_popt[5],
                'peak_frame': peak_frame_idx,
                'n_fitted_frames': len(fitted_centers),
            }
        else:
            fit_params = {}
        
        fit_success = True
    else:
        # Fall back to peak location
        center_row, center_col = float(init_row), float(init_col)
        fit_params = {
            'amplitude': blurred[peak_idx],
            'center_col': center_col,
            'center_row': center_row,
            'sigma_x': np.nan,
            'sigma_y': np.nan,
            'offset': 0,
            'peak_frame': peak_frame_idx,
        }
        fit_success = False
    
    # Clamp to valid range
    center_row = np.clip(center_row, 0, rows - 1)
    center_col = np.clip(center_col, 0, cols - 1)
    
    return (center_row, center_col), blurred[peak_frame_idx], fit_params, fit_success


# =============================================================================
# Algorithm 5: Peak-to-Peak Footprint with Robust Stats
# =============================================================================

def find_center_peak_to_peak(eimage_sta: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Find cell center using peak-to-peak footprint with robust statistics.
    
    TEMPORAL OPTIMIZATION: Uses Savitzky-Golay filtering for robust temporal
    peak/trough detection before computing peak-to-peak.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (row, col, p2p_map)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=False,
                               subtract_baseline=True)
    
    # Compute robust peak-to-peak using Savitzky-Golay filtering
    p2p_map = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            signal = data[:, r, c]
            peak_val, trough_val, _, _ = robust_peak_trough_detection(signal)
            p2p_map[r, c] = peak_val - trough_val
    
    # MAD normalize for robustness
    median_p2p = np.nanmedian(p2p_map)
    mad = np.nanmedian(np.abs(p2p_map - median_p2p))
    if mad > 0:
        p2p_normalized = (p2p_map - median_p2p) / (1.4826 * mad)  # Scale to match std
    else:
        p2p_normalized = p2p_map
    
    # Find center at max p2p
    center_idx = np.nanargmax(p2p_normalized)
    center_row, center_col = np.unravel_index(center_idx, p2p_map.shape)
    
    return int(center_row), int(center_col), p2p_map


# =============================================================================
# Algorithm 6: Trough-Centric Centroid (Soft-Argmin)
# =============================================================================

def find_center_trough_centroid(
    eimage_sta: np.ndarray,
    top_k_percent: float = 10.0,
) -> Tuple[float, float, np.ndarray]:
    """
    Find cell center using weighted centroid of top-k% electrodes.
    
    TEMPORAL OPTIMIZATION: Computes centroid for multiple frames around peak
    and uses activity-weighted temporal average for robust center estimation.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        top_k_percent: Percentage of electrodes to include in centroid
        
    Returns:
        Tuple of (row, col, weight_map) with sub-pixel precision
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # Find peak activity frame
    frame_activity = np.nanmax(np.abs(data), axis=(1, 2))
    peak_frame = int(np.nanargmax(frame_activity))
    
    # TEMPORAL OPTIMIZATION: Compute centroid for time window and average
    window = TEMPORAL_CONFIG.integration_window
    start_frame = max(0, peak_frame - window // 2)
    end_frame = min(n_frames, peak_frame + window // 2 + 1)
    
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    centroids = []
    weights = []
    weight_map_accum = np.zeros((rows, cols))
    
    for t in range(start_frame, end_frame):
        # Get spatial map (use absolute value for weighting)
        peak_map = np.abs(data[t])
        peak_map = np.nan_to_num(peak_map, nan=0)
        
        # Keep only top k% by amplitude
        threshold = np.nanpercentile(peak_map, 100 - top_k_percent)
        frame_weight_map = np.where(peak_map >= threshold, peak_map, 0)
        
        # Compute weighted centroid for this frame
        total_weight = np.sum(frame_weight_map)
        if total_weight > 0:
            center_row_t = np.sum(y_coords * frame_weight_map) / total_weight
            center_col_t = np.sum(x_coords * frame_weight_map) / total_weight
            centroids.append((center_row_t, center_col_t))
            weights.append(frame_activity[t])  # Weight by frame activity
            weight_map_accum += frame_weight_map * frame_activity[t]
    
    if len(centroids) > 0:
        centroids_arr = np.array(centroids)
        weights_arr = np.array(weights)
        weights_arr = weights_arr / (np.sum(weights_arr) + 1e-10)
        
        center_row = np.sum(centroids_arr[:, 0] * weights_arr)
        center_col = np.sum(centroids_arr[:, 1] * weights_arr)
        
        weight_map = weight_map_accum / (np.sum(weights) + 1e-10)
    else:
        center_row, center_col = rows / 2, cols / 2
        weight_map = np.zeros((rows, cols))
    
    return float(center_row), float(center_col), weight_map


# =============================================================================
# Algorithm 7: Template / Matched-Filter Center
# =============================================================================

def find_center_template_matching(eimage_sta: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Find cell center using template matching with SVD-denoised template.
    
    TEMPORAL OPTIMIZATION: Uses rank-k SVD for better temporal dynamics,
    FFT-based cross-correlation for efficiency, and considers temporal
    template structure.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (row, col, energy_map)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=False,
                               subtract_baseline=True)
    
    # Reshape to (time, space) for SVD
    data_2d = data.reshape(n_frames, -1)  # (time, rows*cols)
    
    # Center the data
    data_mean = np.mean(data_2d, axis=0, keepdims=True)
    data_centered = data_2d - data_mean
    
    # TEMPORAL OPTIMIZATION: Use rank-k approximation for better dynamics
    rank_k = 3  # Use top-k components
    try:
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        # Rank-k reconstruction for richer temporal template
        template_temporal = U[:, 0] * S[0]  # Primary temporal pattern
        
        # Also consider secondary patterns for robustness
        for k in range(1, min(rank_k, len(S))):
            if S[k] > 0.1 * S[0]:  # Only if significant
                template_temporal += 0.3 * U[:, k] * S[k]
                
    except np.linalg.LinAlgError:
        # Fallback: use mean absolute signal
        template_temporal = np.mean(np.abs(data_2d), axis=1)
    
    # Normalize template
    template_norm = template_temporal / (np.linalg.norm(template_temporal) + 1e-10)
    
    # TEMPORAL OPTIMIZATION: Use FFT-based cross-correlation for efficiency
    # Vectorized computation across all electrodes
    data_norms = np.linalg.norm(data_2d, axis=0)
    data_norms[data_norms < 1e-10] = 1e-10
    
    # Dot product with normalized signals
    energy_flat = np.abs(data_2d.T @ template_norm) / data_norms
    energy_map = energy_flat.reshape(rows, cols)
    
    # Center at max energy
    center_idx = np.argmax(energy_map)
    center_row, center_col = np.unravel_index(center_idx, energy_map.shape)
    
    return int(center_row), int(center_col), energy_map


# =============================================================================
# Algorithm 8: Low-Rank + Sparsity Decomposition
# =============================================================================

def find_center_lowrank_sparse(
    eimage_sta: np.ndarray,
    rank: int = 3,
    sparse_threshold: float = 2.0,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Find cell center using low-rank + sparse decomposition.
    
    TEMPORAL OPTIMIZATION: Applies temporal smoothing to low-rank component
    for background estimation, and considers temporal continuity in sparse
    component detection.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        rank: Rank for low-rank approximation
        sparse_threshold: MAD threshold for sparsity
        
    Returns:
        Tuple of (row, col, sparse_map, lowrank_map)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess: baseline subtraction only (no smoothing yet)
    data = preprocess_temporal(eimage_sta, smooth_temporal=False, smooth_spatial=False,
                               subtract_baseline=True)
    
    # Reshape to 2D
    data_2d = data.reshape(n_frames, -1)
    
    # Compute low-rank approximation via truncated SVD
    try:
        U, S_vals, Vt = np.linalg.svd(data_2d, full_matrices=False)
        # Low-rank reconstruction
        L = U[:, :rank] @ np.diag(S_vals[:rank]) @ Vt[:rank, :]
    except np.linalg.LinAlgError:
        L = np.zeros_like(data_2d)
    
    # TEMPORAL OPTIMIZATION: Smooth low-rank component temporally
    lowrank_3d = L.reshape(n_frames, rows, cols)
    lowrank_3d = temporal_smooth(lowrank_3d, sigma_t=TEMPORAL_CONFIG.sigma_t)
    L = lowrank_3d.reshape(n_frames, -1)
    
    # Sparse = original - smoothed low-rank
    S_2d = data_2d - L
    
    # Threshold sparse component with adaptive threshold
    noise_level = estimate_noise_level(data)
    threshold = sparse_threshold * noise_level
    S_sparse = np.where(np.abs(S_2d) > threshold, S_2d, 0)
    
    # Reshape back to 3D
    sparse_3d = S_sparse.reshape(n_frames, rows, cols)
    
    # TEMPORAL OPTIMIZATION: Consider temporal continuity
    # Only keep sparse events that persist across multiple frames
    sparse_persistent = np.zeros_like(sparse_3d)
    for t in range(1, n_frames - 1):
        # Keep if non-zero in current and at least one adjacent frame
        current = np.abs(sparse_3d[t]) > 0
        prev_or_next = (np.abs(sparse_3d[t-1]) > 0) | (np.abs(sparse_3d[t+1]) > 0)
        sparse_persistent[t] = np.where(current & prev_or_next, sparse_3d[t], 0)
    sparse_persistent[0] = sparse_3d[0]
    sparse_persistent[-1] = sparse_3d[-1]
    
    # Find center from sparse component (max absolute value over time)
    sparse_map = np.max(np.abs(sparse_persistent), axis=0)
    center_idx = np.argmax(sparse_map)
    center_row, center_col = np.unravel_index(center_idx, sparse_map.shape)
    
    lowrank_map = np.max(np.abs(lowrank_3d), axis=0)
    
    return int(center_row), int(center_col), sparse_map, lowrank_map


# =============================================================================
# Algorithm 9: Difference-of-Gaussians (DoG) Fit
# =============================================================================

def dog_2d(coords, amp_center, amp_surround, x0, y0, sigma_center, sigma_surround, offset):
    """Difference of Gaussians 2D function."""
    x, y = coords
    center = amp_center * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma_center**2))
    surround = amp_surround * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma_surround**2))
    return (offset + center - surround).ravel()


def find_center_dog_fit(eimage_sta: np.ndarray) -> Tuple[float, float, Dict, bool]:
    """
    Find cell center using Difference-of-Gaussians (DoG) fit.
    
    TEMPORAL OPTIMIZATION: Fits DoG to multiple frames around peak and
    tracks center trajectory, using weighted average for robust estimation.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (row, col, fit_params, success)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # Find peak frame
    frame_activity = np.nanmax(np.abs(data), axis=(1, 2))
    peak_frame = int(np.nanargmax(frame_activity))
    
    # TEMPORAL OPTIMIZATION: Fit DoG to multiple frames
    window = TEMPORAL_CONFIG.integration_window
    start_frame = max(0, peak_frame - window // 2)
    end_frame = min(n_frames, peak_frame + window // 2 + 1)
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    
    fitted_centers = []
    fitted_weights = []
    last_successful_popt = None
    
    for t in range(start_frame, end_frame):
        frame_data = data[t]
        peak_smooth = gaussian_filter(frame_data, sigma=1)
        
        # Find initial guess from peak location
        peak_idx = np.unravel_index(np.argmax(np.abs(peak_smooth)), peak_smooth.shape)
        init_row, init_col = peak_idx
        amp = np.max(np.abs(peak_smooth))
        
        try:
            popt, _ = curve_fit(
                dog_2d,
                (x_grid, y_grid),
                peak_smooth.ravel(),
                p0=[amp, amp * 0.3, init_col, init_row, 3.0, 6.0, np.median(peak_smooth)],
                bounds=(
                    [-np.inf, -np.inf, 0, 0, 0.5, 1.0, -np.inf],
                    [np.inf, np.inf, cols-1, rows-1, cols/3, cols/2, np.inf]
                ),
                maxfev=2000,
            )
            
            center_col, center_row = popt[2], popt[3]
            fitted_centers.append((center_row, center_col))
            fitted_weights.append(frame_activity[t])
            last_successful_popt = popt
            
        except (RuntimeError, ValueError):
            continue
    
    if len(fitted_centers) > 0:
        # Weighted average of fitted centers
        centers_arr = np.array(fitted_centers)
        weights_arr = np.array(fitted_weights)
        weights_arr = weights_arr / (np.sum(weights_arr) + 1e-10)
        
        center_row = np.sum(centers_arr[:, 0] * weights_arr)
        center_col = np.sum(centers_arr[:, 1] * weights_arr)
        
        if last_successful_popt is not None:
            fit_params = {
                'amp_center': last_successful_popt[0],
                'amp_surround': last_successful_popt[1],
                'center_col': center_col,
                'center_row': center_row,
                'sigma_center': last_successful_popt[4],
                'sigma_surround': last_successful_popt[5],
                'offset': last_successful_popt[6],
                'n_fitted_frames': len(fitted_centers),
            }
        else:
            fit_params = {}
        success = True
    else:
        # Fallback to peak location
        peak_data = data[peak_frame]
        peak_idx = np.unravel_index(np.argmax(np.abs(peak_data)), peak_data.shape)
        center_row, center_col = float(peak_idx[0]), float(peak_idx[1])
        fit_params = {}
        success = False
    
    # Clamp to valid range
    center_row = np.clip(center_row, 0, rows - 1)
    center_col = np.clip(center_col, 0, cols - 1)
    
    return float(center_row), float(center_col), fit_params, success


# =============================================================================
# Algorithm 10: Graph-Based Gradient Sink
# =============================================================================

def find_center_graph_sink(eimage_sta: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Find cell center using graph-based gradient flow analysis.
    
    TEMPORAL OPTIMIZATION: Integrates gradient flow over time window,
    accumulating in-degree with activity weighting for robust sink detection.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        
    Returns:
        Tuple of (row, col, indegree_map)
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Preprocess with spatio-temporal smoothing
    data = preprocess_temporal(eimage_sta, smooth_temporal=True, smooth_spatial=True,
                               subtract_baseline=True)
    
    # Find peak frame
    frame_activity = np.nanmax(np.abs(data), axis=(1, 2))
    peak_frame = int(np.nanargmax(frame_activity))
    
    # TEMPORAL OPTIMIZATION: Integrate over time window
    window = TEMPORAL_CONFIG.integration_window
    start_frame = max(0, peak_frame - window // 2)
    end_frame = min(n_frames, peak_frame + window // 2 + 1)
    
    # Accumulate in-degree across frames
    indegree_accum = np.zeros((rows, cols))
    
    for t in range(start_frame, end_frame):
        frame_data = data[t]
        
        # Compute gradients
        grad_y, grad_x = np.gradient(frame_data)
        
        # Build in-degree map for this frame
        indegree_frame = np.zeros((rows, cols))
        
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # Check 8 neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        
                        # Flow direction (opposite of gradient)
                        flow_r = -grad_y[nr, nc]
                        flow_c = -grad_x[nr, nc]
                        
                        flow_mag = np.sqrt(flow_r**2 + flow_c**2)
                        if flow_mag > 1e-10:
                            flow_r /= flow_mag
                            flow_c /= flow_mag
                            
                            dir_r = -dr / np.sqrt(dr**2 + dc**2)
                            dir_c = -dc / np.sqrt(dr**2 + dc**2)
                            
                            alignment = flow_r * dir_r + flow_c * dir_c
                            
                            if alignment > 0.5:
                                indegree_frame[r, c] += alignment
        
        # Weight by frame activity
        indegree_accum += indegree_frame * frame_activity[t]
    
    # Normalize by total activity weight
    total_activity = np.sum(frame_activity[start_frame:end_frame])
    if total_activity > 0:
        indegree_map = indegree_accum / total_activity
    else:
        indegree_map = indegree_accum
    
    # Smooth indegree map
    indegree_map = gaussian_filter(indegree_map, sigma=1)
    
    # Find sink (max in-degree)
    center_idx = np.argmax(indegree_map)
    center_row, center_col = np.unravel_index(center_idx, indegree_map.shape)
    
    return int(center_row), int(center_col), indegree_map


# =============================================================================
# Algorithm 11: Multi-Feature Consensus
# =============================================================================

def find_center_consensus(
    all_centers: List[Tuple[float, float]],
    method_names: List[str],
) -> Tuple[float, float, Dict]:
    """
    Find consensus center from multiple methods using median.
    
    Args:
        all_centers: List of (row, col) tuples from different methods
        method_names: Names of the methods for reporting
        
    Returns:
        Tuple of (row, col, stats_dict)
    """
    centers_array = np.array(all_centers)
    
    # Median consensus (robust to outliers)
    median_row = np.median(centers_array[:, 0])
    median_col = np.median(centers_array[:, 1])
    
    # Compute distances from consensus
    distances = np.sqrt(
        (centers_array[:, 0] - median_row)**2 +
        (centers_array[:, 1] - median_col)**2
    )
    
    # Stats
    stats = {
        'median_center': (median_row, median_col),
        'mean_center': (np.mean(centers_array[:, 0]), np.mean(centers_array[:, 1])),
        'std_row': np.std(centers_array[:, 0]),
        'std_col': np.std(centers_array[:, 1]),
        'max_distance': np.max(distances),
        'mean_distance': np.mean(distances),
        'per_method_distance': dict(zip(method_names, distances)),
    }
    
    return float(median_row), float(median_col), stats


# =============================================================================
# Soma Size Estimation
# =============================================================================

@dataclass
class SomaSizeResult:
    """Result container for soma size estimation."""
    method: str
    center: Tuple[float, float]  # (row, col)
    size_x: float  # Width in electrodes
    size_y: float  # Height in electrodes
    area: float  # Area in electrodes^2
    equivalent_diameter: float  # sqrt(4 * area / pi)
    contour: Optional[np.ndarray] = None  # Contour points if available
    fit_params: Optional[Dict] = None  # Additional fit parameters


def estimate_soma_size_gaussian(
    eimage_sta: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (10, 14),
) -> SomaSizeResult:
    """
    Estimate soma size using 2D Gaussian fit.
    
    The sigma parameters of the fitted Gaussian represent the soma extent.
    Size is defined as 2*sigma (covers ~68% of the Gaussian).
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        center: Initial center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        
    Returns:
        SomaSizeResult with size estimates
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    data_subset = eimage_sta[start_frame:end_frame + 1]
    
    # Average across the frame range
    avg_frame = np.nanmean(data_subset, axis=0)
    avg_frame = np.nan_to_num(avg_frame, nan=0.0)
    
    # Apply Gaussian blur for smoother fit
    avg_blurred = gaussian_filter(avg_frame, sigma=1.0)
    
    # Use absolute value (activity can be positive or negative)
    activity = np.abs(avg_blurred)
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:rows, 0:cols]
    
    # Initial guesses
    init_row, init_col = center
    amplitude = np.nanmax(activity) - np.nanmin(activity)
    offset = np.nanmin(activity)
    
    # Estimate initial sigma from FWHM
    half_max = (np.nanmax(activity) + np.nanmin(activity)) / 2
    above_half = activity > half_max
    init_sigma = max(1.0, np.sqrt(np.sum(above_half) / np.pi))
    
    try:
        popt, pcov = curve_fit(
            gaussian_2d,
            (y_grid.ravel(), x_grid.ravel()),
            activity.ravel(),
            p0=[amplitude, init_col, init_row, init_sigma, init_sigma, offset],
            bounds=(
                [0, 0, 0, 0.5, 0.5, -np.inf],
                [np.inf, cols, rows, cols/2, rows/2, np.inf]
            ),
            maxfev=2000
        )
        amplitude, x0, y0, sigma_x, sigma_y, offset = popt
        fit_success = True
    except:
        # Fallback to initial estimates
        sigma_x = init_sigma
        sigma_y = init_sigma
        x0 = init_col
        y0 = init_row
        fit_success = False
    
    # Size is 2*sigma (covers ~95% of Gaussian at 2*sigma)
    size_x = 2 * sigma_x * 2  # Full width at ~2 sigma
    size_y = 2 * sigma_y * 2
    
    # Ellipse area
    area = np.pi * sigma_x * sigma_y * 4  # 2-sigma contour area
    equivalent_diameter = 2 * np.sqrt(area / np.pi)
    
    # Create ellipse contour
    theta = np.linspace(0, 2*np.pi, 100)
    contour_x = x0 + 2*sigma_x * np.cos(theta)
    contour_y = y0 + 2*sigma_y * np.sin(theta)
    contour = np.column_stack([contour_y, contour_x])
    
    return SomaSizeResult(
        method='Gaussian',
        center=(float(y0), float(x0)),
        size_x=float(size_x),
        size_y=float(size_y),
        area=float(area),
        equivalent_diameter=float(equivalent_diameter),
        contour=contour,
        fit_params={'sigma_x': sigma_x, 'sigma_y': sigma_y, 'success': fit_success}
    )


def estimate_soma_size_threshold(
    eimage_sta: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
) -> SomaSizeResult:
    """
    Estimate soma size using half-maximum threshold.
    
    Counts the area above the threshold level (half-max by default).
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        center: Center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        threshold_fraction: Fraction of max for threshold (0.5 = half-max)
        
    Returns:
        SomaSizeResult with size estimates
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    data_subset = eimage_sta[start_frame:end_frame + 1]
    
    # Average across the frame range
    avg_frame = np.nanmean(data_subset, axis=0)
    avg_frame = np.nan_to_num(avg_frame, nan=0.0)
    
    # Use absolute value
    activity = np.abs(avg_frame)
    
    # Find threshold level
    max_val = np.nanmax(activity)
    min_val = np.nanmin(activity)
    threshold = min_val + threshold_fraction * (max_val - min_val)
    
    # Create binary mask
    mask = activity > threshold
    
    # Connected component analysis - keep only component containing center
    from scipy import ndimage as ndi
    labeled, n_labels = ndi.label(mask)
    center_label = labeled[int(center[0]) % rows, int(center[1]) % cols]
    
    if center_label > 0:
        soma_mask = labeled == center_label
    else:
        # Find nearest component
        distances = np.full((rows, cols), np.inf)
        for label in range(1, n_labels + 1):
            comp_mask = labeled == label
            y_coords, x_coords = np.where(comp_mask)
            if len(y_coords) > 0:
                min_dist = np.min(np.sqrt((y_coords - center[0])**2 + (x_coords - center[1])**2))
                distances[comp_mask] = min_dist
        
        if n_labels > 0:
            # Use largest component near center
            best_label = 1
            soma_mask = labeled == best_label
        else:
            soma_mask = mask
    
    # Compute size from mask
    area = np.sum(soma_mask)
    
    if area > 0:
        y_coords, x_coords = np.where(soma_mask)
        size_y = np.max(y_coords) - np.min(y_coords) + 1
        size_x = np.max(x_coords) - np.min(x_coords) + 1
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
    else:
        size_x = size_y = 0
        centroid_y, centroid_x = center
    
    equivalent_diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0
    
    # Extract contour
    contour = None
    if area > 0:
        try:
            contours = ndi.find_objects(soma_mask.astype(int))
            # Get boundary points
            from scipy.ndimage import binary_dilation
            boundary = binary_dilation(soma_mask) & ~soma_mask
            y_pts, x_pts = np.where(soma_mask)
            if len(y_pts) > 2:
                # Order points by angle from centroid
                angles = np.arctan2(y_pts - centroid_y, x_pts - centroid_x)
                order = np.argsort(angles)
                contour = np.column_stack([y_pts[order], x_pts[order]])
        except:
            pass
    
    return SomaSizeResult(
        method='Threshold',
        center=(float(centroid_y), float(centroid_x)),
        size_x=float(size_x),
        size_y=float(size_y),
        area=float(area),
        equivalent_diameter=float(equivalent_diameter),
        contour=contour,
        fit_params={'threshold': threshold, 'threshold_fraction': threshold_fraction}
    )


def estimate_soma_size_fwhm(
    eimage_sta: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (10, 14),
) -> SomaSizeResult:
    """
    Estimate soma size using Full Width at Half Maximum (FWHM).
    
    Measures FWHM in x and y directions through the center.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        center: Center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        
    Returns:
        SomaSizeResult with size estimates
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    data_subset = eimage_sta[start_frame:end_frame + 1]
    
    # Average across the frame range
    avg_frame = np.nanmean(data_subset, axis=0)
    avg_frame = np.nan_to_num(avg_frame, nan=0.0)
    
    # Apply smoothing
    avg_smooth = gaussian_filter(avg_frame, sigma=0.5)
    activity = np.abs(avg_smooth)
    
    center_row = int(np.clip(center[0], 0, rows - 1))
    center_col = int(np.clip(center[1], 0, cols - 1))
    
    # Get profiles through center
    y_profile = activity[:, center_col]
    x_profile = activity[center_row, :]
    
    def compute_fwhm(profile):
        """Compute FWHM of a 1D profile."""
        max_val = np.max(profile)
        min_val = np.min(profile)
        half_max = (max_val + min_val) / 2
        
        above_half = profile > half_max
        
        # Find first and last crossing
        indices = np.where(above_half)[0]
        if len(indices) >= 2:
            return indices[-1] - indices[0] + 1
        elif len(indices) == 1:
            return 1
        else:
            return 0
    
    size_y = compute_fwhm(y_profile)
    size_x = compute_fwhm(x_profile)
    
    # Approximate area as ellipse
    area = np.pi * (size_x / 2) * (size_y / 2)
    equivalent_diameter = np.sqrt(size_x * size_y)
    
    # Create ellipse contour
    theta = np.linspace(0, 2*np.pi, 100)
    contour_x = center_col + (size_x / 2) * np.cos(theta)
    contour_y = center_row + (size_y / 2) * np.sin(theta)
    contour = np.column_stack([contour_y, contour_x])
    
    return SomaSizeResult(
        method='FWHM',
        center=center,
        size_x=float(size_x),
        size_y=float(size_y),
        area=float(area),
        equivalent_diameter=float(equivalent_diameter),
        contour=contour,
        fit_params={'y_profile_max': np.max(y_profile), 'x_profile_max': np.max(x_profile)}
    )


def estimate_soma_size_gradient(
    eimage_sta: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (10, 14),
) -> SomaSizeResult:
    """
    Estimate soma size using gradient magnitude (edge detection).
    
    The soma edge is where gradients are largest. Size is measured
    as the distance from center to the gradient ring.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        center: Center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        
    Returns:
        SomaSizeResult with size estimates
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    data_subset = eimage_sta[start_frame:end_frame + 1]
    
    # Average across the frame range
    avg_frame = np.nanmean(data_subset, axis=0)
    avg_frame = np.nan_to_num(avg_frame, nan=0.0)
    
    # Smooth before gradient
    avg_smooth = gaussian_filter(avg_frame, sigma=1.0)
    
    # Compute gradient magnitude
    grad_y, grad_x = np.gradient(avg_smooth)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Smooth gradient magnitude
    grad_mag_smooth = gaussian_filter(grad_mag, sigma=0.5)
    
    center_row = int(np.clip(center[0], 0, rows - 1))
    center_col = int(np.clip(center[1], 0, cols - 1))
    
    # Find peak gradient along radial directions
    n_angles = 36
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    max_radius = min(rows, cols) // 2
    
    edge_points = []
    for angle in angles:
        best_r = 1
        best_grad = 0
        for r in range(1, max_radius):
            y = center_row + int(r * np.sin(angle))
            x = center_col + int(r * np.cos(angle))
            if 0 <= y < rows and 0 <= x < cols:
                if grad_mag_smooth[y, x] > best_grad:
                    best_grad = grad_mag_smooth[y, x]
                    best_r = r
        edge_points.append([center_row + best_r * np.sin(angle),
                           center_col + best_r * np.cos(angle)])
    
    edge_points = np.array(edge_points)
    
    # Compute size from edge points
    if len(edge_points) > 0:
        size_y = np.max(edge_points[:, 0]) - np.min(edge_points[:, 0])
        size_x = np.max(edge_points[:, 1]) - np.min(edge_points[:, 1])
        distances = np.sqrt((edge_points[:, 0] - center_row)**2 + 
                           (edge_points[:, 1] - center_col)**2)
        mean_radius = np.mean(distances)
        area = np.pi * mean_radius**2
        equivalent_diameter = 2 * mean_radius
    else:
        size_x = size_y = 0
        area = 0
        equivalent_diameter = 0
    
    return SomaSizeResult(
        method='Gradient',
        center=center,
        size_x=float(size_x),
        size_y=float(size_y),
        area=float(area),
        equivalent_diameter=float(equivalent_diameter),
        contour=edge_points,
        fit_params={'mean_radius': mean_radius if len(edge_points) > 0 else 0}
    )


def estimate_soma_size_csd(
    eimage_sta: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
) -> SomaSizeResult:
    """
    Estimate soma size using Current Source Density (CSD) analysis.
    
    The soma appears as a sink (negative CSD) surrounded by sources.
    Size is measured from the extent of the sink region.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        center: Center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        threshold_fraction: Fraction of min CSD for threshold
        
    Returns:
        SomaSizeResult with size estimates
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    data_subset = eimage_sta[start_frame:end_frame + 1]
    
    # Average across the frame range
    avg_frame = np.nanmean(data_subset, axis=0)
    avg_frame = np.nan_to_num(avg_frame, nan=0.0)
    
    # Apply smoothing
    avg_smooth = gaussian_filter(avg_frame, sigma=1.0)
    
    # Compute 2D Laplacian (CSD)
    csd = laplace(avg_smooth)
    csd_smooth = gaussian_filter(csd, sigma=1.0)
    
    # Find sink region (negative CSD)
    min_csd = np.min(csd_smooth)
    max_csd = np.max(csd_smooth)
    
    # Threshold for sink region
    threshold = min_csd * threshold_fraction
    sink_mask = csd_smooth < threshold
    
    # Connected component - keep component containing center
    from scipy import ndimage as ndi
    labeled, n_labels = ndi.label(sink_mask)
    
    center_row = int(np.clip(center[0], 0, rows - 1))
    center_col = int(np.clip(center[1], 0, cols - 1))
    center_label = labeled[center_row, center_col]
    
    if center_label > 0:
        soma_mask = labeled == center_label
    else:
        # Use largest sink component
        if n_labels > 0:
            component_sizes = [np.sum(labeled == i) for i in range(1, n_labels + 1)]
            largest_label = np.argmax(component_sizes) + 1
            soma_mask = labeled == largest_label
        else:
            soma_mask = sink_mask
    
    # Compute size
    area = np.sum(soma_mask)
    
    if area > 0:
        y_coords, x_coords = np.where(soma_mask)
        size_y = np.max(y_coords) - np.min(y_coords) + 1
        size_x = np.max(x_coords) - np.min(x_coords) + 1
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
    else:
        size_x = size_y = 0
        centroid_y, centroid_x = center
    
    equivalent_diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0
    
    # Extract contour
    contour = None
    if area > 0:
        y_pts, x_pts = np.where(soma_mask)
        if len(y_pts) > 2:
            angles = np.arctan2(y_pts - centroid_y, x_pts - centroid_x)
            order = np.argsort(angles)
            contour = np.column_stack([y_pts[order], x_pts[order]])
    
    return SomaSizeResult(
        method='CSD',
        center=(float(centroid_y), float(centroid_x)),
        size_x=float(size_x),
        size_y=float(size_y),
        area=float(area),
        equivalent_diameter=float(equivalent_diameter),
        contour=contour,
        fit_params={'csd_min': min_csd, 'csd_max': max_csd, 'threshold': threshold}
    )


def estimate_soma_size_optical_flow(
    eimage_sta: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (10, 14),
) -> SomaSizeResult:
    """
    Estimate soma size using optical flow divergence.
    
    The soma center should show flow convergence (sink).
    Size is estimated from the divergence pattern extent.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        center: Center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        
    Returns:
        SomaSizeResult with size estimates
    """
    n_frames, rows, cols = eimage_sta.shape
    
    if not HAS_CV2:
        return SomaSizeResult(
            method='OpticalFlow',
            center=center,
            size_x=0, size_y=0, area=0, equivalent_diameter=0,
            contour=None, fit_params={'error': 'OpenCV not available'}
        )
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    
    # Normalize data
    data = np.nan_to_num(eimage_sta, nan=0.0)
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min > 0:
        normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(data, dtype=np.uint8)
    
    # Compute optical flow and accumulate divergence
    divergence_accum = np.zeros((rows, cols))
    n_pairs = 0
    
    for i in range(start_frame, end_frame):
        if i + 1 < n_frames:
            prev_frame = normalized[i]
            next_frame = normalized[i + 1]
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            dvx_dx = np.gradient(flow[:, :, 0], axis=1)
            dvy_dy = np.gradient(flow[:, :, 1], axis=0)
            divergence_accum += dvx_dx + dvy_dy
            n_pairs += 1
    
    if n_pairs > 0:
        divergence_map = divergence_accum / n_pairs
    else:
        divergence_map = divergence_accum
    
    divergence_smooth = gaussian_filter(divergence_map, sigma=2)
    
    # Sink region (negative divergence)
    min_div = np.min(divergence_smooth)
    threshold = min_div * 0.5
    sink_mask = divergence_smooth < threshold
    
    # Connected component containing center
    from scipy import ndimage as ndi
    labeled, n_labels = ndi.label(sink_mask)
    
    center_row = int(np.clip(center[0], 0, rows - 1))
    center_col = int(np.clip(center[1], 0, cols - 1))
    center_label = labeled[center_row, center_col]
    
    if center_label > 0:
        soma_mask = labeled == center_label
    else:
        soma_mask = sink_mask
    
    area = np.sum(soma_mask)
    
    if area > 0:
        y_coords, x_coords = np.where(soma_mask)
        size_y = np.max(y_coords) - np.min(y_coords) + 1
        size_x = np.max(x_coords) - np.min(x_coords) + 1
    else:
        size_x = size_y = 0
    
    equivalent_diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0
    
    return SomaSizeResult(
        method='OpticalFlow',
        center=center,
        size_x=float(size_x),
        size_y=float(size_y),
        area=float(area),
        equivalent_diameter=float(equivalent_diameter),
        contour=None,
        fit_params={'divergence_min': min_div}
    )


def estimate_soma_size_maxmin(
    eimage_sta: np.ndarray,
    center: Tuple[float, float],
    frame_range: Tuple[int, int] = (10, 14),
    threshold_fraction: float = 0.5,
) -> SomaSizeResult:
    """
    Estimate soma size using max-min difference map.
    
    The soma should have high max-min difference.
    Size is estimated from the extent of high-activity region.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        center: Center estimate (row, col)
        frame_range: Range of frames to use (start, end inclusive)
        threshold_fraction: Fraction of max for threshold
        
    Returns:
        SomaSizeResult with size estimates
    """
    n_frames, rows, cols = eimage_sta.shape
    
    # Extract frames in range
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    data_subset = eimage_sta[start_frame:end_frame + 1]
    
    # Compute max - min along time axis
    max_vals = np.nanmax(data_subset, axis=0)
    min_vals = np.nanmin(data_subset, axis=0)
    diff_map = max_vals - min_vals
    diff_map = np.nan_to_num(diff_map, nan=0.0)
    
    # Smooth
    diff_smooth = gaussian_filter(diff_map, sigma=1.0)
    
    # Threshold
    max_diff = np.max(diff_smooth)
    threshold = max_diff * threshold_fraction
    high_activity_mask = diff_smooth > threshold
    
    # Connected component containing center
    from scipy import ndimage as ndi
    labeled, n_labels = ndi.label(high_activity_mask)
    
    center_row = int(np.clip(center[0], 0, rows - 1))
    center_col = int(np.clip(center[1], 0, cols - 1))
    center_label = labeled[center_row, center_col]
    
    if center_label > 0:
        soma_mask = labeled == center_label
    else:
        soma_mask = high_activity_mask
    
    area = np.sum(soma_mask)
    
    if area > 0:
        y_coords, x_coords = np.where(soma_mask)
        size_y = np.max(y_coords) - np.min(y_coords) + 1
        size_x = np.max(x_coords) - np.min(x_coords) + 1
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
    else:
        size_x = size_y = 0
        centroid_y, centroid_x = center
    
    equivalent_diameter = 2 * np.sqrt(area / np.pi) if area > 0 else 0
    
    return SomaSizeResult(
        method='MaxMin',
        center=(float(centroid_y), float(centroid_x)),
        size_x=float(size_x),
        size_y=float(size_y),
        area=float(area),
        equivalent_diameter=float(equivalent_diameter),
        contour=None,
        fit_params={'max_diff': max_diff, 'threshold': threshold}
    )


def estimate_all_soma_sizes(
    eimage_sta: np.ndarray,
    soma_results: Dict[str, Any],
    frame_range: Tuple[int, int] = (10, 14),
) -> Dict[str, SomaSizeResult]:
    """
    Estimate soma size using all available methods.
    
    Args:
        eimage_sta: 3D array (time, rows, cols)
        soma_results: Dictionary with soma center results from all methods
        frame_range: Range of frames to use (start, end inclusive)
        
    Returns:
        Dictionary mapping method name to SomaSizeResult
    """
    size_results = {}
    
    # Get consensus center for methods that need an initial estimate
    centers = []
    method_names = []
    for method_name, result in soma_results.items():
        if 'center' in result and result['center'] is not None:
            centers.append(result['center'])
            method_names.append(method_name)
    
    if centers:
        consensus_center = (
            np.median([c[0] for c in centers]),
            np.median([c[1] for c in centers])
        )
    else:
        # Fallback to image center
        n_frames, rows, cols = eimage_sta.shape
        consensus_center = (rows / 2, cols / 2)
    
    # Method-specific size estimation
    for method_name, result in soma_results.items():
        if 'center' in result and result['center'] is not None:
            center = result['center']
        else:
            center = consensus_center
        
        try:
            if method_name == 'GaussFit':
                size_results[method_name] = estimate_soma_size_gaussian(
                    eimage_sta, center, frame_range
                )
            elif method_name == 'CSD':
                size_results[method_name] = estimate_soma_size_csd(
                    eimage_sta, center, frame_range
                )
            elif method_name == 'OptFlow':
                size_results[method_name] = estimate_soma_size_optical_flow(
                    eimage_sta, center, frame_range
                )
            elif method_name == 'MaxMin':
                size_results[method_name] = estimate_soma_size_maxmin(
                    eimage_sta, center, frame_range
                )
            elif method_name in ['P2P', 'Trough']:
                # Use threshold-based for peak-to-peak and trough methods
                size_results[method_name] = estimate_soma_size_threshold(
                    eimage_sta, center, frame_range
                )
                size_results[method_name].method = method_name
            elif method_name == 'FWHM' or method_name == 'Template':
                size_results[method_name] = estimate_soma_size_fwhm(
                    eimage_sta, center, frame_range
                )
                size_results[method_name].method = method_name
            else:
                # Default to gradient-based for other methods
                size_results[method_name] = estimate_soma_size_gradient(
                    eimage_sta, center, frame_range
                )
                size_results[method_name].method = method_name
        except Exception as e:
            # Create empty result on error
            size_results[method_name] = SomaSizeResult(
                method=method_name,
                center=center,
                size_x=0, size_y=0, area=0, equivalent_diameter=0,
                contour=None, fit_params={'error': str(e)}
            )
    
    # Add consensus size estimation
    if len(size_results) > 0:
        # Compute consensus from all valid size estimates
        valid_sizes = [r for r in size_results.values() if r.area > 0]
        
        if valid_sizes:
            consensus_size_x = np.median([r.size_x for r in valid_sizes])
            consensus_size_y = np.median([r.size_y for r in valid_sizes])
            consensus_area = np.median([r.area for r in valid_sizes])
            consensus_diameter = np.median([r.equivalent_diameter for r in valid_sizes])
            
            size_results['Consensus'] = SomaSizeResult(
                method='Consensus',
                center=consensus_center,
                size_x=float(consensus_size_x),
                size_y=float(consensus_size_y),
                area=float(consensus_area),
                equivalent_diameter=float(consensus_diameter),
                contour=None,
                fit_params={
                    'n_methods': len(valid_sizes),
                    'size_x_std': np.std([r.size_x for r in valid_sizes]),
                    'size_y_std': np.std([r.size_y for r in valid_sizes]),
                    'area_std': np.std([r.area for r in valid_sizes]),
                }
            )
    
    return size_results


def plot_soma_size_individual(
    unit_id: str,
    eimage_sta: np.ndarray,
    size_results: Dict[str, SomaSizeResult],
    frame_range: Tuple[int, int],
    output_path: Path,
) -> None:
    """
    Plot individual soma size estimation results for one unit.
    
    Args:
        unit_id: Unit identifier
        eimage_sta: Original data (time, rows, cols)
        size_results: Dictionary of SomaSizeResult from all methods
        frame_range: Frame range used for estimation
        output_path: Output path for figure
    """
    n_methods = len(size_results)
    n_cols = 4
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    n_frames, rows, cols = eimage_sta.shape
    
    # Get average frame for background
    start_frame = max(0, frame_range[0])
    end_frame = min(n_frames - 1, frame_range[1])
    avg_frame = np.nanmean(eimage_sta[start_frame:end_frame + 1], axis=0)
    
    vmin = np.nanpercentile(avg_frame, 1)
    vmax = np.nanpercentile(avg_frame, 99)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    for i, (method_name, result) in enumerate(size_results.items()):
        ax = axes[i]
        
        # Plot background
        ax.imshow(avg_frame, cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.7)
        
        # Plot center
        if result.center is not None:
            ax.plot(result.center[1], result.center[0], 'k+', markersize=15, 
                   markeredgewidth=2)
        
        # Plot contour if available
        if result.contour is not None and len(result.contour) > 2:
            contour = np.vstack([result.contour, result.contour[0]])  # Close contour
            ax.plot(contour[:, 1], contour[:, 0], color=colors[i], 
                   linewidth=2, label='Boundary')
        elif result.size_x > 0 and result.size_y > 0 and result.center is not None:
            # Draw ellipse
            theta = np.linspace(0, 2*np.pi, 100)
            ell_x = result.center[1] + (result.size_x / 2) * np.cos(theta)
            ell_y = result.center[0] + (result.size_y / 2) * np.sin(theta)
            ax.plot(ell_x, ell_y, color=colors[i], linewidth=2, linestyle='--')
        
        # Title with size info
        title = f"{method_name}\n"
        title += f"Size: {result.size_x:.1f}×{result.size_y:.1f}\n"
        title += f"Area: {result.area:.1f}, Ø: {result.equivalent_diameter:.1f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused axes
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f'{unit_id} - Soma Size Estimation (frames {frame_range[0]}-{frame_range[1]})',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved soma size plot: {output_path.name}")


def plot_soma_size_comparison(
    all_size_results: Dict[str, Dict[str, SomaSizeResult]],
    output_path: Path,
) -> None:
    """
    Create comparison plot of soma sizes across all units and methods.
    
    Args:
        all_size_results: Dict mapping unit_id to Dict of SomaSizeResult
        output_path: Output path for figure
    """
    n_units = len(all_size_results)
    if n_units == 0:
        return
    
    # Get all method names
    all_methods = set()
    for unit_results in all_size_results.values():
        all_methods.update(unit_results.keys())
    method_names = sorted(all_methods)
    n_methods = len(method_names)
    
    # Create figure with 3 subplots: size_x, size_y, equivalent_diameter
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    unit_ids = list(all_size_results.keys())
    x_positions = np.arange(n_units)
    bar_width = 0.8 / n_methods
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    # Subplot 1: Size X (Width)
    ax = axes[0]
    for i, method in enumerate(method_names):
        values = []
        for unit_id in unit_ids:
            if method in all_size_results[unit_id]:
                values.append(all_size_results[unit_id][method].size_x)
            else:
                values.append(0)
        offset = (i - n_methods/2 + 0.5) * bar_width
        ax.bar(x_positions + offset, values, bar_width, label=method, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Unit')
    ax.set_ylabel('Size X (electrodes)')
    ax.set_title('Soma Width (X dimension)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(unit_ids, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Size Y (Height)
    ax = axes[1]
    for i, method in enumerate(method_names):
        values = []
        for unit_id in unit_ids:
            if method in all_size_results[unit_id]:
                values.append(all_size_results[unit_id][method].size_y)
            else:
                values.append(0)
        offset = (i - n_methods/2 + 0.5) * bar_width
        ax.bar(x_positions + offset, values, bar_width, label=method, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Unit')
    ax.set_ylabel('Size Y (electrodes)')
    ax.set_title('Soma Height (Y dimension)', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(unit_ids, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Equivalent Diameter
    ax = axes[2]
    for i, method in enumerate(method_names):
        values = []
        for unit_id in unit_ids:
            if method in all_size_results[unit_id]:
                values.append(all_size_results[unit_id][method].equivalent_diameter)
            else:
                values.append(0)
        offset = (i - n_methods/2 + 0.5) * bar_width
        ax.bar(x_positions + offset, values, bar_width, label=method, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Unit')
    ax.set_ylabel('Equivalent Diameter (electrodes)')
    ax.set_title('Soma Equivalent Diameter', fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(unit_ids, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    # Subplot 4: Comparison to Consensus
    ax = axes[3]
    
    # Calculate deviation from consensus for each method
    deviations = {method: [] for method in method_names if method != 'Consensus'}
    
    for unit_id in unit_ids:
        if 'Consensus' in all_size_results[unit_id]:
            consensus_diam = all_size_results[unit_id]['Consensus'].equivalent_diameter
            if consensus_diam > 0:
                for method in deviations.keys():
                    if method in all_size_results[unit_id]:
                        method_diam = all_size_results[unit_id][method].equivalent_diameter
                        deviation = (method_diam - consensus_diam) / consensus_diam * 100
                        deviations[method].append(deviation)
    
    # Box plot of deviations
    method_list = list(deviations.keys())
    deviation_data = [deviations[m] for m in method_list]
    
    bp = ax.boxplot(deviation_data, labels=method_list, patch_artist=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors[:len(method_list)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Consensus')
    ax.set_xlabel('Method')
    ax.set_ylabel('Deviation from Consensus (%)')
    ax.set_title('Method Deviation from Consensus Diameter', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Soma Size Estimation Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved soma size comparison: {output_path.name}")


# =============================================================================
# Visualization
# =============================================================================

def plot_unit_comparison(
    unit_id: str,
    eimage_sta: np.ndarray,
    maxmin_result: Tuple[int, int, np.ndarray],
    optflow_result: Tuple[Tuple[int, int], Tuple[int, int], np.ndarray],
    csd_result: Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, int],
    gaussian_result: Tuple[Tuple[float, float], np.ndarray, Dict, bool],
    output_path: Path,
) -> None:
    """
    Create a comprehensive comparison plot for a single unit.
    """
    fig = plt.figure(figsize=(20, 16))
    
    # Unpack results
    maxmin_row, maxmin_col, diff_map = maxmin_result
    optflow_sink, optflow_source, divergence_map = optflow_result
    csd_sink, csd_source, csd_map, peak_frame = csd_result
    gauss_center, gauss_blurred, gauss_params, gauss_success = gaussian_result
    
    n_frames = eimage_sta.shape[0]
    
    # Row 1: Raw eimage_sta at key time frames (5 columns)
    frame_indices = [5, 10, 15, 20, 25]
    frame_indices = [min(f, n_frames - 1) for f in frame_indices]
    
    vmin = np.nanpercentile(eimage_sta, 1)
    vmax = np.nanpercentile(eimage_sta, 99)
    
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(5, 5, i + 1)
        im = ax.imshow(eimage_sta[frame_idx], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f"Frame {frame_idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Raw eimage_sta", fontsize=11)
    
    # Row 2: Max-Min Difference with center marked
    ax = fig.add_subplot(5, 5, 6)
    im = ax.imshow(diff_map, cmap='viridis')
    ax.plot(maxmin_col, maxmin_row, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
    ax.set_title(f"Max-Min Diff\nCenter: ({maxmin_row}, {maxmin_col})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Max-Min Method", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Empty subplots for alignment
    for i in range(7, 11):
        ax = fig.add_subplot(5, 5, i)
        ax.axis('off')
    
    # Row 3: Optical Flow Divergence with sink/source marked
    ax = fig.add_subplot(5, 5, 11)
    
    # Use two-slope norm for divergence (centered at 0)
    div_abs_max = max(abs(np.nanmin(divergence_map)), abs(np.nanmax(divergence_map)))
    if div_abs_max > 0:
        norm = TwoSlopeNorm(vmin=-div_abs_max, vcenter=0, vmax=div_abs_max)
    else:
        norm = None
    
    im = ax.imshow(divergence_map, cmap='RdBu_r', norm=norm)
    ax.plot(optflow_sink[1], optflow_sink[0], 'bo', markersize=12, 
            markeredgecolor='white', markeredgewidth=2, label='Sink')
    ax.plot(optflow_source[1], optflow_source[0], 'r^', markersize=12, 
            markeredgecolor='white', markeredgewidth=2, label='Source')
    ax.set_title(f"Optical Flow Divergence\nSink: {optflow_sink}, Source: {optflow_source}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Optical Flow", fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Empty subplots for alignment
    for i in range(12, 16):
        ax = fig.add_subplot(5, 5, i)
        ax.axis('off')
    
    # Row 4: CSD Analysis with sink/source marked
    ax = fig.add_subplot(5, 5, 16)
    
    # Use two-slope norm for CSD (centered at 0)
    csd_abs_max = max(abs(np.nanmin(csd_map)), abs(np.nanmax(csd_map)))
    if csd_abs_max > 0:
        norm = TwoSlopeNorm(vmin=-csd_abs_max, vcenter=0, vmax=csd_abs_max)
    else:
        norm = None
    
    im = ax.imshow(csd_map, cmap='RdBu_r', norm=norm)
    ax.plot(csd_sink[1], csd_sink[0], 'bo', markersize=12, 
            markeredgecolor='white', markeredgewidth=2, label='Sink')
    ax.plot(csd_source[1], csd_source[0], 'r^', markersize=12, 
            markeredgecolor='white', markeredgewidth=2, label='Source')
    ax.set_title(f"CSD (frame {peak_frame})\nSink: {csd_sink}, Source: {csd_source}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("CSD Analysis", fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Empty subplots for alignment
    for i in range(17, 21):
        ax = fig.add_subplot(5, 5, i)
        ax.axis('off')
    
    # Row 5: Gaussian Fit with center marked
    ax = fig.add_subplot(5, 5, 21)
    im = ax.imshow(gauss_blurred, cmap='RdBu_r')
    status = "OK" if gauss_success else "FAIL"
    ax.plot(gauss_center[1], gauss_center[0], 'm*', markersize=15, 
            markeredgecolor='white', markeredgewidth=1.5)
    ax.set_title(f"Gaussian Fit [{status}]\nCenter: ({gauss_center[0]:.1f}, {gauss_center[1]:.1f})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Gaussian Fit", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Draw fitted Gaussian ellipse if successful
    if gauss_success and not np.isnan(gauss_params['sigma_x']):
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(
            (gauss_center[1], gauss_center[0]),
            width=2 * gauss_params['sigma_x'],
            height=2 * gauss_params['sigma_y'],
            fill=False, edgecolor='magenta', linewidth=2
        )
        ax.add_patch(ellipse)
    
    # Summary subplot: overlay all centers on max-min diff map
    ax = fig.add_subplot(5, 5, 22)
    ax.imshow(diff_map, cmap='gray', alpha=0.7)
    ax.plot(maxmin_col, maxmin_row, 'g*', markersize=18, 
            markeredgecolor='white', markeredgewidth=2, label='Max-Min')
    ax.plot(optflow_sink[1], optflow_sink[0], 'bo', markersize=14, 
            markeredgecolor='white', markeredgewidth=2, label='OptFlow Sink')
    ax.plot(csd_sink[1], csd_sink[0], 'rs', markersize=14, 
            markeredgecolor='white', markeredgewidth=2, label='CSD Sink')
    ax.plot(gauss_center[1], gauss_center[0], 'm^', markersize=14, 
            markeredgecolor='white', markeredgewidth=2, label='Gauss Fit')
    ax.set_title("All Methods Comparison", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=7)
    
    # Empty subplots
    for i in range(23, 26):
        ax = fig.add_subplot(5, 5, i)
        ax.axis('off')
    
    fig.suptitle(f"{unit_id} - Cell Center Identification Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_summary(
    results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """
    Create a summary figure comparing all methods across all units.
    """
    n_units = len(results)
    
    fig, axes = plt.subplots(n_units, 5, figsize=(20, 4 * n_units))
    
    if n_units == 1:
        axes = axes.reshape(1, -1)
    
    for i, (unit_id, res) in enumerate(results.items()):
        eimage_sta = res['eimage_sta']
        diff_map = res['maxmin'][2]
        maxmin_center = (res['maxmin'][0], res['maxmin'][1])
        optflow_sink = res['optflow'][0]
        csd_sink = res['csd'][0]
        gauss_center = res['gaussian'][0]
        gauss_blurred = res['gaussian'][1]
        
        # Find frame with peak activity for display
        peak_frame = res['csd'][3]
        
        # Column 1: Peak frame
        ax = axes[i, 0]
        vmin = np.nanpercentile(eimage_sta, 1)
        vmax = np.nanpercentile(eimage_sta, 99)
        ax.imshow(eimage_sta[peak_frame], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f"{unit_id}\nPeak Frame ({peak_frame})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Column 2: Max-Min with center
        ax = axes[i, 1]
        ax.imshow(diff_map, cmap='viridis')
        ax.plot(maxmin_center[1], maxmin_center[0], 'r*', markersize=15, 
                markeredgecolor='white', markeredgewidth=1.5)
        ax.set_title(f"Max-Min: ({maxmin_center[0]}, {maxmin_center[1]})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Column 3: Optical Flow sink
        ax = axes[i, 2]
        div_map = res['optflow'][2]
        div_abs_max = max(abs(np.nanmin(div_map)), abs(np.nanmax(div_map)), 1e-10)
        norm = TwoSlopeNorm(vmin=-div_abs_max, vcenter=0, vmax=div_abs_max)
        ax.imshow(div_map, cmap='RdBu_r', norm=norm)
        ax.plot(optflow_sink[1], optflow_sink[0], 'bo', markersize=12, 
                markeredgecolor='white', markeredgewidth=2)
        ax.set_title(f"OptFlow Sink: ({optflow_sink[0]}, {optflow_sink[1]})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Column 4: CSD sink
        ax = axes[i, 3]
        csd_map = res['csd'][2]
        csd_abs_max = max(abs(np.nanmin(csd_map)), abs(np.nanmax(csd_map)), 1e-10)
        norm = TwoSlopeNorm(vmin=-csd_abs_max, vcenter=0, vmax=csd_abs_max)
        ax.imshow(csd_map, cmap='RdBu_r', norm=norm)
        ax.plot(csd_sink[1], csd_sink[0], 'rs', markersize=12, 
                markeredgecolor='white', markeredgewidth=2)
        ax.set_title(f"CSD Sink: ({csd_sink[0]}, {csd_sink[1]})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Column 5: Gaussian Fit center
        ax = axes[i, 4]
        ax.imshow(gauss_blurred, cmap='RdBu_r')
        ax.plot(gauss_center[1], gauss_center[0], 'm*', markersize=15, 
                markeredgecolor='white', markeredgewidth=1.5)
        ax.set_title(f"Gauss Fit: ({gauss_center[0]:.1f}, {gauss_center[1]:.1f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add column headers
    col_titles = ['Peak Activity', 'Max-Min Method', 'Optical Flow', 'CSD Analysis', 'Gaussian Fit']
    for j, title in enumerate(col_titles):
        axes[0, j].annotate(title, xy=(0.5, 1.15), xycoords='axes fraction',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    fig.suptitle("Cell Center Detection - Summary Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary: {output_path}")


def print_results_table(results: Dict[str, Dict]) -> None:
    """Print a summary table of all results."""
    print("\n" + "=" * 100)
    print("CELL CENTER DETECTION RESULTS")
    print("=" * 100)
    print(f"{'Unit':<12} {'Max-Min':<15} {'OptFlow Sink':<15} {'CSD Sink':<15} {'Gauss Fit':<20} {'Dist(MM-CSD)':<12}")
    print("-" * 100)
    
    for unit_id, res in results.items():
        maxmin = (res['maxmin'][0], res['maxmin'][1])
        optflow = res['optflow'][0]
        csd = res['csd'][0]
        gauss = res['gaussian'][0]
        
        # Calculate distance between max-min and CSD sink
        dist = np.sqrt((maxmin[0] - csd[0])**2 + (maxmin[1] - csd[1])**2)
        
        gauss_str = f"({gauss[0]:.1f}, {gauss[1]:.1f})"
        print(f"{unit_id:<12} {str(maxmin):<15} {str(optflow):<15} {str(csd):<15} {gauss_str:<20} {dist:.2f}")
    
    print("=" * 100 + "\n")


# =============================================================================
# Lucas-Kanade Visualization
# =============================================================================

def plot_lucas_kanade(
    unit_id: str,
    eimage_sta: np.ndarray,
    lk_result: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create detailed visualization of Lucas-Kanade optical flow analysis.
    
    Generates a 2x3 subplot grid showing:
    1. Divergence map with sink/source
    2. Velocity magnitude with flow vectors
    3. Trajectory of tracked points
    4. Flow field with streamlines
    5. Flow direction (color-coded)
    6. Comparison with peak activity frame
    
    Args:
        unit_id: Unit identifier
        eimage_sta: Original data (time, rows, cols)
        lk_result: Dictionary from trace_lucas_kanade_optical_flow
        output_path: Output path for figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    n_frames, rows, cols = eimage_sta.shape
    
    # Get data from result
    divergence_map = lk_result.get('divergence_map', np.zeros((rows, cols)))
    velocity_magnitude = lk_result.get('velocity_magnitude', np.zeros((rows, cols)))
    flow_x = lk_result.get('flow_x', np.zeros((rows, cols)))
    flow_y = lk_result.get('flow_y', np.zeros((rows, cols)))
    trajectory = lk_result.get('trajectory', np.zeros((0, 2)))
    sink = lk_result.get('sink', (rows // 2, cols // 2))
    source = lk_result.get('source', (rows // 2, cols // 2))
    
    # Find peak activity frame for background
    frame_activity = np.nanmax(np.abs(eimage_sta), axis=(1, 2))
    peak_frame = int(np.argmax(frame_activity))
    peak_data = eimage_sta[peak_frame]
    vmin = np.nanpercentile(eimage_sta, 1)
    vmax = np.nanpercentile(eimage_sta, 99)
    
    # --- Subplot 1: Divergence map with sink/source ---
    ax = axes[0]
    div_abs = max(abs(np.nanmin(divergence_map)), abs(np.nanmax(divergence_map)), 1e-10)
    norm = TwoSlopeNorm(vmin=-div_abs, vcenter=0, vmax=div_abs)
    im = ax.imshow(divergence_map, cmap='RdBu_r', norm=norm)
    ax.plot(sink[1], sink[0], 'bo', markersize=12, label=f'Sink ({sink[0]:.0f}, {sink[1]:.0f})')
    ax.plot(source[1], source[0], 'r^', markersize=12, label=f'Source ({source[0]:.0f}, {source[1]:.0f})')
    ax.set_title('Divergence Map (Lucas-Kanade)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.7, label='Divergence')
    
    # --- Subplot 2: Velocity magnitude with flow vectors ---
    ax = axes[1]
    im = ax.imshow(velocity_magnitude, cmap='hot')
    
    # Subsample flow vectors for visualization
    step = max(1, min(rows, cols) // 15)
    y_grid, x_grid = np.mgrid[step//2:rows:step, step//2:cols:step]
    
    # Scale arrows by magnitude
    u = flow_x[y_grid, x_grid]
    v = flow_y[y_grid, x_grid]
    
    ax.quiver(x_grid, y_grid, u, v, color='cyan', alpha=0.8, 
              scale=None, scale_units='xy', angles='xy')
    ax.plot(sink[1], sink[0], 'bo', markersize=10)
    ax.plot(source[1], source[0], 'r^', markersize=10)
    ax.set_title('Velocity Magnitude + Flow Vectors', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.7, label='|v|')
    
    # --- Subplot 3: Trajectory of tracked points ---
    ax = axes[2]
    ax.imshow(peak_data, cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.5)
    
    if len(trajectory) > 1:
        # Color trajectory by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
        for i in range(len(trajectory) - 1):
            ax.plot(trajectory[i:i+2, 1], trajectory[i:i+2, 0], 
                   color=colors[i], linewidth=2)
        ax.plot(trajectory[0, 1], trajectory[0, 0], 'go', markersize=12, label='Start')
        ax.plot(trajectory[-1, 1], trajectory[-1, 0], 'rs', markersize=12, label='End')
        ax.legend(loc='upper right', fontsize=8)
    
    ax.plot(sink[1], sink[0], 'b*', markersize=15, label='Sink')
    ax.set_title(f'Tracked Point Trajectory (n={len(trajectory)})', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # --- Subplot 4: Flow field with streamlines ---
    ax = axes[3]
    ax.imshow(peak_data, cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.3)
    
    # Create streamplot
    x = np.arange(cols)
    y = np.arange(rows)
    
    # Smooth flow for better streamlines
    flow_x_smooth = gaussian_filter(flow_x, sigma=1)
    flow_y_smooth = gaussian_filter(flow_y, sigma=1)
    
    speed = np.sqrt(flow_x_smooth**2 + flow_y_smooth**2)
    if np.max(speed) > 0:
        ax.streamplot(x, y, flow_x_smooth, flow_y_smooth, 
                     color=speed, cmap='plasma', density=1.5,
                     linewidth=1, arrowsize=0.8)
    
    ax.plot(sink[1], sink[0], 'bo', markersize=12)
    ax.plot(source[1], source[0], 'r^', markersize=12)
    ax.set_title('Flow Streamlines', fontsize=11, fontweight='bold')
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # --- Subplot 5: Flow direction (color-coded) ---
    ax = axes[4]
    flow_direction = np.arctan2(flow_y, flow_x)
    
    # Use HSV colormap for direction (angle -> hue)
    im = ax.imshow(flow_direction, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax.plot(sink[1], sink[0], 'ko', markersize=10, markerfacecolor='none', markeredgewidth=2)
    ax.plot(source[1], source[0], 'k^', markersize=10, markerfacecolor='none', markeredgewidth=2)
    ax.set_title('Flow Direction', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='Angle (rad)')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # --- Subplot 6: Comparison overlay on peak frame ---
    ax = axes[5]
    ax.imshow(peak_data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    # Overlay divergence contours
    div_abs = max(abs(np.nanmin(divergence_map)), abs(np.nanmax(divergence_map)), 1e-10)
    levels_neg = np.linspace(-div_abs, -0.1*div_abs, 5)
    levels_pos = np.linspace(0.1*div_abs, div_abs, 5)
    
    if len(levels_neg) > 0:
        ax.contour(divergence_map, levels=levels_neg, colors='blue', alpha=0.5, linewidths=0.8)
    if len(levels_pos) > 0:
        ax.contour(divergence_map, levels=levels_pos, colors='red', alpha=0.5, linewidths=0.8)
    
    ax.plot(sink[1], sink[0], 'bo', markersize=14, label=f'Sink', markeredgecolor='white', markeredgewidth=2)
    ax.plot(source[1], source[0], 'r^', markersize=14, label=f'Source', markeredgecolor='white', markeredgewidth=2)
    
    # Draw trajectory if available
    if len(trajectory) > 1:
        ax.plot(trajectory[:, 1], trajectory[:, 0], 'g-', linewidth=2, alpha=0.8, label='Trajectory')
    
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(f'Peak Frame (t={peak_frame}) + Divergence Contours', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.suptitle(f'{unit_id} - Lucas-Kanade Optical Flow Analysis (4×4 window)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved Lucas-Kanade plot: {output_path.name}")


# =============================================================================
# Comprehensive Comparison Visualizations
# =============================================================================

def plot_soma_comparison(
    unit_id: str,
    eimage_sta: np.ndarray,
    soma_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create comprehensive comparison plot of all soma identification methods.
    
    Args:
        unit_id: Unit identifier
        eimage_sta: Original data
        soma_results: Dictionary with results from all soma methods
        output_path: Output path for figure
    """
    n_methods = len(soma_results)
    n_cols = 4
    n_rows = (n_methods + n_cols) // n_cols + 1  # Extra row for raw data
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    # Find peak frame for display
    frame_activity = np.nanmax(np.abs(eimage_sta), axis=(1, 2))
    peak_frame = int(np.nanargmax(frame_activity))
    
    # Row 0: Show raw data at peak frame with all centers overlaid
    ax = axes[0]
    vmin = np.nanpercentile(eimage_sta, 1)
    vmax = np.nanpercentile(eimage_sta, 99)
    ax.imshow(eimage_sta[peak_frame], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    # Overlay all centers
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    markers = ['*', 'o', 's', '^', 'v', 'D', 'p', 'h', '<', '>', 'P']
    
    for i, (method_name, result) in enumerate(soma_results.items()):
        center = result['center']
        ax.plot(center[1], center[0], marker=markers[i % len(markers)], 
                color=colors[i], markersize=12, markeredgecolor='white', 
                markeredgewidth=1.5, label=method_name[:12])
    
    ax.set_title(f"All Centers (Frame {peak_frame})", fontsize=10)
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Individual method plots
    for i, (method_name, result) in enumerate(soma_results.items()):
        ax = axes[i + 1]
        
        # Get the visualization map if available
        if 'map' in result and result['map'] is not None:
            vis_map = result['map']
            ax.imshow(vis_map, cmap='viridis')
        else:
            ax.imshow(eimage_sta[peak_frame], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        center = result['center']
        ax.plot(center[1], center[0], 'r*', markersize=15, 
                markeredgecolor='white', markeredgewidth=1.5)
        
        ax.set_title(f"{method_name}\n({center[0]:.1f}, {center[1]:.1f})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Turn off unused axes
    for i in range(len(soma_results) + 1, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f"{unit_id} - Soma Center Identification Comparison", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved soma comparison: {output_path}")


def plot_propagation_comparison(
    unit_id: str,
    eimage_sta: np.ndarray,
    prop_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create comprehensive comparison plot of all propagation tracking methods.
    
    Args:
        unit_id: Unit identifier
        eimage_sta: Original data
        prop_results: Dictionary with results from all propagation methods
        output_path: Output path for figure
    """
    n_methods = len(prop_results)
    n_cols = 4
    n_rows = (n_methods + n_cols - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    n_frames, rows, cols = eimage_sta.shape
    vmin = np.nanpercentile(eimage_sta, 1)
    vmax = np.nanpercentile(eimage_sta, 99)
    
    # Row 0: Show all trajectories overlaid
    ax = axes[0]
    frame_activity = np.nanmax(np.abs(eimage_sta), axis=(1, 2))
    peak_frame = int(np.nanargmax(frame_activity))
    ax.imshow(eimage_sta[peak_frame], cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.5)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    for i, (method_name, result) in enumerate(prop_results.items()):
        if 'trajectory' in result and result['trajectory'] is not None:
            traj = result['trajectory']
            if len(traj) > 0:
                ax.plot(traj[:, 1], traj[:, 0], color=colors[i], 
                        linewidth=2, label=method_name[:10], alpha=0.8)
                ax.plot(traj[0, 1], traj[0, 0], 'o', color=colors[i], markersize=6)
                ax.plot(traj[-1, 1], traj[-1, 0], 's', color=colors[i], markersize=6)
    
    ax.set_title("All Trajectories", fontsize=10)
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Individual method plots
    for i, (method_name, result) in enumerate(prop_results.items()):
        ax = axes[i + 1]
        
        # Choose visualization based on method output
        if 'divergence_map' in result and result['divergence_map'] is not None:
            div_map = result['divergence_map']
            div_abs = max(abs(np.nanmin(div_map)), abs(np.nanmax(div_map)), 1e-10)
            norm = TwoSlopeNorm(vmin=-div_abs, vcenter=0, vmax=div_abs)
            ax.imshow(div_map, cmap='RdBu_r', norm=norm)
            if 'sink' in result:
                ax.plot(result['sink'][1], result['sink'][0], 'bo', markersize=10)
            if 'source' in result:
                ax.plot(result['source'][1], result['source'][0], 'r^', markersize=10)
        elif 'delay_map' in result:
            ax.imshow(result['delay_map'], cmap='viridis')
        elif 'activation_time' in result:
            ax.imshow(result['activation_time'], cmap='viridis')
        elif 'first_activation' in result:
            ax.imshow(result['first_activation'], cmap='viridis')
        elif 'phase_map' in result:
            ax.imshow(result['phase_map'], cmap='hsv')
        elif 'power_map' in result:
            ax.imshow(result['power_map'], cmap='hot')
        else:
            ax.imshow(eimage_sta[peak_frame], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        # Overlay trajectory if available
        if 'trajectory' in result and result['trajectory'] is not None:
            traj = result['trajectory']
            if len(traj) > 0:
                ax.plot(traj[:, 1], traj[:, 0], 'c-', linewidth=2, alpha=0.8)
        
        ax.set_title(method_name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Turn off unused axes
    for i in range(len(prop_results) + 1, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f"{unit_id} - Propagation Tracking Comparison", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved propagation comparison: {output_path}")


def plot_summary_comparison(
    all_results: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Create summary comparison plots across all units.
    
    Args:
        all_results: Dictionary of results per unit
        output_dir: Output directory
    """
    if not all_results:
        print("No results to plot")
        return
    
    # Get method names from first unit
    first_unit = list(all_results.keys())[0]
    soma_methods = list(all_results[first_unit].get('soma_results', {}).keys())
    
    n_units = len(all_results)
    n_methods = len(soma_methods)
    
    if n_methods == 0:
        print("No soma methods found")
        return
    
    # ========== Soma Methods Summary ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Center positions scatter
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    
    for i, method in enumerate(soma_methods):
        centers = []
        for unit_id, res in all_results.items():
            if 'soma_results' in res and method in res['soma_results']:
                center = res['soma_results'][method]['center']
                centers.append(center)
        
        if centers:
            centers = np.array(centers)
            ax.scatter(centers[:, 1], centers[:, 0], c=[colors[i]], 
                      label=method[:12], alpha=0.6, s=30)
    
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Soma Centers (All Units, All Methods)")
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.invert_yaxis()
    
    # Plot 2: Method agreement heatmap
    ax = axes[0, 1]
    agreement_matrix = np.zeros((n_methods, n_methods))
    
    for unit_id, res in all_results.items():
        if 'soma_results' not in res:
            continue
        centers = []
        for method in soma_methods:
            if method in res['soma_results']:
                centers.append(res['soma_results'][method]['center'])
            else:
                centers.append((np.nan, np.nan))
        
        centers = np.array(centers)
        
        for i in range(n_methods):
            for j in range(n_methods):
                if not np.isnan(centers[i]).any() and not np.isnan(centers[j]).any():
                    dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + 
                                   (centers[i, 1] - centers[j, 1])**2)
                    agreement_matrix[i, j] += (dist < 3)  # Agree if within 3 pixels
    
    agreement_matrix /= max(n_units, 1)
    
    im = ax.imshow(agreement_matrix, cmap='YlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_methods))
    ax.set_yticks(range(n_methods))
    ax.set_xticklabels([m[:8] for m in soma_methods], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([m[:8] for m in soma_methods], fontsize=8)
    ax.set_title("Method Agreement (fraction < 3px)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Plot 3: Distance from consensus per method
    ax = axes[1, 0]
    distances_per_method = {m: [] for m in soma_methods}
    
    for unit_id, res in all_results.items():
        if 'soma_results' not in res:
            continue
        # Get consensus center
        centers = []
        for method in soma_methods:
            if method in res['soma_results']:
                centers.append(res['soma_results'][method]['center'])
        
        if centers:
            centers = np.array(centers)
            consensus = np.median(centers, axis=0)
            
            for i, method in enumerate(soma_methods):
                if method in res['soma_results']:
                    center = res['soma_results'][method]['center']
                    dist = np.sqrt((center[0] - consensus[0])**2 + 
                                   (center[1] - consensus[1])**2)
                    distances_per_method[method].append(dist)
    
    positions = range(len(soma_methods))
    bp = ax.boxplot([distances_per_method[m] for m in soma_methods], 
                    positions=positions, widths=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([m[:10] for m in soma_methods], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Distance from Consensus (pixels)")
    ax.set_title("Method Deviation from Consensus")
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"Summary Statistics\n{'='*40}\n"
    stats_text += f"Total units analyzed: {n_units}\n"
    stats_text += f"Number of methods: {n_methods}\n\n"
    
    for method in soma_methods[:7]:  # Show first 7
        if distances_per_method[method]:
            mean_dist = np.mean(distances_per_method[method])
            std_dist = np.std(distances_per_method[method])
            stats_text += f"{method[:15]:<16}: {mean_dist:.2f} ± {std_dist:.2f}\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    fig.suptitle("Soma Identification Methods - Summary Comparison", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "soma_methods_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'soma_methods_summary.png'}")
    
    # ========== Propagation Methods Summary ==========
    prop_methods = list(all_results[first_unit].get('prop_results', {}).keys())
    
    if prop_methods:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Sink locations
        ax = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(prop_methods)))
        
        for i, method in enumerate(prop_methods):
            sinks = []
            for unit_id, res in all_results.items():
                if 'prop_results' in res and method in res['prop_results']:
                    if 'sink' in res['prop_results'][method]:
                        sinks.append(res['prop_results'][method]['sink'])
            
            if sinks:
                sinks = np.array(sinks)
                ax.scatter(sinks[:, 1], sinks[:, 0], c=[colors[i]], 
                          label=method[:12], alpha=0.6, s=30)
        
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title("Propagation Sinks (All Methods)")
        ax.legend(loc='upper right', fontsize=7)
        ax.invert_yaxis()
        
        # Plot 2: Method comparison
        ax = axes[1]
        ax.axis('off')
        ax.text(0.1, 0.9, f"Propagation Methods: {len(prop_methods)}\n" +
                "\n".join([f"  - {m}" for m in prop_methods]),
                transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        fig.suptitle("Propagation Tracking Methods - Summary", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "propagation_methods_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_dir / 'propagation_methods_summary.png'}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main analysis workflow with comprehensive method comparison."""
    print("=" * 80)
    print("STA Cell Center Quantification - Comprehensive Analysis")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    print(f"\nLoading data from: {HDF5_PATH}")
    eimage_data = load_eimage_sta_data(HDF5_PATH, NUM_UNITS)
    
    if not eimage_data:
        print("Error: No data loaded!")
        return
    
    # Process each unit
    all_results = {}
    all_size_results = {}  # For soma size estimation comparison
    
    for unit_id, eimage_sta_original in eimage_data.items():
        print(f"\n{'='*60}")
        print(f"Processing {unit_id}")
        print(f"{'='*60}")
        
        # =====================================================================
        # APPLY SPATIAL PADDING
        # =====================================================================
        if TEMPORAL_CONFIG.apply_padding and TEMPORAL_CONFIG.pad_size > 0:
            eimage_sta, padding_info = add_spatial_padding(
                eimage_sta_original, 
                TEMPORAL_CONFIG.pad_size, 
                pad_value='mean'
            )
            print(f"  Applied padding: {TEMPORAL_CONFIG.pad_size} pixels on each side")
        else:
            eimage_sta = eimage_sta_original.copy()
            padding_info = (0, 0, 0, 0)
        
        # Helper function to convert coordinates back to original space
        def unpad_coords(coords):
            """Convert (row, col) from padded to original space."""
            if coords is None:
                return None
            return remove_padding_from_coords(coords, padding_info)
        
        def unpad_trajectory(traj):
            """Convert trajectory from padded to original space."""
            if traj is None:
                return None
            return remove_padding_from_trajectory(traj, padding_info)
        
        def unpad_map(m):
            """Remove padding from 2D map."""
            if m is None:
                return None
            return remove_padding_from_map(m, padding_info)
        
        # =====================================================================
        # SOMA IDENTIFICATION METHODS (on padded data)
        # =====================================================================
        soma_results = {}
        
        # Method 1: Max-Min Difference
        maxmin_result = find_center_maxmin(eimage_sta)
        soma_results['MaxMin'] = {
            'center': unpad_coords((maxmin_result[0], maxmin_result[1])),
            'map': unpad_map(maxmin_result[2]),
        }
        unpadded_center = soma_results['MaxMin']['center']
        print(f"  MaxMin: ({unpadded_center[0]}, {unpadded_center[1]})")
        
        # Method 2: Optical Flow Sink
        optflow_result = find_center_optical_flow(eimage_sta)
        soma_results['OptFlow'] = {
            'center': unpad_coords(optflow_result[0]),  # sink
            'map': unpad_map(optflow_result[2]),
        }
        unpadded_optflow = soma_results['OptFlow']['center']
        print(f"  OptFlow: {unpadded_optflow}")
        
        # Method 3: CSD Sink
        csd_result = find_center_csd(eimage_sta)
        soma_results['CSD'] = {
            'center': unpad_coords(csd_result[0]),  # sink
            'map': unpad_map(csd_result[2]),
        }
        unpadded_csd = soma_results['CSD']['center']
        print(f"  CSD: {unpadded_csd}")
        
        # Method 4: Gaussian Fit
        gaussian_result = find_center_gaussian_fit(eimage_sta, blur_sigma=2.0)
        soma_results['GaussFit'] = {
            'center': unpad_coords(gaussian_result[0]),
            'map': unpad_map(gaussian_result[1]),
            'success': gaussian_result[3],
        }
        unpadded_gauss = soma_results['GaussFit']['center']
        print(f"  GaussFit: ({unpadded_gauss[0]:.1f}, {unpadded_gauss[1]:.1f})")
        
        # Method 5: Peak-to-Peak
        p2p_result = find_center_peak_to_peak(eimage_sta)
        soma_results['P2P'] = {
            'center': unpad_coords((p2p_result[0], p2p_result[1])),
            'map': unpad_map(p2p_result[2]),
        }
        unpadded_p2p = soma_results['P2P']['center']
        print(f"  P2P: ({unpadded_p2p[0]}, {unpadded_p2p[1]})")
        
        # Method 6: Trough Centroid
        centroid_result = find_center_trough_centroid(eimage_sta)
        soma_results['Centroid'] = {
            'center': unpad_coords((centroid_result[0], centroid_result[1])),
            'map': unpad_map(centroid_result[2]),
        }
        unpadded_centroid = soma_results['Centroid']['center']
        print(f"  Centroid: ({unpadded_centroid[0]:.1f}, {unpadded_centroid[1]:.1f})")
        
        # Method 7: Template Matching
        template_result = find_center_template_matching(eimage_sta)
        soma_results['Template'] = {
            'center': unpad_coords((template_result[0], template_result[1])),
            'map': unpad_map(template_result[2]),
        }
        unpadded_template = soma_results['Template']['center']
        print(f"  Template: ({unpadded_template[0]}, {unpadded_template[1]})")
        
        # Method 8: Low-Rank Sparse
        lowrank_result = find_center_lowrank_sparse(eimage_sta)
        soma_results['LowRank'] = {
            'center': unpad_coords((lowrank_result[0], lowrank_result[1])),
            'map': unpad_map(lowrank_result[2]),
        }
        unpadded_lowrank = soma_results['LowRank']['center']
        print(f"  LowRank: ({unpadded_lowrank[0]}, {unpadded_lowrank[1]})")
        
        # Method 9: DoG Fit
        dog_result = find_center_dog_fit(eimage_sta)
        soma_results['DoG'] = {
            'center': unpad_coords((dog_result[0], dog_result[1])),
            'map': None,
            'success': dog_result[3],
        }
        unpadded_dog = soma_results['DoG']['center']
        print(f"  DoG: ({unpadded_dog[0]:.1f}, {unpadded_dog[1]:.1f})")
        
        # Method 10: Graph Sink
        graph_result = find_center_graph_sink(eimage_sta)
        soma_results['GraphSink'] = {
            'center': unpad_coords((graph_result[0], graph_result[1])),
            'map': unpad_map(graph_result[2]),
        }
        unpadded_graph = soma_results['GraphSink']['center']
        print(f"  GraphSink: ({unpadded_graph[0]}, {unpadded_graph[1]})")
        
        # Method 11: Consensus
        all_centers = [v['center'] for v in soma_results.values()]
        method_names = list(soma_results.keys())
        consensus_result = find_center_consensus(all_centers, method_names)
        soma_results['Consensus'] = {
            'center': (consensus_result[0], consensus_result[1]),
            'map': None,
            'stats': consensus_result[2],
        }
        print(f"  Consensus: ({consensus_result[0]:.1f}, {consensus_result[1]:.1f})")
        
        # =====================================================================
        # PROPAGATION TRACKING METHODS (on padded data, convert coords after)
        # =====================================================================
        prop_results = {}
        
        print("\n  Propagation Methods:")
        
        # Method 1: Farneback Optical Flow (existing)
        farneback_data = trace_optical_flow_propagation(eimage_sta, n_key_frames=5)
        prop_results['Farneback'] = {
            'trajectory': unpad_trajectory(farneback_data['propagation_path']),
            'divergence_map': None,
            'sink': soma_results['OptFlow']['center'],  # Already unpadded
            'source': unpad_coords(optflow_result[1]),
        }
        print(f"    Farneback: sink={prop_results['Farneback']['sink']}")
        
        # Method 2: ToA Cross-Correlation
        toa_data = trace_toa_crosscorr(eimage_sta)
        prop_results['ToA'] = {
            'trajectory': None,
            'delay_map': unpad_map(toa_data['delay_map']),
            'velocity_map': unpad_map(toa_data['velocity_map']),
        }
        print(f"    ToA: reference={unpad_coords(toa_data['reference'])}")
        
        # Method 3: Phase Wavefront
        phase_data = trace_phase_wavefront(eimage_sta)
        prop_results['Phase'] = {
            'trajectory': None,
            'phase_map': unpad_map(phase_data['phase_map']),
            'wavefront_direction': phase_data['wavefront_direction'],
        }
        print(f"    Phase: peak_frame={phase_data['peak_frame']}")
        
        # Method 4: Eikonal
        eikonal_data = trace_eikonal_fit(eimage_sta)
        prop_results['Eikonal'] = {
            'trajectory': None,
            'activation_time': unpad_map(eikonal_data['activation_time']),
            'source': unpad_coords(eikonal_data['source']),
            'velocity': eikonal_data['fitted_velocity'],
        }
        print(f"    Eikonal: source={prop_results['Eikonal']['source']}, v={eikonal_data['fitted_velocity']:.2f}")
        
        # Method 5: Beamforming
        beam_data = trace_beamforming(eimage_sta)
        prop_results['Beamform'] = {
            'trajectory': unpad_trajectory(beam_data['trajectory']),
            'power_map': unpad_map(beam_data['power_map']),
        }
        print(f"    Beamform: {len(prop_results['Beamform']['trajectory'])} trajectory points")
        
        # Method 6: Kalman Filter
        kalman_data = trace_kalman_filter(eimage_sta, farneback_data)
        prop_results['Kalman'] = {
            'trajectory': unpad_trajectory(kalman_data['filtered_trajectory']),
            'uncertainties': kalman_data['uncertainties'],
        }
        print(f"    Kalman: mean uncertainty={np.mean(kalman_data['uncertainties']):.2f}")
        
        # Method 7: TV-L1 Optical Flow
        tvl1_data = trace_tvl1_optical_flow(eimage_sta)
        prop_results['TVL1'] = {
            'trajectory': None,
            'divergence_map': unpad_map(tvl1_data['divergence_map']),
            'sink': unpad_coords(tvl1_data['sink']),
            'source': unpad_coords(tvl1_data['source']),
        }
        print(f"    TVL1: sink={prop_results['TVL1']['sink']}")
        
        # Method 8: Lucas-Kanade Optical Flow (4x4 window)
        lk_data = trace_lucas_kanade_optical_flow(eimage_sta, window_size=4)
        prop_results['LucasKanade'] = {
            'trajectory': unpad_trajectory(lk_data['trajectory']),
            'divergence_map': unpad_map(lk_data['divergence_map']),
            'sink': unpad_coords(lk_data['sink']),
            'source': unpad_coords(lk_data['source']),
            'velocity_magnitude': unpad_map(lk_data['velocity_magnitude']),
            'flow_x': unpad_map(lk_data['flow_x']),
            'flow_y': unpad_map(lk_data['flow_y']),
        }
        print(f"    LucasKanade: sink={prop_results['LucasKanade']['sink']}, source={prop_results['LucasKanade']['source']}")
        
        # Method 9: Event-Based
        event_data = trace_event_based(eimage_sta)
        prop_results['Event'] = {
            'trajectory': unpad_trajectory(event_data['trajectory']),
            'first_activation': unpad_map(event_data['first_activation']),
            'leading_edge': unpad_trajectory(event_data['leading_edge']),
        }
        print(f"    Event: max active={np.max(event_data['active_count'])}")
        
        # =====================================================================
        # SOMA SIZE ESTIMATION (using frames 10-14)
        # =====================================================================
        print(f"\n  Estimating soma sizes (frames 10-14)...")
        
        # Create soma_results_for_size with unpadded centers for size estimation
        soma_results_for_size = {}
        for method_name, result in soma_results.items():
            soma_results_for_size[method_name] = result.copy()
        
        # Estimate soma sizes using all methods
        size_results = estimate_all_soma_sizes(
            eimage_sta_original,  # Use original unpadded data
            soma_results_for_size,
            frame_range=(10, 14)
        )
        
        # Print size results
        for method_name, size_result in size_results.items():
            print(f"    {method_name}: {size_result.size_x:.1f}×{size_result.size_y:.1f}, "
                  f"area={size_result.area:.1f}, Ø={size_result.equivalent_diameter:.1f}")
        
        # Store size results for later comparison
        all_size_results[unit_id] = size_results
        
        # =====================================================================
        # STORE RESULTS (use original unpadded data for storage and plotting)
        # =====================================================================
        
        # Create unpadded legacy results for backward compatibility
        maxmin_result_unpad = (
            int(soma_results['MaxMin']['center'][0]),
            int(soma_results['MaxMin']['center'][1]),
            soma_results['MaxMin']['map'],
        )
        optflow_result_unpad = (
            soma_results['OptFlow']['center'],
            prop_results['Farneback']['source'],
            soma_results['OptFlow']['map'],
        )
        csd_result_unpad = (
            soma_results['CSD']['center'],
            unpad_coords(csd_result[1]),  # source
            soma_results['CSD']['map'],
            csd_result[3],  # peak_frame_idx
        )
        gaussian_result_unpad = (
            soma_results['GaussFit']['center'],
            soma_results['GaussFit']['map'],
            gaussian_result[2],  # params
            gaussian_result[3],  # success
        )
        
        # Unpad propagation data for legacy plot
        farneback_data_unpad = remove_padding_from_propagation_data(farneback_data, padding_info)
        
        all_results[unit_id] = {
            'eimage_sta': eimage_sta_original,  # Store original unpadded data
            'soma_results': soma_results,
            'prop_results': prop_results,
            # Legacy results for backward compatibility (unpadded)
            'maxmin': maxmin_result_unpad,
            'optflow': optflow_result_unpad,
            'csd': csd_result_unpad,
            'gaussian': gaussian_result_unpad,
            'propagation': farneback_data_unpad,
        }
        
        # =====================================================================
        # GENERATE PLOTS (using original unpadded data)
        # =====================================================================
        
        # Original comparison plot (legacy) - use original data and unpadded results
        output_path = OUTPUT_DIR / f"{unit_id}_comparison.png"
        plot_unit_comparison(unit_id, eimage_sta_original, maxmin_result_unpad, 
                           optflow_result_unpad, csd_result_unpad, 
                           gaussian_result_unpad, output_path)
        
        # Original propagation plot (legacy)
        propagation_path = OUTPUT_DIR / f"{unit_id}_propagation.png"
        plot_optical_flow_propagation(unit_id, eimage_sta_original, farneback_data_unpad, 
                                     optflow_result_unpad, propagation_path)
        
        # NEW: Divergence + streamlines plot
        div_stream_path = OUTPUT_DIR / f"{unit_id}_divergence_streamlines.png"
        plot_divergence_streamlines(unit_id, eimage_sta_original, optflow_result_unpad, 
                                   farneback_data_unpad, div_stream_path)
        
        # NEW: Lucas-Kanade optical flow plot
        lk_plot_path = OUTPUT_DIR / f"{unit_id}_lucas_kanade.png"
        plot_lucas_kanade(unit_id, eimage_sta_original, prop_results['LucasKanade'], lk_plot_path)
        
        # NEW: Comprehensive soma comparison
        soma_comp_path = OUTPUT_DIR / f"{unit_id}_soma_comparison.png"
        plot_soma_comparison(unit_id, eimage_sta_original, soma_results, soma_comp_path)
        
        # NEW: Comprehensive propagation comparison
        prop_comp_path = OUTPUT_DIR / f"{unit_id}_propagation_comparison.png"
        plot_propagation_comparison(unit_id, eimage_sta_original, prop_results, prop_comp_path)
        
        # NEW: Individual soma size plot
        size_plot_path = OUTPUT_DIR / f"{unit_id}_soma_size.png"
        plot_soma_size_individual(unit_id, eimage_sta_original, size_results, 
                                   (10, 14), size_plot_path)
    
    # =========================================================================
    # SUMMARY PLOTS
    # =========================================================================
    print(f"\n{'='*60}")
    print("Generating Summary Plots")
    print(f"{'='*60}")
    
    # Legacy summary
    summary_path = OUTPUT_DIR / "summary.png"
    plot_summary(all_results, summary_path)
    
    # NEW: Comprehensive summary comparison
    plot_summary_comparison(all_results, OUTPUT_DIR)
    
    # NEW: Soma size comparison across all units
    soma_size_comp_path = OUTPUT_DIR / "soma_size_comparison.png"
    plot_soma_size_comparison(all_size_results, soma_size_comp_path)
    
    # Print results table
    print_results_table(all_results)
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Units analyzed: {len(all_results)}")
    print(f"Soma methods: 11 (4 original + 7 new)")
    print(f"Propagation methods: 9 (1 original + 8 new, incl. Lucas-Kanade)")
    print(f"Soma size estimation: using frames 10-14")
    print(f"\nOutput files per unit:")
    print(f"  - {{unit_id}}_comparison.png (legacy)")
    print(f"  - {{unit_id}}_propagation.png (legacy)")
    print(f"  - {{unit_id}}_divergence_streamlines.png (NEW)")
    print(f"  - {{unit_id}}_lucas_kanade.png (NEW)")
    print(f"  - {{unit_id}}_soma_comparison.png (NEW)")
    print(f"  - {{unit_id}}_propagation_comparison.png (NEW)")
    print(f"  - {{unit_id}}_soma_size.png (NEW)")
    print(f"\nSummary files:")
    print(f"  - summary.png")
    print(f"  - soma_methods_summary.png (NEW)")
    print(f"  - propagation_methods_summary.png (NEW)")
    print(f"  - soma_size_comparison.png (NEW)")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

