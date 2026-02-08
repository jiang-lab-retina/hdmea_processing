"""
Standalone script to calculate iprgc_2hz_QI from mini H5 files.

This script is self-contained and does not depend on any other project files.

Usage:
    python calculate_iprgc_qi.py                          # Process all mini H5 files
    python calculate_iprgc_qi.py path/to/file_mini.h5     # Process specific file
    python calculate_iprgc_qi.py --output results.csv     # Save results to CSV
"""

import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy import signal
from scipy import stats
from typing import Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

IPRGC_TARGET_RATE_HZ = 60.0       # Target binning rate for firing rate
IPRGC_EXPECTED_BINS = 7200        # Expected bins (= 120 seconds × 60 Hz)
IPRGC_LENGTH_TOLERANCE = 0.10    # ±10% tolerance for valid trials
MOVIE_NAME = "iprgc_test"


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

def bessel_lowpass_filter(
    trace: np.ndarray,
    cutoff_freq: float = 2.0,
    order: int = 5,
    sampling_rate: float = 60.0,
) -> np.ndarray:
    """
    Apply Bessel low-pass filter to a trace.
    
    Parameters
    ----------
    trace : np.ndarray
        1D array of firing rate data
    cutoff_freq : float
        Cutoff frequency in Hz (default 2 Hz for iprgc_2hz_QI)
    order : int
        Filter order (default 5)
    sampling_rate : float
        Sampling rate in Hz (default 60 Hz)
        
    Returns
    -------
    np.ndarray
        Filtered trace
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    
    # Ensure cutoff is valid (must be < 1 for normalized frequency)
    if normalized_cutoff >= 1.0:
        raise ValueError(
            f"Cutoff frequency {cutoff_freq} Hz is too high for sampling rate "
            f"{sampling_rate} Hz (Nyquist = {nyquist} Hz)"
        )
    
    b, a = signal.bessel(order, normalized_cutoff, btype='low', analog=False, output='ba')
    filtered_trace = signal.filtfilt(b, a, trace)
    return filtered_trace


# =============================================================================
# QUALITY INDEX FUNCTIONS
# =============================================================================

def get_quality_index_pearson(array: np.ndarray) -> float:
    """
    Calculate quality index as mean Pearson correlation of trials vs mean.
    
    QI = mean(corr(trial_i, mean_trace)) for all trials
    
    Parameters
    ----------
    array : np.ndarray
        2D array of shape (n_trials, n_timepoints)
        
    Returns
    -------
    float
        Quality index value (mean correlation coefficient)
    """
    if array.ndim != 2 or array.shape[0] < 2:
        return np.nan
    
    # Check for NaN values
    if np.isnan(array).any():
        return np.nan
    
    mean_trace = array.mean(axis=0)
    
    # Check if mean trace has zero variance (constant)
    if np.std(mean_trace) == 0:
        return np.nan
    
    correlations = []
    for trial in array:
        # Check if trial has zero variance
        if np.std(trial) == 0:
            continue
        r, _ = stats.pearsonr(trial, mean_trace)
        if not np.isnan(r):
            correlations.append(r)
    
    if len(correlations) == 0:
        return np.nan
    
    return np.mean(correlations)


# =============================================================================
# FIRING RATE CALCULATION
# =============================================================================

def get_sample_based_firing_rate(
    spike_times: np.ndarray,
    trial_start_sample: int,
    trial_end_sample: int,
    target_rate_hz: float = 60.0,
    acquisition_rate: float = 20000.0,
) -> Tuple[np.ndarray, int]:
    """
    Convert spike times to sample-based firing rate.
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times in samples (at acquisition rate)
    trial_start_sample : int
        Start sample of the trial
    trial_end_sample : int
        End sample of the trial
    target_rate_hz : float
        Target binning rate in Hz (default 60 Hz)
    acquisition_rate : float
        Acquisition rate in Hz (default 20000 Hz)
        
    Returns
    -------
    Tuple[np.ndarray, int]
        (firing_rate_array, n_bins)
    """
    samples_per_bin = acquisition_rate / target_rate_hz
    relative_spikes = spike_times.astype(np.int64) - int(trial_start_sample)
    trial_duration = int(trial_end_sample) - int(trial_start_sample)
    mask = (relative_spikes >= 0) & (relative_spikes < trial_duration)
    trial_spikes = relative_spikes[mask]
    
    n_bins = int(np.ceil(trial_duration / samples_per_bin))
    bin_edges = np.arange(0, n_bins + 1) * samples_per_bin
    counts, _ = np.histogram(trial_spikes, bins=bin_edges)
    
    bin_duration_s = 1.0 / target_rate_hz
    firing_rate = counts.astype(np.float32) / bin_duration_s
    return firing_rate, n_bins


# =============================================================================
# IPRGC QI CALCULATION
# =============================================================================

def compute_iprgc_2hz_qi(
    trials_array: np.ndarray,
    filter_order: int = 5,
    sampling_rate: float = 60.0,
) -> float:
    """
    Compute iprgc_2hz_QI from firing rate trials array.
    
    Parameters
    ----------
    trials_array : np.ndarray
        2D array of shape (n_trials, n_timepoints) containing firing rates
    filter_order : int
        Bessel filter order (default 5)
    sampling_rate : float
        Sampling rate of firing rate data in Hz (default 60 Hz)
        
    Returns
    -------
    float
        iprgc_2hz_QI value
    """
    n_trials, n_timepoints = trials_array.shape
    
    # Time constants
    samples_baseline = int(1 * sampling_rate)   # Last 1 second for baseline (60 samples)
    samples_start = int(2 * sampling_rate)      # Start at 2 seconds (120 samples)
    
    # Baseline subtraction: zero by mean of last second
    for i in range(n_trials):
        baseline_mean = np.mean(trials_array[i, -samples_baseline:])
        trials_array[i] = trials_array[i] - baseline_mean
    
    # Apply 2 Hz lowpass filter
    try:
        filtered_2hz = np.zeros_like(trials_array)
        for i in range(n_trials):
            filtered_2hz[i] = bessel_lowpass_filter(
                trials_array[i],
                cutoff_freq=2.0,
                order=filter_order,
                sampling_rate=sampling_rate,
            )
        
        # Trim from 2 seconds to end
        trimmed_2hz = filtered_2hz[:, samples_start:]
        
        # Check if any trial has mean firing rate < 1 Hz (use raw trimmed data)
        raw_trimmed_2hz = trials_array[:, samples_start:]
        has_low_firing_trial = any(np.mean(raw_trimmed_2hz[i]) < 1.0 for i in range(n_trials))
        
        # If any trial is constant, all near zero, or low firing rate, set QI to 0
        has_constant_trial = any(np.std(trimmed_2hz[i]) == 0 for i in range(n_trials))
        all_near_zero = np.max(np.abs(trimmed_2hz)) < 1.0
        
        if has_constant_trial or all_near_zero or has_low_firing_trial:
            return 0.0
        
        return get_quality_index_pearson(trimmed_2hz)
        
    except Exception:
        return np.nan


# =============================================================================
# H5 FILE PROCESSING
# =============================================================================

def process_unit_from_h5(
    h5_file: h5py.File,
    unit_id: str,
    acquisition_rate: float,
) -> Optional[float]:
    """
    Process a single unit from an H5 file and compute iprgc_2hz_QI.
    
    Parameters
    ----------
    h5_file : h5py.File
        Open H5 file handle
    unit_id : str
        Unit identifier
    acquisition_rate : float
        Acquisition rate in Hz
        
    Returns
    -------
    Optional[float]
        iprgc_2hz_QI value, or None if data is missing
    """
    unit_path = f"units/{unit_id}"
    iprgc_path = f"{unit_path}/spike_times_sectioned/{MOVIE_NAME}"
    
    # Check if iprgc_test data exists
    trials_path = f"{iprgc_path}/trials_spike_times"
    starts_path = f"{iprgc_path}/trials_start_end"
    
    if trials_path not in h5_file or starts_path not in h5_file:
        return None
    
    trials_start_end = h5_file[starts_path][:]
    trials_group = h5_file[trials_path]
    
    # Compute expected bin range for trial validation
    samples_per_bin = acquisition_rate / IPRGC_TARGET_RATE_HZ
    min_expected = int(IPRGC_EXPECTED_BINS * (1 - IPRGC_LENGTH_TOLERANCE))
    max_expected = int(IPRGC_EXPECTED_BINS * (1 + IPRGC_LENGTH_TOLERANCE))
    
    # Collect valid trials
    valid_firing_rates = []
    
    for trial_idx_str in trials_group.keys():
        trial_idx = int(trial_idx_str)
        trial_ds = trials_group[trial_idx_str]
        
        # Handle scalar (empty) vs array datasets
        if trial_ds.shape == ():
            spike_times = np.array([])
        else:
            spike_times = trial_ds[:]
        
        trial_start, trial_end = trials_start_end[trial_idx]
        
        # Compute firing rate
        fr, n_bins = get_sample_based_firing_rate(
            spike_times, trial_start, trial_end,
            IPRGC_TARGET_RATE_HZ, acquisition_rate
        )
        
        # Validate trial length (±10% of expected)
        if not (min_expected <= n_bins <= max_expected):
            continue
        
        valid_firing_rates.append(fr)
    
    if len(valid_firing_rates) < 2:
        return None
    
    # Trim all trials to minimum length
    min_len = min(len(fr) for fr in valid_firing_rates)
    trials_array = np.vstack([fr[:min_len] for fr in valid_firing_rates])
    
    # Compute QI
    return compute_iprgc_2hz_qi(trials_array)


def process_mini_h5(h5_path: Path) -> dict:
    """
    Process a mini H5 file and compute iprgc_2hz_QI for all units.
    
    Parameters
    ----------
    h5_path : Path
        Path to the mini H5 file
        
    Returns
    -------
    dict
        Dictionary mapping unit_id to iprgc_2hz_QI value
    """
    results = {}
    
    with h5py.File(h5_path, "r") as f:
        # Get acquisition rate from metadata
        if "metadata/acquisition_rate" not in f:
            print(f"  WARNING: No acquisition_rate in {h5_path.name}, using default 20000 Hz")
            acquisition_rate = 20000.0
        else:
            acquisition_rate = float(f["metadata/acquisition_rate"][()])
        
        # Get all unit IDs
        if "units" not in f:
            print(f"  No units found in {h5_path.name}")
            return results
        
        unit_ids = list(f["units"].keys())
        
        for unit_id in unit_ids:
            qi = process_unit_from_h5(f, unit_id, acquisition_rate)
            if qi is not None:
                results[unit_id] = qi
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate iprgc_2hz_QI from mini H5 files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python calculate_iprgc_qi.py                          # Process all mini H5 files
    python calculate_iprgc_qi.py path/to/file_mini.h5     # Process specific file
    python calculate_iprgc_qi.py --output results.csv     # Save results to CSV
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Input H5 file or directory. Default: ./export/mini/"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output CSV file path. If not specified, results are printed to console."
    )
    
    args = parser.parse_args()
    
    # Determine input files
    if args.input is None:
        # Default: look for mini H5 files in export/mini/
        mini_dir = Path(__file__).parent / "export" / "mini"
        if not mini_dir.exists():
            print(f"Error: Default directory not found: {mini_dir}")
            print("Please specify an input file or directory.")
            return
        h5_files = sorted(mini_dir.glob("*_mini.h5"))
    elif args.input.is_file():
        h5_files = [args.input]
    elif args.input.is_dir():
        h5_files = sorted(args.input.glob("*.h5"))
    else:
        print(f"Error: Input not found: {args.input}")
        return
    
    if not h5_files:
        print("No H5 files found.")
        return
    
    print(f"Processing {len(h5_files)} H5 file(s)...")
    print("-" * 60)
    
    all_results = []
    
    for h5_path in h5_files:
        print(f"\nProcessing: {h5_path.name}")
        results = process_mini_h5(h5_path)
        
        if results:
            # Add filename to results
            filename = h5_path.stem.replace("_mini", "")
            for unit_id, qi in results.items():
                all_results.append({
                    "filename": filename,
                    "unit_id": unit_id,
                    "row_index": f"{filename}_unit_{unit_id}",
                    "iprgc_2hz_QI": qi
                })
            
            # Print summary
            qi_values = list(results.values())
            valid_qi = [q for q in qi_values if not np.isnan(q)]
            print(f"  Units with iprgc data: {len(results)}")
            print(f"  Valid QI values: {len(valid_qi)}")
            if valid_qi:
                print(f"  QI range: {min(valid_qi):.4f} - {max(valid_qi):.4f}")
                print(f"  QI mean: {np.mean(valid_qi):.4f}")
        else:
            print("  No iprgc_test data found")
    
    print("\n" + "-" * 60)
    print(f"Total units processed: {len(all_results)}")
    
    # Output results
    if args.output:
        # Save to CSV
        import csv
        with open(args.output, "w", newline="") as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=["row_index", "filename", "unit_id", "iprgc_2hz_QI"])
                writer.writeheader()
                writer.writerows(all_results)
        print(f"Results saved to: {args.output}")
    else:
        # Print to console
        if all_results:
            print("\nResults:")
            print("-" * 60)
            print(f"{'row_index':<50} {'iprgc_2hz_QI':>12}")
            print("-" * 60)
            for r in all_results[:20]:  # Show first 20
                print(f"{r['row_index']:<50} {r['iprgc_2hz_QI']:>12.4f}")
            if len(all_results) > 20:
                print(f"... and {len(all_results) - 20} more rows")
            print("-" * 60)


if __name__ == "__main__":
    main()
