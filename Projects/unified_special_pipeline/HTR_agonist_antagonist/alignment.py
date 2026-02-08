"""
HTR Agonist/Antagonist Alignment Pipeline

This module aligns units across before/after recording pairs based on:
- Coordinate proximity (row, col)
- Response signature similarity
- Waveform signature similarity

Pairing logic:
- "Before" file: Condition contains play_optimization_set6_a_ipRGC_without_step()
- "After" file: Condition contains play_optimization_set6_a_ipRGC_manual() 
  AND comes after the before file (by timestamp) AND shares the same Chip number
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import h5py

# Import config
try:
    from .specific_config import (
        OUTPUT_DIR, GSHEET_CSV_PATH, PROJECT_ROOT,
    )
except ImportError:
    from specific_config import (
        OUTPUT_DIR, GSHEET_CSV_PATH, PROJECT_ROOT,
    )

# =============================================================================
# Constants
# =============================================================================

BEFORE_CONDITION = "play_optimization_set6_a_ipRGC_without_step"
AFTER_CONDITION = "play_optimization_set6_a_ipRGC_manual"

# Alignment parameters
DEFAULT_QUALITY_THRESHOLD = 0.05
DEFAULT_WAVEFORM_WEIGHT = 10.0
DEFAULT_ITERATION_DISTANCES = [0, 1, 2]

# Step light response parameters
STEP_SECTION_NAME = "step_up_5s_5i_b0_3x"  # Section name for step light
STEP_BIN_SIZE_MS = 50  # Bin size for PSTH in milliseconds
STEP_SAMPLE_RATE = 20000  # Sample rate in Hz

# Cell type labeling parameters (based on step response)
# Time is in bins (50ms per bin, so 10 bins = 500ms)
# Step stimulus: 5s ON (0-5000ms), 5s OFF (5000-10000ms)
# With 50ms binning:
#   ON window: 1-3s = bins [20, 60] (sustained ON response)
#   OFF window: 6-8s = bins [120, 160] (1-3s after step OFF at 5s)
ON_CELL_PARAMS = {
    "quality_threshold": 0.1,
    "response_range": [20, 60],     # 1000-3000ms (1-3s sustained ON response)
    "baseline_range": [0, 10],      # 0-500ms (early response as baseline)
    "response_std_threshold": 0.1,
}

OFF_CELL_PARAMS = {
    "quality_threshold": 0.1,
    "response_range": [120, 160],   # 6000-8000ms (1-3s after step OFF at 5s)
    "baseline_range": [100, 110],   # 5000-5500ms (early OFF response as baseline)
    "response_std_threshold": 0.1,
}

# Output directory for aligned files
ALIGNED_OUTPUT_DIR = OUTPUT_DIR / "aligned"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FilePair:
    """Represents a before/after file pair for alignment."""
    before_file: str  # gsheet filename format (dots)
    after_file: str   # gsheet filename format (dots)
    before_h5: Path   # H5 file path
    after_h5: Path    # H5 file path
    chip: str         # Chip number
    before_time: datetime
    after_time: datetime


@dataclass
class UnitData:
    """Data for a single unit."""
    unit_id: str
    row: int
    col: int
    waveform: np.ndarray
    response_signature: np.ndarray
    quality_index: float
    spike_times: Optional[np.ndarray] = None
    cell_type: str = "unknown"  # ON, OFF, ON_OFF, or unknown
    step_response_trials: Optional[np.ndarray] = None  # Per-trial step responses


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(level: int = logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# =============================================================================
# Filename Utilities
# =============================================================================

def gsheet_to_h5_filename(gsheet_filename: str) -> str:
    """
    Convert gsheet filename format to H5 filename format.
    
    Example: "2025.10.01.09.33.32.Rec.cmcr" -> "2025.10.01-09.33.32-Rec.h5"
    """
    # Remove .cmcr extension
    base = gsheet_filename.replace(".cmcr", "")
    
    # Parse the date.time.Rec pattern and convert to dashes
    # Format: YYYY.MM.DD.HH.MM.SS.Rec -> YYYY.MM.DD-HH.MM.SS-Rec
    parts = base.split(".")
    if len(parts) >= 7:
        # parts[0:3] = date, parts[3:6] = time, parts[6] = "Rec"
        date_part = ".".join(parts[0:3])
        time_part = ".".join(parts[3:6])
        rec_part = parts[6] if len(parts) > 6 else "Rec"
        h5_name = f"{date_part}-{time_part}-{rec_part}.h5"
    else:
        # Fallback: just replace extension
        h5_name = base + ".h5"
    
    return h5_name


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from gsheet filename.
    
    Example: "2025.10.01.09.33.32.Rec.cmcr" -> datetime(2025, 10, 1, 9, 33, 32)
    """
    try:
        # Remove extension
        base = filename.replace(".cmcr", "").replace(".h5", "")
        # Handle both formats
        base = base.replace("-", ".")
        parts = base.split(".")
        
        if len(parts) >= 6:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            second = int(parts[5])
            return datetime(year, month, day, hour, minute, second)
    except (ValueError, IndexError):
        pass
    return None


# =============================================================================
# Pair Discovery
# =============================================================================

def load_gsheet_dataframe() -> Optional[pd.DataFrame]:
    """Load the gsheet CSV file."""
    logger = logging.getLogger(__name__)
    
    if not GSHEET_CSV_PATH.exists():
        logger.error(f"Gsheet CSV not found: {GSHEET_CSV_PATH}")
        return None
    
    try:
        df = pd.read_csv(GSHEET_CSV_PATH)
        logger.info(f"Loaded gsheet with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading gsheet: {e}")
        return None


def find_alignment_pairs(
    gsheet_df: pd.DataFrame,
    h5_folder: Path = OUTPUT_DIR,
) -> List[FilePair]:
    """
    Find before/after file pairs from gsheet data.
    
    Pairing logic:
    - Before: Condition contains BEFORE_CONDITION
    - After: Condition contains AFTER_CONDITION, same Chip, later timestamp
    
    Args:
        gsheet_df: DataFrame with gsheet data
        h5_folder: Folder containing H5 files
    
    Returns:
        List of FilePair objects
    """
    logger = logging.getLogger(__name__)
    pairs = []
    
    # Filter rows with target conditions
    before_mask = gsheet_df['Condition'].astype(str).str.contains(BEFORE_CONDITION, na=False)
    after_mask = gsheet_df['Condition'].astype(str).str.contains(AFTER_CONDITION, na=False)
    
    before_rows = gsheet_df[before_mask].copy()
    after_rows = gsheet_df[after_mask].copy()
    
    logger.info(f"Found {len(before_rows)} 'before' files, {len(after_rows)} 'after' files")
    
    # Add timestamp column
    before_rows['_timestamp'] = before_rows['File_name'].apply(extract_timestamp_from_filename)
    after_rows['_timestamp'] = after_rows['File_name'].apply(extract_timestamp_from_filename)
    
    # Group by Chip
    for chip in before_rows['Chip'].dropna().unique():
        chip_str = str(chip).strip()
        
        # Get before files for this chip
        chip_before = before_rows[before_rows['Chip'].astype(str).str.strip() == chip_str]
        chip_after = after_rows[after_rows['Chip'].astype(str).str.strip() == chip_str]
        
        if chip_after.empty:
            logger.debug(f"No 'after' files for Chip {chip_str}")
            continue
        
        # For each before file, find the next after file
        for _, before_row in chip_before.iterrows():
            before_file = str(before_row['File_name'])
            before_time = before_row['_timestamp']
            
            if before_time is None:
                logger.warning(f"Could not parse timestamp for {before_file}")
                continue
            
            # Find the first after file that comes after this before file
            valid_afters = chip_after[chip_after['_timestamp'] > before_time]
            
            if valid_afters.empty:
                logger.debug(f"No 'after' file found after {before_file}")
                continue
            
            # Get the earliest after file
            valid_afters = valid_afters.sort_values('_timestamp')
            after_row = valid_afters.iloc[0]
            after_file = str(after_row['File_name'])
            after_time = after_row['_timestamp']
            
            # Check if H5 files exist
            before_h5_name = gsheet_to_h5_filename(before_file)
            after_h5_name = gsheet_to_h5_filename(after_file)
            
            before_h5_path = h5_folder / before_h5_name
            after_h5_path = h5_folder / after_h5_name
            
            if not before_h5_path.exists():
                logger.warning(f"Before H5 not found: {before_h5_path}")
                continue
            
            if not after_h5_path.exists():
                logger.warning(f"After H5 not found: {after_h5_path}")
                continue
            
            pair = FilePair(
                before_file=before_file,
                after_file=after_file,
                before_h5=before_h5_path,
                after_h5=after_h5_path,
                chip=chip_str,
                before_time=before_time,
                after_time=after_time,
            )
            pairs.append(pair)
            
            logger.info(f"Found pair (Chip {chip_str}): {before_file} -> {after_file}")
    
    logger.info(f"Total pairs found: {len(pairs)}")
    return pairs


# =============================================================================
# H5 Data Loading
# =============================================================================

def compute_step_light_mean_response(
    unit_group: h5py.Group,
    section_name: str = STEP_SECTION_NAME,
    bin_size_ms: float = STEP_BIN_SIZE_MS,
    sample_rate: int = STEP_SAMPLE_RATE,
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Compute mean response to step light from trial spike times.
    
    Args:
        unit_group: H5 group for the unit
        section_name: Name of the step light section
        bin_size_ms: Bin size for PSTH in milliseconds
        sample_rate: Sample rate in Hz
    
    Returns:
        Tuple of (mean_response, quality_index, trial_responses)
        trial_responses is a 2D array (n_trials, n_bins) for cell type labeling
    """
    default_response = np.zeros(100)
    
    if 'spike_times_sectioned' not in unit_group:
        return default_response, 0.0, None
    
    sectioned = unit_group['spike_times_sectioned']
    
    if section_name not in sectioned:
        return default_response, 0.0, None
    
    section = sectioned[section_name]
    
    if 'trials_start_end' not in section or 'trials_spike_times' not in section:
        return default_response, 0.0, None
    
    # Get trial timing
    trials_start_end = section['trials_start_end'][:]  # shape: (n_trials, 2)
    n_trials = len(trials_start_end)
    
    if n_trials == 0:
        return default_response, 0.0, None
    
    # Calculate trial duration and number of bins
    trial_durations = trials_start_end[:, 1] - trials_start_end[:, 0]
    max_duration = int(np.max(trial_durations))
    
    # Convert to ms and calculate bins
    bin_size_samples = int(bin_size_ms * sample_rate / 1000)
    n_bins = max_duration // bin_size_samples + 1
    
    if n_bins <= 0:
        return default_response, 0.0, None
    
    # Collect PSTH for each trial
    trials_group = section['trials_spike_times']
    trial_responses = []
    
    for trial_idx in range(n_trials):
        trial_key = str(trial_idx)
        if trial_key not in trials_group:
            continue
        
        spike_times = trials_group[trial_key][:]
        trial_start = trials_start_end[trial_idx, 0]
        trial_end = trials_start_end[trial_idx, 1]
        
        # Filter spikes within trial window
        valid_spikes = spike_times[(spike_times >= trial_start) & (spike_times < trial_end)]
        
        # Convert to relative times (samples from trial start)
        relative_times = valid_spikes - trial_start
        
        # Bin the spikes
        bins = np.arange(0, max_duration + bin_size_samples, bin_size_samples)
        hist, _ = np.histogram(relative_times, bins=bins)
        
        # Convert to firing rate (spikes per second)
        firing_rate = hist * (1000 / bin_size_ms)
        
        trial_responses.append(firing_rate)
    
    if not trial_responses:
        return default_response, 0.0, None
    
    # Pad responses to same length and stack
    max_len = max(len(r) for r in trial_responses)
    padded_responses = np.zeros((len(trial_responses), max_len))
    for i, r in enumerate(trial_responses):
        padded_responses[i, :len(r)] = r
    
    # Calculate mean response
    mean_response = padded_responses.mean(axis=0)
    
    # Calculate quality index (variance of mean / mean of variances)
    if padded_responses.shape[0] > 1:
        var_of_mean = np.var(mean_response)
        mean_of_var = np.mean(np.var(padded_responses, axis=1))
        if mean_of_var > 0:
            quality_index = var_of_mean / mean_of_var
        else:
            quality_index = 0.1 if var_of_mean > 0 else 0.0
    else:
        # Single trial - use variance of response as proxy
        quality_index = 0.1 if np.var(mean_response) > 0 else 0.0
    
    return mean_response, quality_index, padded_responses


def get_quality_index(array: np.ndarray) -> float:
    """
    Calculate quality index for a unit's response array.
    
    Quality index = variance of mean response / mean of trial variances
    Higher values indicate more consistent, stimulus-locked responses.
    
    Args:
        array: 2D array of shape (n_trials, n_timepoints)
    
    Returns:
        Quality index (float), or NaN if invalid
    """
    if array.ndim != 2 or array.shape[0] < 2:
        return np.nan
    
    # Mean across trials (response signature)
    mean_across_trials = array.mean(axis=0)
    
    # Variance across timepoints for each trial
    var_across_timepoints = array.var(axis=1)
    
    # Check for NaN
    if np.isnan(mean_across_trials).any() or np.isnan(var_across_timepoints).any():
        return np.nan
    
    # Mean of trial variances
    mean_var = np.mean(var_across_timepoints)
    
    if mean_var == 0:
        return np.nan
    
    # Variance ratio
    quality_index = np.var(mean_across_trials) / mean_var
    return quality_index


def cell_type_identifier(
    trace_array: np.ndarray,
    quality_threshold: float = 0.1,
    response_range: List[int] = [2, 4],
    baseline_range: List[int] = [0, 2],
    response_std_threshold: float = 0.1,
) -> bool:
    """
    Identify if a cell responds to stimulus based on trace array.
    
    Args:
        trace_array: 2D array of shape (n_trials, n_timepoints)
        quality_threshold: Minimum quality index required
        response_range: [start, end] indices for response window
        baseline_range: [start, end] indices for baseline window
        response_std_threshold: Response must exceed baseline by this many stds
    
    Returns:
        True if cell is responsive, False otherwise
    """
    if trace_array is None or trace_array.ndim != 2 or trace_array.shape[0] < 1:
        return False
    
    n_bins = trace_array.shape[1]
    
    # Adjust ranges if they exceed array bounds
    resp_start = max(0, response_range[0])
    resp_end = min(n_bins, response_range[1])
    base_start = max(0, baseline_range[0])
    base_end = min(n_bins, baseline_range[1]) if baseline_range[1] else n_bins
    
    if resp_end <= resp_start or base_end <= base_start:
        return False
    
    # Check quality threshold
    if quality_threshold is not None:
        quality_index = get_quality_index(trace_array)
        if np.isnan(quality_index) or quality_index < quality_threshold:
            return False
    
    # Check response vs baseline
    if response_std_threshold is not None:
        response_mean = np.mean(trace_array[:, resp_start:resp_end])
        baseline_mean = np.mean(trace_array[:, base_start:base_end])
        baseline_std = np.std(trace_array[:, base_start:base_end])
        
        if baseline_std == 0:
            # If no variability in baseline, check if response is higher
            if response_mean <= baseline_mean:
                return False
        else:
            if response_mean < baseline_mean + response_std_threshold * baseline_std:
                return False
    
    return True


def label_cell_type(
    step_response_trials: Optional[np.ndarray],
    on_params: Dict = None,
    off_params: Dict = None,
) -> str:
    """
    Label cell type based on step response.
    
    Args:
        step_response_trials: 2D array of shape (n_trials, n_timepoints)
        on_params: Parameters for ON cell identification
        off_params: Parameters for OFF cell identification
    
    Returns:
        Cell type: "ON", "OFF", "ON_OFF", or "unknown"
    """
    if step_response_trials is None or len(step_response_trials) == 0:
        return "unknown"
    
    if on_params is None:
        on_params = ON_CELL_PARAMS
    if off_params is None:
        off_params = OFF_CELL_PARAMS
    
    is_on = cell_type_identifier(
        step_response_trials,
        quality_threshold=on_params.get("quality_threshold", 0.1),
        response_range=on_params.get("response_range", [2, 4]),
        baseline_range=on_params.get("baseline_range", [0, 2]),
        response_std_threshold=on_params.get("response_std_threshold", 0.1),
    )
    
    is_off = cell_type_identifier(
        step_response_trials,
        quality_threshold=off_params.get("quality_threshold", 0.1),
        response_range=off_params.get("response_range", [100, 110]),
        baseline_range=off_params.get("baseline_range", [95, 100]),
        response_std_threshold=off_params.get("response_std_threshold", 0.1),
    )
    
    if is_on and is_off:
        return "ON_OFF"
    elif is_on and not is_off:
        return "ON"
    elif not is_on and is_off:
        return "OFF"
    else:
        return "unknown"


def load_unit_data_from_h5(
    h5_path: Path,
    step_section_name: str = STEP_SECTION_NAME,
    label_cell_types: bool = True,
) -> Dict[str, UnitData]:
    """
    Load unit data from an H5 file.
    
    Uses waveform and step light mean response for alignment.
    Labels cell types (ON, OFF, ON_OFF, unknown) based on step response.
    
    Args:
        h5_path: Path to H5 file
        step_section_name: Name of step light section for response signature
        label_cell_types: If True, compute cell type labels
    
    Returns:
        Dictionary mapping unit_id to UnitData
    """
    logger = logging.getLogger(__name__)
    units = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'units' not in f:
                logger.warning(f"No 'units' group in {h5_path}")
                return units
            
            units_group = f['units']
            step_found_count = 0
            cell_type_counts = {"ON": 0, "OFF": 0, "ON_OFF": 0, "unknown": 0}
            
            for unit_id in units_group.keys():
                unit_group = units_group[unit_id]
                
                # Get coordinates
                row = int(unit_group.attrs.get('row', 0))
                col = int(unit_group.attrs.get('col', 0))
                
                # Get waveform
                waveform = np.zeros(50)  # Default
                if 'waveform' in unit_group:
                    waveform = unit_group['waveform'][:]
                elif 'mean_waveform' in unit_group:
                    waveform = unit_group['mean_waveform'][:]
                
                # Get response signature from step light mean response
                response_signature, quality_index, trial_responses = compute_step_light_mean_response(
                    unit_group, 
                    section_name=step_section_name,
                )
                
                cell_type = "unknown"
                step_response_trials = None
                
                if len(response_signature) > 0 and np.any(response_signature != 0):
                    step_found_count += 1
                    step_response_trials = trial_responses
                    
                    # Label cell type based on step response
                    if label_cell_types and trial_responses is not None:
                        cell_type = label_cell_type(trial_responses)
                        cell_type_counts[cell_type] += 1
                else:
                    # Fallback to firing_rate_10hz if step data not available
                    if 'firing_rate_10hz' in unit_group:
                        firing_rate = unit_group['firing_rate_10hz'][:]
                        response_signature = firing_rate
                        if len(firing_rate) > 0 and np.std(firing_rate) > 0:
                            quality_index = np.var(firing_rate) / (np.mean(firing_rate) + 1e-6)
                            quality_index = min(quality_index, 10.0) / 10.0
                        else:
                            quality_index = 0.1
                    cell_type_counts["unknown"] += 1
                
                # Get spike times if available
                spike_times = None
                if 'spike_times' in unit_group:
                    spike_times = unit_group['spike_times'][:]
                
                units[unit_id] = UnitData(
                    unit_id=unit_id,
                    row=row,
                    col=col,
                    waveform=waveform,
                    response_signature=response_signature,
                    quality_index=quality_index,
                    spike_times=spike_times,
                    cell_type=cell_type,
                    step_response_trials=step_response_trials,
                )
            
            logger.info(f"Loaded {len(units)} units from {h5_path.name} ({step_found_count} with step response)")
            if label_cell_types:
                logger.info(f"  Cell types: ON={cell_type_counts['ON']}, OFF={cell_type_counts['OFF']}, ON_OFF={cell_type_counts['ON_OFF']}, unknown={cell_type_counts['unknown']}")
    
    except Exception as e:
        logger.error(f"Error loading {h5_path}: {e}")
    
    return units


# =============================================================================
# Quality Filtering
# =============================================================================

def filter_high_quality_units(
    units: Dict[str, UnitData],
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> List[str]:
    """
    Filter units by quality index.
    
    Args:
        units: Dictionary of unit data
        quality_threshold: Minimum quality index
    
    Returns:
        List of high-quality unit IDs
    """
    high_quality = []
    for unit_id, data in units.items():
        if not np.isnan(data.quality_index) and data.quality_index >= quality_threshold:
            high_quality.append(unit_id)
    
    return sorted(high_quality)


# =============================================================================
# Unit Alignment
# =============================================================================

def get_unit_pair_list(
    ref_units: Dict[str, UnitData],
    align_units: Dict[str, UnitData],
    ref_keys: List[str],
    align_keys: List[str],
    waveform_weight: float = DEFAULT_WAVEFORM_WEIGHT,
    distance_threshold: int = 0,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """
    Find matching unit pairs between reference and alignment datasets.
    
    Args:
        ref_units: Reference dataset units
        align_units: Alignment dataset units
        ref_keys: List of reference unit IDs to match
        align_keys: List of alignment unit IDs to consider
        waveform_weight: Weight for waveform similarity vs response
        distance_threshold: Maximum distance (in pixels) for coordinate matching
        verbose: Print matching details
    
    Returns:
        List of (ref_unit_id, align_unit_id) pairs
    """
    logger = logging.getLogger(__name__)
    final_pairs = []
    
    for ref_key in ref_keys:
        ref_data = ref_units[ref_key]
        ref_coord = np.array([ref_data.row, ref_data.col])
        
        candidates = []
        response_diffs = []
        waveform_diffs = []
        sum_diffs = []
        
        for align_key in align_keys:
            align_data = align_units[align_key]
            align_coord = np.array([align_data.row, align_data.col])
            
            # Check coordinate distance
            distance = np.linalg.norm(ref_coord - align_coord)
            if distance <= np.sqrt(distance_threshold):
                candidates.append(align_key)
                
                # Calculate response difference
                # Normalize signatures for comparison
                ref_resp = ref_data.response_signature
                align_resp = align_data.response_signature
                
                # Handle different lengths
                min_len = min(len(ref_resp), len(align_resp))
                if min_len > 0:
                    response_diff = np.linalg.norm(ref_resp[:min_len] - align_resp[:min_len])
                else:
                    response_diff = np.inf
                response_diffs.append(response_diff)
                
                # Calculate waveform difference
                ref_wave = ref_data.waveform
                align_wave = align_data.waveform
                min_len = min(len(ref_wave), len(align_wave))
                if min_len > 0:
                    waveform_diff = np.linalg.norm(ref_wave[:min_len] - align_wave[:min_len])
                else:
                    waveform_diff = np.inf
                waveform_diffs.append(waveform_diff)
                
                # Combined score
                sum_diff = response_diff + waveform_diff * waveform_weight
                sum_diffs.append(sum_diff)
        
        # Select best match
        if len(candidates) == 1:
            final_pairs.append((ref_key, candidates[0]))
            if verbose:
                logger.debug(f"Matched: {ref_key} -> {candidates[0]}")
        
        elif len(candidates) > 1:
            # Choose by combined score
            if np.argmin(response_diffs) == np.argmin(waveform_diffs):
                best_idx = np.argmin(response_diffs)
            else:
                best_idx = np.argmin(sum_diffs)
            
            final_pairs.append((ref_key, candidates[best_idx]))
            if verbose:
                logger.debug(f"Matched (multi): {ref_key} -> {candidates[best_idx]}")
    
    return final_pairs


def generate_alignment_links(
    ref_units: Dict[str, UnitData],
    align_units: Dict[str, UnitData],
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    waveform_weight: float = DEFAULT_WAVEFORM_WEIGHT,
    iteration_distances: List[int] = None,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Generate alignment links between two datasets using iterative matching.
    
    Args:
        ref_units: Reference (before) dataset units
        align_units: Alignment (after) dataset units
        quality_threshold: Minimum quality index for inclusion
        waveform_weight: Weight for waveform similarity
        iteration_distances: Distance thresholds for iterative matching
    
    Returns:
        Tuple of (pairs, ref_high_quality, align_high_quality)
    """
    logger = logging.getLogger(__name__)
    
    if iteration_distances is None:
        iteration_distances = DEFAULT_ITERATION_DISTANCES
    
    # Filter high-quality units
    ref_hq = filter_high_quality_units(ref_units, quality_threshold)
    align_hq = filter_high_quality_units(align_units, quality_threshold)
    
    logger.info(f"High-quality units: ref={len(ref_hq)}, align={len(align_hq)}")
    
    final_pairs = []
    remaining_ref = ref_hq.copy()
    remaining_align = align_hq.copy()
    
    for i, distance in enumerate(iteration_distances):
        logger.info(f"Iteration {i+1}: distance_threshold={distance}")
        
        pairs = get_unit_pair_list(
            ref_units, align_units,
            remaining_ref, remaining_align,
            waveform_weight=waveform_weight,
            distance_threshold=distance,
        )
        
        final_pairs.extend(pairs)
        
        # Remove matched units from remaining lists
        matched_ref = {p[0] for p in pairs}
        matched_align = {p[1] for p in pairs}
        
        remaining_ref = [k for k in remaining_ref if k not in matched_ref]
        remaining_align = [k for k in remaining_align if k not in matched_align]
        
        logger.info(f"  Matched: {len(pairs)}, remaining: ref={len(remaining_ref)}, align={len(remaining_align)}")
    
    logger.info(f"Total matched pairs: {len(final_pairs)} / {len(ref_hq)} ref units ({100*len(final_pairs)/max(1,len(ref_hq)):.1f}%)")
    
    return final_pairs, ref_hq, align_hq


def build_alignment_dataframe(
    pairs: List[Tuple[str, str]],
    before_file: str,
    after_file: str,
) -> pd.DataFrame:
    """
    Build a DataFrame of alignment pairs.
    
    Args:
        pairs: List of (before_unit, after_unit) tuples
        before_file: Name of before file
        after_file: Name of after file
    
    Returns:
        DataFrame with columns for before and after unit IDs
    """
    if not pairs:
        return pd.DataFrame(columns=[before_file, after_file])
    
    data = {
        before_file: [p[0] for p in pairs],
        after_file: [p[1] for p in pairs],
    }
    return pd.DataFrame(data)


# =============================================================================
# Merged Output
# =============================================================================

def save_aligned_pair(
    pair: FilePair,
    pairs: List[Tuple[str, str]],
    ref_units: Dict[str, UnitData],
    align_units: Dict[str, UnitData],
    ref_hq: List[str],
    align_hq: List[str],
    output_dir: Path = ALIGNED_OUTPUT_DIR,
) -> Path:
    """
    Save aligned pair data to a merged H5 file.
    
    Args:
        pair: FilePair object
        pairs: List of matched unit pairs
        ref_units: Reference units data
        align_units: Alignment units data
        ref_hq: High-quality reference unit IDs
        align_hq: High-quality alignment unit IDs
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    before_stem = pair.before_h5.stem
    after_stem = pair.after_h5.stem
    output_name = f"{before_stem}_to_{after_stem}_aligned.h5"
    output_path = output_dir / output_name
    
    logger.info(f"Saving aligned data to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['before_file'] = pair.before_file
        f.attrs['after_file'] = pair.after_file
        f.attrs['before_h5'] = str(pair.before_h5)
        f.attrs['after_h5'] = str(pair.after_h5)
        f.attrs['chip'] = pair.chip
        f.attrs['before_time'] = pair.before_time.isoformat()
        f.attrs['after_time'] = pair.after_time.isoformat()
        f.attrs['num_pairs'] = len(pairs)
        f.attrs['num_ref_hq'] = len(ref_hq)
        f.attrs['num_align_hq'] = len(align_hq)
        
        # Alignment connections
        connections = f.create_group('connections')
        if pairs:
            ref_ids = np.array([p[0] for p in pairs], dtype='S')
            align_ids = np.array([p[1] for p in pairs], dtype='S')
            connections.create_dataset('before_units', data=ref_ids)
            connections.create_dataset('after_units', data=align_ids)
        
        # High-quality unit lists
        connections.create_dataset('before_hq_units', data=np.array(ref_hq, dtype='S'))
        connections.create_dataset('after_hq_units', data=np.array(align_hq, dtype='S'))
        
        # Paired unit data
        paired_units = f.create_group('paired_units')
        
        # Count cell types in aligned pairs
        cell_type_counts = {"ON": 0, "OFF": 0, "ON_OFF": 0, "unknown": 0}
        
        for i, (ref_id, align_id) in enumerate(pairs):
            pair_group = paired_units.create_group(f'pair_{i:04d}')
            pair_group.attrs['before_unit'] = ref_id
            pair_group.attrs['after_unit'] = align_id
            
            # Before unit data (reference for cell type)
            if ref_id in ref_units:
                ref_data = ref_units[ref_id]
                before_grp = pair_group.create_group('before')
                before_grp.attrs['unit_id'] = ref_id
                before_grp.attrs['row'] = ref_data.row
                before_grp.attrs['col'] = ref_data.col
                before_grp.attrs['quality_index'] = ref_data.quality_index
                before_grp.attrs['cell_type'] = ref_data.cell_type  # Cell type from before file
                before_grp.create_dataset('waveform', data=ref_data.waveform)
                before_grp.create_dataset('response_signature', data=ref_data.response_signature)
                
                # Save step response trials if available
                if ref_data.step_response_trials is not None:
                    before_grp.create_dataset('step_response_trials', data=ref_data.step_response_trials)
                
                # Track cell type counts
                cell_type_counts[ref_data.cell_type] += 1
                
                # Also set cell type at pair level (from before file)
                pair_group.attrs['cell_type'] = ref_data.cell_type
            
            # After unit data
            if align_id in align_units:
                align_data = align_units[align_id]
                after_grp = pair_group.create_group('after')
                after_grp.attrs['unit_id'] = align_id
                after_grp.attrs['row'] = align_data.row
                after_grp.attrs['col'] = align_data.col
                after_grp.attrs['quality_index'] = align_data.quality_index
                after_grp.attrs['cell_type'] = align_data.cell_type
                after_grp.create_dataset('waveform', data=align_data.waveform)
                after_grp.create_dataset('response_signature', data=align_data.response_signature)
                
                # Save step response trials if available
                if align_data.step_response_trials is not None:
                    after_grp.create_dataset('step_response_trials', data=align_data.step_response_trials)
        
        # Save cell type summary
        f.attrs['cell_type_on'] = cell_type_counts['ON']
        f.attrs['cell_type_off'] = cell_type_counts['OFF']
        f.attrs['cell_type_on_off'] = cell_type_counts['ON_OFF']
        f.attrs['cell_type_unknown'] = cell_type_counts['unknown']
    
    logger.info(f"Saved {len(pairs)} aligned pairs to {output_path.name}")
    return output_path


# =============================================================================
# Main Pipeline
# =============================================================================

def process_alignment_pair(
    pair: FilePair,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    waveform_weight: float = DEFAULT_WAVEFORM_WEIGHT,
    iteration_distances: List[int] = None,
    output_dir: Path = ALIGNED_OUTPUT_DIR,
) -> Optional[Path]:
    """
    Process a single alignment pair.
    
    Args:
        pair: FilePair to process
        quality_threshold: Quality index threshold
        waveform_weight: Waveform weight
        iteration_distances: Distance thresholds
        output_dir: Output directory
    
    Returns:
        Path to output file, or None if failed
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing pair: {pair.before_file} -> {pair.after_file}")
    
    # Load unit data
    ref_units = load_unit_data_from_h5(pair.before_h5)
    align_units = load_unit_data_from_h5(pair.after_h5)
    
    if not ref_units:
        logger.error(f"No units loaded from {pair.before_h5}")
        return None
    
    if not align_units:
        logger.error(f"No units loaded from {pair.after_h5}")
        return None
    
    # Generate alignment
    pairs, ref_hq, align_hq = generate_alignment_links(
        ref_units, align_units,
        quality_threshold=quality_threshold,
        waveform_weight=waveform_weight,
        iteration_distances=iteration_distances,
    )
    
    # Save output
    output_path = save_aligned_pair(
        pair, pairs,
        ref_units, align_units,
        ref_hq, align_hq,
        output_dir=output_dir,
    )
    
    return output_path


def run_alignment_pipeline(
    h5_folder: Path = OUTPUT_DIR,
    output_dir: Path = ALIGNED_OUTPUT_DIR,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    waveform_weight: float = DEFAULT_WAVEFORM_WEIGHT,
    iteration_distances: List[int] = None,
) -> List[Path]:
    """
    Run the full alignment pipeline.
    
    Args:
        h5_folder: Folder containing H5 files
        output_dir: Output folder for aligned files
        quality_threshold: Quality index threshold
        waveform_weight: Waveform weight
        iteration_distances: Distance thresholds
    
    Returns:
        List of output file paths
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("HTR Alignment Pipeline")
    logger.info("=" * 60)
    
    # Load gsheet
    gsheet_df = load_gsheet_dataframe()
    if gsheet_df is None:
        return []
    
    # Find pairs
    pairs = find_alignment_pairs(gsheet_df, h5_folder)
    
    if not pairs:
        logger.warning("No alignment pairs found")
        return []
    
    # Process each pair
    output_files = []
    
    for i, pair in enumerate(pairs):
        logger.info(f"\n[{i+1}/{len(pairs)}] Processing alignment pair")
        
        try:
            output_path = process_alignment_pair(
                pair,
                quality_threshold=quality_threshold,
                waveform_weight=waveform_weight,
                iteration_distances=iteration_distances,
                output_dir=output_dir,
            )
            
            if output_path:
                output_files.append(output_path)
        
        except Exception as e:
            logger.error(f"Error processing pair: {e}")
            continue
    
    logger.info("=" * 60)
    logger.info(f"Alignment complete: {len(output_files)}/{len(pairs)} pairs processed")
    logger.info("=" * 60)
    
    return output_files


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Align units across before/after recording pairs"
    )
    parser.add_argument(
        "--h5-folder", type=Path, default=OUTPUT_DIR,
        help=f"Folder containing H5 files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--output", type=Path, default=ALIGNED_OUTPUT_DIR,
        help=f"Output directory (default: {ALIGNED_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=DEFAULT_QUALITY_THRESHOLD,
        help=f"Quality index threshold (default: {DEFAULT_QUALITY_THRESHOLD})"
    )
    parser.add_argument(
        "--waveform-weight", type=float, default=DEFAULT_WAVEFORM_WEIGHT,
        help=f"Waveform weight (default: {DEFAULT_WAVEFORM_WEIGHT})"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    print("=" * 70)
    print("HTR Alignment Pipeline")
    print("=" * 70)
    print(f"H5 Folder:    {args.h5_folder}")
    print(f"Output:       {args.output}")
    print(f"Quality:      {args.quality_threshold}")
    print(f"Waveform Wt:  {args.waveform_weight}")
    print(f"Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run pipeline
    output_files = run_alignment_pipeline(
        h5_folder=args.h5_folder,
        output_dir=args.output,
        quality_threshold=args.quality_threshold,
        waveform_weight=args.waveform_weight,
    )
    
    print()
    print("=" * 70)
    print(f"Complete: {len(output_files)} aligned files created")
    for f in output_files:
        print(f"  - {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
