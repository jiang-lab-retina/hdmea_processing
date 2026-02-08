"""
Middle Recordings Alignment Pipeline (Default Method)

Uses BEFORE→AFTER alignment as the primary chain:
1. Load BEFORE→AFTER aligned pairs from alignment.py output
2. For each aligned pair, find corresponding units in MIDDLE recordings
3. If a middle recording is missing a unit, fill with zeros (empty spike times)
4. Save with cell type from BEFORE file

This ensures all units that have BEFORE→AFTER alignment also have data
for all middle recordings (either real or zero-filled).

Alternative method (strict chain matching) is in middle_recordings_alignment_strict.py
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

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
DEFAULT_ITERATION_DISTANCES = [0, 1, 2]

# Output directories
ALIGNED_OUTPUT_DIR = OUTPUT_DIR / "aligned_middle"
BEFORE_AFTER_ALIGNED_DIR = OUTPUT_DIR / "aligned"

# Step response parameters for cell type classification
STEP_STIMULUS = "step_up_5s_5i_b0_3x"
BIN_SIZE_MS = 50
TRIAL_DURATION_MS = 10000

# Cell type classification parameters (1-3s ON, 6-8s OFF)
# Updated with minimum thresholds to prevent false positives
ON_CELL_PARAMS = {
    "quality_threshold": 0.1,
    "response_range": [20, 60],     # 1-3s (bins at 50ms)
    "baseline_range": [0, 10],
    "response_std_threshold": 0.1,
    "min_response_hz": 5.0,         # Minimum response rate in Hz
    "min_diff_hz": 3.0,             # Minimum difference from baseline
}

OFF_CELL_PARAMS = {
    "quality_threshold": 0.1,
    "response_range": [120, 160],   # 6-8s (bins at 50ms)
    "baseline_range": [100, 110],
    "response_std_threshold": 0.1,
    "min_response_hz": 5.0,         # Minimum response rate in Hz
    "min_diff_hz": 3.0,             # Minimum difference from baseline
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChipRecordings:
    """All recordings for a single chip."""
    chip: str
    genotype: str
    before_file: Optional[str] = None
    before_h5_path: Optional[Path] = None
    after_file: Optional[str] = None
    after_h5_path: Optional[Path] = None
    middle_files: List[str] = field(default_factory=list)
    middle_h5_paths: List[Path] = field(default_factory=list)


@dataclass
class UnitData:
    """Data for a single unit."""
    unit_id: str
    row: int
    col: int
    waveform: np.ndarray
    mean_firing_rate: float = 0.0
    spike_times: Optional[np.ndarray] = None
    cell_type: str = "unknown"


@dataclass
class AlignedPair:
    """Before→After aligned pair with cell type."""
    before_unit_id: str
    after_unit_id: str
    cell_type: str
    row: int
    col: int


# =============================================================================
# Utility Functions
# =============================================================================

def gsheet_to_h5_filename(gsheet_name: str) -> str:
    """Convert gsheet filename format to H5 filename format."""
    parts = gsheet_name.replace('.cmcr', '').split('.')
    if len(parts) >= 6:
        date_part = f"{parts[0]}.{parts[1]}.{parts[2]}"
        time_part = f"{parts[3]}.{parts[4]}.{parts[5]}"
        suffix = parts[6] if len(parts) > 6 else "Rec"
        return f"{date_part}-{time_part}-{suffix}.h5"
    return gsheet_name.replace('.cmcr', '.h5')


def extract_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp from filename."""
    try:
        base = filename.replace('.cmcr', '').replace('.h5', '')
        parts = base.split('.')
        if len(parts) >= 6:
            date_str = f"{parts[0]}.{parts[1]}.{parts[2]}"
            time_str = f"{parts[3]}.{parts[4]}.{parts[5]}"
            return datetime.strptime(f"{date_str} {time_str}", "%Y.%m.%d %H.%M.%S")
    except:
        pass
    return None


# =============================================================================
# Cell Type Classification from Step Response
# =============================================================================

def compute_step_response_psth(
    spike_times_trials: List[np.ndarray],
    trial_start_end: np.ndarray,
    bin_size_ms: int = BIN_SIZE_MS,
    trial_duration_ms: int = TRIAL_DURATION_MS,
) -> np.ndarray:
    """Compute PSTH from trial spike times."""
    n_bins = trial_duration_ms // bin_size_ms
    bin_counts = np.zeros(n_bins)
    n_trials = len(spike_times_trials)
    
    if n_trials == 0:
        return bin_counts
    
    for trial_idx, trial_spikes in enumerate(spike_times_trials):
        if trial_idx >= len(trial_start_end):
            continue
        
        start_time = trial_start_end[trial_idx, 0]
        
        for spike in trial_spikes:
            relative_time_ms = (spike - start_time) / 20.0  # 20kHz to ms
            if 0 <= relative_time_ms < trial_duration_ms:
                bin_idx = int(relative_time_ms // bin_size_ms)
                if 0 <= bin_idx < n_bins:
                    bin_counts[bin_idx] += 1
    
    bin_size_sec = bin_size_ms / 1000.0
    firing_rate = bin_counts / (n_trials * bin_size_sec)
    return firing_rate


def get_quality_index(response: np.ndarray) -> float:
    """Calculate quality index (variance ratio)."""
    if len(response) == 0:
        return 0.0
    mean_val = np.mean(response)
    if mean_val == 0:
        return 0.0
    variance = np.var(response)
    return variance / (mean_val ** 2)


def classify_cell_type(
    step_response: np.ndarray,
    on_params: dict = ON_CELL_PARAMS,
    off_params: dict = OFF_CELL_PARAMS,
) -> str:
    """Classify cell type based on step response."""
    if len(step_response) < 160:
        return "unknown"
    
    # ON response
    on_range = on_params["response_range"]
    on_baseline = on_params["baseline_range"]
    on_response = step_response[on_range[0]:on_range[1]]
    on_base = step_response[on_baseline[0]:on_baseline[1]]
    
    on_mean = np.mean(on_response)
    on_base_mean = np.mean(on_base)
    on_diff = on_mean - on_base_mean
    on_qi = get_quality_index(on_response)
    
    min_response_on = on_params.get("min_response_hz", 5.0)
    min_diff_on = on_params.get("min_diff_hz", 3.0)
    
    is_on = (
        on_mean >= min_response_on and
        on_diff >= min_diff_on and
        on_qi > on_params["quality_threshold"]
    )
    
    # OFF response
    off_range = off_params["response_range"]
    off_baseline = off_params["baseline_range"]
    off_response = step_response[off_range[0]:off_range[1]]
    off_base = step_response[off_baseline[0]:off_baseline[1]]
    
    off_mean = np.mean(off_response)
    off_base_mean = np.mean(off_base)
    off_diff = off_mean - off_base_mean
    off_qi = get_quality_index(off_response)
    
    min_response_off = off_params.get("min_response_hz", 5.0)
    min_diff_off = off_params.get("min_diff_hz", 3.0)
    
    is_off = (
        off_mean >= min_response_off and
        off_diff >= min_diff_off and
        off_qi > off_params["quality_threshold"]
    )
    
    if is_on and is_off:
        return "ON_OFF"
    elif is_on:
        return "ON"
    elif is_off:
        return "OFF"
    else:
        return "unknown"


# =============================================================================
# Load Before→After Alignment
# =============================================================================

def load_before_after_alignment(aligned_h5_path: Path) -> Tuple[List[AlignedPair], str, str]:
    """Load BEFORE→AFTER aligned pairs from alignment.py output.
    
    Returns:
        Tuple of (aligned_pairs, before_file, after_file)
    """
    logger = logging.getLogger(__name__)
    pairs = []
    before_file = ""
    after_file = ""
    
    try:
        with h5py.File(aligned_h5_path, 'r') as f:
            before_file = f.attrs.get('before_file', '')
            after_file = f.attrs.get('after_file', '')
            chip = f.attrs.get('chip', '')
            
            if 'paired_units' not in f:
                return pairs, before_file, after_file
            
            for pair_key in f['paired_units'].keys():
                pair_group = f['paired_units'][pair_key]
                before_unit = pair_group.attrs.get('before_unit', '')
                after_unit = pair_group.attrs.get('after_unit', '')
                cell_type = pair_group.attrs.get('cell_type', 'unknown')
                
                # Get coordinates from before unit
                row = 0
                col = 0
                if 'before' in pair_group:
                    row = int(pair_group['before'].attrs.get('row', 0))
                    col = int(pair_group['before'].attrs.get('col', 0))
                
                pairs.append(AlignedPair(
                    before_unit_id=before_unit,
                    after_unit_id=after_unit,
                    cell_type=cell_type,
                    row=row,
                    col=col,
                ))
            
            logger.info(f"Loaded {len(pairs)} aligned pairs from {aligned_h5_path.name}")
    
    except Exception as e:
        logger.error(f"Error loading {aligned_h5_path}: {e}")
    
    return pairs, before_file, after_file


# =============================================================================
# Discover Middle Recordings
# =============================================================================

def find_chip_recordings(h5_folder: Path = None) -> Dict[str, ChipRecordings]:
    """Find all recordings for each chip.
    
    Args:
        h5_folder: Folder containing H5 files (default: OUTPUT_DIR)
    """
    logger = logging.getLogger(__name__)
    h5_folder = h5_folder or OUTPUT_DIR
    
    if not GSHEET_CSV_PATH.exists():
        logger.error(f"Gsheet CSV not found: {GSHEET_CSV_PATH}")
        return {}
    
    # Detect date pattern from H5 files in the folder
    h5_files = list(h5_folder.glob('*.h5'))
    if not h5_files:
        logger.warning(f"No H5 files found in {h5_folder}")
        return {}
    
    # Extract date patterns from filenames (e.g., "2025.10.01" or "2025.10.02")
    date_patterns = set()
    for h5_file in h5_files:
        # Extract date from filename like "2025.10.02-09.43.24-Rec.h5"
        parts = h5_file.stem.split('-')
        if len(parts) >= 1:
            date_part = parts[0]  # e.g., "2025.10.02"
            date_patterns.add(date_part)
    
    logger.info(f"Detected date patterns in H5 folder: {date_patterns}")
    
    df = pd.read_csv(GSHEET_CSV_PATH)
    
    # Filter for any of the detected dates
    if date_patterns:
        date_filter = '|'.join(date_patterns)
        df = df[df['File_name'].str.contains(date_filter, na=False, regex=True)].copy()
    else:
        logger.warning("Could not detect date patterns from H5 files")
        return {}
    
    before_rows = df[df['Condition'].str.contains(BEFORE_CONDITION, na=False)]
    chips_with_before = before_rows['Chip'].unique()
    
    results = {}
    
    for chip in chips_with_before:
        chip_df = df[df['Chip'] == chip].sort_values('File_name')
        
        if len(chip_df) < 3:
            continue
        
        genotype = chip_df['Genotype'].iloc[0]
        if pd.isna(genotype):
            genotype = "unknown"
        
        chip_rec = ChipRecordings(chip=str(chip), genotype=str(genotype))
        
        for _, row in chip_df.iterrows():
            fname = row['File_name']
            cond = str(row['Condition']) if pd.notna(row['Condition']) else ""
            h5_path = h5_folder / gsheet_to_h5_filename(fname)
            
            if BEFORE_CONDITION in cond:
                chip_rec.before_file = fname
                if h5_path.exists():
                    chip_rec.before_h5_path = h5_path
            elif AFTER_CONDITION in cond:
                chip_rec.after_file = fname
                if h5_path.exists():
                    chip_rec.after_h5_path = h5_path
            else:
                if h5_path.exists():
                    chip_rec.middle_files.append(fname)
                    chip_rec.middle_h5_paths.append(h5_path)
        
        if chip_rec.before_h5_path and chip_rec.middle_h5_paths:
            results[str(chip)] = chip_rec
            logger.info(f"Chip {chip}: before + {len(chip_rec.middle_files)} middle + after, genotype={chip_rec.genotype}")
    
    return results


# =============================================================================
# Unit Matching for Middle Recordings
# =============================================================================

def load_units_from_h5(h5_path: Path) -> Dict[str, UnitData]:
    """Load unit data from H5 file."""
    logger = logging.getLogger(__name__)
    units = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'units' not in f:
                return units
            
            for unit_id in f['units'].keys():
                unit_group = f['units'][unit_id]
                
                row = int(unit_group.attrs.get('row', 0))
                col = int(unit_group.attrs.get('col', 0))
                
                waveform = np.array([])
                if 'waveform' in unit_group:
                    waveform = unit_group['waveform'][:]
                elif 'mean_waveform' in unit_group:
                    waveform = unit_group['mean_waveform'][:]
                
                mean_firing_rate = 0.0
                if 'firing_rate_10hz' in unit_group:
                    fr = unit_group['firing_rate_10hz'][:]
                    mean_firing_rate = np.mean(fr) if len(fr) > 0 else 0.0
                
                spike_times = None
                if 'spike_times' in unit_group:
                    spike_times = unit_group['spike_times'][:]
                
                units[unit_id] = UnitData(
                    unit_id=unit_id,
                    row=row,
                    col=col,
                    waveform=waveform,
                    mean_firing_rate=mean_firing_rate,
                    spike_times=spike_times,
                )
    
    except Exception as e:
        logger.error(f"Error loading {h5_path}: {e}")
    
    return units


def find_matching_unit(
    target_row: int,
    target_col: int,
    target_waveform: np.ndarray,
    units: Dict[str, UnitData],
    distance_thresholds: List[int] = DEFAULT_ITERATION_DISTANCES,
) -> Optional[str]:
    """Find a matching unit in a recording by location and waveform.
    
    Returns:
        unit_id if found, None otherwise
    """
    target_coord = np.array([target_row, target_col])
    
    for dist_thresh in distance_thresholds:
        candidates = []
        waveform_diffs = []
        
        for unit_id, unit_data in units.items():
            coord = np.array([unit_data.row, unit_data.col])
            distance = np.linalg.norm(target_coord - coord)
            
            if distance <= np.sqrt(dist_thresh):
                candidates.append(unit_id)
                
                # Calculate waveform difference
                ref_wave = target_waveform
                align_wave = unit_data.waveform
                min_len = min(len(ref_wave), len(align_wave))
                
                if min_len > 0:
                    ref_norm = ref_wave[:min_len] / (np.max(np.abs(ref_wave[:min_len])) + 1e-10)
                    align_norm = align_wave[:min_len] / (np.max(np.abs(align_wave[:min_len])) + 1e-10)
                    diff = np.linalg.norm(ref_norm - align_norm)
                    waveform_diffs.append(diff)
                else:
                    waveform_diffs.append(np.inf)
        
        if candidates:
            best_idx = np.argmin(waveform_diffs)
            return candidates[best_idx]
    
    return None


# =============================================================================
# Process Chip with Zero-Filled Missing Recordings
# =============================================================================

def process_chip_with_zerofill(
    chip_rec: ChipRecordings,
    aligned_pairs: List[AlignedPair],
    output_folder: Path = None,
) -> Optional[Path]:
    """Process a chip's middle recordings with zero-fill for missing data.
    
    Uses BEFORE→AFTER alignment as the primary chain.
    For each aligned pair, finds the unit in each middle recording.
    If not found, fills with zeros (empty spike times).
    
    Args:
        chip_rec: Chip recordings data
        aligned_pairs: BEFORE→AFTER aligned pairs
        output_folder: Output directory (default: ALIGNED_OUTPUT_DIR)
    """
    output_folder = output_folder or ALIGNED_OUTPUT_DIR
    logger = logging.getLogger(__name__)
    
    if not chip_rec.middle_h5_paths:
        logger.warning(f"No middle recordings for chip {chip_rec.chip}")
        return None
    
    logger.info(f"\nProcessing Chip {chip_rec.chip} with zero-fill method")
    logger.info(f"  BEFORE→AFTER pairs: {len(aligned_pairs)}")
    logger.info(f"  Middle recordings: {len(chip_rec.middle_h5_paths)}")
    
    # Load the before file to get waveforms for matching
    before_units = load_units_from_h5(chip_rec.before_h5_path)
    
    # Load all middle recordings
    middle_units_list = []
    for h5_path in chip_rec.middle_h5_paths:
        units = load_units_from_h5(h5_path)
        middle_units_list.append(units)
        logger.info(f"  Loaded {len(units)} units from {h5_path.name}")
    
    # For each aligned pair, find or zero-fill middle recordings
    chains = []  # List of dicts: {before_unit, after_unit, cell_type, middle_units}
    
    matched_count = 0
    zerofill_count = 0
    
    skipped_no_first_match = 0
    
    for pair in aligned_pairs:
        # Get waveform from before unit for matching
        if pair.before_unit_id not in before_units:
            continue
        
        before_unit = before_units[pair.before_unit_id]
        
        # Find matching unit in each middle recording
        middle_matches = []
        skip_this_chain = False
        
        for rec_idx, middle_units in enumerate(middle_units_list):
            match_id = find_matching_unit(
                target_row=pair.row,
                target_col=pair.col,
                target_waveform=before_unit.waveform,
                units=middle_units,
            )
            
            if match_id:
                middle_matches.append({
                    'unit_id': match_id,
                    'spike_times': middle_units[match_id].spike_times,
                    'zero_filled': False,
                })
                matched_count += 1
            else:
                # FIRST middle recording: must have a real match, no zero-fill allowed
                if rec_idx == 0:
                    skip_this_chain = True
                    skipped_no_first_match += 1
                    break
                
                # REST of middle recordings (2nd, 3rd, etc.): zero-fill allowed
                middle_matches.append({
                    'unit_id': None,
                    'spike_times': np.array([]),  # Empty array
                    'zero_filled': True,
                })
                zerofill_count += 1
        
        # Skip this chain if first middle recording didn't match
        if skip_this_chain:
            continue
        
        chains.append({
            'before_unit': pair.before_unit_id,
            'after_unit': pair.after_unit_id,
            'cell_type': pair.cell_type,
            'row': pair.row,
            'col': pair.col,
            'waveform': before_unit.waveform,
            'middle_matches': middle_matches,
        })
    
    logger.info(f"  Matched: {matched_count}, Zero-filled: {zerofill_count}, Skipped (no first match): {skipped_no_first_match}")
    
    # Save aligned data
    if not chains:
        logger.warning(f"No chains to save for chip {chip_rec.chip}")
        return None
    
    # Name output file by first and last middle file names
    first_middle = Path(chip_rec.middle_files[0]).stem if chip_rec.middle_files else "unknown"
    last_middle = Path(chip_rec.middle_files[-1]).stem if chip_rec.middle_files else "unknown"
    output_name = f"{first_middle}_to_{last_middle}_middle_aligned.h5"
    output_path = output_folder / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['chip'] = chip_rec.chip
        f.attrs['genotype'] = chip_rec.genotype
        f.attrs['before_file'] = chip_rec.before_file if chip_rec.before_file else ""
        f.attrs['after_file'] = chip_rec.after_file if chip_rec.after_file else ""
        f.attrs['num_recordings'] = len(chip_rec.middle_files)
        f.attrs['num_chains'] = len(chains)
        f.attrs['middle_files'] = [str(f) for f in chip_rec.middle_files]
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['alignment_method'] = 'before_after_chain_zerofill'
        f.attrs['matched_count'] = matched_count
        f.attrs['zerofill_count'] = zerofill_count
        
        # Save aligned units
        aligned_units = f.create_group('aligned_units')
        
        for chain_idx, chain in enumerate(chains):
            chain_group = aligned_units.create_group(f'chain_{chain_idx:04d}')
            chain_group.attrs['reference_unit'] = chain['before_unit']
            chain_group.attrs['after_unit'] = chain['after_unit']
            chain_group.attrs['cell_type'] = chain['cell_type']
            chain_group.attrs['row'] = chain['row']
            chain_group.attrs['col'] = chain['col']
            
            # Save waveform from before file
            chain_group.create_dataset('waveform', data=chain['waveform'])
            
            # Save middle recording data
            middle_unit_ids = []
            zero_filled_flags = []
            
            for rec_idx, match in enumerate(chain['middle_matches']):
                rec_group = chain_group.create_group(f'recording_{rec_idx}')
                rec_group.attrs['file'] = chip_rec.middle_files[rec_idx]
                rec_group.attrs['unit_id'] = match['unit_id'] if match['unit_id'] else ""
                rec_group.attrs['zero_filled'] = match['zero_filled']
                
                # Save spike times (either real or empty)
                if match['spike_times'] is not None:
                    rec_group.create_dataset('spike_times', data=match['spike_times'])
                else:
                    rec_group.create_dataset('spike_times', data=np.array([]))
                
                middle_unit_ids.append(match['unit_id'] if match['unit_id'] else "")
                zero_filled_flags.append(match['zero_filled'])
            
            chain_group.attrs['middle_unit_ids'] = middle_unit_ids
            chain_group.attrs['zero_filled_flags'] = zero_filled_flags
    
    # Count cell types
    ct_counts = {}
    for chain in chains:
        ct = chain['cell_type']
        ct_counts[ct] = ct_counts.get(ct, 0) + 1
    
    logger.info(f"Saved {len(chains)} chains to {output_path.name}, cell types: {ct_counts}")
    
    return output_path


# =============================================================================
# Main Pipeline
# =============================================================================

def run_alignment_pipeline(
    h5_folder: Path = None,
    aligned_folder: Path = None,
    output_folder: Path = None,
) -> List[Path]:
    """Run the full alignment pipeline using BEFORE→AFTER chain with zero-fill.
    
    Args:
        h5_folder: Folder containing H5 files (default: OUTPUT_DIR)
        aligned_folder: Folder with BEFORE→AFTER aligned files (default: h5_folder/aligned)
        output_folder: Output directory (default: h5_folder/aligned_middle)
    """
    logger = logging.getLogger(__name__)
    
    # Set defaults
    h5_folder = h5_folder or OUTPUT_DIR
    aligned_folder = aligned_folder or (h5_folder / "aligned")
    output_folder = output_folder or (h5_folder / "aligned_middle")
    
    # First, ensure alignment.py has been run
    aligned_files = list(aligned_folder.glob('*.h5'))
    if not aligned_files:
        logger.info("No BEFORE→AFTER aligned files found. Running alignment.py first...")
        # Import and run alignment
        try:
            from alignment import run_alignment_pipeline as run_before_after_alignment
            run_before_after_alignment(h5_folder=h5_folder, output_dir=aligned_folder)
            aligned_files = list(aligned_folder.glob('*.h5'))
        except Exception as e:
            logger.error(f"Failed to run alignment.py: {e}")
            return []
    
    logger.info(f"Found {len(aligned_files)} BEFORE→AFTER aligned files")
    
    # Find all chip recordings
    chip_recordings = find_chip_recordings(h5_folder=h5_folder)
    logger.info(f"Found {len(chip_recordings)} chips with middle recordings")
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process each chip
    output_files = []
    
    for chip, chip_rec in chip_recordings.items():
        # Find the corresponding BEFORE→AFTER alignment file
        before_h5_stem = chip_rec.before_h5_path.stem if chip_rec.before_h5_path else ""
        after_h5_stem = chip_rec.after_h5_path.stem if chip_rec.after_h5_path else ""
        
        aligned_file = None
        for af in aligned_files:
            if before_h5_stem in af.name and 'aligned' in af.name:
                aligned_file = af
                break
        
        if not aligned_file:
            logger.warning(f"No BEFORE→AFTER alignment found for chip {chip}")
            continue
        
        # Load aligned pairs
        aligned_pairs, _, _ = load_before_after_alignment(aligned_file)
        
        if not aligned_pairs:
            logger.warning(f"No aligned pairs in {aligned_file.name}")
            continue
        
        # Process with zero-fill
        output = process_chip_with_zerofill(chip_rec, aligned_pairs, output_folder=output_folder)
        if output:
            output_files.append(output)
    
    logger.info("=" * 60)
    logger.info(f"Alignment complete: {len(output_files)}/{len(chip_recordings)} chips processed")
    logger.info("=" * 60)
    
    return output_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Middle Recordings Alignment Pipeline (BEFORE→AFTER chain with zero-fill)"
    )
    parser.add_argument(
        "--h5-folder", type=Path, default=OUTPUT_DIR,
        help=f"Folder containing H5 files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--aligned-folder", type=Path, default=None,
        help="Folder containing BEFORE→AFTER aligned files (default: h5_folder/aligned)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: h5_folder/aligned_middle)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set defaults based on h5_folder
    h5_folder = args.h5_folder
    aligned_folder = args.aligned_folder or (h5_folder / "aligned")
    output_folder = args.output or (h5_folder / "aligned_middle")
    
    output_files = run_alignment_pipeline(
        h5_folder=h5_folder,
        aligned_folder=aligned_folder,
        output_folder=output_folder,
    )
    
    print("=" * 70)
    print("Middle Recordings Alignment Pipeline (Default: BEFORE->AFTER chain)")
    print("=" * 70)
    print(f"H5 Folder:    {h5_folder}")
    print(f"Aligned:      {aligned_folder}")
    print(f"Output:       {output_folder}")
    print(f"Method:       BEFORE->AFTER chain with zero-fill for missing middle")
    print("=" * 70)
    
    if output_files:
        print(f"\nComplete: {len(output_files)} aligned files created")
        for f in output_files:
            print(f"  - {f.name}")
    else:
        print("\nNo aligned files created")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
