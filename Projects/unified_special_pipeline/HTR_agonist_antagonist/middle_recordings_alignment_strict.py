"""
Middle Recordings Alignment Pipeline

Aligns the FIRST (before) file to all MIDDLE recordings.
For each chip:
- First recording: BEFORE (play_optimization_set6_a_ipRGC_without_step) - REFERENCE for cell types
- Middle recordings: agonist CP 1um recordings - ALIGNED
- Last recording: AFTER (play_optimization_set6_a_ipRGC_manual) - EXCLUDED

Cell types are determined from the BEFORE file's step response.
Alignment is based on location and waveform similarity.
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

# Output directory
ALIGNED_OUTPUT_DIR = OUTPUT_DIR / "aligned_middle"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChipRecordings:
    """All recordings for a single chip."""
    chip: str
    genotype: str
    before_file: Optional[str] = None
    before_h5_path: Optional[Path] = None  # H5 path for before file
    after_file: Optional[str] = None
    middle_files: List[str] = field(default_factory=list)
    middle_h5_paths: List[Path] = field(default_factory=list)


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


# Output directory for step-aligned files
ALIGNED_STEP_DIR = OUTPUT_DIR / "aligned"


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
            relative_time_ms = (spike - start_time) / 20.0  # Convert frames to ms at 20kHz
            if 0 <= relative_time_ms < trial_duration_ms:
                bin_idx = int(relative_time_ms // bin_size_ms)
                if 0 <= bin_idx < n_bins:
                    bin_counts[bin_idx] += 1
    
    # Convert to Hz
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
    """Classify cell type based on step response.
    
    A cell is classified as ON/OFF/ON_OFF only if:
    1. Response rate exceeds minimum threshold (min_response_hz)
    2. Difference from baseline exceeds minimum (min_diff_hz)
    3. Quality index exceeds threshold
    """
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
    
    # ON criteria: must meet ALL conditions
    min_response_on = on_params.get("min_response_hz", 5.0)
    min_diff_on = on_params.get("min_diff_hz", 3.0)
    
    is_on = (
        on_mean >= min_response_on and           # Response must be strong enough
        on_diff >= min_diff_on and               # Difference must be significant
        on_qi > on_params["quality_threshold"]   # Quality must be good
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
    
    # OFF criteria: must meet ALL conditions
    min_response_off = off_params.get("min_response_hz", 5.0)
    min_diff_off = off_params.get("min_diff_hz", 3.0)
    
    is_off = (
        off_mean >= min_response_off and         # Response must be strong enough
        off_diff >= min_diff_off and             # Difference must be significant
        off_qi > off_params["quality_threshold"] # Quality must be good
    )
    
    if is_on and is_off:
        return "ON_OFF"
    elif is_on:
        return "ON"
    elif is_off:
        return "OFF"
    else:
        return "unknown"


def load_unit_data_with_cell_type(h5_path: Path) -> Dict[str, UnitData]:
    """Load unit data from H5 file and classify cell types from step response."""
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
                
                # Load waveform
                waveform = np.array([])
                if 'waveform' in unit_group:
                    waveform = unit_group['waveform'][:]
                elif 'mean_waveform' in unit_group:
                    waveform = unit_group['mean_waveform'][:]
                
                # Load firing rate
                mean_firing_rate = 0.0
                if 'firing_rate_10hz' in unit_group:
                    fr = unit_group['firing_rate_10hz'][:]
                    mean_firing_rate = np.mean(fr) if len(fr) > 0 else 0.0
                
                # Load spike times
                spike_times = None
                if 'spike_times' in unit_group:
                    spike_times = unit_group['spike_times'][:]
                
                # Classify cell type from step response
                cell_type = "unknown"
                step_path = f'spike_times_sectioned/{STEP_STIMULUS}'
                if step_path in unit_group:
                    step_group = unit_group[step_path]
                    
                    # Load trial spike times
                    trial_spikes = []
                    if 'trials_spike_times' in step_group:
                        trials_group = step_group['trials_spike_times']
                        for trial_key in sorted(trials_group.keys(), key=lambda x: int(x)):
                            trial_spikes.append(trials_group[trial_key][:])
                    
                    # Load trial start/end times
                    if 'trials_start_end' in step_group and len(trial_spikes) > 0:
                        trial_start_end = step_group['trials_start_end'][:]
                        
                        # Compute PSTH and classify
                        psth = compute_step_response_psth(trial_spikes, trial_start_end)
                        cell_type = classify_cell_type(psth)
                
                units[unit_id] = UnitData(
                    unit_id=unit_id,
                    row=row,
                    col=col,
                    waveform=waveform,
                    mean_firing_rate=mean_firing_rate,
                    spike_times=spike_times,
                    cell_type=cell_type,
                )
            
            # Count cell types
            ct_counts = {}
            for u in units.values():
                ct_counts[u.cell_type] = ct_counts.get(u.cell_type, 0) + 1
            
            logger.info(f"Loaded {len(units)} units from {h5_path.name}, cell types: {ct_counts}")
    
    except Exception as e:
        logger.error(f"Error loading {h5_path}: {e}")
    
    return units


# =============================================================================
# Discover Middle Recordings
# =============================================================================

def find_middle_recordings() -> List[ChipRecordings]:
    """Find middle recordings for each chip (excluding first and last)."""
    logger = logging.getLogger(__name__)
    
    if not GSHEET_CSV_PATH.exists():
        logger.error(f"Gsheet CSV not found: {GSHEET_CSV_PATH}")
        return []
    
    df = pd.read_csv(GSHEET_CSV_PATH)
    
    # Filter for relevant date (2025.10.01)
    df = df[df['File_name'].str.contains('2025.10.01', na=False)].copy()
    
    # Find chips that have before condition
    before_rows = df[df['Condition'].str.contains(BEFORE_CONDITION, na=False)]
    chips_with_before = before_rows['Chip'].unique()
    
    results = []
    
    for chip in chips_with_before:
        chip_df = df[df['Chip'] == chip].sort_values('File_name')
        
        if len(chip_df) < 3:
            continue  # Need at least 3 recordings to have a middle
        
        # Get genotype
        genotype = chip_df['Genotype'].iloc[0]
        if pd.isna(genotype):
            genotype = "unknown"
        
        chip_rec = ChipRecordings(chip=str(chip), genotype=str(genotype))
        
        for _, row in chip_df.iterrows():
            fname = row['File_name']
            cond = str(row['Condition']) if pd.notna(row['Condition']) else ""
            
            if BEFORE_CONDITION in cond:
                chip_rec.before_file = fname
                # Set before H5 path
                before_h5 = OUTPUT_DIR / gsheet_to_h5_filename(fname)
                if before_h5.exists():
                    chip_rec.before_h5_path = before_h5
            elif AFTER_CONDITION in cond:
                chip_rec.after_file = fname
            else:
                # This is a middle file
                h5_path = OUTPUT_DIR / gsheet_to_h5_filename(fname)
                if h5_path.exists():
                    chip_rec.middle_files.append(fname)
                    chip_rec.middle_h5_paths.append(h5_path)
        
        # Only include if we have both before file and middle files
        if chip_rec.middle_files and chip_rec.before_h5_path:
            results.append(chip_rec)
            logger.info(f"Chip {chip}: 1 before + {len(chip_rec.middle_files)} middle recordings, genotype={chip_rec.genotype}")
    
    return results


# =============================================================================
# Unit Data Loading
# =============================================================================

def load_unit_data_from_h5(h5_path: Path) -> Dict[str, UnitData]:
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
            
            logger.info(f"Loaded {len(units)} units from {h5_path.name}")
    
    except Exception as e:
        logger.error(f"Error loading {h5_path}: {e}")
    
    return units


# =============================================================================
# Unit Alignment
# =============================================================================

def align_units_pairwise(
    ref_units: Dict[str, UnitData],
    align_units: Dict[str, UnitData],
    distance_threshold: int = 0,
) -> List[Tuple[str, str]]:
    """Find matching unit pairs based on location and waveform."""
    pairs = []
    
    for ref_key, ref_data in ref_units.items():
        ref_coord = np.array([ref_data.row, ref_data.col])
        
        best_match = None
        best_diff = np.inf
        
        for align_key, align_data in align_units.items():
            align_coord = np.array([align_data.row, align_data.col])
            
            distance = np.linalg.norm(ref_coord - align_coord)
            if distance <= np.sqrt(distance_threshold):
                # Calculate waveform difference
                ref_wave = ref_data.waveform
                align_wave = align_data.waveform
                min_len = min(len(ref_wave), len(align_wave))
                
                if min_len > 0:
                    ref_norm = ref_wave[:min_len] / (np.max(np.abs(ref_wave[:min_len])) + 1e-10)
                    align_norm = align_wave[:min_len] / (np.max(np.abs(align_wave[:min_len])) + 1e-10)
                    waveform_diff = np.linalg.norm(ref_norm - align_norm)
                    
                    if waveform_diff < best_diff:
                        best_diff = waveform_diff
                        best_match = align_key
        
        if best_match:
            pairs.append((ref_key, best_match))
    
    return pairs


def align_multiple_recordings(
    units_list: List[Dict[str, UnitData]],
    distance_thresholds: List[int] = None,
) -> Dict[str, List[str]]:
    """
    Align units across multiple recordings.
    Returns a dictionary mapping reference unit ID to list of matched unit IDs.
    """
    logger = logging.getLogger(__name__)
    
    if distance_thresholds is None:
        distance_thresholds = DEFAULT_ITERATION_DISTANCES
    
    if len(units_list) < 2:
        return {}
    
    # Use first recording as reference
    ref_units = units_list[0]
    
    # Alignment chains: ref_unit_id -> [matched_id_in_rec_1, matched_id_in_rec_2, ...]
    chains = {uid: [uid] for uid in ref_units.keys()}
    
    for i, align_units in enumerate(units_list[1:], 1):
        logger.info(f"Aligning recording {i+1} to reference...")
        
        # Build current reference from chains
        current_ref = {}
        for ref_id, chain in chains.items():
            if chain[-1] is not None and chain[-1] in units_list[i-1]:
                current_ref[ref_id] = units_list[i-1][chain[-1]]
        
        # Align with iterative distance
        all_pairs = []
        remaining_ref = list(current_ref.keys())
        remaining_align = list(align_units.keys())
        
        for dist_thresh in distance_thresholds:
            pairs = align_units_pairwise(
                {k: current_ref[k] for k in remaining_ref if k in current_ref},
                {k: align_units[k] for k in remaining_align},
                distance_threshold=dist_thresh,
            )
            
            matched_ref = set()
            matched_align = set()
            for ref_id, align_id in pairs:
                if ref_id not in matched_ref and align_id not in matched_align:
                    all_pairs.append((ref_id, align_id))
                    matched_ref.add(ref_id)
                    matched_align.add(align_id)
            
            remaining_ref = [k for k in remaining_ref if k not in matched_ref]
            remaining_align = [k for k in remaining_align if k not in matched_align]
        
        # Update chains
        pair_dict = dict(all_pairs)
        for ref_id in chains:
            if ref_id in pair_dict:
                chains[ref_id].append(pair_dict[ref_id])
            else:
                chains[ref_id].append(None)
        
        logger.info(f"  Matched: {len(all_pairs)} units")
    
    # Filter to only complete chains (matched in all recordings)
    complete_chains = {k: v for k, v in chains.items() if None not in v}
    logger.info(f"Complete chains: {len(complete_chains)} / {len(chains)}")
    
    return complete_chains


# =============================================================================
# Save Aligned Data
# =============================================================================

def save_aligned_middle_with_before(
    output_path: Path,
    chip_rec: ChipRecordings,
    before_units: Dict[str, UnitData],
    middle_units_list: List[Dict[str, UnitData]],
    chains: Dict[str, List[str]],
) -> None:
    """Save aligned middle recordings data with cell types from before file.
    
    chains: {before_unit_id: [before_id, middle_1_id, middle_2_id, ...]}
    """
    logger = logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Save metadata
        f.attrs['chip'] = chip_rec.chip
        f.attrs['genotype'] = chip_rec.genotype
        f.attrs['before_file'] = chip_rec.before_file if chip_rec.before_file else ""
        f.attrs['num_recordings'] = len(chip_rec.middle_files)
        f.attrs['num_chains'] = len(chains)
        f.attrs['middle_files'] = [str(p) for p in chip_rec.middle_files]
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['alignment_method'] = 'before_to_middle_chain'
        
        # Save aligned units
        aligned_units = f.create_group('aligned_units')
        
        for chain_idx, (ref_id, unit_ids) in enumerate(chains.items()):
            chain_group = aligned_units.create_group(f'chain_{chain_idx:04d}')
            chain_group.attrs['reference_unit'] = ref_id
            
            # unit_ids[0] is from before file, unit_ids[1:] are from middle files
            middle_unit_ids = unit_ids[1:]  # Skip the before unit ID
            chain_group.attrs['unit_ids'] = middle_unit_ids
            
            # Get cell type from before file's unit
            cell_type = "unknown"
            if ref_id in before_units:
                cell_type = before_units[ref_id].cell_type
            chain_group.attrs['cell_type'] = cell_type
            
            # Save data from each middle recording (skip before file)
            for rec_idx, unit_id in enumerate(middle_unit_ids):
                rec_group = chain_group.create_group(f'recording_{rec_idx}')
                rec_group.attrs['file'] = chip_rec.middle_files[rec_idx]
                rec_group.attrs['unit_id'] = unit_id
                
                if rec_idx < len(middle_units_list) and unit_id in middle_units_list[rec_idx]:
                    unit_data = middle_units_list[rec_idx][unit_id]
                    rec_group.attrs['row'] = unit_data.row
                    rec_group.attrs['col'] = unit_data.col
                    rec_group.attrs['mean_firing_rate'] = unit_data.mean_firing_rate
                    rec_group.attrs['cell_type'] = cell_type  # Use before file's cell type
                    rec_group.create_dataset('waveform', data=unit_data.waveform)
                    if unit_data.spike_times is not None:
                        rec_group.create_dataset('spike_times', data=unit_data.spike_times)
    
    # Count cell types
    ct_counts = {}
    for ref_id in chains:
        if ref_id in before_units:
            ct = before_units[ref_id].cell_type
            ct_counts[ct] = ct_counts.get(ct, 0) + 1
    
    logger.info(f"Saved {len(chains)} aligned chains to {output_path.name}, cell types: {ct_counts}")


# =============================================================================
# Main Pipeline
# =============================================================================

def process_chip_recordings(chip_rec: ChipRecordings) -> Optional[Path]:
    """Process middle recordings for a single chip.
    
    Uses the BEFORE file as reference for:
    1. Cell type classification (from step response)
    2. Alignment reference (first in chain)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"\nProcessing Chip {chip_rec.chip}")
    logger.info(f"  Genotype: {chip_rec.genotype}")
    logger.info(f"  Before file: {chip_rec.before_file}")
    logger.info(f"  Middle recordings: {len(chip_rec.middle_files)}")
    
    # Load before file with cell type classification from step response
    if not chip_rec.before_h5_path:
        logger.warning(f"No before file H5 for chip {chip_rec.chip}")
        return None
    
    before_units = load_unit_data_with_cell_type(chip_rec.before_h5_path)
    if not before_units:
        logger.warning(f"Failed to load before file for chip {chip_rec.chip}")
        return None
    
    # Count cell types from before file
    ct_counts = {}
    for u in before_units.values():
        ct_counts[u.cell_type] = ct_counts.get(u.cell_type, 0) + 1
    logger.info(f"  Cell types from before: {ct_counts}")
    
    # Load all middle recordings
    middle_units_list = []
    for h5_path in chip_rec.middle_h5_paths:
        units = load_unit_data_from_h5(h5_path)
        if units:
            middle_units_list.append(units)
    
    if not middle_units_list:
        logger.warning(f"No middle recordings loaded for chip {chip_rec.chip}")
        return None
    
    # Create units list with before as first (reference)
    # Before file is the reference, middle files follow
    units_list = [before_units] + middle_units_list
    
    # Align: before -> middle_1 -> middle_2 -> ...
    chains = align_multiple_recordings(units_list)
    
    if not chains:
        logger.warning(f"No complete alignment chains for chip {chip_rec.chip}")
        return None
    
    # Save (only save middle recordings data, but with cell types from before)
    output_name = f"chip_{chip_rec.chip}_middle_aligned.h5"
    output_path = ALIGNED_OUTPUT_DIR / output_name
    
    save_aligned_middle_with_before(output_path, chip_rec, before_units, middle_units_list, chains)
    
    return output_path


def run_alignment_pipeline() -> List[Path]:
    """Run the full alignment pipeline."""
    logger = logging.getLogger(__name__)
    
    # Find middle recordings
    chip_recordings = find_middle_recordings()
    logger.info(f"\nFound {len(chip_recordings)} chips with middle recordings")
    
    if not chip_recordings:
        return []
    
    # Create output directory
    ALIGNED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each chip
    output_files = []
    for chip_rec in chip_recordings:
        output = process_chip_recordings(chip_rec)
        if output:
            output_files.append(output)
    
    logger.info("=" * 60)
    logger.info(f"Alignment complete: {len(output_files)}/{len(chip_recordings)} chips processed")
    logger.info("=" * 60)
    
    return output_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Middle Recordings Alignment Pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    output_files = run_alignment_pipeline()
    
    print("=" * 70)
    print("Middle Recordings Alignment Pipeline")
    print("=" * 70)
    print(f"Output:       {ALIGNED_OUTPUT_DIR}")
    print(f"Method:       Location + Waveform Chain Alignment")
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
