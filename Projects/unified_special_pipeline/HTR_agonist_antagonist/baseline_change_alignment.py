"""
Baseline Change Alignment Pipeline

Aligns units across before/after recording pairs based on:
- Coordinate proximity (row, col electrode position)
- Waveform signature similarity

This is a simpler alignment that doesn't require step response data,
suitable for baseline comparisons.

Pairing logic:
- "Before" file: Condition contains play_optimization_set6_a_ipRGC_without_step()
- "After" file: Condition contains play_optimization_set6_a_ipRGC_manual() 
  AND comes after the before file (by timestamp) AND shares the same Chip number
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
DEFAULT_QUALITY_THRESHOLD = 0.01  # Lower threshold since we don't use step response
DEFAULT_WAVEFORM_WEIGHT = 1.0  # Only waveform matters
DEFAULT_ITERATION_DISTANCES = [0, 1, 2]

# Output directory for aligned files
ALIGNED_OUTPUT_DIR = OUTPUT_DIR / "aligned_baseline"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FilePair:
    """Represents a before/after file pair for alignment."""
    before_file: str
    after_file: str
    before_h5: Path
    after_h5: Path
    chip: str
    before_time: datetime
    after_time: datetime


@dataclass
class UnitData:
    """Data for a single unit (simplified for baseline alignment)."""
    unit_id: str
    row: int
    col: int
    waveform: np.ndarray
    mean_firing_rate: float = 0.0
    spike_times: Optional[np.ndarray] = None


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
# Gsheet and Pair Discovery
# =============================================================================

def load_gsheet_dataframe() -> pd.DataFrame:
    """Load the gsheet table."""
    if GSHEET_CSV_PATH.exists():
        return pd.read_csv(GSHEET_CSV_PATH)
    return pd.DataFrame()


def find_alignment_pairs() -> List[FilePair]:
    """Find before/after alignment pairs from gsheet."""
    logger = logging.getLogger(__name__)
    
    df = load_gsheet_dataframe()
    if df.empty:
        logger.warning("Empty gsheet dataframe")
        return []
    
    pairs = []
    
    # Find all before files
    before_rows = df[df['Condition'].str.contains(BEFORE_CONDITION, na=False)]
    after_rows = df[df['Condition'].str.contains(AFTER_CONDITION, na=False)]
    
    logger.info(f"Found {len(before_rows)} before rows and {len(after_rows)} after rows")
    
    for _, before_row in before_rows.iterrows():
        before_file = before_row['File_name']
        before_chip = str(before_row['Chip'])
        before_time = extract_timestamp_from_filename(before_file)
        
        if before_time is None:
            continue
        
        # Find matching after file (same chip, after timestamp)
        for _, after_row in after_rows.iterrows():
            after_file = after_row['File_name']
            after_chip = str(after_row['Chip'])
            after_time = extract_timestamp_from_filename(after_file)
            
            if after_time is None:
                continue
            
            if before_chip == after_chip and after_time > before_time:
                before_h5 = OUTPUT_DIR / gsheet_to_h5_filename(before_file)
                after_h5 = OUTPUT_DIR / gsheet_to_h5_filename(after_file)
                
                if before_h5.exists() and after_h5.exists():
                    pairs.append(FilePair(
                        before_file=before_file,
                        after_file=after_file,
                        before_h5=before_h5,
                        after_h5=after_h5,
                        chip=before_chip,
                        before_time=before_time,
                        after_time=after_time,
                    ))
                    logger.info(f"Found pair: {before_file} -> {after_file} (Chip: {before_chip})")
                    break  # Only take first matching after
    
    return pairs


# =============================================================================
# Unit Data Loading (Simplified)
# =============================================================================

def load_unit_data_from_h5(h5_path: Path) -> Dict[str, UnitData]:
    """
    Load unit data from H5 file (simplified for baseline alignment).
    Only loads waveform and position data.
    """
    logger = logging.getLogger(__name__)
    units = {}
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'units' not in f:
                logger.warning(f"No units group in {h5_path}")
                return units
            
            for unit_id in f['units'].keys():
                unit_group = f['units'][unit_id]
                
                # Get coordinates
                row = int(unit_group.attrs.get('row', 0))
                col = int(unit_group.attrs.get('col', 0))
                
                # Get waveform
                waveform = np.array([])
                if 'waveform' in unit_group:
                    waveform = unit_group['waveform'][:]
                elif 'mean_waveform' in unit_group:
                    waveform = unit_group['mean_waveform'][:]
                
                # Get mean firing rate
                mean_firing_rate = 0.0
                if 'firing_rate_10hz' in unit_group:
                    fr = unit_group['firing_rate_10hz'][:]
                    mean_firing_rate = np.mean(fr) if len(fr) > 0 else 0.0
                
                # Get spike times
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

def get_unit_pair_list(
    ref_units: Dict[str, UnitData],
    align_units: Dict[str, UnitData],
    ref_keys: List[str],
    align_keys: List[str],
    distance_threshold: int = 0,
) -> List[Tuple[str, str]]:
    """
    Find matching unit pairs based on location and waveform.
    
    Args:
        ref_units: Reference dataset units
        align_units: Alignment dataset units  
        ref_keys: List of reference unit IDs to match
        align_keys: List of alignment unit IDs to consider
        distance_threshold: Maximum distance for coordinate matching
    
    Returns:
        List of (ref_unit_id, align_unit_id) pairs
    """
    final_pairs = []
    
    for ref_key in ref_keys:
        ref_data = ref_units[ref_key]
        ref_coord = np.array([ref_data.row, ref_data.col])
        
        candidates = []
        waveform_diffs = []
        
        for align_key in align_keys:
            align_data = align_units[align_key]
            align_coord = np.array([align_data.row, align_data.col])
            
            # Check coordinate distance
            distance = np.linalg.norm(ref_coord - align_coord)
            if distance <= np.sqrt(distance_threshold):
                candidates.append(align_key)
                
                # Calculate waveform difference
                ref_wave = ref_data.waveform
                align_wave = align_data.waveform
                min_len = min(len(ref_wave), len(align_wave))
                if min_len > 0:
                    # Normalize waveforms before comparison
                    ref_norm = ref_wave[:min_len] / (np.max(np.abs(ref_wave[:min_len])) + 1e-10)
                    align_norm = align_wave[:min_len] / (np.max(np.abs(align_wave[:min_len])) + 1e-10)
                    waveform_diff = np.linalg.norm(ref_norm - align_norm)
                else:
                    waveform_diff = np.inf
                waveform_diffs.append(waveform_diff)
        
        # Select best match
        if candidates:
            best_idx = np.argmin(waveform_diffs)
            final_pairs.append((ref_key, candidates[best_idx]))
    
    return final_pairs


def generate_alignment_links(
    ref_units: Dict[str, UnitData],
    align_units: Dict[str, UnitData],
    distance_thresholds: List[int] = None,
) -> List[Tuple[str, str]]:
    """
    Generate alignment links using iterative matching.
    
    Args:
        ref_units: Reference units
        align_units: Alignment units
        distance_thresholds: List of distance thresholds for iterations
    
    Returns:
        List of matched (ref_unit_id, align_unit_id) pairs
    """
    logger = logging.getLogger(__name__)
    
    if distance_thresholds is None:
        distance_thresholds = DEFAULT_ITERATION_DISTANCES
    
    # Use all units (no quality filtering for baseline)
    ref_keys = list(ref_units.keys())
    align_keys = list(align_units.keys())
    
    logger.info(f"Units: ref={len(ref_keys)}, align={len(align_keys)}")
    
    all_pairs = []
    
    for dist_thresh in distance_thresholds:
        logger.info(f"Iteration: distance_threshold={dist_thresh}")
        
        pairs = get_unit_pair_list(
            ref_units, align_units,
            ref_keys, align_keys,
            distance_threshold=dist_thresh,
        )
        
        # Remove matched units for next iteration
        matched_ref = set()
        matched_align = set()
        for ref_id, align_id in pairs:
            if ref_id not in matched_ref and align_id not in matched_align:
                all_pairs.append((ref_id, align_id))
                matched_ref.add(ref_id)
                matched_align.add(align_id)
        
        ref_keys = [k for k in ref_keys if k not in matched_ref]
        align_keys = [k for k in align_keys if k not in matched_align]
        
        logger.info(f"  Matched: {len(matched_ref)}, remaining: ref={len(ref_keys)}, align={len(align_keys)}")
        
        if not ref_keys or not align_keys:
            break
    
    logger.info(f"Total matched pairs: {len(all_pairs)} / {len(ref_units)} ref units "
               f"({len(all_pairs)/len(ref_units)*100:.1f}%)")
    
    return all_pairs


# =============================================================================
# Save Aligned Data
# =============================================================================

def save_aligned_pair(
    output_path: Path,
    ref_units: Dict[str, UnitData],
    align_units: Dict[str, UnitData],
    pairs: List[Tuple[str, str]],
    file_pair: FilePair,
) -> None:
    """Save aligned pair data to H5 file."""
    logger = logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Save metadata
        f.attrs['before_file'] = file_pair.before_file
        f.attrs['after_file'] = file_pair.after_file
        f.attrs['before_h5'] = str(file_pair.before_h5)
        f.attrs['after_h5'] = str(file_pair.after_h5)
        f.attrs['chip'] = file_pair.chip
        f.attrs['before_time'] = file_pair.before_time.isoformat()
        f.attrs['after_time'] = file_pair.after_time.isoformat()
        f.attrs['num_pairs'] = len(pairs)
        f.attrs['num_ref_units'] = len(ref_units)
        f.attrs['num_align_units'] = len(align_units)
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['alignment_method'] = 'location_waveform'
        
        # Save paired units
        paired_units = f.create_group('paired_units')
        
        for i, (ref_id, align_id) in enumerate(pairs):
            pair_group = paired_units.create_group(f'pair_{i:04d}')
            pair_group.attrs['before_unit'] = ref_id
            pair_group.attrs['after_unit'] = align_id
            
            if ref_id in ref_units:
                ref_data = ref_units[ref_id]
                before_grp = pair_group.create_group('before')
                before_grp.attrs['row'] = ref_data.row
                before_grp.attrs['col'] = ref_data.col
                before_grp.attrs['mean_firing_rate'] = ref_data.mean_firing_rate
                before_grp.create_dataset('waveform', data=ref_data.waveform)
                if ref_data.spike_times is not None:
                    before_grp.create_dataset('spike_times', data=ref_data.spike_times)
            
            if align_id in align_units:
                align_data = align_units[align_id]
                after_grp = pair_group.create_group('after')
                after_grp.attrs['row'] = align_data.row
                after_grp.attrs['col'] = align_data.col
                after_grp.attrs['mean_firing_rate'] = align_data.mean_firing_rate
                after_grp.create_dataset('waveform', data=align_data.waveform)
                if align_data.spike_times is not None:
                    after_grp.create_dataset('spike_times', data=align_data.spike_times)
    
    logger.info(f"Saved {len(pairs)} aligned pairs to {output_path.name}")


# =============================================================================
# Main Pipeline
# =============================================================================

def process_alignment_pair(pair: FilePair) -> Optional[Path]:
    """Process a single alignment pair."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing pair: {pair.before_file} -> {pair.after_file}")
    
    # Load units
    ref_units = load_unit_data_from_h5(pair.before_h5)
    align_units = load_unit_data_from_h5(pair.after_h5)
    
    if not ref_units or not align_units:
        logger.warning(f"Empty units for pair: {pair.before_file}")
        return None
    
    # Generate alignment
    pairs = generate_alignment_links(ref_units, align_units)
    
    if not pairs:
        logger.warning(f"No matching pairs found for: {pair.before_file}")
        return None
    
    # Save aligned data
    output_name = f"{pair.before_h5.stem}_to_{pair.after_h5.stem}_baseline.h5"
    output_path = ALIGNED_OUTPUT_DIR / output_name
    
    save_aligned_pair(output_path, ref_units, align_units, pairs, pair)
    
    return output_path


def run_alignment_pipeline() -> List[Path]:
    """Run the full alignment pipeline."""
    logger = logging.getLogger(__name__)
    
    # Find pairs
    pairs = find_alignment_pairs()
    logger.info(f"Found {len(pairs)} alignment pairs")
    
    if not pairs:
        logger.warning("No alignment pairs found")
        return []
    
    # Create output directory
    ALIGNED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each pair
    output_files = []
    for i, pair in enumerate(pairs):
        logger.info(f"\n[{i+1}/{len(pairs)}] Processing alignment pair")
        output = process_alignment_pair(pair)
        if output:
            output_files.append(output)
    
    logger.info("=" * 60)
    logger.info(f"Alignment complete: {len(output_files)}/{len(pairs)} pairs processed")
    logger.info("=" * 60)
    
    return output_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Baseline Change Alignment Pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    output_files = run_alignment_pipeline()
    
    # Print summary
    print("=" * 70)
    print("Baseline Change Alignment Pipeline")
    print("=" * 70)
    print(f"H5 Folder:    {OUTPUT_DIR}")
    print(f"Output:       {ALIGNED_OUTPUT_DIR}")
    print(f"Method:       Location + Waveform")
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
