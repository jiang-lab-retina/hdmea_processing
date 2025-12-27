#!/usr/bin/env python
"""
Test script for bad lane replacement in STA data.

This script:
1. Reads Bad_lanes from metadata/gsheet_row/Bad_lanes
2. Replaces bad columns (lanes) in STA data with the mean of the whole STA
3. Creates before/after visualization plots

Lane numbers are 1-indexed and correspond to x-axis columns in the STA data.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import h5py


# =============================================================================
# Configuration
# =============================================================================

# Input folder containing HDF5 files
INPUT_FOLDER = Path(r"M:\Python_Project\Data_Processing_2027\Projects\add_manual_label_cell_type\manual_label_20251225")

# Test files with bad lanes (found by scanning the folder)
TEST_FILES = [
    "2024.03.20-10.08.11-Rec.h5",   # Bad lanes: 13
    "2024.03.20-13.41.33-Rec.h5",   # Bad lanes: 29
    "2024.03.25-13.44.04-Rec.h5",   # Bad lanes: 39, 47
    "2024.03.25-15.38.58-Rec.h5",   # Bad lanes: 1, 65
    "2024.03.05-10.07.59-Rec.h5",   # Bad lanes: 43
]

# Output folder for plots
OUTPUT_FOLDER = Path(__file__).parent / "bad_lanes_test"


# =============================================================================
# Bad Lanes Processing
# =============================================================================

def parse_bad_lanes(bad_lanes_str: str) -> list[int]:
    """
    Parse bad lanes string to list of 0-indexed column indices.
    
    Args:
        bad_lanes_str: Comma-separated string of lane numbers (1-indexed)
                      e.g., "33, 34, 37, 39, 45, 47"
    
    Returns:
        List of 0-indexed column indices
    """
    if not bad_lanes_str or bad_lanes_str.strip() == "":
        return []
    
    lanes = []
    for part in bad_lanes_str.split(","):
        part = part.strip()
        if part:
            try:
                # Convert from 1-indexed to 0-indexed
                lane_1indexed = int(part)
                lane_0indexed = lane_1indexed - 1
                lanes.append(lane_0indexed)
            except ValueError:
                print(f"  Warning: Could not parse lane number: '{part}'")
    
    return sorted(lanes)


def read_bad_lanes_from_hdf5(hdf5_path: Path) -> list[int]:
    """
    Read Bad_lanes from HDF5 metadata.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        List of 0-indexed bad lane column indices
    """
    with h5py.File(hdf5_path, "r") as f:
        if "metadata/gsheet_row/Bad_lanes" not in f:
            print("  No Bad_lanes field found in metadata")
            return []
        
        bad_lanes_data = f["metadata/gsheet_row/Bad_lanes"][()]
        
        # Handle bytes vs string
        if isinstance(bad_lanes_data, bytes):
            bad_lanes_str = bad_lanes_data.decode("utf-8")
        elif isinstance(bad_lanes_data, np.ndarray):
            if bad_lanes_data.size > 0:
                val = bad_lanes_data.flat[0]
                if isinstance(val, bytes):
                    bad_lanes_str = val.decode("utf-8")
                else:
                    bad_lanes_str = str(val)
            else:
                bad_lanes_str = ""
        else:
            bad_lanes_str = str(bad_lanes_data)
        
        print(f"  Raw Bad_lanes value: '{bad_lanes_str}'")
        return parse_bad_lanes(bad_lanes_str)


def replace_bad_lanes(sta_data: np.ndarray, bad_lanes: list[int]) -> np.ndarray:
    """
    Replace bad lane columns with the mean of the entire STA data.
    
    Args:
        sta_data: STA array with shape (time, height, width) or (time, y, x)
        bad_lanes: List of 0-indexed column (x) indices to replace
        
    Returns:
        Modified STA data with bad lanes replaced by mean
    """
    if len(bad_lanes) == 0:
        return sta_data.copy()
    
    # Make a copy to avoid modifying original
    result = sta_data.copy()
    
    # Calculate mean of entire STA
    sta_mean = np.mean(sta_data)
    
    # STA shape is typically (time, y, x) where x is the column axis
    # Lane numbers correspond to x-axis (columns)
    n_cols = sta_data.shape[-1]  # Last dimension is x (columns)
    
    print(f"  STA shape: {sta_data.shape}")
    print(f"  STA mean value: {sta_mean:.6f}")
    print(f"  Bad lanes (0-indexed): {bad_lanes}")
    print(f"  Valid column range: 0-{n_cols-1}")
    
    # Replace bad columns
    replaced = 0
    for lane_idx in bad_lanes:
        if 0 <= lane_idx < n_cols:
            # Replace the entire column across all time points and y positions
            result[:, :, lane_idx] = sta_mean
            replaced += 1
        else:
            print(f"  Warning: Lane index {lane_idx} out of range (0-{n_cols-1})")
    
    print(f"  Replaced {replaced} lanes with mean value")
    
    return result


# =============================================================================
# Visualization
# =============================================================================

def visualize_bad_lanes(
    sta_before: np.ndarray,
    sta_after: np.ndarray,
    bad_lanes: list[int],
    unit_id: str,
    output_path: Path,
    time_slice: int = None,
):
    """
    Create before/after visualization of bad lane replacement.
    
    Args:
        sta_before: Original STA data
        sta_after: STA data with bad lanes replaced
        bad_lanes: List of 0-indexed bad lane indices
        unit_id: Unit identifier for title
        output_path: Path to save the figure
        time_slice: Time index to visualize (default: use time with max variance)
    """
    # Find time slice with maximum variance if not specified
    if time_slice is None:
        variance_per_time = np.var(sta_before, axis=(1, 2))
        time_slice = int(np.argmax(variance_per_time))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get data for the selected time slice
    before_slice = sta_before[time_slice]
    after_slice = sta_after[time_slice]
    diff_slice = after_slice - before_slice
    
    # Use mean ± 2*std for color scale to highlight bad lanes
    # (bad lanes often have extreme values outside normal range)
    combined = np.concatenate([before_slice.flatten(), after_slice.flatten()])
    data_mean = np.mean(combined)
    data_std = np.std(combined)
    n_std = 2  # Number of standard deviations for range
    vmin = data_mean - n_std * data_std
    vmax = data_mean + n_std * data_std
    
    # Plot before
    im0 = axes[0].imshow(before_slice, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Before (t={time_slice})")
    axes[0].set_xlabel("X (columns/lanes)")
    axes[0].set_ylabel("Y")
    plt.colorbar(im0, ax=axes[0])
    
    # Mark bad lanes with vertical lines
    for lane in bad_lanes:
        if 0 <= lane < before_slice.shape[1]:
            axes[0].axvline(x=lane, color='red', linewidth=0.5, alpha=0.7)
    
    # Plot after
    im1 = axes[1].imshow(after_slice, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"After (bad lanes → mean)")
    axes[1].set_xlabel("X (columns/lanes)")
    axes[1].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[1])
    
    # Mark bad lanes
    for lane in bad_lanes:
        if 0 <= lane < after_slice.shape[1]:
            axes[1].axvline(x=lane, color='red', linewidth=0.5, alpha=0.7)
    
    # Plot difference - use mean ± 2*std for symmetric scale
    diff_std = np.std(diff_slice)
    diff_range = n_std * diff_std
    im2 = axes[2].imshow(diff_slice, aspect='auto', cmap='RdBu_r', 
                         vmin=-diff_range, vmax=diff_range)
    axes[2].set_title("Difference (After - Before)")
    axes[2].set_xlabel("X (columns/lanes)")
    axes[2].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[2])
    
    fig.suptitle(f"Bad Lane Replacement - Unit {unit_id}\nBad lanes (1-indexed): {[l+1 for l in bad_lanes]}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_path}")


def visualize_column_profiles(
    sta_before: np.ndarray,
    sta_after: np.ndarray,
    bad_lanes: list[int],
    unit_id: str,
    output_path: Path,
    time_slice: int = None,
):
    """
    Create column profile visualization showing mean values per column.
    """
    if time_slice is None:
        variance_per_time = np.var(sta_before, axis=(1, 2))
        time_slice = int(np.argmax(variance_per_time))
    
    # Mean across y-axis for each column (x)
    before_profile = np.mean(sta_before[time_slice], axis=0)
    after_profile = np.mean(sta_after[time_slice], axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    x = np.arange(len(before_profile))
    
    ax.plot(x, before_profile, 'b-', label='Before', alpha=0.7)
    ax.plot(x, after_profile, 'g-', label='After', alpha=0.7)
    
    # Mark bad lanes
    for lane in bad_lanes:
        if 0 <= lane < len(before_profile):
            ax.axvline(x=lane, color='red', linewidth=1, linestyle='--', alpha=0.5)
    
    # Mark first bad lane for legend
    if bad_lanes:
        ax.axvline(x=-100, color='red', linewidth=1, linestyle='--', label='Bad lanes')
    
    ax.set_xlabel("Column (lane) index (0-indexed)")
    ax.set_ylabel("Mean value")
    ax.set_title(f"Column Profile - Unit {unit_id} (t={time_slice})\nBad lanes (1-indexed): {[l+1 for l in bad_lanes]}")
    ax.legend()
    ax.set_xlim(0, len(before_profile)-1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_path}")


# =============================================================================
# Main
# =============================================================================

def process_single_file(source_file: Path, output_subfolder: Path) -> bool:
    """
    Process a single HDF5 file for bad lanes test.
    
    Args:
        source_file: Path to source HDF5 file
        output_subfolder: Folder to save plots for this file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'-' * 50}")
    print(f"File: {source_file.name}")
    print(f"{'-' * 50}")
    
    if not source_file.exists():
        print(f"  ERROR: File not found")
        return False
    
    # Read bad lanes
    bad_lanes = read_bad_lanes_from_hdf5(source_file)
    
    if not bad_lanes:
        print("  No bad lanes found, skipping")
        return False
    
    print(f"  Bad lanes (1-indexed): {[l+1 for l in bad_lanes]}")
    
    # Create output subfolder
    output_subfolder.mkdir(parents=True, exist_ok=True)
    
    # Process first unit with STA data
    with h5py.File(source_file, "r") as f:
        if "units" not in f:
            print("  ERROR: No 'units' group in file")
            return False
        
        unit_ids = list(f["units"].keys())
        processed = 0
        
        for unit_id in unit_ids:
            sta_path = f"units/{unit_id}/features/eimage_sta/data"
            if sta_path not in f:
                continue
            
            # Read STA data
            sta_before = f[sta_path][:]
            print(f"  Unit {unit_id}: STA shape {sta_before.shape}")
            
            # Replace bad lanes
            sta_after = replace_bad_lanes(sta_before, bad_lanes)
            
            # Create visualizations
            plot_path = output_subfolder / f"bad_lanes_{unit_id}.png"
            visualize_bad_lanes(sta_before, sta_after, bad_lanes, unit_id, plot_path)
            
            profile_path = output_subfolder / f"profile_{unit_id}.png"
            visualize_column_profiles(sta_before, sta_after, bad_lanes, unit_id, profile_path)
            
            processed += 1
            if processed >= 1:  # Only process 1 unit per file for speed
                break
        
        if processed == 0:
            print("  No units with STA data found")
            return False
    
    return True


def main():
    """Run bad lanes test on multiple files."""
    print("=" * 60)
    print("Bad Lanes Replacement Test - Multiple Files")
    print("=" * 60)
    
    # Validate input folder
    if not INPUT_FOLDER.exists():
        print(f"ERROR: Input folder not found: {INPUT_FOLDER}")
        return 1
    
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Test files: {len(TEST_FILES)}")
    
    # Create main output folder
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Process each test file
    successful = 0
    for i, filename in enumerate(TEST_FILES, 1):
        print(f"\n[{i}/{len(TEST_FILES)}]", end="")
        
        source_file = INPUT_FOLDER / filename
        # Use date as subfolder name
        date_part = filename.split("-Rec")[0]
        output_subfolder = OUTPUT_FOLDER / date_part
        
        if process_single_file(source_file, output_subfolder):
            successful += 1
    
    print("\n" + "=" * 60)
    print("Bad Lanes Test Complete!")
    print(f"Processed: {successful}/{len(TEST_FILES)} files")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

