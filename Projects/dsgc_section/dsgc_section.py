"""
Development and test script for DSGC Direction Sectioning.

This script:
1. Copies the test HDF5 file to the export folder
2. Runs section_by_direction on the copy
3. Prints result statistics
4. Verifies output structure

Usage:
    python dsgc_section.py
"""

import sys
from pathlib import Path

# Add src to path for development
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import h5py
import numpy as np

from hdmea.features import section_by_direction, DIRECTION_LIST


# =============================================================================
# Configuration
# =============================================================================

TEST_SOURCE_FILE = Path(
    r"M:\Python_Project\Data_Processing_2027\Projects\ap_trace_hdf5"
    r"\export_ap_tracking_20251226\2024.08.08-10.40.20-Rec.h5"
)

EXPORT_DIR = Path(__file__).parent / "export"
OUTPUT_FILE = EXPORT_DIR / TEST_SOURCE_FILE.name

MOVIE_NAME = "moving_h_bar_s5_d8_3x"

# Use _hd version (single pixel) instead of _area_hd (25x25 area average)
ON_OFF_DICT_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)


def main():
    """Run direction sectioning on test file and verify results."""
    print("=" * 70)
    print("DSGC Direction Sectioning Test")
    print("=" * 70)
    
    # Check source file exists
    if not TEST_SOURCE_FILE.exists():
        print(f"ERROR: Test source file not found: {TEST_SOURCE_FILE}")
        return 1
    
    print(f"\nSource file: {TEST_SOURCE_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Ensure export directory exists
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run section_by_direction with output_path (copies source first)
    print("\n--- Running section_by_direction ---")
    
    result = section_by_direction(
        TEST_SOURCE_FILE,
        movie_name=MOVIE_NAME,
        on_off_dict_path=ON_OFF_DICT_PATH,  # Use _hd version (single pixel detection)
        padding_frames=10,
        force=True,  # Overwrite if exists
        output_path=OUTPUT_FILE,
        # unit_ids=["unit_002"],  # Uncomment to test single unit
    )
    
    # Print results
    print("\n--- Results ---")
    print(f"Units processed: {result.units_processed}")
    print(f"Units skipped: {result.units_skipped}")
    print(f"Padding frames: {result.padding_frames}")
    print(f"Elapsed time: {result.elapsed_seconds:.2f}s")
    
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings[:5]:  # Show first 5
            print(f"  - {w}")
        if len(result.warnings) > 5:
            print(f"  ... and {len(result.warnings) - 5} more")
    
    if result.skipped_units:
        print(f"\nSkipped units: {result.skipped_units[:10]}")
    
    # Verify output structure
    print("\n--- Verifying Output Structure ---")
    
    with h5py.File(OUTPUT_FILE, "r") as f:
        units = list(f["units"].keys())
        print(f"Total units in file: {len(units)}")
        
        # Find units with direction_section
        units_with_section = []
        for unit_id in units:
            section_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_NAME}/direction_section"
            if section_path in f:
                units_with_section.append(unit_id)
        
        print(f"Units with direction_section: {len(units_with_section)}")
        
        if units_with_section:
            # Check first unit's structure
            unit_id = units_with_section[0]
            section_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_NAME}/direction_section"
            section_group = f[section_path]
            
            print(f"\nSample unit: {unit_id}")
            print(f"  Attributes:")
            for key, val in section_group.attrs.items():
                print(f"    {key}: {val}")
            
            print(f"  Direction groups: {list(section_group.keys())}")
            
            # Check one direction
            dir_group = section_group["0"]
            print(f"\n  Direction 0°:")
            print(f"    trials: {list(dir_group['trials'].keys())}")
            
            # Count spikes per trial
            total_spikes = 0
            for rep in ["0", "1", "2"]:
                n_spikes = len(dir_group[f"trials/{rep}"][:])
                total_spikes += n_spikes
                print(f"      Trial {rep}: {n_spikes} spikes")
            
            print(f"    section_bounds: {dir_group['section_bounds'][:]}")
            
            # Summary across all directions
            print(f"\n  Spike counts by direction:")
            for direction in DIRECTION_LIST:
                dir_grp = section_group[str(direction)]
                total = sum(len(dir_grp[f"trials/{r}"][:]) for r in ["0", "1", "2"])
                print(f"    {direction:3d}°: {total:4d} spikes")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

