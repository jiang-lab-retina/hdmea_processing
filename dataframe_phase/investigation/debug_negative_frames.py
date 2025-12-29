"""
Debug script to investigate why 2024.02.28-11.19.39-Rec has negative frame counts
for direction 45°.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import h5py
import numpy as np

# Configuration
project_root = Path(__file__).parent.parent.parent
HDF5_DIR = project_root / "Projects/unified_pipeline/export_all_steps"

PROBLEM_FILE = "2024.02.28-11.19.39-Rec.h5"
GOOD_FILE = "2024.02.26-10.53.19-Rec.h5"  # A file that works correctly
MOVIE_NAME = "moving_h_bar_s5_d8_3x"


def investigate_file(h5_path: Path):
    """Investigate direction section data in an HDF5 file."""
    print(f"\n{'='*70}")
    print(f"Investigating: {h5_path.name}")
    print(f"{'='*70}")
    
    with h5py.File(h5_path, "r") as f:
        # Get first unit's direction section
        unit_ids = list(f["units"].keys())
        sample_unit = unit_ids[0]
        
        dir_section_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_NAME}/direction_section"
        
        if dir_section_path not in f:
            print(f"No direction_section found!")
            return
        
        dir_group = f[dir_section_path]
        
        # Get attributes
        if "_attrs" in dir_group:
            attrs = dir_group["_attrs"]
            print(f"\nUnit: {sample_unit}")
            print(f"Cell center: ({attrs['cell_center_row'][()]}, {attrs['cell_center_col'][()]})")
            print(f"Padding frames: {attrs['padding_frames'][()]}")
        
        # Check all directions
        print(f"\nSection bounds per direction:")
        print("-" * 60)
        
        for direction in sorted([d for d in dir_group.keys() if d != "_attrs"], key=int):
            dir_data = dir_group[direction]
            bounds = dir_data["section_bounds"][:]
            
            print(f"\nDirection {direction}°:")
            for rep_idx, (start, end) in enumerate(bounds):
                n_frames = end - start
                status = "✓" if n_frames > 0 else "✗ NEGATIVE!"
                print(f"  Rep {rep_idx}: frames {start} - {end} (n={n_frames}) {status}")
        
        # Check direction 45 specifically
        print(f"\n\n{'='*70}")
        print("Direction 45° Deep Dive:")
        print(f"{'='*70}")
        
        dir_45 = dir_group["45"]
        bounds_45 = dir_45["section_bounds"][:]
        
        print(f"\nSection bounds for 45°:")
        print(f"  Raw data: {bounds_45}")
        
        for rep_idx, (start, end) in enumerate(bounds_45):
            print(f"\n  Rep {rep_idx}:")
            print(f"    Start frame (movie-relative): {start}")
            print(f"    End frame (movie-relative): {end}")
            print(f"    Duration: {end - start} frames")
            
            if start < 0 or end < 0:
                print(f"    ✗ NEGATIVE FRAME INDICES!")
            if end < start:
                print(f"    ✗ END < START!")


def compare_files():
    """Compare the problem file with a good file."""
    problem_path = HDF5_DIR / PROBLEM_FILE
    good_path = HDF5_DIR / GOOD_FILE
    
    print(f"\n{'#'*70}")
    print("COMPARISON: Good vs Problem File")
    print(f"{'#'*70}")
    
    # Investigate both
    investigate_file(good_path)
    investigate_file(problem_path)
    
    # Compare cell centers
    print(f"\n\n{'='*70}")
    print("Cell Center Comparison (first few units):")
    print(f"{'='*70}")
    
    with h5py.File(good_path, "r") as good_f, h5py.File(problem_path, "r") as prob_f:
        good_units = list(good_f["units"].keys())[:5]
        prob_units = list(prob_f["units"].keys())[:5]
        
        print(f"\nGood file ({GOOD_FILE}):")
        for unit_id in good_units:
            path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_NAME}/direction_section/_attrs"
            if path.replace("/_attrs", "") in good_f:
                if "_attrs" in good_f[path.replace("/_attrs", "")]:
                    attrs = good_f[path.replace("/_attrs", "") + "/_attrs"]
                    row = attrs["cell_center_row"][()]
                    col = attrs["cell_center_col"][()]
                    print(f"  {unit_id}: center=({row}, {col})")
        
        print(f"\nProblem file ({PROBLEM_FILE}):")
        for unit_id in prob_units:
            path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_NAME}/direction_section/_attrs"
            if path.replace("/_attrs", "") in prob_f:
                if "_attrs" in prob_f[path.replace("/_attrs", "")]:
                    attrs = prob_f[path.replace("/_attrs", "") + "/_attrs"]
                    row = attrs["cell_center_row"][()]
                    col = attrs["cell_center_col"][()]
                    print(f"  {unit_id}: center=({row}, {col})")
    
    # Check if it's a specific unit or all units
    print(f"\n\n{'='*70}")
    print("Checking if problem is unit-specific or global:")
    print(f"{'='*70}")
    
    with h5py.File(problem_path, "r") as f:
        units_with_negative = []
        units_checked = 0
        
        for unit_id in f["units"].keys():
            path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_NAME}/direction_section/45/section_bounds"
            if path in f:
                bounds = f[path][:]
                units_checked += 1
                for start, end in bounds:
                    if end - start < 0:
                        units_with_negative.append(unit_id)
                        break
        
        print(f"\nUnits checked: {units_checked}")
        print(f"Units with negative frames in dir 45°: {len(units_with_negative)}")
        
        if len(units_with_negative) == units_checked:
            print("  → Problem affects ALL units (likely on/off dict issue)")
        elif len(units_with_negative) == 0:
            print("  → No units have negative frames!")
        else:
            print(f"  → Problem affects some units: {units_with_negative[:5]}...")
    
    # Check the on_off_dict lookup
    print(f"\n\n{'='*70}")
    print("Investigating possible causes:")
    print(f"{'='*70}")
    
    with h5py.File(problem_path, "r") as f:
        # Get a sample unit with the problem
        sample_unit = list(f["units"].keys())[0]
        dir_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_NAME}/direction_section"
        
        attrs = f[dir_path + "/_attrs"]
        cell_row = attrs["cell_center_row"][()]
        cell_col = attrs["cell_center_col"][()]
        padding = attrs["padding_frames"][()]
        
        print(f"\nSample unit: {sample_unit}")
        print(f"Cell center: ({cell_row}, {cell_col})")
        print(f"Padding frames: {padding}")
        
        # Get bounds for direction 45
        bounds_45 = f[dir_path + "/45/section_bounds"][:]
        
        print(f"\nDirection 45° bounds: {bounds_45}")
        
        # Calculate what on/off times must have been
        # bounds = [on - padding, off + padding]
        # So: on = start + padding, off = end - padding
        for rep, (start, end) in enumerate(bounds_45):
            on_time = start + padding
            off_time = end - padding
            print(f"\n  Rep {rep}:")
            print(f"    Calculated on_time: {on_time}")
            print(f"    Calculated off_time: {off_time}")
            print(f"    Duration (off - on): {off_time - on_time}")
            
            if off_time < on_time:
                print(f"    ✗ off_time < on_time! on/off dict has wrong values")


def check_on_off_dict():
    """Check the on_off_dict file for issues."""
    print(f"\n\n{'='*70}")
    print("Checking on_off_dict file:")
    print(f"{'='*70}")
    
    # Look for on_off_dict file
    on_off_dict_path = project_root / "Projects/unified_pipeline/on_off_dict/moving_h_bar_s5_d8_3x.npy"
    
    if not on_off_dict_path.exists():
        print(f"on_off_dict not found at: {on_off_dict_path}")
        # Try to find it
        possible_paths = list((project_root / "Projects/unified_pipeline").rglob("*on_off*"))
        if possible_paths:
            print(f"Found possible files: {possible_paths}")
        return
    
    print(f"Found: {on_off_dict_path}")
    
    # Load and inspect
    on_off_dict = np.load(on_off_dict_path, allow_pickle=True).item()
    
    print(f"\nTotal pixels: {len(on_off_dict)}")
    
    # Sample a few entries
    sample_keys = list(on_off_dict.keys())[:3]
    print(f"\nSample entries:")
    for key in sample_keys:
        val = on_off_dict[key]
        print(f"\n  Pixel {key}:")
        print(f"    on_peak_location: {val.get('on_peak_location', 'N/A')}")
        print(f"    off_peak_location: {val.get('off_peak_location', 'N/A')}")
    
    # Check for any entries where off < on
    print(f"\nChecking for problematic entries (off < on):")
    problems = []
    for key, val in on_off_dict.items():
        on_times = val.get('on_peak_location', [])
        off_times = val.get('off_peak_location', [])
        
        if len(on_times) > 0 and len(off_times) > 0:
            for i, (on, off) in enumerate(zip(on_times, off_times)):
                if off < on:
                    problems.append((key, i, on, off))
    
    if problems:
        print(f"\n  Found {len(problems)} problematic entries!")
        for key, idx, on, off in problems[:10]:
            print(f"    Pixel {key}, trial {idx}: on={on}, off={off}, diff={off-on}")
    else:
        print("  No problematic entries found in on_off_dict")


if __name__ == "__main__":
    compare_files()
    check_on_off_dict()
    
    print(f"\n\n{'#'*70}")
    print("CONCLUSION")
    print(f"{'#'*70}")

