"""
Recreate moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl (single-pixel version)
using the exact method from the legacy code, then compare with the existing version.

Based on: Legacy_code/Data_Processing_2025/Processing_2025/Design_Stimulation_Pattern/Quick_check.py
Lines 892-935

This is the SINGLE-PIXEL version (no RF averaging).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial

# Paths
STIMULUS_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x.npy"
)

LEGACY_DICT_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NEW_DICT_PATH = OUTPUT_DIR / "moving_h_bar_s5_d8_3x_on_off_dict_hd_recreated.pkl"


def process_single_row(x, ds_stim):
    """
    Process a single row of pixels using SINGLE-PIXEL method.
    This is the exact algorithm from legacy code (lines 892-935).
    
    No RF averaging - just the single pixel trace.
    """
    height, width = ds_stim.shape[1], ds_stim.shape[2]
    row_results = {}
    
    for y in range(width):
        # Single pixel trace - NO averaging
        trace = ds_stim[:, x, y].astype(np.float64)
        
        # Compute derivative to detect edges
        # Legacy: trace = np.concatenate([np.zeros(1), trace]); trace = np.diff(trace)
        trace = np.concatenate([np.zeros(1), trace])
        trace = np.diff(trace)
        
        on_peak_location = []
        off_peak_location = []
        counter = 0
        
        # Process 3 repetitions × 8 directions = 24 trials
        for j in range(3):  # 3 repetitions
            counter = counter + 120  # Skip pre-margin (120 frames)
            
            for i in range(8):  # 8 directions
                start = counter
                counter = counter + 4400 / 8  # 550 frames per direction
                end = counter
                
                # ON: Maximum positive derivative (bar entering pixel)
                # Legacy: on_peak_iter = np.argmax(trace[int(start):int(end)])
                on_peak_iter = np.argmax(trace[int(start):int(end)])
                
                # OFF: Maximum negative derivative (bar leaving pixel)
                # Legacy: off_peak_iter = np.argmax(-trace[int(start):int(end)])
                off_peak_iter = np.argmax(-trace[int(start):int(end)])
                
                # Store: int(on_peak_iter + start)
                on_peak_location.append(int(on_peak_iter + start))
                off_peak_location.append(int(off_peak_iter + start))
        
        # Store in results
        row_results[(x, y)] = {
            "on_peak_location": on_peak_location,
            "off_peak_location": off_peak_location
        }
    
    return row_results


def create_on_off_dict_hd_parallel(stimulus_path: Path) -> dict:
    """
    Create on_off_dict_hd (single-pixel) using multiprocessing.
    Uses 80% of CPU cores.
    """
    print(f"Loading stimulus: {stimulus_path}")
    ds_stim = np.load(stimulus_path)
    print(f"Stimulus shape: {ds_stim.shape}")
    
    height = ds_stim.shape[1]
    
    # Calculate number of workers (80% of CPU cores)
    total_cores = multiprocessing.cpu_count()
    n_workers = max(1, int(total_cores * 0.8))
    print(f"CPU cores: {total_cores}, using {n_workers} workers (80%)")
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_row, ds_stim=ds_stim)
    
    # Process all rows in parallel
    print(f"\nProcessing {height} rows in parallel (single-pixel method)...")
    
    ds_on_dict_hd = {}
    
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, range(height)),
            total=height,
            desc="Processing rows"
        ))
    
    # Merge all results
    print("Merging results...")
    for row_result in results:
        ds_on_dict_hd.update(row_result)
    
    return ds_on_dict_hd


def compare_dicts(new_dict: dict, legacy_dict: dict) -> dict:
    """Compare recreated dictionary with legacy version."""
    print("\n" + "=" * 70)
    print("COMPARISON: Recreated vs Legacy")
    print("=" * 70)
    
    # Check keys
    new_keys = set(new_dict.keys())
    legacy_keys = set(legacy_dict.keys())
    
    print(f"\nKey comparison:")
    print(f"  Recreated keys: {len(new_keys)}")
    print(f"  Legacy keys: {len(legacy_keys)}")
    print(f"  Keys match: {new_keys == legacy_keys}")
    
    if new_keys != legacy_keys:
        missing_in_new = legacy_keys - new_keys
        extra_in_new = new_keys - legacy_keys
        print(f"  Missing in recreated: {len(missing_in_new)}")
        print(f"  Extra in recreated: {len(extra_in_new)}")
    
    # Compare values
    differences = {
        'on_mismatch': [],
        'off_mismatch': [],
        'total_checked': 0,
        'exact_match': 0
    }
    
    common_keys = new_keys & legacy_keys
    
    for key in common_keys:
        differences['total_checked'] += 1
        
        new_on = new_dict[key]['on_peak_location']
        legacy_on = legacy_dict[key]['on_peak_location']
        new_off = new_dict[key]['off_peak_location']
        legacy_off = legacy_dict[key]['off_peak_location']
        
        on_match = new_on == legacy_on
        off_match = new_off == legacy_off
        
        if on_match and off_match:
            differences['exact_match'] += 1
        else:
            if not on_match:
                differences['on_mismatch'].append({
                    'key': key,
                    'new': new_on,
                    'legacy': legacy_on
                })
            
            if not off_match:
                differences['off_mismatch'].append({
                    'key': key,
                    'new': new_off,
                    'legacy': legacy_off
                })
    
    print(f"\nValue comparison:")
    print(f"  Total pixels checked: {differences['total_checked']}")
    print(f"  Exact matches: {differences['exact_match']}")
    print(f"  ON mismatches: {len(differences['on_mismatch'])}")
    print(f"  OFF mismatches: {len(differences['off_mismatch'])}")
    
    if len(differences['on_mismatch']) == 0 and len(differences['off_mismatch']) == 0:
        print("\n  ✓ PERFECT MATCH! Recreated dictionary is identical to legacy.")
    else:
        print("\n  ✗ DIFFERENCES FOUND!")
        
        # Show first few mismatches
        if differences['on_mismatch']:
            print("\n  Sample ON mismatches:")
            for m in differences['on_mismatch'][:3]:
                print(f"    Pixel {m['key']}:")
                print(f"      Recreated: {m['new'][:4]}...")
                print(f"      Legacy:    {m['legacy'][:4]}...")
        
        if differences['off_mismatch']:
            print("\n  Sample OFF mismatches:")
            for m in differences['off_mismatch'][:3]:
                print(f"    Pixel {m['key']}:")
                print(f"      Recreated: {m['new'][:4]}...")
                print(f"      Legacy:    {m['legacy'][:4]}...")
    
    return differences


def main():
    print("#" * 70)
    print("RECREATE on_off_dict_hd.pkl (SINGLE-PIXEL) FROM LEGACY CODE")
    print("(Multiprocessing optimized - 80% of CPU cores)")
    print("#" * 70)
    
    # Check if stimulus file exists
    if not STIMULUS_PATH.exists():
        print(f"✗ Stimulus file not found: {STIMULUS_PATH}")
        return
    
    # Create new dictionary using exact legacy method with parallel processing
    print("\n" + "=" * 70)
    print("Step 1: Creating new dictionary using legacy single-pixel method")
    print("=" * 70)
    
    import time
    start_time = time.time()
    
    new_dict = create_on_off_dict_hd_parallel(STIMULUS_PATH)
    
    elapsed = time.time() - start_time
    print(f"\nCreated dictionary with {len(new_dict)} pixels in {elapsed:.1f} seconds")
    
    # Save new dictionary
    print(f"\nSaving to: {NEW_DICT_PATH}")
    with open(NEW_DICT_PATH, 'wb') as f:
        pickle.dump(new_dict, f)
    print("✓ Saved")
    
    # Load legacy dictionary
    print("\n" + "=" * 70)
    print("Step 2: Loading legacy dictionary for comparison")
    print("=" * 70)
    
    if not LEGACY_DICT_PATH.exists():
        print(f"✗ Legacy dictionary not found: {LEGACY_DICT_PATH}")
        return
    
    print(f"Loading: {LEGACY_DICT_PATH}")
    with open(LEGACY_DICT_PATH, 'rb') as f:
        legacy_dict = pickle.load(f)
    print(f"Loaded {len(legacy_dict)} pixels")
    
    # Compare
    differences = compare_dicts(new_dict, legacy_dict)
    
    # Summary
    print("\n" + "#" * 70)
    print("SUMMARY")
    print("#" * 70)
    
    if len(differences['on_mismatch']) == 0 and len(differences['off_mismatch']) == 0:
        print(f"""
✓ SUCCESS! The recreated dictionary matches the legacy version exactly.

Files:
  - Legacy:    {LEGACY_DICT_PATH}
  - Recreated: {NEW_DICT_PATH}

Both contain {len(new_dict)} pixels with identical ON/OFF values.
Time: {elapsed:.1f} seconds
""")
    else:
        total_mismatches = len(differences['on_mismatch']) + len(differences['off_mismatch'])
        print(f"""
! WARNING: Found {total_mismatches} differences between recreated and legacy.

This may be due to:
  - Different numpy version
  - Different random seed behavior
  - Floating point precision differences

Please investigate the differences.
""")


if __name__ == "__main__":
    main()

