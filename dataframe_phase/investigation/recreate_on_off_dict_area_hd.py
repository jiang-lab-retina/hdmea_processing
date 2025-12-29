"""
Recreate moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl using the exact method
from the legacy code, then compare with the existing version.

OPTIMIZED with multiprocessing (80% of CPU cores).
Algorithm is IDENTICAL to legacy code.
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
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl"
)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NEW_DICT_PATH = OUTPUT_DIR / "moving_h_bar_s5_d8_3x_on_off_dict_area_hd_bar_height_3.pkl"
BAR_HEIGHT = 3

def get_all_max_indices_np(trace):
    """
    Find all indices where the trace equals its maximum value.
    Returns the first index if there are multiple.
    
    This is the exact function from the legacy code.
    """
    max_value = np.max(trace)
    max_indices = np.where(trace == max_value)[0]
    return max_value, max_indices


def process_single_row(x, ds_stim, receptive_field_side_length):
    """
    Process a single row of pixels.
    This is the exact algorithm from legacy code, just for one row.
    
    Returns: dict of {(x, y): {'on_peak_location': [...], 'off_peak_location': [...]}}
    """
    height, width = ds_stim.shape[1], ds_stim.shape[2]
    row_results = {}
    
    for y in range(width):
        # Handle edge cases - clamp to valid range
        # This is the exact edge handling from legacy code
        if x < receptive_field_side_length:
            x_location = receptive_field_side_length
        elif x > height - receptive_field_side_length:
            x_location = height - receptive_field_side_length
        else:
            x_location = x
            
        if y < receptive_field_side_length:
            y_location = receptive_field_side_length
        elif y > width - receptive_field_side_length:
            y_location = width - receptive_field_side_length
        else:
            y_location = y
        
        # Average a 50x50 (25*2) area around the pixel to simulate receptive field
        trace = ds_stim[
            :, 
            x_location - receptive_field_side_length : x_location + receptive_field_side_length, 
            y_location - receptive_field_side_length : y_location + receptive_field_side_length
        ].mean(axis=(1, 2))
        
        # Compute derivative to detect edges
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
                on_max_value, on_peak_iter = get_all_max_indices_np(trace[int(start):int(end)])
                
                # OFF: Maximum negative derivative (bar leaving pixel)
                off_max_value, off_peak_iter = get_all_max_indices_np(-trace[int(start):int(end)])
                
                # Store first peak index (relative to trial start) + start offset
                on_peak_location.append(int(on_peak_iter[0] + start))
                off_peak_location.append(int(off_peak_iter[0] + start))
        
        # Store in results
        row_results[(x, y)] = {
            "on_peak_location": on_peak_location,
            "off_peak_location": off_peak_location
        }
    
    return row_results


def create_on_off_dict_area_hd_parallel(stimulus_path: Path) -> dict:
    """
    Create on_off_dict using the exact area-based method from legacy code.
    Uses multiprocessing with 80% of CPU cores.
    """
    print(f"Loading stimulus: {stimulus_path}")
    ds_stim = np.load(stimulus_path)
    print(f"Stimulus shape: {ds_stim.shape}")
    
    # Parameters from legacy code
    bar_height = BAR_HEIGHT
    receptive_field_side_length = int(bar_height / 2)  # = 25 pixels
    
    print(f"Bar height: {bar_height}")
    print(f"Receptive field side length: {receptive_field_side_length}")
    
    height = ds_stim.shape[1]
    
    # Calculate number of workers (80% of CPU cores)
    total_cores = multiprocessing.cpu_count()
    n_workers = max(1, int(total_cores * 0.8))
    print(f"CPU cores: {total_cores}, using {n_workers} workers (80%)")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_row,
        ds_stim=ds_stim,
        receptive_field_side_length=receptive_field_side_length
    )
    
    # Process all rows in parallel
    print(f"\nProcessing {height} rows in parallel...")
    
    ds_on_dict_area_hd = {}
    
    with multiprocessing.Pool(processes=n_workers) as pool:
        # Use imap for progress bar
        results = list(tqdm(
            pool.imap(process_func, range(height)),
            total=height,
            desc="Processing rows"
        ))
    
    # Merge all results
    print("Merging results...")
    for row_result in results:
        ds_on_dict_area_hd.update(row_result)
    
    return ds_on_dict_area_hd


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
        'total_checked': 0
    }
    
    common_keys = new_keys & legacy_keys
    
    for key in common_keys:
        differences['total_checked'] += 1
        
        new_on = new_dict[key]['on_peak_location']
        legacy_on = legacy_dict[key]['on_peak_location']
        new_off = new_dict[key]['off_peak_location']
        legacy_off = legacy_dict[key]['off_peak_location']
        
        if new_on != legacy_on:
            differences['on_mismatch'].append({
                'key': key,
                'new': new_on,
                'legacy': legacy_on
            })
        
        if new_off != legacy_off:
            differences['off_mismatch'].append({
                'key': key,
                'new': new_off,
                'legacy': legacy_off
            })
    
    print(f"\nValue comparison:")
    print(f"  Total pixels checked: {differences['total_checked']}")
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
    print("RECREATE on_off_dict_area_hd.pkl FROM LEGACY CODE")
    print("(Multiprocessing optimized - 80% of CPU cores)")
    print("#" * 70)
    
    # Check if stimulus file exists
    if not STIMULUS_PATH.exists():
        print(f"✗ Stimulus file not found: {STIMULUS_PATH}")
        return
    
    # Create new dictionary using exact legacy method with parallel processing
    print("\n" + "=" * 70)
    print("Step 1: Creating new dictionary using legacy method (parallel)")
    print("=" * 70)
    
    import time
    start_time = time.time()
    
    new_dict = create_on_off_dict_area_hd_parallel(STIMULUS_PATH)
    
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
