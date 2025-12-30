"""
DOUBLE-REFINED: Create moving_h_bar_s5_d8_3x_on_off_dict_hd_double_refined.pkl

Combines the strengths of both detection methods:
- Single-pixel method: Accurate timing but may choose wrong peak when multiple exist
- Area-based method (50x50): Reliable peak selection as reference

Algorithm:
1. For each pixel, find ALL significant onset candidates from single-pixel trace
2. Use area-based (50x50) detection to get a reliable reference onset time
3. Select the single-pixel candidate closest in time to the area-based reference
4. Use the refined OFF detection (first significant negative derivative after ON)

This gives both accurate timing AND reliable peak selection.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
import numpy as np
from scipy.signal import find_peaks
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

NEW_DICT_PATH = OUTPUT_DIR / "moving_h_bar_s5_d8_3x_on_off_dict_hd_double_refined.pkl"

# Area-based detection parameters
BAR_HEIGHT = 50  # 50x50 receptive field for reliable peak selection
RECEPTIVE_FIELD_SIDE_LENGTH = int(BAR_HEIGHT / 2)  # = 25 pixels


def find_first_off_peak(trace_segment, on_peak_idx, on_peak_value):
    """
    Find the FIRST significant negative derivative peak after ON.
    
    Strategy: Find the first frame where the derivative drops below
    a threshold (50% of the ON peak magnitude, but negative).
    If no such frame exists, fall back to the maximum negative derivative.
    """
    # Threshold: negative derivative with at least 50% of ON peak magnitude
    threshold = -abs(on_peak_value) * 0.5
    
    # Also set a minimum absolute threshold for cases where ON peak is small
    min_threshold = -50.0
    threshold = min(threshold, min_threshold)
    
    # Search region: after ON peak
    search_region = trace_segment[on_peak_idx + 1:]
    
    if len(search_region) == 0:
        return len(trace_segment) - 1
    
    # Find first frame below threshold
    below_threshold = np.where(search_region <= threshold)[0]
    
    if len(below_threshold) > 0:
        # Return the first frame that crosses the threshold
        return below_threshold[0] + on_peak_idx + 1
    else:
        # Fallback: find the maximum negative derivative after ON
        return np.argmax(-search_region) + on_peak_idx + 1


def find_onset_candidates(segment, min_prominence_ratio=0.3):
    """
    Find ALL significant onset candidates in a derivative segment.
    
    Uses scipy.signal.find_peaks with prominence-based filtering.
    Returns array of peak indices within the segment.
    """
    if len(segment) == 0:
        return np.array([0])
    
    max_val = np.max(segment)
    if max_val <= 0:
        # No positive peaks, return index of max (least negative)
        return np.array([np.argmax(segment)])
    
    # Find all local maxima with minimum prominence
    min_prominence = max_val * min_prominence_ratio
    peaks, properties = find_peaks(segment, prominence=min_prominence)
    
    if len(peaks) == 0:
        # No peaks found with prominence filter, fall back to argmax
        return np.array([np.argmax(segment)])
    
    return peaks


def get_area_trace(ds_stim, x, y, height, width, rf_side_length):
    """
    Get the averaged trace over a receptive field area around (x, y).
    Handles edge cases by clamping to valid range.
    """
    # Handle edge cases - clamp to valid range
    if x < rf_side_length:
        x_location = rf_side_length
    elif x > height - rf_side_length:
        x_location = height - rf_side_length
    else:
        x_location = x
        
    if y < rf_side_length:
        y_location = rf_side_length
    elif y > width - rf_side_length:
        y_location = width - rf_side_length
    else:
        y_location = y
    
    # Average the receptive field area
    trace = ds_stim[
        :, 
        x_location - rf_side_length : x_location + rf_side_length, 
        y_location - rf_side_length : y_location + rf_side_length
    ].mean(axis=(1, 2))
    
    return trace


def process_single_row(x, ds_stim, rf_side_length):
    """
    Process a single row of pixels using DOUBLE-REFINED method.
    
    For each pixel:
    1. Find ALL onset candidates from single-pixel trace (accurate timing)
    2. Get reference onset from area-based trace (reliable selection)
    3. Select single-pixel candidate closest to area-based reference
    4. Apply refined OFF detection
    """
    height, width = ds_stim.shape[1], ds_stim.shape[2]
    row_results = {}
    
    for y in range(width):
        # === SINGLE-PIXEL TRACE (for accurate timing) ===
        single_trace = ds_stim[:, x, y].astype(np.float64)
        single_trace = np.concatenate([np.zeros(1), single_trace])
        single_deriv = np.diff(single_trace)
        
        # === AREA-BASED TRACE (for reliable peak selection) ===
        area_trace = get_area_trace(ds_stim, x, y, height, width, rf_side_length)
        area_trace = area_trace.astype(np.float64)
        area_trace = np.concatenate([np.zeros(1), area_trace])
        area_deriv = np.diff(area_trace)
        
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
                
                # Extract segments
                single_segment = single_deriv[int(start):int(end)]
                area_segment = area_deriv[int(start):int(end)]
                
                # === STEP 1: Find ALL onset candidates from single-pixel ===
                candidates = find_onset_candidates(single_segment)
                
                # === STEP 2: Get reference onset from area-based ===
                area_ref_idx = np.argmax(area_segment)
                
                # === STEP 3: Select closest single-pixel candidate to area reference ===
                if len(candidates) == 1:
                    # Only one candidate, use it directly
                    on_peak_iter = candidates[0]
                else:
                    # Multiple candidates, select closest to area-based reference
                    distances = np.abs(candidates - area_ref_idx)
                    closest_idx = np.argmin(distances)
                    on_peak_iter = candidates[closest_idx]
                
                on_peak_value = single_segment[on_peak_iter]
                on_peak_abs = int(on_peak_iter + start)
                
                # === STEP 4: OFF detection using refined method ===
                off_peak_iter = find_first_off_peak(single_segment, on_peak_iter, on_peak_value)
                off_peak_abs = int(off_peak_iter + start)
                
                on_peak_location.append(on_peak_abs)
                off_peak_location.append(off_peak_abs)
        
        # Store in results
        row_results[(x, y)] = {
            "on_peak_location": on_peak_location,
            "off_peak_location": off_peak_location
        }
    
    return row_results


def create_on_off_dict_hd_parallel(stimulus_path: Path) -> dict:
    """
    Create on_off_dict_hd using double-refined method with multiprocessing.
    Uses 80% of CPU cores.
    """
    print(f"Loading stimulus: {stimulus_path}")
    ds_stim = np.load(stimulus_path)
    print(f"Stimulus shape: {ds_stim.shape}")
    
    height = ds_stim.shape[1]
    
    print(f"\nDouble-refined parameters:")
    print(f"  Bar height (for area): {BAR_HEIGHT}")
    print(f"  Receptive field side length: {RECEPTIVE_FIELD_SIDE_LENGTH}")
    print(f"  Area size: {BAR_HEIGHT}x{BAR_HEIGHT} pixels")
    
    # Calculate number of workers (80% of CPU cores)
    total_cores = multiprocessing.cpu_count()
    n_workers = max(1, int(total_cores * 0.8))
    print(f"  CPU cores: {total_cores}, using {n_workers} workers (80%)")
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_row, 
        ds_stim=ds_stim,
        rf_side_length=RECEPTIVE_FIELD_SIDE_LENGTH
    )
    
    # Process all rows in parallel
    print(f"\nProcessing {height} rows in parallel (double-refined method)...")
    
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
    print("COMPARISON: Double-Refined vs Legacy")
    print("=" * 70)
    
    # Check keys
    new_keys = set(new_dict.keys())
    legacy_keys = set(legacy_dict.keys())
    
    print(f"\nKey comparison:")
    print(f"  Double-refined keys: {len(new_keys)}")
    print(f"  Legacy keys: {len(legacy_keys)}")
    print(f"  Keys match: {new_keys == legacy_keys}")
    
    if new_keys != legacy_keys:
        missing_in_new = legacy_keys - new_keys
        extra_in_new = new_keys - legacy_keys
        print(f"  Missing in double-refined: {len(missing_in_new)}")
        print(f"  Extra in double-refined: {len(extra_in_new)}")
    
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
    print(f"  ON differences: {len(differences['on_mismatch'])}")
    print(f"  OFF differences: {len(differences['off_mismatch'])}")
    
    if len(differences['on_mismatch']) == 0 and len(differences['off_mismatch']) == 0:
        print("\n  Note: Identical to legacy (expected if single-peak cases dominate)")
    else:
        print("\n  Differences found (expected - these are improved detections):")
        print(f"    ON improvements: {len(differences['on_mismatch'])} pixels")
        print(f"    OFF adjustments: {len(differences['off_mismatch'])} pixels")
        
        # Show first few differences
        if differences['on_mismatch'][:3]:
            print("\n  Sample ON differences:")
            for m in differences['on_mismatch'][:3]:
                print(f"    Pixel {m['key']}:")
                print(f"      Double-refined: {m['new'][:4]}...")
                print(f"      Legacy:         {m['legacy'][:4]}...")
    
    return differences


def main():
    print("#" * 70)
    print("DOUBLE-REFINED on_off_dict_hd.pkl")
    print("(Single-pixel timing + Area-based peak selection)")
    print("(Multiprocessing optimized - 80% of CPU cores)")
    print("#" * 70)
    
    # Check if stimulus file exists
    if not STIMULUS_PATH.exists():
        print(f"✗ Stimulus file not found: {STIMULUS_PATH}")
        return
    
    # Create new dictionary using double-refined method
    print("\n" + "=" * 70)
    print("Step 1: Creating dictionary with double-refined method")
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
    
    # Load legacy dictionary for comparison
    print("\n" + "=" * 70)
    print("Step 2: Loading legacy dictionary for comparison")
    print("=" * 70)
    
    if not LEGACY_DICT_PATH.exists():
        print(f"✗ Legacy dictionary not found: {LEGACY_DICT_PATH}")
        print("  Skipping comparison.")
    else:
        print(f"Loading: {LEGACY_DICT_PATH}")
        with open(LEGACY_DICT_PATH, 'rb') as f:
            legacy_dict = pickle.load(f)
        print(f"Loaded {len(legacy_dict)} pixels")
        
        # Compare
        differences = compare_dicts(new_dict, legacy_dict)
    
    # Validate: check that all entries have OFF > ON
    print("\n" + "=" * 70)
    print("Step 3: Validating double-refined dictionary (OFF > ON for all)")
    print("=" * 70)
    
    problematic_pixels = 0
    DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
    problems_by_dir = {d: 0 for d in DIRECTION_LIST}
    
    for key, pixel_data in new_dict.items():
        on_times = pixel_data['on_peak_location']
        off_times = pixel_data['off_peak_location']
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            if off <= on:
                problematic_pixels += 1
                direction = DIRECTION_LIST[trial_idx % 8]
                problems_by_dir[direction] += 1
    
    print(f"\nValidation results:")
    for d in DIRECTION_LIST:
        status = "✓" if problems_by_dir[d] == 0 else f"✗ {problems_by_dir[d]}"
        print(f"  Direction {d:>3}°: {status}")
    
    print(f"\n  Total problematic entries: {problematic_pixels}")
    
    # Summary
    print("\n" + "#" * 70)
    print("SUMMARY")
    print("#" * 70)
    
    if problematic_pixels == 0:
        print(f"""
✓ SUCCESS! The double-refined dictionary has ZERO problematic pixels.

Method:
  - Single-pixel traces for accurate timing
  - Area-based (50x50) for reliable peak selection
  - Refined OFF detection (first significant negative after ON)

Files:
  - Legacy:         {LEGACY_DICT_PATH}
  - Double-refined: {NEW_DICT_PATH}

Double-refined dictionary contains {len(new_dict)} pixels.
All entries have OFF > ON (constraint satisfied).
Time: {elapsed:.1f} seconds
""")
    else:
        print(f"""
! WARNING: Still found {problematic_pixels} problematic entries.

Please investigate further.
""")


if __name__ == "__main__":
    main()

