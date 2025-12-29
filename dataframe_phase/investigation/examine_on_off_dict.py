"""
Examine the on_off_dict to check pixel (270, 30) for direction 45° issue.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
import numpy as np

# Path to on_off_dict
ON_OFF_DICT_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

# Direction mapping (8 directions × 3 repetitions = 24 trials)
# Trial index = direction_idx + rep_idx * 8
# Direction order: 0, 45, 90, 135, 180, 225, 270, 315
DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8
N_REPETITIONS = 3


def load_on_off_dict(path: Path):
    """Load on/off timing dictionary from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_trial_indices_for_direction(direction: int) -> list:
    """Get trial indices for a specific direction."""
    direction_idx = DIRECTION_LIST.index(direction)
    return [direction_idx + rep * N_DIRECTIONS for rep in range(N_REPETITIONS)]


def examine_pixel(on_off_dict, row: int, col: int):
    """Examine on/off times for a specific pixel."""
    print(f"\n{'='*70}")
    print(f"Examining pixel ({row}, {col})")
    print(f"{'='*70}")
    
    # Try both key orders
    key1 = (row, col)
    key2 = (col, row)
    
    if key1 in on_off_dict:
        key = key1
        print(f"Found with key: {key1}")
    elif key2 in on_off_dict:
        key = key2
        print(f"Found with FLIPPED key: {key2}")
    else:
        print(f"✗ Pixel NOT FOUND in on_off_dict!")
        print(f"  Tried: {key1} and {key2}")
        return None
    
    pixel_data = on_off_dict[key]
    
    on_times = np.array(pixel_data['on_peak_location'])
    off_times = np.array(pixel_data['off_peak_location'])
    
    print(f"\nNumber of trials: {len(on_times)}")
    print(f"on_peak_location: {on_times}")
    print(f"off_peak_location: {off_times}")
    
    # Check each direction
    print(f"\n{'-'*70}")
    print("Per-Direction Analysis:")
    print(f"{'-'*70}")
    
    for direction in DIRECTION_LIST:
        trial_indices = get_trial_indices_for_direction(direction)
        
        print(f"\nDirection {direction}°:")
        print(f"  Trial indices: {trial_indices}")
        
        for i, trial_idx in enumerate(trial_indices):
            if trial_idx < len(on_times) and trial_idx < len(off_times):
                on = on_times[trial_idx]
                off = off_times[trial_idx]
                duration = off - on
                status = "✓" if duration > 0 else "✗ NEGATIVE!"
                print(f"    Rep {i} (trial {trial_idx}): on={on:>5}, off={off:>5}, duration={duration:>4} {status}")
            else:
                print(f"    Rep {i} (trial {trial_idx}): OUT OF BOUNDS")
    
    return pixel_data


def compare_pixels(on_off_dict, pixels: list):
    """Compare on/off times for multiple pixels."""
    print(f"\n\n{'#'*70}")
    print("COMPARISON OF MULTIPLE PIXELS")
    print(f"{'#'*70}")
    
    # Focus on direction 45
    direction = 45
    trial_indices = get_trial_indices_for_direction(direction)
    
    print(f"\nDirection {direction}° (trial indices: {trial_indices}):")
    print("-" * 80)
    print(f"{'Pixel':<15} | {'Rep 0':^20} | {'Rep 1':^20} | {'Rep 2':^20}")
    print("-" * 80)
    
    for row, col in pixels:
        key1 = (row, col)
        key2 = (col, row)
        
        if key1 in on_off_dict:
            pixel_data = on_off_dict[key1]
        elif key2 in on_off_dict:
            pixel_data = on_off_dict[key2]
        else:
            print(f"({row:>3}, {col:>3})     | {'NOT FOUND':^20} | {'':^20} | {'':^20}")
            continue
        
        on_times = pixel_data['on_peak_location']
        off_times = pixel_data['off_peak_location']
        
        entries = []
        for trial_idx in trial_indices:
            if trial_idx < len(on_times):
                on = on_times[trial_idx]
                off = off_times[trial_idx]
                dur = off - on
                status = "!" if dur < 0 else ""
                entries.append(f"{on}->{off} ({dur:+d}){status}")
            else:
                entries.append("N/A")
        
        print(f"({row:>3}, {col:>3})     | {entries[0]:^20} | {entries[1]:^20} | {entries[2]:^20}")


def find_problematic_pixels(on_off_dict):
    """Find all pixels with negative durations."""
    print(f"\n\n{'#'*70}")
    print("SEARCHING FOR ALL PROBLEMATIC PIXELS")
    print(f"{'#'*70}")
    
    problems = []
    
    for key, pixel_data in on_off_dict.items():
        on_times = pixel_data['on_peak_location']
        off_times = pixel_data['off_peak_location']
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            if off < on:
                direction_idx = trial_idx % N_DIRECTIONS
                rep_idx = trial_idx // N_DIRECTIONS
                direction = DIRECTION_LIST[direction_idx]
                
                problems.append({
                    'pixel': key,
                    'trial_idx': trial_idx,
                    'direction': direction,
                    'rep': rep_idx,
                    'on': on,
                    'off': off,
                    'duration': off - on,
                })
    
    if not problems:
        print("\n✓ No problematic pixels found!")
        return
    
    print(f"\nFound {len(problems)} entries with negative duration:")
    print("-" * 80)
    
    # Group by pixel
    from collections import defaultdict
    by_pixel = defaultdict(list)
    for p in problems:
        by_pixel[p['pixel']].append(p)
    
    print(f"\nAffected pixels: {len(by_pixel)}")
    
    for pixel, entries in by_pixel.items():
        print(f"\n  Pixel {pixel}:")
        for e in entries:
            print(f"    Dir {e['direction']:>3}°, Rep {e['rep']}: on={e['on']}, off={e['off']}, dur={e['duration']}")


def main():
    print("=" * 70)
    print("On/Off Dictionary Examination")
    print("=" * 70)
    
    print(f"\nLoading: {ON_OFF_DICT_PATH}")
    
    if not ON_OFF_DICT_PATH.exists():
        print(f"✗ File not found!")
        return
    
    on_off_dict = load_on_off_dict(ON_OFF_DICT_PATH)
    print(f"✓ Loaded {len(on_off_dict)} pixels")
    
    # Sample some keys to understand format
    sample_keys = list(on_off_dict.keys())[:5]
    print(f"\nSample pixel keys: {sample_keys}")
    
    # Examine the problem pixel
    examine_pixel(on_off_dict, 270, 30)
    
    # Compare with nearby pixels
    nearby_pixels = [
        (270, 30),   # Problem pixel
        (270, 50),   # Same row, different col
        (270, 170),  # Same row, col from good unit
        (250, 30),   # Nearby row
        (290, 30),   # Nearby row
    ]
    compare_pixels(on_off_dict, nearby_pixels)
    
    # Find all problematic pixels
    find_problematic_pixels(on_off_dict)
    
    print("\n" + "=" * 70)
    print("Examination Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

