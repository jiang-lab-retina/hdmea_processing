"""
Analyze moving_h_bar_s5_d8_3x_on_off_dict_area_hd_bar_height_3.pkl
for problematic points (swapped ON/OFF values).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
from collections import defaultdict
import numpy as np

# Paths
BAR_HEIGHT_3_PATH = Path(__file__).parent / "output" / "moving_h_bar_s5_d8_3x_on_off_dict_area_hd_bar_height_3.pkl"

LEGACY_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl"
)

SINGLE_PIXEL_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8


def analyze_dict(dict_path: Path, name: str):
    """Analyze an on_off_dict for problematic entries."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"Path: {dict_path}")
    print(f"{'='*70}")
    
    if not dict_path.exists():
        print(f"  ✗ File not found!")
        return None, None
    
    with open(dict_path, 'rb') as f:
        on_off_dict = pickle.load(f)
    
    print(f"  Total pixels: {len(on_off_dict)}")
    
    # Count problems by direction
    problems_by_dir = defaultdict(int)
    pixels_by_dir = defaultdict(set)
    all_problems = []
    
    for key, pixel_data in on_off_dict.items():
        on_times = pixel_data['on_peak_location']
        off_times = pixel_data['off_peak_location']
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            if off < on:
                direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
                problems_by_dir[direction] += 1
                pixels_by_dir[direction].add(key)
                all_problems.append({
                    'pixel': key,
                    'direction': direction,
                    'trial_idx': trial_idx,
                    'on': on,
                    'off': off,
                    'diff': off - on
                })
    
    # Summary
    print(f"\n  Problematic entries by direction:")
    print(f"  {'-'*50}")
    
    total_problems = 0
    for d in DIRECTION_LIST:
        n_entries = problems_by_dir[d]
        n_pixels = len(pixels_by_dir[d])
        total_problems += n_entries
        
        if n_entries > 0:
            print(f"    Direction {d:>3}°: {n_entries:>5} entries across {n_pixels:>4} pixels")
        else:
            print(f"    Direction {d:>3}°: ✓ No problems")
    
    print(f"  {'-'*50}")
    
    if total_problems > 0:
        total_pixels = len(set.union(*pixels_by_dir.values())) if pixels_by_dir else 0
        print(f"    TOTAL: {total_problems} entries across {total_pixels} unique pixels")
    else:
        print(f"    ✓ NO PROBLEMS FOUND!")
    
    return on_off_dict, pixels_by_dir


def compare_specific_pixel(dicts: dict, row: int, col: int):
    """Compare a specific pixel across dictionaries."""
    print(f"\n{'='*70}")
    print(f"Comparing pixel ({row}, {col})")
    print(f"{'='*70}")
    
    key = (row, col)
    
    for name, d in dicts.items():
        if d is None:
            continue
            
        if key not in d:
            print(f"\n  {name}: Pixel not found!")
            continue
        
        pd = d[key]
        on_times = pd['on_peak_location']
        off_times = pd['off_peak_location']
        
        # Count problems
        problems = sum(1 for on, off in zip(on_times, off_times) if off < on)
        
        print(f"\n  {name}: ({problems} problems)")
        print(f"  {'Trial':<6} {'Dir':<5} {'ON':>6} {'OFF':>6} {'Dur':>6} {'Status':<10}")
        print(f"  {'-'*45}")
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
            dur = off - on
            status = "✓" if dur > 0 else "✗"
            
            # Only show first rep for brevity
            if trial_idx < 8:
                print(f"  {trial_idx:<6} {direction:>3}°  {on:>6} {off:>6} {dur:>+6} {status}")


def main():
    print("#" * 70)
    print("ANALYSIS: bar_height=3 vs bar_height=50 vs single-pixel")
    print("#" * 70)
    
    # Load all dictionaries
    dicts = {}
    pixels_by_dir = {}
    
    # Bar height 3
    d, p = analyze_dict(BAR_HEIGHT_3_PATH, "Bar Height 3 (RF=1)")
    if d:
        dicts["bar_height_3"] = d
        pixels_by_dir["bar_height_3"] = p
    
    # Legacy (bar height 50)
    d, p = analyze_dict(LEGACY_PATH, "Bar Height 50 (RF=25) - Legacy")
    if d:
        dicts["bar_height_50"] = d
        pixels_by_dir["bar_height_50"] = p
    
    # Single pixel
    d, p = analyze_dict(SINGLE_PIXEL_PATH, "Single Pixel (no RF averaging)")
    if d:
        dicts["single_pixel"] = d
        pixels_by_dir["single_pixel"] = p
    
    # Summary comparison
    print("\n" + "#" * 70)
    print("SUMMARY COMPARISON")
    print("#" * 70)
    
    print("\n  Problematic pixels per direction:")
    print(f"  {'Direction':<12} {'Single-Pixel':>14} {'Bar H=3':>12} {'Bar H=50':>12}")
    print(f"  {'-'*55}")
    
    for d in DIRECTION_LIST:
        sp = len(pixels_by_dir.get("single_pixel", {}).get(d, set()))
        h3 = len(pixels_by_dir.get("bar_height_3", {}).get(d, set()))
        h50 = len(pixels_by_dir.get("bar_height_50", {}).get(d, set()))
        
        print(f"  {d:>3}°          {sp:>14} {h3:>12} {h50:>12}")
    
    # Total
    sp_total = sum(len(v) for v in pixels_by_dir.get("single_pixel", {}).values())
    h3_total = sum(len(v) for v in pixels_by_dir.get("bar_height_3", {}).values())
    h50_total = sum(len(v) for v in pixels_by_dir.get("bar_height_50", {}).values())
    
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<12} {sp_total:>14} {h3_total:>12} {h50_total:>12}")
    
    # Compare specific problematic pixel
    print("\n" + "#" * 70)
    print("SPECIFIC PIXEL COMPARISON")
    print("#" * 70)
    
    compare_specific_pixel(dicts, 270, 30)
    
    # Conclusion
    print("\n" + "#" * 70)
    print("CONCLUSION")
    print("#" * 70)
    
    print(f"""
Receptive Field Size Comparison:
--------------------------------
- Single-pixel (RF=1x1): {sp_total} affected pixels
- Bar Height 3 (RF=3x3): {h3_total} affected pixels  
- Bar Height 50 (RF=50x50): {h50_total} affected pixels
""")
    
    if h3_total > h50_total:
        print(f"  ! Bar height 3 has MORE problems ({h3_total}) than bar height 50 ({h50_total})")
        print(f"    The larger RF averaging helps smooth out ambiguous peaks.")
    elif h3_total < h50_total:
        print(f"  ✓ Bar height 3 has FEWER problems ({h3_total}) than bar height 50 ({h50_total})")
    else:
        print(f"  = Both have the same number of problems ({h3_total})")


if __name__ == "__main__":
    main()

