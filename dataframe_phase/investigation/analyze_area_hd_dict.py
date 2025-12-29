"""
Analyze the area-based on_off_dict to check if it has the same problems
as the single-pixel version.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
from collections import defaultdict
import numpy as np

# Paths to both dictionaries
SINGLE_PIXEL_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

AREA_BASED_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl"
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
        total_pixels = len(set.union(*pixels_by_dir.values()))
        print(f"    TOTAL: {total_problems} entries across {total_pixels} unique pixels")
    else:
        print(f"    ✓ NO PROBLEMS FOUND!")
    
    return on_off_dict, pixels_by_dir


def compare_specific_pixel(single_dict, area_dict, row, col):
    """Compare a specific pixel between both dictionaries."""
    print(f"\n{'='*70}")
    print(f"Comparing pixel ({row}, {col})")
    print(f"{'='*70}")
    
    key = (row, col)
    
    for name, d in [("Single-pixel", single_dict), ("Area-based", area_dict)]:
        if d is None:
            continue
            
        if key not in d:
            print(f"\n  {name}: Pixel not found!")
            continue
        
        pd = d[key]
        on_times = pd['on_peak_location']
        off_times = pd['off_peak_location']
        
        print(f"\n  {name}:")
        print(f"  {'Trial':<8} {'Dir':<6} {'Rep':<4} {'ON':>6} {'OFF':>6} {'Dur':>6} {'Status':<10}")
        print(f"  {'-'*50}")
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
            rep = trial_idx // N_DIRECTIONS
            dur = off - on
            status = "✓" if dur > 0 else "✗ SWAPPED"
            
            print(f"  {trial_idx:<8} {direction:>3}°   {rep:<4} {on:>6} {off:>6} {dur:>+6} {status}")


def main():
    print("#" * 70)
    print("COMPARISON: Single-Pixel vs Area-Based on_off_dict")
    print("#" * 70)
    
    # Analyze both dictionaries
    single_dict, single_problems = analyze_dict(SINGLE_PIXEL_PATH, "Single-Pixel (_hd.pkl)")
    area_dict, area_problems = analyze_dict(AREA_BASED_PATH, "Area-Based (_area_hd.pkl)")
    
    # Summary comparison
    print("\n" + "#" * 70)
    print("SUMMARY COMPARISON")
    print("#" * 70)
    
    print("\n  Problematic pixels per direction:")
    print(f"  {'Direction':<12} {'Single-Pixel':>15} {'Area-Based':>15} {'Improvement':>15}")
    print(f"  {'-'*60}")
    
    for d in DIRECTION_LIST:
        single_n = len(single_problems[d]) if single_problems else 0
        area_n = len(area_problems[d]) if area_problems else 0
        
        if single_n > 0:
            improvement = f"{100 * (single_n - area_n) / single_n:.1f}% better"
        else:
            improvement = "N/A"
        
        print(f"  {d:>3}°         {single_n:>15} {area_n:>15} {improvement:>15}")
    
    # Compare specific problematic pixels
    problem_pixels = [
        (270, 30),   # From unit_001 in problem recording
        (200, 100),  # Sample from affected region
        (150, 150),  # Center pixel (should be fine)
    ]
    
    print("\n" + "#" * 70)
    print("SPECIFIC PIXEL COMPARISON")
    print("#" * 70)
    
    for row, col in problem_pixels:
        compare_specific_pixel(single_dict, area_dict, row, col)
    
    # Conclusion
    print("\n" + "#" * 70)
    print("CONCLUSION")
    print("#" * 70)
    
    single_total = sum(len(v) for v in single_problems.values()) if single_problems else 0
    area_total = sum(len(v) for v in area_problems.values()) if area_problems else 0
    
    if area_total == 0 and single_total > 0:
        print("""
✓ The area-based dictionary has NO problematic pixels!
  
  RECOMMENDATION: Switch to using the area-based dictionary:
  
  In Projects/unified_pipeline/config.py, change:
  
  DSGC_ON_OFF_DICT_PATH = Path(
      r"M:\\Python_Project\\Data_Processing_2025\\Design_Stimulation_Pattern"
      r"\\Data\\Stimulations\\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl"
  )
""")
    elif area_total < single_total:
        reduction = 100 * (single_total - area_total) / single_total
        print(f"""
  The area-based dictionary has {reduction:.1f}% fewer problems:
  - Single-pixel: {single_total} affected pixels
  - Area-based: {area_total} affected pixels
  
  Consider using the area-based dictionary for better results.
""")
    else:
        print(f"""
  Both dictionaries have similar issues:
  - Single-pixel: {single_total} affected pixels
  - Area-based: {area_total} affected pixels
""")


if __name__ == "__main__":
    main()

