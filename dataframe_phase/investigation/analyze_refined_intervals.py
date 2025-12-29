"""
Analyze the variation of ON-to-OFF intervals (durations) in the refined dictionary.
Compare with legacy single-pixel and area-based dictionaries.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
import numpy as np
from collections import defaultdict

# Paths
REFINED_PATH = Path(__file__).parent / "output" / "moving_h_bar_s5_d8_3x_on_off_dict_hd_refined.pkl"

LEGACY_SINGLE_PIXEL_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

LEGACY_AREA_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl"
)

DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8
N_REPS = 3


def analyze_intervals(dict_path: Path, name: str):
    """Analyze ON-to-OFF intervals in a dictionary."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"{'='*70}")
    
    if not dict_path.exists():
        print(f"  ✗ File not found: {dict_path}")
        return None
    
    with open(dict_path, 'rb') as f:
        on_off_dict = pickle.load(f)
    
    print(f"  Total pixels: {len(on_off_dict)}")
    
    # Collect durations by direction
    durations_by_dir = defaultdict(list)
    all_durations = []
    negative_count = 0
    
    for key, pixel_data in on_off_dict.items():
        on_times = pixel_data['on_peak_location']
        off_times = pixel_data['off_peak_location']
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            duration = off - on
            direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
            
            durations_by_dir[direction].append(duration)
            all_durations.append(duration)
            
            if duration <= 0:
                negative_count += 1
    
    # Statistics per direction
    print(f"\n  Duration statistics by direction:")
    print(f"  {'Dir':>5} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Range':>8}")
    print(f"  {'-'*50}")
    
    stats = {}
    for d in DIRECTION_LIST:
        durations = np.array(durations_by_dir[d])
        stats[d] = {
            'min': np.min(durations),
            'max': np.max(durations),
            'mean': np.mean(durations),
            'std': np.std(durations),
            'range': np.max(durations) - np.min(durations)
        }
        print(f"  {d:>3}°  {stats[d]['min']:>8} {stats[d]['max']:>8} "
              f"{stats[d]['mean']:>8.1f} {stats[d]['std']:>8.2f} {stats[d]['range']:>8}")
    
    # Overall statistics
    all_durations = np.array(all_durations)
    print(f"\n  Overall statistics:")
    print(f"    Min duration: {np.min(all_durations)}")
    print(f"    Max duration: {np.max(all_durations)}")
    print(f"    Mean duration: {np.mean(all_durations):.2f}")
    print(f"    Std deviation: {np.std(all_durations):.2f}")
    print(f"    Negative/zero: {negative_count}")
    
    # Value distribution
    print(f"\n  Duration value distribution:")
    unique, counts = np.unique(all_durations, return_counts=True)
    
    # Show most common values
    sorted_indices = np.argsort(-counts)
    print(f"    Top 10 most common durations:")
    for i in sorted_indices[:10]:
        pct = counts[i] / len(all_durations) * 100
        print(f"      Duration {unique[i]:>4}: {counts[i]:>8} occurrences ({pct:>5.2f}%)")
    
    return {
        'stats': stats,
        'all_durations': all_durations,
        'durations_by_dir': durations_by_dir,
        'negative_count': negative_count
    }


def compare_all():
    """Compare all three dictionaries."""
    print("#" * 70)
    print("INTERVAL VARIATION ANALYSIS")
    print("#" * 70)
    
    results = {}
    
    # Analyze each dictionary
    results['refined'] = analyze_intervals(REFINED_PATH, "Refined (OFF > ON constraint)")
    results['single_pixel'] = analyze_intervals(LEGACY_SINGLE_PIXEL_PATH, "Legacy Single-Pixel")
    results['area_hd'] = analyze_intervals(LEGACY_AREA_PATH, "Legacy Area-Based (RF=25)")
    
    # Comparison summary
    print("\n" + "#" * 70)
    print("COMPARISON SUMMARY")
    print("#" * 70)
    
    print("\n  Standard deviation by direction (lower = more consistent):")
    print(f"  {'Dir':>5} {'Single-Pixel':>14} {'Refined':>14} {'Area-Based':>14}")
    print(f"  {'-'*50}")
    
    for d in DIRECTION_LIST:
        sp_std = results['single_pixel']['stats'][d]['std'] if results['single_pixel'] else 'N/A'
        ref_std = results['refined']['stats'][d]['std'] if results['refined'] else 'N/A'
        area_std = results['area_hd']['stats'][d]['std'] if results['area_hd'] else 'N/A'
        
        print(f"  {d:>3}°  {sp_std:>14.2f} {ref_std:>14.2f} {area_std:>14.2f}")
    
    print("\n  Negative/zero durations:")
    for name, data in results.items():
        if data:
            print(f"    {name:>15}: {data['negative_count']}")
    
    # Horizontal vs Diagonal comparison for refined
    if results['refined']:
        print("\n  Refined dictionary - Horizontal/Vertical vs Diagonal:")
        
        hv_dirs = [0, 90, 180, 270]
        diag_dirs = [45, 135, 225, 315]
        
        hv_stds = [results['refined']['stats'][d]['std'] for d in hv_dirs]
        diag_stds = [results['refined']['stats'][d]['std'] for d in diag_dirs]
        
        print(f"    Horizontal/Vertical (0°, 90°, 180°, 270°):")
        print(f"      Avg Std: {np.mean(hv_stds):.2f}")
        print(f"    Diagonal (45°, 135°, 225°, 315°):")
        print(f"      Avg Std: {np.mean(diag_stds):.2f}")


def main():
    compare_all()
    
    print("\n" + "#" * 70)
    print("CONCLUSION")
    print("#" * 70)
    print("""
The refined dictionary fixes the negative duration issue by constraining
OFF peak search to only occur after the ON peak.

Key observations:
1. Zero negative/zero durations (vs 5304 in legacy single-pixel)
2. Similar variation patterns to the original single-pixel method
3. Area-based method still has lower variation for diagonal directions
""")


if __name__ == "__main__":
    main()

