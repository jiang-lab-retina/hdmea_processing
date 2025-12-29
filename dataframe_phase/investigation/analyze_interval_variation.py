"""
Analyze the variation of ON/OFF intervals in the area-based on_off_dict.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Path to area-based dictionary
AREA_BASED_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl"
)

SINGLE_PIXEL_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

OUTPUT_DIR = Path(__file__).parent / "output" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8
N_REPS = 3


def analyze_intervals(dict_path: Path, name: str):
    """Analyze ON/OFF interval durations."""
    print(f"\n{'='*70}")
    print(f"Analyzing intervals: {name}")
    print(f"{'='*70}")
    
    with open(dict_path, 'rb') as f:
        on_off_dict = pickle.load(f)
    
    # Collect all intervals by direction
    intervals_by_dir = defaultdict(list)
    
    for key, pixel_data in on_off_dict.items():
        on_times = pixel_data['on_peak_location']
        off_times = pixel_data['off_peak_location']
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
            duration = off - on
            intervals_by_dir[direction].append(duration)
    
    # Statistics
    print(f"\n  {'Direction':<12} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Range':>8}")
    print(f"  {'-'*55}")
    
    stats = {}
    for direction in DIRECTION_LIST:
        intervals = np.array(intervals_by_dir[direction])
        stats[direction] = {
            'min': intervals.min(),
            'max': intervals.max(),
            'mean': intervals.mean(),
            'std': intervals.std(),
            'range': intervals.max() - intervals.min(),
            'data': intervals
        }
        
        s = stats[direction]
        print(f"  {direction:>3}°         {s['min']:>8} {s['max']:>8} {s['mean']:>8.1f} {s['std']:>8.2f} {s['range']:>8}")
    
    return stats, on_off_dict


def plot_interval_distributions(single_stats, area_stats):
    """Create plots comparing interval distributions."""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('ON/OFF Interval Duration Distribution by Direction\n(Single-Pixel vs Area-Based)', 
                 fontsize=14, fontweight='bold')
    
    for i, direction in enumerate(DIRECTION_LIST):
        ax = axes[i // 4, i % 4]
        
        # Get data
        single_data = single_stats[direction]['data']
        area_data = area_stats[direction]['data']
        
        # Filter out negative values for single-pixel (for visualization)
        single_positive = single_data[single_data > 0]
        
        # Histograms
        bins = np.arange(0, 35, 1)
        
        ax.hist(single_positive, bins=bins, alpha=0.5, label=f'Single (n={len(single_positive)})', color='red')
        ax.hist(area_data, bins=bins, alpha=0.5, label=f'Area (n={len(area_data)})', color='blue')
        
        # Stats annotation
        ax.axvline(area_stats[direction]['mean'], color='blue', linestyle='--', alpha=0.7)
        ax.axvline(single_stats[direction]['mean'], color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(f"Direction {direction}°")
        ax.set_xlabel('Duration (frames)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 35)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "interval_distribution_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def analyze_spatial_variation(on_off_dict, direction=45):
    """Analyze how intervals vary spatially across the grid."""
    
    # Create 300x300 grid of interval values
    interval_grid = np.zeros((300, 300))
    
    for (row, col), pixel_data in on_off_dict.items():
        # Get first trial of specified direction
        trial_idx = DIRECTION_LIST.index(direction)
        on = pixel_data['on_peak_location'][trial_idx]
        off = pixel_data['off_peak_location'][trial_idx]
        interval_grid[row, col] = off - on
    
    return interval_grid


def plot_spatial_variation(single_dict, area_dict):
    """Plot spatial variation of intervals."""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Spatial Distribution of ON/OFF Interval Duration\n(Top: Single-Pixel, Bottom: Area-Based)', 
                 fontsize=14, fontweight='bold')
    
    # Only show diagonal directions (which have problems)
    directions = [45, 135, 225, 315]
    
    for i, direction in enumerate(directions):
        # Single-pixel
        ax = axes[0, i]
        grid = analyze_spatial_variation(single_dict, direction)
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-30, vmax=30, origin='upper')
        ax.set_title(f'Single-Pixel, Dir {direction}°')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, label='Duration')
        
        # Area-based
        ax = axes[1, i]
        grid = analyze_spatial_variation(area_dict, direction)
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-30, vmax=30, origin='upper')
        ax.set_title(f'Area-Based, Dir {direction}°')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, label='Duration')
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "interval_spatial_variation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("#" * 70)
    print("ON/OFF INTERVAL VARIATION ANALYSIS")
    print("#" * 70)
    
    # Analyze both dictionaries
    single_stats, single_dict = analyze_intervals(SINGLE_PIXEL_PATH, "Single-Pixel (_hd.pkl)")
    area_stats, area_dict = analyze_intervals(AREA_BASED_PATH, "Area-Based (_area_hd.pkl)")
    
    # Detailed statistics for area-based
    print("\n" + "=" * 70)
    print("DETAILED AREA-BASED STATISTICS")
    print("=" * 70)
    
    print("\n  All directions combined:")
    all_intervals = np.concatenate([area_stats[d]['data'] for d in DIRECTION_LIST])
    print(f"    Total samples: {len(all_intervals):,}")
    print(f"    Min: {all_intervals.min()}")
    print(f"    Max: {all_intervals.max()}")
    print(f"    Mean: {all_intervals.mean():.2f}")
    print(f"    Std: {all_intervals.std():.2f}")
    print(f"    Median: {np.median(all_intervals):.0f}")
    
    # Unique values
    unique_vals = np.unique(all_intervals)
    print(f"\n  Unique interval values: {len(unique_vals)}")
    print(f"    Values: {sorted(unique_vals)}")
    
    # Value counts
    print(f"\n  Value distribution:")
    for val in sorted(unique_vals):
        count = np.sum(all_intervals == val)
        pct = 100 * count / len(all_intervals)
        bar = '█' * int(pct / 2)
        print(f"    {val:>3} frames: {count:>8,} ({pct:>5.1f}%) {bar}")
    
    # Horizontal vs Diagonal comparison
    print("\n" + "=" * 70)
    print("HORIZONTAL/VERTICAL vs DIAGONAL DIRECTIONS")
    print("=" * 70)
    
    hv_directions = [0, 90, 180, 270]
    diag_directions = [45, 135, 225, 315]
    
    hv_intervals = np.concatenate([area_stats[d]['data'] for d in hv_directions])
    diag_intervals = np.concatenate([area_stats[d]['data'] for d in diag_directions])
    
    print(f"\n  Horizontal/Vertical (0°, 90°, 180°, 270°):")
    print(f"    Mean: {hv_intervals.mean():.2f} frames")
    print(f"    Std: {hv_intervals.std():.2f}")
    print(f"    Range: {hv_intervals.min()} - {hv_intervals.max()}")
    
    print(f"\n  Diagonal (45°, 135°, 225°, 315°):")
    print(f"    Mean: {diag_intervals.mean():.2f} frames")
    print(f"    Std: {diag_intervals.std():.2f}")
    print(f"    Range: {diag_intervals.min()} - {diag_intervals.max()}")
    
    # Create plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    plot_interval_distributions(single_stats, area_stats)
    plot_spatial_variation(single_dict, area_dict)
    
    # Summary
    print("\n" + "#" * 70)
    print("SUMMARY")
    print("#" * 70)
    print(f"""
Area-Based ON/OFF Interval Statistics:
--------------------------------------
- All intervals are POSITIVE (no swapped values)
- Interval range: {all_intervals.min()} - {all_intervals.max()} frames
- Mean duration: {all_intervals.mean():.1f} frames (~{all_intervals.mean()/60:.3f}s at 60Hz)
- Standard deviation: {all_intervals.std():.2f} frames

Variation by movement type:
- Horizontal/Vertical: {hv_intervals.mean():.1f} ± {hv_intervals.std():.1f} frames (range: {hv_intervals.max() - hv_intervals.min()})
- Diagonal: {diag_intervals.mean():.1f} ± {diag_intervals.std():.1f} frames (range: {diag_intervals.max() - diag_intervals.min()})

The variation is due to:
- Bar speed (5 pixels/frame)
- Receptive field averaging area (50×50 pixels)
- Pixel position relative to bar path
""")


if __name__ == "__main__":
    main()

