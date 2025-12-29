"""
Investigate the 460-duration problem in the refined dictionary.

These are the 5,304 entries (1,768 pixels × 3 reps) that have abnormally
long durations because the OFF peak search starts after ON, but finds
a peak near the end of the segment.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Paths
REFINED_PATH = Path(__file__).parent / "output" / "moving_h_bar_s5_d8_3x_on_off_dict_hd_refined.pkl"
LEGACY_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)
STIMULUS_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x.npy"
)

OUTPUT_DIR = Path(__file__).parent / "output" / "plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8


def find_460_pixels():
    """Find all pixels that have duration=460."""
    print("Loading refined dictionary...")
    with open(REFINED_PATH, 'rb') as f:
        refined_dict = pickle.load(f)
    
    print("Loading legacy dictionary...")
    with open(LEGACY_PATH, 'rb') as f:
        legacy_dict = pickle.load(f)
    
    # Find pixels with duration=460
    pixels_460 = defaultdict(list)  # {pixel: [(trial_idx, direction, refined_dur, legacy_dur)]}
    
    for key, pixel_data in refined_dict.items():
        on_times = pixel_data['on_peak_location']
        off_times = pixel_data['off_peak_location']
        
        legacy_on = legacy_dict[key]['on_peak_location']
        legacy_off = legacy_dict[key]['off_peak_location']
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            duration = off - on
            legacy_dur = legacy_off[trial_idx] - legacy_on[trial_idx]
            
            if duration >= 400:  # Abnormally long
                direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
                pixels_460[key].append({
                    'trial_idx': trial_idx,
                    'direction': direction,
                    'rep': trial_idx // N_DIRECTIONS,
                    'refined_on': on,
                    'refined_off': off,
                    'refined_dur': duration,
                    'legacy_on': legacy_on[trial_idx],
                    'legacy_off': legacy_off[trial_idx],
                    'legacy_dur': legacy_dur
                })
    
    print(f"\nFound {len(pixels_460)} pixels with duration >= 400")
    
    # Summary by direction
    dir_counts = defaultdict(int)
    for pixel, trials in pixels_460.items():
        for t in trials:
            dir_counts[t['direction']] += 1
    
    print("\nAffected entries by direction:")
    for d in DIRECTION_LIST:
        print(f"  {d:>3}°: {dir_counts[d]}")
    
    return pixels_460, refined_dict, legacy_dict


def analyze_pixel_trace(pixel, pixels_460, ds_stim):
    """Analyze the intensity trace for a specific pixel."""
    x, y = pixel
    
    # Get pixel trace
    trace = ds_stim[:, x, y].astype(np.float64)
    
    # Compute derivative
    trace_diff = np.concatenate([np.zeros(1), trace])
    trace_diff = np.diff(trace_diff)
    
    # Get affected trials
    affected = pixels_460[pixel]
    
    print(f"\n{'='*70}")
    print(f"Pixel ({x}, {y})")
    print(f"{'='*70}")
    
    for info in affected:
        trial_idx = info['trial_idx']
        direction = info['direction']
        rep = info['rep']
        
        # Calculate segment bounds (same as in the algorithm)
        counter = 0
        for j in range(rep + 1):
            if j < rep:
                counter += 120 + 4400
            else:
                counter += 120
                for i in range(direction // 45 + 1):
                    if i < direction // 45:
                        counter += 4400 / 8
                    else:
                        start = counter
                        counter += 4400 / 8
                        end = counter
        
        start = int(start)
        end = int(end)
        segment_trace = trace[start:end]
        segment_diff = trace_diff[start:end]
        
        # Find ON peak (max positive derivative)
        on_peak_idx = np.argmax(segment_diff)
        on_peak_abs = on_peak_idx + start
        
        # Find OFF peak after ON (max negative derivative)
        off_search_start = on_peak_idx + 1
        if off_search_start < len(segment_diff):
            off_peak_idx = np.argmax(-segment_diff[off_search_start:]) + off_search_start
        else:
            off_peak_idx = len(segment_diff) - 1
        off_peak_abs = off_peak_idx + start
        
        print(f"\n  Trial {trial_idx} (Dir {direction}°, Rep {rep}):")
        print(f"    Segment: frames {start} to {end} ({end-start} frames)")
        print(f"    ON peak at frame {on_peak_abs} (relative: {on_peak_idx})")
        print(f"    OFF peak at frame {off_peak_abs} (relative: {off_peak_idx})")
        print(f"    Duration: {off_peak_abs - on_peak_abs} frames")
        
        # Analyze the derivative trace
        print(f"\n    Derivative analysis:")
        print(f"      Max positive (ON): {segment_diff[on_peak_idx]:.2f} at frame {on_peak_idx}")
        
        # Check what's after ON peak
        after_on = segment_diff[on_peak_idx+1:]
        if len(after_on) > 0:
            min_after = np.min(after_on)
            min_idx_after = np.argmin(after_on) + on_peak_idx + 1
            print(f"      Min negative after ON: {min_after:.2f} at frame {min_idx_after}")
            print(f"      Max negative after ON: {-np.max(-after_on):.2f}")
        
        # Check the full derivative range
        print(f"      Full segment derivative range: [{segment_diff.min():.2f}, {segment_diff.max():.2f}]")
        
        # Check if the trace is flat (no clear ON/OFF)
        trace_range = segment_trace.max() - segment_trace.min()
        print(f"      Intensity range: {trace_range:.2f} (min: {segment_trace.min():.2f}, max: {segment_trace.max():.2f})")
        
        if trace_range < 10:
            print(f"      ⚠ FLAT TRACE - pixel may not be covered by the bar!")
        
        return start, end, on_peak_idx, off_peak_idx, segment_trace, segment_diff, direction


def plot_problematic_pixel(pixel, pixels_460, ds_stim):
    """Create a visualization of a problematic pixel."""
    x, y = pixel
    trace = ds_stim[:, x, y].astype(np.float64)
    trace_diff = np.concatenate([np.zeros(1), trace])
    trace_diff = np.diff(trace_diff)
    
    affected = pixels_460[pixel]
    info = affected[0]  # Take first affected trial
    
    trial_idx = info['trial_idx']
    direction = info['direction']
    rep = info['rep']
    
    # Calculate segment bounds
    counter = 0
    for j in range(rep + 1):
        if j < rep:
            counter += 120 + 4400
        else:
            counter += 120
            for i in range((direction // 45) + 1):
                if i < (direction // 45):
                    counter += 4400 / 8
                else:
                    start = counter
                    counter += 4400 / 8
                    end = counter
    
    start = int(start)
    end = int(end)
    segment_trace = trace[start:end]
    segment_diff = trace_diff[start:end]
    
    # Find peaks
    on_peak_idx = np.argmax(segment_diff)
    off_search_start = on_peak_idx + 1
    if off_search_start < len(segment_diff):
        off_peak_idx = np.argmax(-segment_diff[off_search_start:]) + off_search_start
    else:
        off_peak_idx = len(segment_diff) - 1
    
    # Also find where the true OFF would be (before ON constraint)
    true_off_idx = np.argmax(-segment_diff)
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    frames = np.arange(len(segment_trace))
    
    # Plot 1: Intensity trace
    ax1 = axes[0]
    ax1.plot(frames, segment_trace, 'b-', linewidth=1.5, label='Intensity')
    ax1.axvline(on_peak_idx, color='green', linestyle='--', linewidth=2, label=f'ON ({on_peak_idx})')
    ax1.axvline(off_peak_idx, color='red', linestyle='--', linewidth=2, label=f'OFF refined ({off_peak_idx})')
    ax1.axvline(true_off_idx, color='orange', linestyle=':', linewidth=2, label=f'OFF original ({true_off_idx})')
    ax1.set_xlabel('Frame (relative to segment start)')
    ax1.set_ylabel('Intensity')
    ax1.set_title(f'Pixel ({x}, {y}) - Direction {direction}° Rep {rep}\nIntensity Trace')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Derivative trace
    ax2 = axes[1]
    ax2.plot(frames, segment_diff, 'purple', linewidth=1.5, label='Derivative')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2.axvline(on_peak_idx, color='green', linestyle='--', linewidth=2, label=f'ON (max+)')
    ax2.axvline(off_peak_idx, color='red', linestyle='--', linewidth=2, label=f'OFF refined (max- after ON)')
    ax2.axvline(true_off_idx, color='orange', linestyle=':', linewidth=2, label=f'OFF original (max-)')
    
    # Shade the search region for OFF
    ax2.axvspan(on_peak_idx, len(segment_diff), alpha=0.2, color='red', label='OFF search region')
    
    ax2.set_xlabel('Frame (relative to segment start)')
    ax2.set_ylabel('Derivative (intensity change)')
    ax2.set_title(f'Derivative Trace - ON at {on_peak_idx}, OFF at {off_peak_idx} (duration: {off_peak_idx - on_peak_idx})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"460_problem_pixel_{x}_{y}_dir{direction}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    return true_off_idx, on_peak_idx, off_peak_idx


def main():
    print("#" * 70)
    print("INVESTIGATING THE 460-DURATION PROBLEM")
    print("#" * 70)
    
    # Find all 460-duration pixels
    pixels_460, refined_dict, legacy_dict = find_460_pixels()
    
    # Load stimulus
    print("\nLoading stimulus...")
    ds_stim = np.load(STIMULUS_PATH)
    print(f"Stimulus shape: {ds_stim.shape}")
    
    # Analyze a few example pixels
    sample_pixels = list(pixels_460.keys())[:3]
    
    print("\n" + "#" * 70)
    print("DETAILED ANALYSIS OF SAMPLE PIXELS")
    print("#" * 70)
    
    for pixel in sample_pixels:
        analyze_pixel_trace(pixel, pixels_460, ds_stim)
        plot_problematic_pixel(pixel, pixels_460, ds_stim)
    
    # Root cause analysis
    print("\n" + "#" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("#" * 70)
    
    # Check what the true OFF peak location is in these pixels
    print("\nChecking TRUE OFF peak (before constraint):")
    
    for pixel in sample_pixels[:1]:
        x, y = pixel
        trace = ds_stim[:, x, y].astype(np.float64)
        trace_diff = np.concatenate([np.zeros(1), trace])
        trace_diff = np.diff(trace_diff)
        
        info = pixels_460[pixel][0]
        direction = info['direction']
        
        # For direction 45, the segment is at a specific location
        # Let's look at the whole trial
        counter = 120  # First rep
        for i in range(direction // 45 + 1):
            if i < direction // 45:
                counter += 4400 / 8
            else:
                start = counter
                counter += 4400 / 8
                end = counter
        
        start = int(start)
        end = int(end)
        segment_diff = trace_diff[start:end]
        
        # Get all local maxima of positive and negative derivative
        on_peak = np.argmax(segment_diff)
        off_peak_orig = np.argmax(-segment_diff)
        
        print(f"\n  Pixel {pixel}, Direction {direction}°:")
        print(f"    ON peak (max positive derivative) at frame {on_peak}")
        print(f"    OFF peak (max negative derivative) at frame {off_peak_orig}")
        
        if off_peak_orig < on_peak:
            print(f"    ⚠ OFF comes BEFORE ON! (diff: {off_peak_orig - on_peak})")
            print(f"    This is why the legacy has negative duration: {off_peak_orig - on_peak}")
            print(f"    The refined algorithm searches after ON, so it finds OFF at {on_peak + np.argmax(-segment_diff[on_peak+1:]) + 1}")
    
    print("\n" + "#" * 70)
    print("CONCLUSION")
    print("#" * 70)
    print("""
ROOT CAUSE:
-----------
For diagonal bar movements at certain pixels, the bar's edge creates:
1. A NEGATIVE derivative spike (as the bar starts to leave) BEFORE
2. A POSITIVE derivative spike (as a different part of bar enters)

This is a physical phenomenon of how diagonal bars sweep across pixels
near the edges of the bar's path.

WHY 460?
--------
- The segment is 550 frames long (4400/8 = 550)
- ON peak is near the start (around frame 90)
- The TRUE OFF peak (negative derivative) is BEFORE ON
- When we search AFTER ON, we find the next negative peak near the end
- 550 - 90 ≈ 460 frames

SOLUTION OPTIONS:
-----------------
1. Use area-based dictionary (RF=25) - smooths out edge effects
2. Accept that these pixels have ambiguous ON/OFF timing
3. Use a different detection method (e.g., threshold-based instead of peak-based)
""")


if __name__ == "__main__":
    main()

