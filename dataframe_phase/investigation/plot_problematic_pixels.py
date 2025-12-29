"""
2D Scatter plot of problematic pixels in on_off_dict.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Configuration
ON_OFF_DICT_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

OUTPUT_DIR = Path(__file__).parent / "output" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8

# Direction colors (using distinct colors for diagonal directions)
DIRECTION_COLORS = {
    0: '#2ecc71',    # green
    45: '#e74c3c',   # red
    90: '#3498db',   # blue
    135: '#f39c12',  # orange
    180: '#9b59b6',  # purple
    225: '#1abc9c',  # teal
    270: '#34495e',  # dark gray
    315: '#e91e63',  # pink
}


def load_problematic_pixels():
    """Load and identify problematic pixels."""
    with open(ON_OFF_DICT_PATH, 'rb') as f:
        on_off_dict = pickle.load(f)
    
    # Collect problematic pixels by direction
    pixels_by_dir = defaultdict(list)
    
    for key, pixel_data in on_off_dict.items():
        on_times = pixel_data['on_peak_location']
        off_times = pixel_data['off_peak_location']
        
        for trial_idx, (on, off) in enumerate(zip(on_times, off_times)):
            if off < on:
                direction = DIRECTION_LIST[trial_idx % N_DIRECTIONS]
                pixels_by_dir[direction].append(key)
                break  # Only count each pixel once per direction
    
    return pixels_by_dir, len(on_off_dict)


def main():
    print("Loading problematic pixels...")
    pixels_by_dir, total_pixels = load_problematic_pixels()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Problematic Pixels in on_off_dict\n(ON/OFF times swapped)', 
                 fontsize=16, fontweight='bold')
    
    # =========================================================================
    # Plot 1: All problematic pixels colored by direction
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('All Problematic Pixels by Direction', fontsize=12)
    
    for direction in [45, 135, 225, 315]:  # Only plot affected directions
        if pixels_by_dir[direction]:
            pixels = np.array(pixels_by_dir[direction])
            rows = pixels[:, 0]
            cols = pixels[:, 1]
            ax1.scatter(cols, rows, c=DIRECTION_COLORS[direction], 
                       label=f'{direction}°', alpha=0.6, s=5)
    
    ax1.set_xlim(0, 300)
    ax1.set_ylim(0, 300)
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.legend(title='Direction', loc='upper right')
    ax1.set_aspect('equal')
    ax1.invert_yaxis()  # Image coordinates
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 2-5: Individual direction plots
    # =========================================================================
    directions = [45, 135, 225, 315]
    
    for i, direction in enumerate(directions):
        ax = fig.add_subplot(2, 2, i + 1) if i == 0 else fig.add_subplot(2, 4, i + 4)
    
    # Create a 2x4 grid for the bottom row
    for i, direction in enumerate(directions):
        ax = fig.add_subplot(2, 4, 5 + i)
        ax.set_title(f'Direction {direction}°', fontsize=11)
        
        if pixels_by_dir[direction]:
            pixels = np.array(pixels_by_dir[direction])
            rows = pixels[:, 0]
            cols = pixels[:, 1]
            ax.scatter(cols, rows, c=DIRECTION_COLORS[direction], alpha=0.7, s=8)
            
            # Add count
            ax.text(0.02, 0.98, f'n={len(pixels)}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "problematic_pixels_scatter.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # =========================================================================
    # Create a more detailed combined plot
    # =========================================================================
    fig2, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle('Problematic Pixels Analysis - ON/OFF Swapped in on_off_dict', 
                  fontsize=14, fontweight='bold')
    
    # Plot 1: All combined
    ax = axes[0, 0]
    ax.set_title('All Affected Pixels (Diagonal Directions Only)', fontsize=11)
    
    all_pixels = []
    for direction in [45, 135, 225, 315]:
        if pixels_by_dir[direction]:
            for p in pixels_by_dir[direction]:
                all_pixels.append(p)
    
    if all_pixels:
        all_pixels = np.array(list(set(map(tuple, all_pixels))))
        ax.scatter(all_pixels[:, 1], all_pixels[:, 0], c='red', alpha=0.5, s=3)
        ax.text(0.02, 0.98, f'Total unique: {len(all_pixels)}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of problem density
    ax = axes[0, 1]
    ax.set_title('Problem Density Heatmap', fontsize=11)
    
    # Create density grid
    density = np.zeros((30, 30))
    for p in all_pixels:
        r, c = int(p[0] // 10), int(p[1] // 10)
        if 0 <= r < 30 and 0 <= c < 30:
            density[r, c] += 1
    
    im = ax.imshow(density, cmap='Reds', origin='upper', extent=[0, 300, 300, 0])
    plt.colorbar(im, ax=ax, label='Count per 10x10 block')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Plot 3: Problem regions overlay
    ax = axes[0, 2]
    ax.set_title('Problem Regions (>0 problems)', fontsize=11)
    
    # Create binary mask
    mask = (density > 0).astype(float)
    ax.imshow(mask, cmap='Reds', origin='upper', extent=[0, 300, 300, 0], alpha=0.7)
    
    # Overlay the 300x300 grid reference
    ax.axhline(150, color='blue', linestyle='--', alpha=0.5, label='Center')
    ax.axvline(150, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend(loc='upper right')
    
    # Bottom row: Individual directions
    for i, direction in enumerate([45, 135, 225, 315]):
        if i < 3:
            ax = axes[1, i]
        else:
            break
            
        ax.set_title(f'Direction {direction}° ({len(pixels_by_dir[direction])} pixels)', fontsize=11)
        
        if pixels_by_dir[direction]:
            pixels = np.array(pixels_by_dir[direction])
            ax.scatter(pixels[:, 1], pixels[:, 0], c=DIRECTION_COLORS[direction], 
                      alpha=0.6, s=8)
            
            # Draw direction arrow
            center = 150
            arrow_len = 50
            angle_rad = np.radians(90 - direction)  # Convert to standard math angle
            dx = arrow_len * np.cos(angle_rad)
            dy = -arrow_len * np.sin(angle_rad)  # Negative because y is inverted
            ax.annotate('', xy=(center + dx, center + dy), xytext=(center, center),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 300)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    # Last subplot: Direction 315
    ax = axes[1, 2]
    ax.clear()
    direction = 315
    ax.set_title(f'Direction {direction}° ({len(pixels_by_dir[direction])} pixels)', fontsize=11)
    
    if pixels_by_dir[direction]:
        pixels = np.array(pixels_by_dir[direction])
        ax.scatter(pixels[:, 1], pixels[:, 0], c=DIRECTION_COLORS[direction], 
                  alpha=0.6, s=8)
        
        # Draw direction arrow
        center = 150
        arrow_len = 50
        angle_rad = np.radians(90 - direction)
        dx = arrow_len * np.cos(angle_rad)
        dy = -arrow_len * np.sin(angle_rad)
        ax.annotate('', xy=(center + dx, center + dy), xytext=(center, center),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path2 = OUTPUT_DIR / "problematic_pixels_analysis.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()
    
    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total pixels in dictionary: {total_pixels}")
    print(f"Total unique affected pixels: {len(all_pixels)}")
    print(f"Percentage affected: {100 * len(all_pixels) / total_pixels:.2f}%")
    
    print("\nBy direction:")
    for d in [45, 135, 225, 315]:
        print(f"  {d}°: {len(pixels_by_dir[d])} pixels")
    
    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

