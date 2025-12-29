"""
Plot heatmaps showing ON/OFF frame numbers across the 300x300 grid.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to area-based dictionary
AREA_BASED_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl"
)

OUTPUT_DIR = Path(__file__).parent / "output" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECTION_LIST = [0, 45, 90, 135, 180, 225, 270, 315]
N_DIRECTIONS = 8


def load_dict(path):
    """Load on_off_dict from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_grids(on_off_dict, trial_idx=0):
    """Create 300x300 grids for ON, OFF, and duration values."""
    on_grid = np.zeros((300, 300))
    off_grid = np.zeros((300, 300))
    duration_grid = np.zeros((300, 300))
    
    for (row, col), pixel_data in on_off_dict.items():
        on = pixel_data['on_peak_location'][trial_idx]
        off = pixel_data['off_peak_location'][trial_idx]
        
        on_grid[row, col] = on
        off_grid[row, col] = off
        duration_grid[row, col] = off - on
    
    return on_grid, off_grid, duration_grid


def plot_single_direction(on_off_dict, direction, rep=0, output_dir=OUTPUT_DIR):
    """Plot ON, OFF, and duration heatmaps for a single direction."""
    trial_idx = DIRECTION_LIST.index(direction) + rep * N_DIRECTIONS
    
    on_grid, off_grid, duration_grid = create_grids(on_off_dict, trial_idx)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Direction {direction}° (Rep {rep+1}) - Frame Number Heatmaps', 
                 fontsize=14, fontweight='bold')
    
    # ON frame heatmap
    ax = axes[0]
    im = ax.imshow(on_grid, cmap='viridis', origin='upper')
    ax.set_title('ON Frame (bar enters)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Frame number')
    
    # OFF frame heatmap
    ax = axes[1]
    im = ax.imshow(off_grid, cmap='viridis', origin='upper')
    ax.set_title('OFF Frame (bar exits)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Frame number')
    
    # Duration heatmap
    ax = axes[2]
    im = ax.imshow(duration_grid, cmap='RdYlGn', origin='upper', vmin=10, vmax=30)
    ax.set_title('Duration (OFF - ON)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='Frames')
    
    plt.tight_layout()
    
    output_path = output_dir / f"heatmap_direction_{direction}_rep{rep}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_directions_on(on_off_dict, output_dir=OUTPUT_DIR):
    """Plot ON frame heatmaps for all 8 directions."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('ON Frame Heatmaps for All Directions (Rep 1)\nColor = Frame when bar enters pixel', 
                 fontsize=14, fontweight='bold')
    
    for i, direction in enumerate(DIRECTION_LIST):
        ax = axes[i // 4, i % 4]
        trial_idx = DIRECTION_LIST.index(direction)  # Rep 0
        
        on_grid, _, _ = create_grids(on_off_dict, trial_idx)
        
        im = ax.imshow(on_grid, cmap='viridis', origin='upper')
        ax.set_title(f'{direction}°')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, label='Frame')
        
        # Add direction arrow
        center = 150
        arrow_len = 40
        angle_rad = np.radians(90 - direction)
        dx = arrow_len * np.cos(angle_rad)
        dy = -arrow_len * np.sin(angle_rad)
        ax.annotate('', xy=(center + dx, center + dy), xytext=(center, center),
                   arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    plt.tight_layout()
    
    output_path = output_dir / "heatmap_all_directions_ON.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_directions_duration(on_off_dict, output_dir=OUTPUT_DIR):
    """Plot duration heatmaps for all 8 directions."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Duration Heatmaps for All Directions (Rep 1)\nColor = Number of frames bar covers pixel', 
                 fontsize=14, fontweight='bold')
    
    for i, direction in enumerate(DIRECTION_LIST):
        ax = axes[i // 4, i % 4]
        trial_idx = DIRECTION_LIST.index(direction)  # Rep 0
        
        _, _, duration_grid = create_grids(on_off_dict, trial_idx)
        
        im = ax.imshow(duration_grid, cmap='RdYlGn', origin='upper', vmin=10, vmax=30)
        ax.set_title(f'{direction}° (mean={duration_grid.mean():.1f})')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, label='Frames')
        
        # Add direction arrow
        center = 150
        arrow_len = 40
        angle_rad = np.radians(90 - direction)
        dx = arrow_len * np.cos(angle_rad)
        dy = -arrow_len * np.sin(angle_rad)
        ax.annotate('', xy=(center + dx, center + dy), xytext=(center, center),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    plt.tight_layout()
    
    output_path = output_dir / "heatmap_all_directions_duration.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_detailed_diagonal(on_off_dict, output_dir=OUTPUT_DIR):
    """Detailed comparison of diagonal directions."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle('Diagonal Directions - Detailed ON/OFF/Duration Heatmaps', 
                 fontsize=14, fontweight='bold')
    
    diagonal_dirs = [45, 135, 225, 315]
    
    for i, direction in enumerate(diagonal_dirs):
        trial_idx = DIRECTION_LIST.index(direction)
        on_grid, off_grid, duration_grid = create_grids(on_off_dict, trial_idx)
        
        # ON
        ax = axes[i, 0]
        im = ax.imshow(on_grid, cmap='viridis', origin='upper')
        ax.set_title(f'{direction}° - ON Frame')
        ax.set_ylabel(f'Direction {direction}°')
        plt.colorbar(im, ax=ax)
        
        # OFF
        ax = axes[i, 1]
        im = ax.imshow(off_grid, cmap='viridis', origin='upper')
        ax.set_title(f'{direction}° - OFF Frame')
        plt.colorbar(im, ax=ax)
        
        # Duration
        ax = axes[i, 2]
        im = ax.imshow(duration_grid, cmap='RdYlGn', origin='upper', vmin=15, vmax=25)
        ax.set_title(f'{direction}° - Duration (mean={duration_grid.mean():.1f})')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    output_path = output_dir / "heatmap_diagonal_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Generating ON/OFF Frame Heatmaps")
    print("=" * 70)
    
    print(f"\nLoading: {AREA_BASED_PATH}")
    on_off_dict = load_dict(AREA_BASED_PATH)
    print(f"Loaded {len(on_off_dict)} pixels")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    # All directions - ON frames
    plot_all_directions_on(on_off_dict)
    
    # All directions - Duration
    plot_all_directions_duration(on_off_dict)
    
    # Detailed diagonal
    plot_detailed_diagonal(on_off_dict)
    
    # Individual direction plots for 45° (the previously problematic direction)
    plot_single_direction(on_off_dict, 45, rep=0)
    
    print("\n" + "=" * 70)
    print("All heatmaps generated!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

