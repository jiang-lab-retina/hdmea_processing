#!/usr/bin/env python
"""
eimage_sta Validation Plots.

Generates visualizations of eimage_sta (spike-triggered average) data:
- STA spatial heatmap at peak time
- STA temporal profile at peak pixel
- Multi-frame STA montage

Usage:
    python plot_eimage_sta.py <hdf5_path> [--output-dir <path>] [--unit <unit_id>]
    
Example:
    python plot_eimage_sta.py ../test_output/2024.08.08-10.40.20-Rec_final.h5
    python plot_eimage_sta.py ../test_output/file.h5 --unit unit_001
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Add src to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# =============================================================================
# Styling
# =============================================================================

plt.style.use("dark_background")

# Custom colormap for STA (blue-white-red diverging)
STA_CMAP = LinearSegmentedColormap.from_list(
    "sta_diverging",
    ["#0d47a1", "#1976d2", "#64b5f6", "#ffffff", "#ef5350", "#c62828", "#7f0000"],
)

BG_COLOR = "#1a1a2e"
PANEL_COLOR = "#16213e"


# =============================================================================
# Data Loading
# =============================================================================

def load_eimage_sta(hdf5_path: Path, unit_id: str) -> Optional[np.ndarray]:
    """Load eimage_sta data for a unit."""
    with h5py.File(hdf5_path, "r") as f:
        sta_path = f"units/{unit_id}/features/eimage_sta/data"
        if sta_path not in f:
            return None
        return f[sta_path][()]


def get_unit_ids_with_sta(hdf5_path: Path) -> list:
    """Get list of unit IDs that have eimage_sta data."""
    unit_ids = []
    with h5py.File(hdf5_path, "r") as f:
        if "units" not in f:
            return []
        for unit_id in f["units"]:
            sta_path = f"units/{unit_id}/features/eimage_sta/data"
            if sta_path in f:
                unit_ids.append(unit_id)
    return sorted(unit_ids)


# =============================================================================
# Plotting Functions
# =============================================================================

def find_peak_location(sta: np.ndarray) -> Tuple[int, int, int]:
    """Find peak location (t, row, col) in STA."""
    # Find absolute maximum
    abs_sta = np.abs(sta)
    peak_idx = np.unravel_index(np.nanargmax(abs_sta), sta.shape)
    return peak_idx  # (t, row, col)


def plot_sta_spatial(ax, sta: np.ndarray, peak_t: int, title: str = "STA Spatial"):
    """Plot spatial heatmap at peak time."""
    frame = sta[peak_t]
    
    # Symmetric colormap limits
    vmax = np.nanmax(np.abs(frame))
    
    im = ax.imshow(frame, cmap=STA_CMAP, vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_title(f"{title} (t={peak_t})", fontsize=10)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return im


def plot_sta_temporal(ax, sta: np.ndarray, peak_row: int, peak_col: int):
    """Plot temporal profile at peak pixel."""
    temporal = sta[:, peak_row, peak_col]
    t = np.arange(len(temporal))
    
    ax.plot(t, temporal, color="#64b5f6", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    
    # Mark peak
    peak_t = np.nanargmax(np.abs(temporal))
    ax.axvline(peak_t, color="#ef5350", linestyle="--", alpha=0.7, label=f"Peak t={peak_t}")
    
    ax.set_xlabel("Frame")
    ax.set_ylabel("STA Value")
    ax.set_title(f"Temporal Profile at ({peak_row}, {peak_col})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)


def plot_sta_montage(ax, sta: np.ndarray, n_frames: int = 9):
    """Plot montage of STA frames."""
    n_total = sta.shape[0]
    
    # Select frames evenly spaced
    if n_total <= n_frames:
        frame_indices = list(range(n_total))
    else:
        frame_indices = np.linspace(0, n_total - 1, n_frames, dtype=int)
    
    # Determine grid size
    n_cols = int(np.ceil(np.sqrt(len(frame_indices))))
    n_rows = int(np.ceil(len(frame_indices) / n_cols))
    
    # Global color limits
    vmax = np.nanmax(np.abs(sta))
    
    # Create montage
    montage_rows = []
    for r in range(n_rows):
        row_frames = []
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx < len(frame_indices):
                row_frames.append(sta[frame_indices[idx]])
            else:
                row_frames.append(np.zeros_like(sta[0]))
        montage_rows.append(np.hstack(row_frames))
    montage = np.vstack(montage_rows)
    
    im = ax.imshow(montage, cmap=STA_CMAP, vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_title(f"STA Montage ({len(frame_indices)} frames)", fontsize=10)
    ax.axis("off")
    
    # Add frame labels
    h, w = sta.shape[1], sta.shape[2]
    for i, fi in enumerate(frame_indices):
        r, c = i // n_cols, i % n_cols
        x = c * w + w // 2
        y = r * h + 5
        ax.text(x, y, f"t={fi}", fontsize=7, ha="center", color="white")
    
    return im


def plot_unit_sta(sta: np.ndarray, unit_id: str, output_dir: Path):
    """Generate complete STA validation plot for one unit."""
    fig = plt.figure(figsize=(14, 5), facecolor=BG_COLOR)
    
    # Find peak
    peak_t, peak_row, peak_col = find_peak_location(sta)
    
    # Create subplots
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(PANEL_COLOR)
    
    # Plot spatial
    plot_sta_spatial(ax1, sta, peak_t, f"{unit_id} STA Spatial")
    
    # Mark peak location on spatial plot
    ax1.scatter([peak_col], [peak_row], marker="+", s=100, c="lime", linewidths=2)
    
    # Plot temporal
    plot_sta_temporal(ax2, sta, peak_row, peak_col)
    
    # Plot montage
    plot_sta_montage(ax3, sta)
    
    # Title
    fig.suptitle(f"{unit_id} - eimage_sta Validation", fontsize=12, fontweight="bold")
    
    # Save
    output_path = output_dir / f"{unit_id}_eimage_sta.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def run_validation(
    hdf5_path: Path,
    output_dir: Optional[Path] = None,
    unit_id: Optional[str] = None,
):
    """Run eimage_sta validation plots."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / hdf5_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get units to process
    if unit_id:
        unit_ids = [unit_id]
    else:
        unit_ids = get_unit_ids_with_sta(hdf5_path)
    
    print(f"Processing {len(unit_ids)} units for eimage_sta validation...")
    
    success = 0
    for uid in tqdm(unit_ids, desc="eimage_sta plots"):
        sta = load_eimage_sta(hdf5_path, uid)
        if sta is None:
            continue
        
        try:
            plot_unit_sta(sta, uid, output_dir)
            success += 1
        except Exception as e:
            print(f"  Error for {uid}: {e}")
    
    print(f"Generated {success}/{len(unit_ids)} plots in {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="eimage_sta validation plots")
    parser.add_argument("hdf5_path", type=Path, help="Path to HDF5 file")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--unit", type=str, help="Single unit ID to process")
    
    args = parser.parse_args()
    
    if not args.hdf5_path.exists():
        print(f"Error: File not found: {args.hdf5_path}")
        return 1
    
    run_validation(args.hdf5_path, args.output_dir, args.unit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

