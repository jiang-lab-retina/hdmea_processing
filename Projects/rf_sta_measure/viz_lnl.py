"""
LNL (Linear-Nonlinear) Fitting Visualization

Visualizes LNL model fitting results for validation:
1. Individual unit nonlinearity curves (histogram + parametric fit)
2. Population summary of fit quality metrics
3. Parameter distributions

Author: Generated for experimental analysis
Date: 2024-12
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Default paths
DEFAULT_HDF5_PATH = Path(__file__).parent / "export" / "2024.09.18-12.17.43-Rec.h5"
STA_FEATURE_NAME = "sta_perfect_dense_noise_15x15_15hz_r42_3min"


def load_lnl_data(hdf5_path: Path) -> Dict[str, Dict]:
    """
    Load LNL fitting data from HDF5 file.
    
    Returns:
        Dict mapping unit_id -> LNL data dict
    """
    lnl_data = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'units' not in f:
            raise ValueError("No 'units' group in HDF5 file")
        
        for unit_id in f['units'].keys():
            lnl_path = f"units/{unit_id}/features/{STA_FEATURE_NAME}/sta_geometry/lnl"
            
            if lnl_path not in f:
                continue
            
            lnl_group = f[lnl_path]
            lnl_data[unit_id] = {
                'a': float(lnl_group['a'][()]),
                'b': float(lnl_group['b'][()]),
                'a_norm': float(lnl_group['a_norm'][()]) if 'a_norm' in lnl_group else 0.0,
                'bits_per_spike': float(lnl_group['bits_per_spike'][()]) if 'bits_per_spike' in lnl_group else 0.0,
                'r_squared': float(lnl_group['r_squared'][()]),
                'rectification_index': float(lnl_group['rectification_index'][()]) if 'rectification_index' in lnl_group else 0.0,
                'nonlinearity_index': float(lnl_group['nonlinearity_index'][()]) if 'nonlinearity_index' in lnl_group else 0.0,
                'threshold_g': float(lnl_group['threshold_g'][()]) if 'threshold_g' in lnl_group else 0.0,
                'log_likelihood': float(lnl_group['log_likelihood'][()]),
                'null_log_likelihood': float(lnl_group['null_log_likelihood'][()]),
                'n_frames': int(lnl_group['n_frames'][()]),
                'n_spikes': int(lnl_group['n_spikes'][()]),
                'g_bin_centers': np.array(lnl_group['g_bin_centers']),
                'rate_vs_g': np.array(lnl_group['rate_vs_g']),
            }
    
    return lnl_data


def plot_unit_nonlinearity(
    lnl: Dict,
    unit_id: str,
    ax: Optional[plt.Axes] = None,
    show_parametric: bool = True,
    percentile_threshold: float = 10.0,
) -> plt.Axes:
    """
    Plot the nonlinearity curve for a single unit.
    
    Shows normalized generator signal (z-scored) vs normalized firing rate (relative to mean).
    Excludes bins in the lower/upper percentile tails where data is sparse.
    
    Args:
        lnl: LNL data dictionary
        unit_id: Unit identifier for title
        ax: Optional matplotlib axes
        show_parametric: Whether to show parametric fit
        percentile_threshold: Exclude data outside this percentile range (default: 10%)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    g_raw = lnl['g_bin_centers']
    rate_raw = lnl['rate_vs_g']
    a_norm = lnl.get('a_norm', lnl['a'])
    bits = lnl.get('bits_per_spike', 0)
    
    if len(g_raw) == 0 or len(rate_raw) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Filter out NaN and zero rates (often from bins with no data)
    valid_mask = np.isfinite(rate_raw) & (rate_raw > 0)
    
    # Exclude bins outside the middle 80% of g range (sparse data at tails)
    g_low = np.percentile(g_raw, percentile_threshold)
    g_high = np.percentile(g_raw, 100 - percentile_threshold)
    range_mask = (g_raw >= g_low) & (g_raw <= g_high)
    
    # Combine masks
    mask = valid_mask & range_mask
    
    if mask.sum() < 3:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return ax
    
    g = g_raw[mask]
    rate = rate_raw[mask]
    
    # Normalize generator signal (z-score)
    g_mean = g.mean()
    g_std = g.std()
    if g_std > 0:
        g_norm = (g - g_mean) / g_std
    else:
        g_norm = g - g_mean
    
    # Normalize firing rate (relative to mean rate)
    mean_rate = rate.mean()
    if mean_rate > 0:
        rate_norm = rate / mean_rate
    else:
        rate_norm = rate
    
    # Plot normalized histogram nonlinearity
    ax.plot(g_norm, rate_norm, 'b.-', linewidth=1.5, markersize=4, label='Nonlinearity')
    
    # Add reference line at mean (rate_norm = 1)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Mean rate')
    ax.axvline(0.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Generator Signal (z-scored)')
    ax.set_ylabel('Firing Rate (rel. to mean)')
    
    # Get nonlinearity metrics
    rect_idx = lnl.get('rectification_index', 0)
    nl_idx = lnl.get('nonlinearity_index', 0)
    
    ax.set_title(f'{unit_id}\nbits={bits:.2f}, NL={nl_idx:.2f}, RI={rect_idx:+.2f}')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add spike count annotation
    ax.annotate(f'n={lnl["n_spikes"]}', xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=8, color='gray')
    
    return ax


def plot_population_summary(lnl_data: Dict[str, Dict], save_path: Optional[Path] = None):
    """
    Plot population summary of LNL fitting quality.
    
    Shows:
    - Distribution of deviance explained
    - Distribution of R²
    - Parameter distributions (a, b)
    - Spike count vs fit quality
    """
    # Extract arrays
    unit_ids = list(lnl_data.keys())
    bits_per_spike = np.array([lnl_data[u].get('bits_per_spike', 0) for u in unit_ids])
    r_squared = np.array([lnl_data[u]['r_squared'] for u in unit_ids])
    a_norm_vals = np.array([lnl_data[u].get('a_norm', 0) for u in unit_ids])
    b_vals = np.array([lnl_data[u]['b'] for u in unit_ids])
    n_spikes = np.array([lnl_data[u]['n_spikes'] for u in unit_ids])
    ll = np.array([lnl_data[u]['log_likelihood'] for u in unit_ids])
    null_ll = np.array([lnl_data[u]['null_log_likelihood'] for u in unit_ids])
    rect_idx = np.array([lnl_data[u].get('rectification_index', 0) for u in unit_ids])
    nl_idx = np.array([lnl_data[u].get('nonlinearity_index', 0) for u in unit_ids])
    
    # Compute log-likelihood improvement
    ll_improvement = ll - null_ll
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Bits per Spike distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(bits_per_spike, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No improvement')
    ax1.axvline(np.median(bits_per_spike), color='orange', linestyle='-', linewidth=2, 
                label=f'Median: {np.median(bits_per_spike):.3f}')
    ax1.set_xlabel('Bits per Spike')
    ax1.set_ylabel('Count')
    ax1.set_title('Information Gain (bits/spike)')
    ax1.legend(fontsize=8)
    
    # 2. R² distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(r_squared, bins=30, color='seagreen', edgecolor='white', alpha=0.8)
    ax2.axvline(np.median(r_squared), color='orange', linestyle='-', linewidth=2,
                label=f'Median: {np.median(r_squared):.3f}')
    ax2.set_xlabel('R² (pred vs obs)')
    ax2.set_ylabel('Count')
    ax2.set_title('R² Distribution')
    ax2.legend(fontsize=8)
    
    # 3. Log-likelihood improvement distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(ll_improvement, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax3.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax3.axvline(np.median(ll_improvement), color='orange', linestyle='-', linewidth=2,
                label=f'Median: {np.median(ll_improvement):.1f}')
    ax3.set_xlabel('LL - Null LL')
    ax3.set_ylabel('Count')
    ax3.set_title('Log-Likelihood Improvement')
    ax3.legend(fontsize=8)
    
    # 4. Parameter 'a_norm' (normalized gain) distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(a_norm_vals, bins=30, color='mediumpurple', edgecolor='white', alpha=0.8)
    ax4.axvline(0, color='red', linestyle='--', linewidth=1.5, label='a=0 (no modulation)')
    ax4.axvline(np.median(a_norm_vals), color='orange', linestyle='-', linewidth=2,
                label=f'Median: {np.median(a_norm_vals):.3f}')
    ax4.set_xlabel('a_norm (effect per std of g)')
    ax4.set_ylabel('Count')
    ax4.set_title('Normalized Gain Parameter')
    ax4.legend(fontsize=8)
    
    # 5. Parameter 'b' (baseline) distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(b_vals, bins=30, color='goldenrod', edgecolor='white', alpha=0.8)
    ax5.axvline(np.median(b_vals), color='orange', linestyle='-', linewidth=2,
                label=f'Median: {np.median(b_vals):.2f}')
    ax5.set_xlabel('Parameter b (baseline)')
    ax5.set_ylabel('Count')
    ax5.set_title('LNP Baseline Parameter')
    ax5.legend(fontsize=8)
    
    # 6. Spike count distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(n_spikes, bins=30, color='teal', edgecolor='white', alpha=0.8)
    ax6.axvline(np.median(n_spikes), color='orange', linestyle='-', linewidth=2,
                label=f'Median: {np.median(n_spikes):.0f}')
    ax6.set_xlabel('Number of Spikes')
    ax6.set_ylabel('Count')
    ax6.set_title('Spike Count Distribution')
    ax6.legend(fontsize=8)
    
    # 7. Rectification Index distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(rect_idx, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax7.axvline(0, color='gray', linestyle='--', linewidth=1.5, label='Symmetric')
    ax7.axvline(np.median(rect_idx), color='orange', linestyle='-', linewidth=2,
                label=f'Median: {np.median(rect_idx):.3f}')
    ax7.set_xlabel('Rectification Index')
    ax7.set_ylabel('Count')
    ax7.set_title('ON/OFF Asymmetry (-1=OFF, +1=ON)')
    ax7.legend(fontsize=8)
    
    # 8. Nonlinearity Index distribution
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(nl_idx, bins=30, color='teal', edgecolor='white', alpha=0.8)
    ax8.axvline(np.median(nl_idx), color='orange', linestyle='-', linewidth=2,
                label=f'Median: {np.median(nl_idx):.3f}')
    ax8.set_xlabel('Nonlinearity Index')
    ax8.set_ylabel('Count')
    ax8.set_title('Curvature (0=linear, 1=highly curved)')
    ax8.legend(fontsize=8)
    
    # 9. Summary statistics text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Compute summary stats
    n_positive_bits = np.sum(bits_per_spike > 0)
    n_good_fit = np.sum(bits_per_spike > 0.5)
    n_on_cells = np.sum(rect_idx > 0.2)
    n_off_cells = np.sum(rect_idx < -0.2)
    
    summary_text = f"""
    LNL Fitting Summary
    ───────────────────
    Total units: {len(unit_ids)}
    
    Bits per Spike:
      Median: {np.median(bits_per_spike):.3f}
      > 0.5 (good): {n_good_fit} ({100*n_good_fit/len(unit_ids):.1f}%)
    
    Nonlinearity Index:
      Median: {np.median(nl_idx):.3f}
      Range: [{nl_idx.min():.2f}, {nl_idx.max():.2f}]
    
    Rectification:
      ON-like (RI>0.2): {n_on_cells}
      OFF-like (RI<-0.2): {n_off_cells}
      Symmetric: {len(unit_ids)-n_on_cells-n_off_cells}
    
    Spike counts:
      Median: {np.median(n_spikes):.0f}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('LNL Model Fitting - Population Summary', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved population summary to: {save_path}")
    
    return fig


def plot_example_units(
    lnl_data: Dict[str, Dict],
    n_examples: int = 12,
    sort_by: str = 'deviance_explained',
    save_path: Optional[Path] = None,
):
    """
    Plot nonlinearity curves for example units.
    
    Args:
        lnl_data: Dict of LNL data per unit
        n_examples: Number of examples to show
        sort_by: Sort criterion ('deviance_explained', 'r_squared', 'n_spikes', 'random')
        save_path: Optional path to save figure
    """
    unit_ids = list(lnl_data.keys())
    
    # Sort units
    if sort_by == 'deviance_explained' or sort_by == 'bits_per_spike':
        # Best fits first (by bits per spike)
        unit_ids = sorted(unit_ids, key=lambda u: lnl_data[u].get('bits_per_spike', 0), reverse=True)
    elif sort_by == 'r_squared':
        unit_ids = sorted(unit_ids, key=lambda u: lnl_data[u]['r_squared'], reverse=True)
    elif sort_by == 'n_spikes':
        unit_ids = sorted(unit_ids, key=lambda u: lnl_data[u]['n_spikes'], reverse=True)
    elif sort_by == 'random':
        np.random.shuffle(unit_ids)
    
    # Select examples
    selected = unit_ids[:n_examples]
    
    # Create subplot grid
    n_cols = 4
    n_rows = int(np.ceil(n_examples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten()
    
    for i, unit_id in enumerate(selected):
        plot_unit_nonlinearity(lnl_data[unit_id], unit_id, ax=axes[i])
    
    # Hide unused axes
    for i in range(len(selected), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'LNL Nonlinearity Curves (sorted by {sort_by})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved example units to: {save_path}")
    
    return fig


def plot_worst_and_best(
    lnl_data: Dict[str, Dict],
    n_each: int = 6,
    save_path: Optional[Path] = None,
):
    """
    Plot comparison of best and worst fits.
    """
    unit_ids = list(lnl_data.keys())
    
    # Sort by bits per spike
    sorted_ids = sorted(unit_ids, key=lambda u: lnl_data[u].get('bits_per_spike', 0), reverse=True)
    
    best_units = sorted_ids[:n_each]
    worst_units = sorted_ids[-n_each:]
    
    fig, axes = plt.subplots(2, n_each, figsize=(4 * n_each, 7))
    
    # Best fits (top row)
    for i, unit_id in enumerate(best_units):
        ax = axes[0, i]
        plot_unit_nonlinearity(lnl_data[unit_id], unit_id, ax=ax)
        if i == 0:
            ax.set_ylabel('BEST FITS\nFiring Rate (Hz)')
    
    # Worst fits (bottom row)
    for i, unit_id in enumerate(worst_units):
        ax = axes[1, i]
        plot_unit_nonlinearity(lnl_data[unit_id], unit_id, ax=ax)
        if i == 0:
            ax.set_ylabel('WORST FITS\nFiring Rate (Hz)')
    
    plt.suptitle('LNL Model: Best vs Worst Fits', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved best/worst comparison to: {save_path}")
    
    return fig


def main():
    """Main visualization workflow."""
    print("=" * 70)
    print("LNL Fitting Validation Visualization")
    print("=" * 70)
    
    hdf5_path = DEFAULT_HDF5_PATH
    print(f"Loading from: {hdf5_path}")
    
    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found: {hdf5_path}")
        return
    
    # Load data
    lnl_data = load_lnl_data(hdf5_path)
    print(f"Loaded LNL data for {len(lnl_data)} units")
    
    # Create output directory
    output_dir = Path(__file__).parent / "viz_output"
    output_dir.mkdir(exist_ok=True)
    
    # 1. Population summary
    print("\nGenerating population summary...")
    plot_population_summary(lnl_data, save_path=output_dir / "lnl_population_summary.png")
    
    # 2. Best fits examples
    print("Generating best fits examples...")
    plot_example_units(lnl_data, n_examples=12, sort_by='deviance_explained',
                       save_path=output_dir / "lnl_best_fits.png")
    
    # 3. Best vs worst comparison
    print("Generating best vs worst comparison...")
    plot_worst_and_best(lnl_data, n_each=6, save_path=output_dir / "lnl_best_vs_worst.png")
    
    # 4. Random examples (for unbiased view)
    print("Generating random examples...")
    plot_example_units(lnl_data, n_examples=12, sort_by='random',
                       save_path=output_dir / "lnl_random_examples.png")
    
    print(f"\n{'=' * 70}")
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 70)
    
    # Close all figures (don't block with plt.show())
    plt.close('all')


if __name__ == "__main__":
    main()

