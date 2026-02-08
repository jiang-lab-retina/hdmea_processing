"""
Visualization of Before/After Alignment Comparison

Compares ON response of ON cells, OFF response of OFF cells,
and both responses for ON_OFF cells between before and after recordings.

Uses analysis windows for step_up_5s_5i_b0_3x stimulus:
- ON window: 1-3 seconds after step onset (sustained ON response)
- OFF window: 6-8 seconds (1-3s after step offset at 5s mark)
- sustained_window_ms: [500, 5000] - sustained response during light

With 50ms binning:
- ON window: bins 20-60 (1000-3000ms from trial start)
- OFF window: bins 120-160 (6000-8000ms, i.e., 1-3s after step offset)
- Baseline: bins 0-2 (pre-stimulus for comparison)
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Analysis windows (in bins, 50ms per bin)
BIN_SIZE_MS = 50
ON_WINDOW_BINS = (20, 60)       # 1000-3000ms (1-3s after step onset)
OFF_WINDOW_BINS = (120, 160)    # 6000-8000ms (6-8s, i.e., 1-3s after step offset at 5s)
SUSTAINED_WINDOW_BINS = (10, 100)  # 500-5000ms sustained

# Output directory
try:
    from specific_config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "output"

ALIGNED_DIR = OUTPUT_DIR / "aligned"
FIGURES_DIR = OUTPUT_DIR / "figures"

# =============================================================================
# Helper Functions
# =============================================================================

def safe_stack_traces(before_traces: list, after_traces: list, n_cells: int):
    """Safely stack before/after traces, handling variable lengths.
    
    Returns:
        Tuple of (before_arr, after_arr, n_cells) or (None, None, 0) if no valid pairs
    """
    from collections import Counter
    
    # Pair up traces and filter for consistent length
    paired_traces = []
    for i in range(min(n_cells, len(before_traces), len(after_traces))):
        b = before_traces[i]
        a = after_traces[i]
        if len(b) > 0 and len(a) > 0:
            # Use minimum length to truncate both to same size
            min_len = min(len(b), len(a))
            paired_traces.append((b[:min_len], a[:min_len]))
    
    if not paired_traces:
        return None, None, 0
    
    # Find most common length and filter
    lengths = [len(p[0]) for p in paired_traces]
    target_len = Counter(lengths).most_common(1)[0][0]
    paired_traces = [p for p in paired_traces if len(p[0]) == target_len]
    
    if not paired_traces:
        return None, None, 0
    
    n_cells = len(paired_traces)
    before_arr = np.array([p[0] for p in paired_traces])
    after_arr = np.array([p[1] for p in paired_traces])
    
    return before_arr, after_arr, n_cells


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CellResponse:
    """Response data for a single cell."""
    unit_id: str
    cell_type: str
    before_on_response: float
    after_on_response: float
    before_off_response: float
    after_off_response: float
    before_sustained: float = 0.0
    after_sustained: float = 0.0


@dataclass
class AlignmentStats:
    """Statistics for an alignment pair."""
    pair_name: str
    chip: str
    n_pairs: int
    n_on: int
    n_off: int
    n_on_off: int
    n_unknown: int
    responses: List[CellResponse] = field(default_factory=list)
    before_genotype: str = "unknown"
    after_genotype: str = "unknown"
    genotype_consistent: bool = True


# =============================================================================
# Response Calculation
# =============================================================================

def calculate_response_amplitude(
    step_response_trials: np.ndarray,
    window_bins: Tuple[int, int],
    baseline_bins: Tuple[int, int] = (0, 2),
) -> float:
    """
    Calculate response amplitude from step response trials.
    
    Args:
        step_response_trials: 2D array (n_trials, n_bins) of firing rates
        window_bins: (start, end) bins for response window
        baseline_bins: (start, end) bins for baseline
    
    Returns:
        Mean response amplitude (response - baseline)
    """
    if step_response_trials is None or len(step_response_trials) == 0:
        return 0.0
    
    n_bins = step_response_trials.shape[1]
    
    # Clip windows to valid range
    resp_start = max(0, window_bins[0])
    resp_end = min(n_bins, window_bins[1])
    base_start = max(0, baseline_bins[0])
    base_end = min(n_bins, baseline_bins[1])
    
    if resp_end <= resp_start or base_end <= base_start:
        return 0.0
    
    # Calculate mean response and baseline across all trials
    mean_response = np.mean(step_response_trials[:, resp_start:resp_end])
    mean_baseline = np.mean(step_response_trials[:, base_start:base_end])
    
    return mean_response - mean_baseline


def calculate_mean_firing_rate(
    step_response_trials: np.ndarray,
    window_bins: Tuple[int, int],
) -> float:
    """
    Calculate mean firing rate in a window.
    
    Args:
        step_response_trials: 2D array (n_trials, n_bins) of firing rates
        window_bins: (start, end) bins for window
    
    Returns:
        Mean firing rate in the window
    """
    if step_response_trials is None or len(step_response_trials) == 0:
        return 0.0
    
    n_bins = step_response_trials.shape[1]
    start = max(0, window_bins[0])
    end = min(n_bins, window_bins[1])
    
    if end <= start:
        return 0.0
    
    return np.mean(step_response_trials[:, start:end])


# =============================================================================
# Data Loading
# =============================================================================

def get_genotype_from_h5(h5_path: Path) -> str:
    """
    Extract genotype from source H5 file.
    
    Args:
        h5_path: Path to source H5 file
    
    Returns:
        Genotype string or 'unknown'
    """
    if not h5_path.exists():
        return "unknown"
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'metadata/gsheet_row/Genotype' in f:
                genotype = f['metadata/gsheet_row/Genotype'][()]
                if isinstance(genotype, bytes):
                    genotype = genotype.decode('utf-8')
                return genotype
    except Exception:
        pass
    
    return "unknown"


def load_alignment_stats(aligned_h5_path: Path) -> AlignmentStats:
    """
    Load alignment statistics and response data from aligned H5 file.
    
    Args:
        aligned_h5_path: Path to aligned H5 file
    
    Returns:
        AlignmentStats object with response data
    """
    logger = logging.getLogger(__name__)
    
    responses = []
    cell_type_counts = {"ON": 0, "OFF": 0, "ON_OFF": 0, "unknown": 0}
    before_genotype = "unknown"
    after_genotype = "unknown"
    genotype_consistent = True
    
    with h5py.File(aligned_h5_path, 'r') as f:
        # Get metadata
        chip = f.attrs.get('chip', 'unknown')
        n_pairs = f.attrs.get('num_pairs', 0)
        pair_name = aligned_h5_path.stem
        
        # Get source H5 paths and extract genotypes
        before_h5_path = f.attrs.get('before_h5', '')
        after_h5_path = f.attrs.get('after_h5', '')
        
        if isinstance(before_h5_path, bytes):
            before_h5_path = before_h5_path.decode('utf-8')
        if isinstance(after_h5_path, bytes):
            after_h5_path = after_h5_path.decode('utf-8')
        
        if before_h5_path:
            before_genotype = get_genotype_from_h5(Path(before_h5_path))
        if after_h5_path:
            after_genotype = get_genotype_from_h5(Path(after_h5_path))
        
        # Check consistency
        if before_genotype != after_genotype:
            genotype_consistent = False
            logger.warning(f"Genotype mismatch in {pair_name}: before='{before_genotype}', after='{after_genotype}'")
        
        # Process paired units
        if 'paired_units' not in f:
            logger.warning(f"No paired_units in {aligned_h5_path}")
            return AlignmentStats(
                pair_name=pair_name, chip=str(chip), n_pairs=0,
                n_on=0, n_off=0, n_on_off=0, n_unknown=0,
                before_genotype=before_genotype, after_genotype=after_genotype,
                genotype_consistent=genotype_consistent
            )
        
        paired_units = f['paired_units']
        
        for pair_key in paired_units.keys():
            pair_group = paired_units[pair_key]
            
            # Get cell type from before unit (pair level or before unit)
            cell_type = pair_group.attrs.get('cell_type', 'unknown')
            if isinstance(cell_type, bytes):
                cell_type = cell_type.decode('utf-8')
            
            cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
            
            # Get before response
            before_on = 0.0
            before_off = 0.0
            before_sustained = 0.0
            
            if 'before' in pair_group and 'step_response_trials' in pair_group['before']:
                before_trials = pair_group['before']['step_response_trials'][:]
                before_on = calculate_mean_firing_rate(before_trials, ON_WINDOW_BINS)
                before_off = calculate_mean_firing_rate(before_trials, OFF_WINDOW_BINS)
                before_sustained = calculate_mean_firing_rate(before_trials, SUSTAINED_WINDOW_BINS)
            
            # Get after response
            after_on = 0.0
            after_off = 0.0
            after_sustained = 0.0
            
            if 'after' in pair_group and 'step_response_trials' in pair_group['after']:
                after_trials = pair_group['after']['step_response_trials'][:]
                after_on = calculate_mean_firing_rate(after_trials, ON_WINDOW_BINS)
                after_off = calculate_mean_firing_rate(after_trials, OFF_WINDOW_BINS)
                after_sustained = calculate_mean_firing_rate(after_trials, SUSTAINED_WINDOW_BINS)
            
            # Get unit IDs
            before_unit = pair_group.attrs.get('before_unit', 'unknown')
            if isinstance(before_unit, bytes):
                before_unit = before_unit.decode('utf-8')
            
            responses.append(CellResponse(
                unit_id=before_unit,
                cell_type=cell_type,
                before_on_response=before_on,
                after_on_response=after_on,
                before_off_response=before_off,
                after_off_response=after_off,
                before_sustained=before_sustained,
                after_sustained=after_sustained,
            ))
    
    return AlignmentStats(
        pair_name=pair_name,
        chip=str(chip),
        n_pairs=len(responses),
        n_on=cell_type_counts.get('ON', 0),
        n_off=cell_type_counts.get('OFF', 0),
        n_on_off=cell_type_counts.get('ON_OFF', 0),
        n_unknown=cell_type_counts.get('unknown', 0),
        responses=responses,
        before_genotype=before_genotype,
        after_genotype=after_genotype,
        genotype_consistent=genotype_consistent,
    )


# =============================================================================
# Visualization
# =============================================================================

def create_comparison_bar_plots(
    stats: AlignmentStats,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create bar plots comparing before/after responses by cell type.
    
    Args:
        stats: AlignmentStats object with response data
        output_dir: Directory to save figures
    
    Returns:
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate responses by cell type
    on_cells = [r for r in stats.responses if r.cell_type == 'ON']
    off_cells = [r for r in stats.responses if r.cell_type == 'OFF']
    on_off_cells = [r for r in stats.responses if r.cell_type == 'ON_OFF']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Before/After Response Comparison\n{stats.pair_name}\n'
                 f'Chip: {stats.chip} | Pairs: {stats.n_pairs} '
                 f'(ON: {stats.n_on}, OFF: {stats.n_off}, ON_OFF: {stats.n_on_off})',
                 fontsize=12)
    
    bar_width = 0.35
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    
    # --- Plot 1: ON cells - ON response ---
    ax1 = axes[0, 0]
    if on_cells:
        before_on = [r.before_on_response for r in on_cells]
        after_on = [r.after_on_response for r in on_cells]
        
        mean_before = np.mean(before_on)
        mean_after = np.mean(after_on)
        sem_before = np.std(before_on) / np.sqrt(len(before_on)) if len(before_on) > 1 else 0
        sem_after = np.std(after_on) / np.sqrt(len(after_on)) if len(after_on) > 1 else 0
        
        x = np.array([0])
        ax1.bar(x - bar_width/2, [mean_before], bar_width, yerr=[sem_before], 
                label='Before', color=colors['before'], capsize=5)
        ax1.bar(x + bar_width/2, [mean_after], bar_width, yerr=[sem_after], 
                label='After', color=colors['after'], capsize=5)
        
        # Add significance indicator
        if len(on_cells) >= 3:
            from scipy import stats as scipy_stats
            try:
                t_stat, p_val = scipy_stats.ttest_rel(before_on, after_on)
                sig_str = f'p={p_val:.3f}' + (' *' if p_val < 0.05 else '')
                ax1.text(0, max(mean_before, mean_after) * 1.1, sig_str, 
                        ha='center', fontsize=10)
            except:
                pass
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(['ON Response'])
    
    ax1.set_ylabel('Firing Rate (Hz)')
    ax1.set_title(f'ON Cells - ON Response (n={len(on_cells)})')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # --- Plot 2: OFF cells - OFF response ---
    ax2 = axes[0, 1]
    if off_cells:
        before_off = [r.before_off_response for r in off_cells]
        after_off = [r.after_off_response for r in off_cells]
        
        mean_before = np.mean(before_off)
        mean_after = np.mean(after_off)
        sem_before = np.std(before_off) / np.sqrt(len(before_off)) if len(before_off) > 1 else 0
        sem_after = np.std(after_off) / np.sqrt(len(after_off)) if len(after_off) > 1 else 0
        
        x = np.array([0])
        ax2.bar(x - bar_width/2, [mean_before], bar_width, yerr=[sem_before], 
                label='Before', color=colors['before'], capsize=5)
        ax2.bar(x + bar_width/2, [mean_after], bar_width, yerr=[sem_after], 
                label='After', color=colors['after'], capsize=5)
        
        if len(off_cells) >= 3:
            try:
                t_stat, p_val = scipy_stats.ttest_rel(before_off, after_off)
                sig_str = f'p={p_val:.3f}' + (' *' if p_val < 0.05 else '')
                ax2.text(0, max(mean_before, mean_after) * 1.1, sig_str, 
                        ha='center', fontsize=10)
            except:
                pass
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(['OFF Response'])
    
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_title(f'OFF Cells - OFF Response (n={len(off_cells)})')
    ax2.legend()
    ax2.set_ylim(bottom=0)
    
    # --- Plot 3: ON_OFF cells - Both responses ---
    ax3 = axes[1, 0]
    if on_off_cells:
        before_on = [r.before_on_response for r in on_off_cells]
        after_on = [r.after_on_response for r in on_off_cells]
        before_off = [r.before_off_response for r in on_off_cells]
        after_off = [r.after_off_response for r in on_off_cells]
        
        x = np.array([0, 1])
        means_before = [np.mean(before_on), np.mean(before_off)]
        means_after = [np.mean(after_on), np.mean(after_off)]
        sems_before = [np.std(before_on) / np.sqrt(len(before_on)) if len(before_on) > 1 else 0,
                       np.std(before_off) / np.sqrt(len(before_off)) if len(before_off) > 1 else 0]
        sems_after = [np.std(after_on) / np.sqrt(len(after_on)) if len(after_on) > 1 else 0,
                      np.std(after_off) / np.sqrt(len(after_off)) if len(after_off) > 1 else 0]
        
        ax3.bar(x - bar_width/2, means_before, bar_width, yerr=sems_before, 
                label='Before', color=colors['before'], capsize=5)
        ax3.bar(x + bar_width/2, means_after, bar_width, yerr=sems_after, 
                label='After', color=colors['after'], capsize=5)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(['ON Response', 'OFF Response'])
    
    ax3.set_ylabel('Firing Rate (Hz)')
    ax3.set_title(f'ON_OFF Cells - Both Responses (n={len(on_off_cells)})')
    ax3.legend()
    ax3.set_ylim(bottom=0)
    
    # --- Plot 4: Summary - All cell types ---
    ax4 = axes[1, 1]
    
    # Calculate percent change for each cell type
    cell_types = ['ON', 'OFF', 'ON_OFF']
    percent_changes = []
    n_cells = []
    
    for ct in cell_types:
        if ct == 'ON':
            cells = on_cells
            before_vals = [r.before_on_response for r in cells]
            after_vals = [r.after_on_response for r in cells]
        elif ct == 'OFF':
            cells = off_cells
            before_vals = [r.before_off_response for r in cells]
            after_vals = [r.after_off_response for r in cells]
        else:  # ON_OFF - use average of ON and OFF
            cells = on_off_cells
            before_vals = [(r.before_on_response + r.before_off_response) / 2 for r in cells]
            after_vals = [(r.after_on_response + r.after_off_response) / 2 for r in cells]
        
        n_cells.append(len(cells))
        
        if cells and np.mean(before_vals) > 0:
            pct_change = (np.mean(after_vals) - np.mean(before_vals)) / np.mean(before_vals) * 100
            percent_changes.append(pct_change)
        else:
            percent_changes.append(0)
    
    x = np.arange(len(cell_types))
    colors_pct = ['#4CAF50' if p >= 0 else '#f44336' for p in percent_changes]
    
    bars = ax4.bar(x, percent_changes, color=colors_pct)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{ct}\n(n={n})' for ct, n in zip(cell_types, n_cells)])
    ax4.set_ylabel('Percent Change (%)')
    ax4.set_title('Response Change: After vs Before')
    
    # Add value labels on bars
    for bar, val in zip(bars, percent_changes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'{stats.pair_name}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_scatter_plots(
    stats: AlignmentStats,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create scatter plots of before vs after responses.
    
    Args:
        stats: AlignmentStats object with response data
        output_dir: Directory to save figures
    
    Returns:
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate responses by cell type
    on_cells = [r for r in stats.responses if r.cell_type == 'ON']
    off_cells = [r for r in stats.responses if r.cell_type == 'OFF']
    on_off_cells = [r for r in stats.responses if r.cell_type == 'ON_OFF']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Before vs After Response Scatter\n{stats.pair_name}', fontsize=12)
    
    # ON cells - ON response
    ax1 = axes[0]
    if on_cells:
        x = [r.before_on_response for r in on_cells]
        y = [r.after_on_response for r in on_cells]
        ax1.scatter(x, y, alpha=0.6, c='#2196F3', s=50)
        
        # Add unity line
        max_val = max(max(x), max(y)) if x and y else 1
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Unity')
        
        # Add regression line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, max_val, 100)
            ax1.plot(x_line, p(x_line), 'r-', alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax1.set_xlabel('Before (Hz)')
    ax1.set_ylabel('After (Hz)')
    ax1.set_title(f'ON Cells - ON Response (n={len(on_cells)})')
    ax1.legend(fontsize=8)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # OFF cells - OFF response
    ax2 = axes[1]
    if off_cells:
        x = [r.before_off_response for r in off_cells]
        y = [r.after_off_response for r in off_cells]
        ax2.scatter(x, y, alpha=0.6, c='#FF5722', s=50)
        
        max_val = max(max(x), max(y)) if x and y else 1
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Unity')
        
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, max_val, 100)
            ax2.plot(x_line, p(x_line), 'r-', alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax2.set_xlabel('Before (Hz)')
    ax2.set_ylabel('After (Hz)')
    ax2.set_title(f'OFF Cells - OFF Response (n={len(off_cells)})')
    ax2.legend(fontsize=8)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    # ON_OFF cells - Combined
    ax3 = axes[2]
    if on_off_cells:
        # ON response
        x_on = [r.before_on_response for r in on_off_cells]
        y_on = [r.after_on_response for r in on_off_cells]
        ax3.scatter(x_on, y_on, alpha=0.6, c='#4CAF50', s=50, label='ON response')
        
        # OFF response
        x_off = [r.before_off_response for r in on_off_cells]
        y_off = [r.after_off_response for r in on_off_cells]
        ax3.scatter(x_off, y_off, alpha=0.6, c='#9C27B0', s=50, marker='^', label='OFF response')
        
        all_vals = x_on + y_on + x_off + y_off
        max_val = max(all_vals) if all_vals else 1
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Unity')
    
    ax3.set_xlabel('Before (Hz)')
    ax3.set_ylabel('After (Hz)')
    ax3.set_title(f'ON_OFF Cells (n={len(on_off_cells)})')
    ax3.legend(fontsize=8)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{stats.pair_name}_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_mean_trace_plots(
    stats: AlignmentStats,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create mean trace plots with std shading for before/after comparison.
    
    Args:
        stats: AlignmentStats object with response data
        output_dir: Directory to save figures
    
    Returns:
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate responses by cell type
    on_cells = [r for r in stats.responses if r.cell_type == 'ON']
    off_cells = [r for r in stats.responses if r.cell_type == 'OFF']
    on_off_cells = [r for r in stats.responses if r.cell_type == 'ON_OFF']
    
    # We need to reload the actual trace data from the H5 file
    aligned_h5_path = ALIGNED_DIR / f"{stats.pair_name}.h5"
    
    # Collect traces by cell type
    traces_by_type = {
        'ON': {'before': [], 'after': []},
        'OFF': {'before': [], 'after': []},
        'ON_OFF': {'before': [], 'after': []},
    }
    
    with h5py.File(aligned_h5_path, 'r') as f:
        if 'paired_units' not in f:
            return None
        
        paired_units = f['paired_units']
        
        for pair_key in paired_units.keys():
            pair_group = paired_units[pair_key]
            
            # Get cell type
            cell_type = pair_group.attrs.get('cell_type', 'unknown')
            if isinstance(cell_type, bytes):
                cell_type = cell_type.decode('utf-8')
            
            if cell_type not in traces_by_type:
                continue
            
            # Get before trace (mean across trials)
            if 'before' in pair_group and 'step_response_trials' in pair_group['before']:
                before_trials = pair_group['before']['step_response_trials'][:]
                # Mean across trials for this cell
                before_mean = np.mean(before_trials, axis=0)
                traces_by_type[cell_type]['before'].append(before_mean)
            
            # Get after trace
            if 'after' in pair_group and 'step_response_trials' in pair_group['after']:
                after_trials = pair_group['after']['step_response_trials'][:]
                after_mean = np.mean(after_trials, axis=0)
                traces_by_type[cell_type]['after'].append(after_mean)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Mean Traces Before/After Comparison\n{stats.pair_name}', fontsize=12)
    
    # Time axis (50ms bins, 10s total = 200 bins)
    # Assuming step is 5s on, 5s off = 10s total
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    
    cell_types = ['ON', 'OFF', 'ON_OFF']
    titles = ['ON Cells', 'OFF Cells', 'ON_OFF Cells']
    
    for ax, ct, title in zip(axes, cell_types, titles):
        before_traces = traces_by_type[ct]['before']
        after_traces = traces_by_type[ct]['after']
        
        n_cells = min(len(before_traces), len(after_traces))
        
        # Use helper to safely stack traces
        before_arr, after_arr, n_cells = safe_stack_traces(before_traces, after_traces, n_cells)
        
        if n_cells == 0 or before_arr is None:
            ax.text(0.5, 0.5, f'No {ct} cells', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{title} (n=0)')
            continue
        
        n_bins = before_arr.shape[1]
        time_ms = np.arange(n_bins) * BIN_SIZE_MS
        time_s = time_ms / 1000
        
        # Before mean and std
        before_mean = np.mean(before_arr, axis=0)
        before_std = np.std(before_arr, axis=0)
        
        # After mean and std
        after_mean = np.mean(after_arr, axis=0)
        after_std = np.std(after_arr, axis=0)
        
        # Plot before
        ax.plot(time_s, before_mean, color=colors['before'], linewidth=2, label='Before')
        ax.fill_between(time_s, before_mean - before_std, before_mean + before_std,
                       color=colors['before'], alpha=0.3)
        
        # Plot after
        ax.plot(time_s, after_mean, color=colors['after'], linewidth=2, label='After')
        ax.fill_between(time_s, after_mean - after_std, after_mean + after_std,
                       color=colors['after'], alpha=0.3)
        
        # Add vertical lines for stimulus
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Step ON')
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Step OFF')
        
        # Add shaded regions for ON/OFF analysis windows
        ax.axvspan(1, 3, color='green', alpha=0.1)  # ON window (1-3s)
        ax.axvspan(6, 8, color='gray', alpha=0.1)   # OFF window (6-8s)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(f'{title} (n={n_cells})')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, time_s[-1] if len(time_s) > 0 else 10)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{stats.pair_name}_mean_traces.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_combined_mean_traces(
    all_stats: List[AlignmentStats],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create combined mean trace plots across all alignment pairs.
    
    Args:
        all_stats: List of AlignmentStats objects
        output_dir: Directory to save figures
    
    Returns:
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all traces by cell type
    traces_by_type = {
        'ON': {'before': [], 'after': []},
        'OFF': {'before': [], 'after': []},
        'ON_OFF': {'before': [], 'after': []},
    }
    
    for stats in all_stats:
        aligned_h5_path = ALIGNED_DIR / f"{stats.pair_name}.h5"
        
        if not aligned_h5_path.exists():
            continue
        
        with h5py.File(aligned_h5_path, 'r') as f:
            if 'paired_units' not in f:
                continue
            
            paired_units = f['paired_units']
            
            for pair_key in paired_units.keys():
                pair_group = paired_units[pair_key]
                
                cell_type = pair_group.attrs.get('cell_type', 'unknown')
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode('utf-8')
                
                if cell_type not in traces_by_type:
                    continue
                
                # Get before trace
                if 'before' in pair_group and 'step_response_trials' in pair_group['before']:
                    before_trials = pair_group['before']['step_response_trials'][:]
                    before_mean = np.mean(before_trials, axis=0)
                    traces_by_type[cell_type]['before'].append(before_mean)
                
                # Get after trace
                if 'after' in pair_group and 'step_response_trials' in pair_group['after']:
                    after_trials = pair_group['after']['step_response_trials'][:]
                    after_mean = np.mean(after_trials, axis=0)
                    traces_by_type[cell_type]['after'].append(after_mean)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    total_pairs = sum(s.n_pairs for s in all_stats)
    fig.suptitle(f'Combined Mean Traces (All {len(all_stats)} Pairs, {total_pairs} Units)', fontsize=12)
    
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    cell_types = ['ON', 'OFF', 'ON_OFF']
    titles = ['ON Cells', 'OFF Cells', 'ON_OFF Cells']
    
    for ax, ct, title in zip(axes, cell_types, titles):
        before_traces = traces_by_type[ct]['before']
        after_traces = traces_by_type[ct]['after']
        
        n_cells = min(len(before_traces), len(after_traces))
        
        # Use helper to safely stack traces
        before_arr, after_arr, n_cells = safe_stack_traces(before_traces, after_traces, n_cells)
        
        if n_cells == 0 or before_arr is None:
            ax.text(0.5, 0.5, f'No {ct} cells', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{title} (n=0)')
            continue
        
        n_bins = before_arr.shape[1]
        time_ms = np.arange(n_bins) * BIN_SIZE_MS
        time_s = time_ms / 1000
        
        # Before mean and SEM (use SEM for combined data)
        before_mean = np.mean(before_arr, axis=0)
        before_sem = np.std(before_arr, axis=0) / np.sqrt(n_cells)
        
        # After mean and SEM
        after_mean = np.mean(after_arr, axis=0)
        after_sem = np.std(after_arr, axis=0) / np.sqrt(n_cells)
        
        # Plot before
        ax.plot(time_s, before_mean, color=colors['before'], linewidth=2, label='Before')
        ax.fill_between(time_s, before_mean - before_sem, before_mean + before_sem,
                       color=colors['before'], alpha=0.3)
        
        # Plot after
        ax.plot(time_s, after_mean, color=colors['after'], linewidth=2, label='After')
        ax.fill_between(time_s, after_mean - after_sem, after_mean + after_sem,
                       color=colors['after'], alpha=0.3)
        
        # Add vertical lines for stimulus
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Step ON')
        ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Step OFF')
        
        # Add shaded regions for ON/OFF analysis windows
        ax.axvspan(1, 3, color='green', alpha=0.1)  # ON window (1-3s)
        ax.axvspan(6, 8, color='gray', alpha=0.1)   # OFF window (6-8s)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(f'{title} (n={n_cells})')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, time_s[-1] if len(time_s) > 0 else 10)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'combined_mean_traces.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_combined_summary(
    all_stats: List[AlignmentStats],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create a combined summary figure for all alignment pairs.
    
    Args:
        all_stats: List of AlignmentStats objects
        output_dir: Directory to save figures
    
    Returns:
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all responses
    all_on_cells = []
    all_off_cells = []
    all_on_off_cells = []
    
    for stats in all_stats:
        all_on_cells.extend([r for r in stats.responses if r.cell_type == 'ON'])
        all_off_cells.extend([r for r in stats.responses if r.cell_type == 'OFF'])
        all_on_off_cells.extend([r for r in stats.responses if r.cell_type == 'ON_OFF'])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    total_pairs = sum(s.n_pairs for s in all_stats)
    fig.suptitle(f'Combined Before/After Comparison\n'
                 f'Total: {total_pairs} aligned pairs from {len(all_stats)} recordings',
                 fontsize=12)
    
    bar_width = 0.35
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    
    # ON cells
    ax1 = axes[0]
    if all_on_cells:
        before_on = [r.before_on_response for r in all_on_cells]
        after_on = [r.after_on_response for r in all_on_cells]
        
        mean_before = np.mean(before_on)
        mean_after = np.mean(after_on)
        sem_before = np.std(before_on) / np.sqrt(len(before_on))
        sem_after = np.std(after_on) / np.sqrt(len(after_on))
        
        x = np.array([0])
        ax1.bar(x - bar_width/2, [mean_before], bar_width, yerr=[sem_before], 
                label='Before', color=colors['before'], capsize=5)
        ax1.bar(x + bar_width/2, [mean_after], bar_width, yerr=[sem_after], 
                label='After', color=colors['after'], capsize=5)
        
        # Statistical test
        if len(all_on_cells) >= 3:
            from scipy import stats as scipy_stats
            try:
                t_stat, p_val = scipy_stats.ttest_rel(before_on, after_on)
                pct_change = (mean_after - mean_before) / mean_before * 100 if mean_before > 0 else 0
                ax1.set_title(f'ON Cells - ON Response\n(n={len(all_on_cells)}, '
                             f'change: {pct_change:.1f}%, p={p_val:.3f})')
            except:
                ax1.set_title(f'ON Cells - ON Response (n={len(all_on_cells)})')
        else:
            ax1.set_title(f'ON Cells - ON Response (n={len(all_on_cells)})')
    
    ax1.set_ylabel('Firing Rate (Hz)')
    ax1.set_xticks([])
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # OFF cells
    ax2 = axes[1]
    if all_off_cells:
        before_off = [r.before_off_response for r in all_off_cells]
        after_off = [r.after_off_response for r in all_off_cells]
        
        mean_before = np.mean(before_off)
        mean_after = np.mean(after_off)
        sem_before = np.std(before_off) / np.sqrt(len(before_off))
        sem_after = np.std(after_off) / np.sqrt(len(after_off))
        
        x = np.array([0])
        ax2.bar(x - bar_width/2, [mean_before], bar_width, yerr=[sem_before], 
                label='Before', color=colors['before'], capsize=5)
        ax2.bar(x + bar_width/2, [mean_after], bar_width, yerr=[sem_after], 
                label='After', color=colors['after'], capsize=5)
        
        if len(all_off_cells) >= 3:
            try:
                t_stat, p_val = scipy_stats.ttest_rel(before_off, after_off)
                pct_change = (mean_after - mean_before) / mean_before * 100 if mean_before > 0 else 0
                ax2.set_title(f'OFF Cells - OFF Response\n(n={len(all_off_cells)}, '
                             f'change: {pct_change:.1f}%, p={p_val:.3f})')
            except:
                ax2.set_title(f'OFF Cells - OFF Response (n={len(all_off_cells)})')
        else:
            ax2.set_title(f'OFF Cells - OFF Response (n={len(all_off_cells)})')
    
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_xticks([])
    ax2.legend()
    ax2.set_ylim(bottom=0)
    
    # ON_OFF cells
    ax3 = axes[2]
    if all_on_off_cells:
        before_on = [r.before_on_response for r in all_on_off_cells]
        after_on = [r.after_on_response for r in all_on_off_cells]
        before_off = [r.before_off_response for r in all_on_off_cells]
        after_off = [r.after_off_response for r in all_on_off_cells]
        
        x = np.array([0, 1])
        means_before = [np.mean(before_on), np.mean(before_off)]
        means_after = [np.mean(after_on), np.mean(after_off)]
        sems_before = [np.std(before_on) / np.sqrt(len(before_on)),
                       np.std(before_off) / np.sqrt(len(before_off))]
        sems_after = [np.std(after_on) / np.sqrt(len(after_on)),
                      np.std(after_off) / np.sqrt(len(after_off))]
        
        ax3.bar(x - bar_width/2, means_before, bar_width, yerr=sems_before, 
                label='Before', color=colors['before'], capsize=5)
        ax3.bar(x + bar_width/2, means_after, bar_width, yerr=sems_after, 
                label='After', color=colors['after'], capsize=5)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(['ON Response', 'OFF Response'])
    
    ax3.set_ylabel('Firing Rate (Hz)')
    ax3.set_title(f'ON_OFF Cells (n={len(all_on_off_cells)})')
    ax3.legend()
    ax3.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'combined_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# Genotype-Grouped Plots
# =============================================================================

def create_genotype_summary(
    stats_by_genotype: Dict[str, List[AlignmentStats]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create summary bar plots grouped by genotype.
    
    Args:
        stats_by_genotype: Dictionary mapping genotype -> list of AlignmentStats
        output_dir: Directory to save figures
    
    Returns:
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(stats_by_genotype.keys())
    n_genotypes = len(genotypes)
    
    if n_genotypes == 0:
        return None
    
    fig, axes = plt.subplots(n_genotypes, 3, figsize=(15, 5 * n_genotypes))
    if n_genotypes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Before/After Comparison by Genotype', fontsize=14, y=1.02)
    
    bar_width = 0.35
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    
    for row, genotype in enumerate(genotypes):
        stats_list = stats_by_genotype[genotype]
        
        # Combine all responses for this genotype
        all_on_cells = []
        all_off_cells = []
        all_on_off_cells = []
        
        for stats in stats_list:
            all_on_cells.extend([r for r in stats.responses if r.cell_type == 'ON'])
            all_off_cells.extend([r for r in stats.responses if r.cell_type == 'OFF'])
            all_on_off_cells.extend([r for r in stats.responses if r.cell_type == 'ON_OFF'])
        
        total_pairs = sum(s.n_pairs for s in stats_list)
        
        # ON cells
        ax1 = axes[row, 0]
        if all_on_cells:
            before_on = [r.before_on_response for r in all_on_cells]
            after_on = [r.after_on_response for r in all_on_cells]
            
            mean_before = np.mean(before_on)
            mean_after = np.mean(after_on)
            sem_before = np.std(before_on) / np.sqrt(len(before_on))
            sem_after = np.std(after_on) / np.sqrt(len(after_on))
            
            x = np.array([0])
            ax1.bar(x - bar_width/2, [mean_before], bar_width, yerr=[sem_before], 
                    label='Before', color=colors['before'], capsize=5)
            ax1.bar(x + bar_width/2, [mean_after], bar_width, yerr=[sem_after], 
                    label='After', color=colors['after'], capsize=5)
            
            # Statistical test
            if len(all_on_cells) >= 3:
                from scipy import stats as scipy_stats
                try:
                    t_stat, p_val = scipy_stats.ttest_rel(before_on, after_on)
                    pct_change = (mean_after - mean_before) / mean_before * 100 if mean_before > 0 else 0
                    ax1.set_title(f'ON Cells (n={len(all_on_cells)})\n'
                                 f'Change: {pct_change:.1f}%, p={p_val:.3f}')
                except:
                    ax1.set_title(f'ON Cells (n={len(all_on_cells)})')
            else:
                ax1.set_title(f'ON Cells (n={len(all_on_cells)})')
        else:
            ax1.set_title('ON Cells (n=0)')
        
        ax1.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
        ax1.set_xticks([])
        ax1.legend()
        ax1.set_ylim(bottom=0)
        
        # OFF cells
        ax2 = axes[row, 1]
        if all_off_cells:
            before_off = [r.before_off_response for r in all_off_cells]
            after_off = [r.after_off_response for r in all_off_cells]
            
            mean_before = np.mean(before_off)
            mean_after = np.mean(after_off)
            sem_before = np.std(before_off) / np.sqrt(len(before_off))
            sem_after = np.std(after_off) / np.sqrt(len(after_off))
            
            x = np.array([0])
            ax2.bar(x - bar_width/2, [mean_before], bar_width, yerr=[sem_before], 
                    label='Before', color=colors['before'], capsize=5)
            ax2.bar(x + bar_width/2, [mean_after], bar_width, yerr=[sem_after], 
                    label='After', color=colors['after'], capsize=5)
            
            if len(all_off_cells) >= 3:
                try:
                    t_stat, p_val = scipy_stats.ttest_rel(before_off, after_off)
                    pct_change = (mean_after - mean_before) / mean_before * 100 if mean_before > 0 else 0
                    ax2.set_title(f'OFF Cells (n={len(all_off_cells)})\n'
                                 f'Change: {pct_change:.1f}%, p={p_val:.3f}')
                except:
                    ax2.set_title(f'OFF Cells (n={len(all_off_cells)})')
            else:
                ax2.set_title(f'OFF Cells (n={len(all_off_cells)})')
        else:
            ax2.set_title('OFF Cells (n=0)')
        
        ax2.set_xticks([])
        ax2.legend()
        ax2.set_ylim(bottom=0)
        
        # ON_OFF cells
        ax3 = axes[row, 2]
        if all_on_off_cells:
            before_on = [r.before_on_response for r in all_on_off_cells]
            after_on = [r.after_on_response for r in all_on_off_cells]
            before_off = [r.before_off_response for r in all_on_off_cells]
            after_off = [r.after_off_response for r in all_on_off_cells]
            
            x = np.array([0, 1])
            means_before = [np.mean(before_on), np.mean(before_off)]
            means_after = [np.mean(after_on), np.mean(after_off)]
            sems_before = [np.std(before_on) / np.sqrt(len(before_on)),
                           np.std(before_off) / np.sqrt(len(before_off))]
            sems_after = [np.std(after_on) / np.sqrt(len(after_on)),
                          np.std(after_off) / np.sqrt(len(after_off))]
            
            ax3.bar(x - bar_width/2, means_before, bar_width, yerr=sems_before, 
                    label='Before', color=colors['before'], capsize=5)
            ax3.bar(x + bar_width/2, means_after, bar_width, yerr=sems_after, 
                    label='After', color=colors['after'], capsize=5)
            
            ax3.set_xticks(x)
            ax3.set_xticklabels(['ON Resp', 'OFF Resp'])
        
        ax3.set_title(f'ON_OFF Cells (n={len(all_on_off_cells)})\nTotal pairs: {total_pairs}')
        ax3.legend()
        ax3.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'genotype_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def collect_traces_by_genotype(
    stats_by_genotype: Dict[str, List[AlignmentStats]],
) -> Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]:
    """
    Collect all traces organized by genotype and cell type.
    
    Returns:
        Dict[genotype][cell_type]['before'/'after'] -> list of traces
    """
    result = {}
    
    for genotype, stats_list in stats_by_genotype.items():
        traces_by_type = {
            'ON': {'before': [], 'after': []},
            'OFF': {'before': [], 'after': []},
            'ON_OFF': {'before': [], 'after': []},
        }
        
        for stats in stats_list:
            aligned_h5_path = ALIGNED_DIR / f"{stats.pair_name}.h5"
            
            if not aligned_h5_path.exists():
                continue
            
            with h5py.File(aligned_h5_path, 'r') as f:
                if 'paired_units' not in f:
                    continue
                
                paired_units = f['paired_units']
                
                for pair_key in paired_units.keys():
                    pair_group = paired_units[pair_key]
                    
                    cell_type = pair_group.attrs.get('cell_type', 'unknown')
                    if isinstance(cell_type, bytes):
                        cell_type = cell_type.decode('utf-8')
                    
                    if cell_type not in traces_by_type:
                        continue
                    
                    # Get before trace
                    if 'before' in pair_group and 'step_response_trials' in pair_group['before']:
                        before_trials = pair_group['before']['step_response_trials'][:]
                        before_mean = np.mean(before_trials, axis=0)
                        traces_by_type[cell_type]['before'].append(before_mean)
                    
                    # Get after trace
                    if 'after' in pair_group and 'step_response_trials' in pair_group['after']:
                        after_trials = pair_group['after']['step_response_trials'][:]
                        after_mean = np.mean(after_trials, axis=0)
                        traces_by_type[cell_type]['after'].append(after_mean)
        
        result[genotype] = traces_by_type
    
    return result


def create_genotype_mean_traces(
    stats_by_genotype: Dict[str, List[AlignmentStats]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create mean trace plots grouped by genotype.
    
    Args:
        stats_by_genotype: Dictionary mapping genotype -> list of AlignmentStats
        output_dir: Directory to save figures
    
    Returns:
        Path to saved figure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(stats_by_genotype.keys())
    n_genotypes = len(genotypes)
    
    if n_genotypes == 0:
        return None
    
    fig, axes = plt.subplots(n_genotypes, 3, figsize=(18, 5 * n_genotypes))
    if n_genotypes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Mean Traces by Genotype (mean +/- SEM)', fontsize=14, y=1.02)
    
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    for row, genotype in enumerate(genotypes):
        stats_list = stats_by_genotype[genotype]
        
        # Collect all traces by cell type for this genotype
        traces_by_type = {
            'ON': {'before': [], 'after': []},
            'OFF': {'before': [], 'after': []},
            'ON_OFF': {'before': [], 'after': []},
        }
        
        for stats in stats_list:
            aligned_h5_path = ALIGNED_DIR / f"{stats.pair_name}.h5"
            
            if not aligned_h5_path.exists():
                continue
            
            with h5py.File(aligned_h5_path, 'r') as f:
                if 'paired_units' not in f:
                    continue
                
                paired_units = f['paired_units']
                
                for pair_key in paired_units.keys():
                    pair_group = paired_units[pair_key]
                    
                    cell_type = pair_group.attrs.get('cell_type', 'unknown')
                    if isinstance(cell_type, bytes):
                        cell_type = cell_type.decode('utf-8')
                    
                    if cell_type not in traces_by_type:
                        continue
                    
                    # Get before trace
                    if 'before' in pair_group and 'step_response_trials' in pair_group['before']:
                        before_trials = pair_group['before']['step_response_trials'][:]
                        before_mean = np.mean(before_trials, axis=0)
                        traces_by_type[cell_type]['before'].append(before_mean)
                    
                    # Get after trace
                    if 'after' in pair_group and 'step_response_trials' in pair_group['after']:
                        after_trials = pair_group['after']['step_response_trials'][:]
                        after_mean = np.mean(after_trials, axis=0)
                        traces_by_type[cell_type]['after'].append(after_mean)
        
        # Plot each cell type
        for col, ct in enumerate(cell_types):
            ax = axes[row, col]
            
            before_traces = traces_by_type[ct]['before']
            after_traces = traces_by_type[ct]['after']
            
            n_cells = min(len(before_traces), len(after_traces))
            
            if n_cells == 0:
                ax.text(0.5, 0.5, f'No {ct} cells', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{ct} Cells (n=0)')
                if col == 0:
                    ax.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
                continue
            
            # Pair up traces and filter for consistent length
            from collections import Counter
            paired_traces = []
            for i in range(n_cells):
                b = before_traces[i]
                a = after_traces[i]
                if len(b) > 0 and len(a) > 0:
                    # Use minimum length to truncate both to same size
                    min_len = min(len(b), len(a))
                    paired_traces.append((b[:min_len], a[:min_len]))
            
            if not paired_traces:
                ax.text(0.5, 0.5, f'No {ct} cells\n(no valid pairs)', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{ct} Cells (n=0)')
                if col == 0:
                    ax.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
                continue
            
            # Find most common length and filter
            lengths = [len(p[0]) for p in paired_traces]
            target_len = Counter(lengths).most_common(1)[0][0]
            paired_traces = [p for p in paired_traces if len(p[0]) == target_len]
            
            if not paired_traces:
                ax.text(0.5, 0.5, f'No {ct} cells\n(inconsistent lengths)', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{ct} Cells (n=0)')
                if col == 0:
                    ax.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
                continue
            
            n_cells = len(paired_traces)
            
            # Stack traces and compute mean/SEM
            before_arr = np.array([p[0] for p in paired_traces])
            after_arr = np.array([p[1] for p in paired_traces])
            
            n_bins = before_arr.shape[1]
            time_ms = np.arange(n_bins) * BIN_SIZE_MS
            time_s = time_ms / 1000
            
            # Before mean and SEM
            before_mean = np.mean(before_arr, axis=0)
            before_sem = np.std(before_arr, axis=0) / np.sqrt(n_cells)
            
            # After mean and SEM
            after_mean = np.mean(after_arr, axis=0)
            after_sem = np.std(after_arr, axis=0) / np.sqrt(n_cells)
            
            # Plot before
            ax.plot(time_s, before_mean, color=colors['before'], linewidth=2, label='Before')
            ax.fill_between(time_s, before_mean - before_sem, before_mean + before_sem,
                           color=colors['before'], alpha=0.3)
            
            # Plot after
            ax.plot(time_s, after_mean, color=colors['after'], linewidth=2, label='After')
            ax.fill_between(time_s, after_mean - after_sem, after_mean + after_sem,
                           color=colors['after'], alpha=0.3)
            
            # Add vertical lines for stimulus
            ax.axvline(x=0, color='green', linestyle='--', alpha=0.5)
            ax.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
            
            # Add shaded regions for ON/OFF windows
            ax.axvspan(1, 3, color='green', alpha=0.1)
            ax.axvspan(6, 8, color='gray', alpha=0.1)
            
            ax.set_xlabel('Time (s)')
            if col == 0:
                ax.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
            ax.set_title(f'{ct} Cells (n={n_cells})')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim(0, time_s[-1] if len(time_s) > 0 else 10)
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'genotype_mean_traces.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_genotype_overlay_traces(
    stats_by_genotype: Dict[str, List[AlignmentStats]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create overlay plots comparing before/after for each genotype.
    Each genotype in a separate row, with before (solid) and after (dashed) overlaid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    traces_by_genotype = collect_traces_by_genotype(stats_by_genotype)
    genotypes = list(stats_by_genotype.keys())
    n_genotypes = len(genotypes)
    
    if n_genotypes == 0:
        return None
    
    # Colors for before/after
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    
    fig, axes = plt.subplots(n_genotypes, 3, figsize=(18, 5 * n_genotypes))
    if n_genotypes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Before vs After Comparison by Genotype (mean ± SEM)', fontsize=14, y=1.02)
    
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    for row, genotype in enumerate(genotypes):
        traces = traces_by_genotype.get(genotype, {})
        
        for col, ct in enumerate(cell_types):
            ax = axes[row, col]
            
            before_traces = traces.get(ct, {}).get('before', [])
            after_traces = traces.get(ct, {}).get('after', [])
            
            n_cells = min(len(before_traces), len(after_traces))
            
            # Use helper to safely stack traces
            before_arr, after_arr, n_cells = safe_stack_traces(before_traces, after_traces, n_cells)
            
            if n_cells == 0 or before_arr is None:
                ax.text(0.5, 0.5, f'No {ct} cells', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{ct} Cells (n=0)')
                if col == 0:
                    ax.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
                continue
            
            n_bins = before_arr.shape[1]
            time_s = np.arange(n_bins) * BIN_SIZE_MS / 1000
            
            # Before mean and SEM
            mean_before = np.mean(before_arr, axis=0)
            sem_before = np.std(before_arr, axis=0) / np.sqrt(n_cells)
            
            # After mean and SEM
            mean_after = np.mean(after_arr, axis=0)
            sem_after = np.std(after_arr, axis=0) / np.sqrt(n_cells)
            
            # Plot before (solid)
            ax.plot(time_s, mean_before, color=colors['before'], linewidth=2, 
                   linestyle='-', label='Before')
            ax.fill_between(time_s, mean_before - sem_before, mean_before + sem_before,
                           color=colors['before'], alpha=0.25)
            
            # Plot after (dashed)
            ax.plot(time_s, mean_after, color=colors['after'], linewidth=2, 
                   linestyle='--', label='After')
            ax.fill_between(time_s, mean_after - sem_after, mean_after + sem_after,
                           color=colors['after'], alpha=0.25)
            
            # Add stimulus markers
            ax.axvline(x=0, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
            ax.axvline(x=5, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
            ax.axvspan(1, 3, color='green', alpha=0.05)
            ax.axvspan(6, 8, color='gray', alpha=0.05)
            
            ax.set_xlabel('Time (s)')
            if col == 0:
                ax.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
            else:
                ax.set_ylabel('Firing Rate (Hz)')
            ax.set_title(f'{ct} Cells (n={n_cells})')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlim(0, 10)
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'genotype_overlay_traces.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_percent_change_comparison(
    stats_by_genotype: Dict[str, List[AlignmentStats]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create bar plots comparing percent change between genotypes.
    """
    from scipy import stats as scipy_stats
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(stats_by_genotype.keys())
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    # Collect percent changes for each cell
    pct_changes = {gt: {ct: {'on': [], 'off': []} for ct in cell_types} for gt in genotypes}
    
    for genotype, stats_list in stats_by_genotype.items():
        for stats in stats_list:
            for r in stats.responses:
                if r.cell_type not in cell_types:
                    continue
                
                # ON response percent change
                if r.before_on_response > 0:
                    pct_on = (r.after_on_response - r.before_on_response) / r.before_on_response * 100
                    pct_changes[genotype][r.cell_type]['on'].append(pct_on)
                
                # OFF response percent change
                if r.before_off_response > 0:
                    pct_off = (r.after_off_response - r.before_off_response) / r.before_off_response * 100
                    pct_changes[genotype][r.cell_type]['off'].append(pct_off)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Percent Change (After - Before) / Before by Genotype', fontsize=12)
    
    bar_width = 0.35
    genotype_colors = ['#2196F3', '#FF5722', '#4CAF50']
    
    for col, ct in enumerate(cell_types):
        ax = axes[col]
        
        if ct == 'ON_OFF':
            # Show both ON and OFF responses
            x = np.arange(len(genotypes) * 2)
            means = []
            sems = []
            labels = []
            colors = []
            
            for i, gt in enumerate(genotypes):
                on_changes = pct_changes[gt][ct]['on']
                off_changes = pct_changes[gt][ct]['off']
                
                means.append(np.mean(on_changes) if on_changes else 0)
                sems.append(np.std(on_changes) / np.sqrt(len(on_changes)) if len(on_changes) > 1 else 0)
                labels.append(f'{gt}\nON (n={len(on_changes)})')
                colors.append(genotype_colors[i % len(genotype_colors)])
                
                means.append(np.mean(off_changes) if off_changes else 0)
                sems.append(np.std(off_changes) / np.sqrt(len(off_changes)) if len(off_changes) > 1 else 0)
                labels.append(f'{gt}\nOFF (n={len(off_changes)})')
                colors.append(genotype_colors[i % len(genotype_colors)])
            
            bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
        else:
            # Show single response type
            x = np.arange(len(genotypes))
            means = []
            sems = []
            labels = []
            n_cells_list = []
            
            response_key = 'on' if ct == 'ON' else 'off'
            
            for gt in genotypes:
                changes = pct_changes[gt][ct][response_key]
                means.append(np.mean(changes) if changes else 0)
                sems.append(np.std(changes) / np.sqrt(len(changes)) if len(changes) > 1 else 0)
                labels.append(f'{gt}\n(n={len(changes)})')
                n_cells_list.append(len(changes))
            
            bars = ax.bar(x, means, yerr=sems, capsize=5, 
                         color=genotype_colors[:len(genotypes)], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            
            # Add statistical comparison if 2 genotypes
            if len(genotypes) == 2:
                g1_changes = pct_changes[genotypes[0]][ct][response_key]
                g2_changes = pct_changes[genotypes[1]][ct][response_key]
                if len(g1_changes) >= 3 and len(g2_changes) >= 3:
                    try:
                        t_stat, p_val = scipy_stats.ttest_ind(g1_changes, g2_changes)
                        sig_str = f'p={p_val:.3f}' + (' *' if p_val < 0.05 else '')
                        ax.text(0.5, 0.95, sig_str, ha='center', va='top', 
                               transform=ax.transAxes, fontsize=10)
                    except:
                        pass
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Percent Change (%)')
        resp_type = 'ON Response' if ct == 'ON' else ('OFF Response' if ct == 'OFF' else 'ON & OFF')
        ax.set_title(f'{ct} Cells - {resp_type}')
    
    plt.tight_layout()
    
    output_path = output_dir / 'genotype_percent_change.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_response_distribution(
    stats_by_genotype: Dict[str, List[AlignmentStats]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create box plots showing response distribution by genotype.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(stats_by_genotype.keys())
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Response Distribution by Genotype', fontsize=14)
    
    genotype_colors = ['#2196F3', '#FF5722', '#4CAF50']
    
    for col, ct in enumerate(cell_types):
        # Row 1: Before responses
        ax_before = axes[0, col]
        # Row 2: After responses
        ax_after = axes[1, col]
        
        before_data = []
        after_data = []
        labels = []
        
        for gt in genotypes:
            stats_list = stats_by_genotype[gt]
            
            before_vals = []
            after_vals = []
            
            for stats in stats_list:
                for r in stats.responses:
                    if r.cell_type != ct:
                        continue
                    
                    if ct == 'ON':
                        before_vals.append(r.before_on_response)
                        after_vals.append(r.after_on_response)
                    elif ct == 'OFF':
                        before_vals.append(r.before_off_response)
                        after_vals.append(r.after_off_response)
                    else:  # ON_OFF - average of both
                        before_vals.append((r.before_on_response + r.before_off_response) / 2)
                        after_vals.append((r.after_on_response + r.after_off_response) / 2)
            
            before_data.append(before_vals)
            after_data.append(after_vals)
            labels.append(f'{gt}\n(n={len(before_vals)})')
        
        # Before box plot
        if any(len(d) > 0 for d in before_data):
            bp1 = ax_before.boxplot([d for d in before_data if len(d) > 0], 
                                    tick_labels=[l for l, d in zip(labels, before_data) if len(d) > 0],
                                    patch_artist=True)
            for patch, color in zip(bp1['boxes'], genotype_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax_before.set_ylabel('Firing Rate (Hz)')
        ax_before.set_title(f'{ct} Cells - BEFORE')
        ax_before.set_ylim(bottom=0)
        
        # After box plot
        if any(len(d) > 0 for d in after_data):
            bp2 = ax_after.boxplot([d for d in after_data if len(d) > 0], 
                                   tick_labels=[l for l, d in zip(labels, after_data) if len(d) > 0],
                                   patch_artist=True)
            for patch, color in zip(bp2['boxes'], genotype_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax_after.set_ylabel('Firing Rate (Hz)')
        ax_after.set_title(f'{ct} Cells - AFTER')
        ax_after.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'genotype_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_paired_change_plot(
    stats_by_genotype: Dict[str, List[AlignmentStats]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create paired before-after plots showing individual cell changes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(stats_by_genotype.keys())
    cell_types = ['ON', 'OFF', 'ON_OFF']
    n_genotypes = len(genotypes)
    
    fig, axes = plt.subplots(n_genotypes, 3, figsize=(15, 5 * n_genotypes))
    if n_genotypes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Paired Before-After Changes by Cell', fontsize=14, y=1.02)
    
    for row, genotype in enumerate(genotypes):
        stats_list = stats_by_genotype[genotype]
        
        for col, ct in enumerate(cell_types):
            ax = axes[row, col]
            
            before_vals = []
            after_vals = []
            
            for stats in stats_list:
                for r in stats.responses:
                    if r.cell_type != ct:
                        continue
                    
                    if ct == 'ON':
                        before_vals.append(r.before_on_response)
                        after_vals.append(r.after_on_response)
                    elif ct == 'OFF':
                        before_vals.append(r.before_off_response)
                        after_vals.append(r.after_off_response)
                    else:
                        before_vals.append((r.before_on_response + r.before_off_response) / 2)
                        after_vals.append((r.after_on_response + r.after_off_response) / 2)
            
            n_cells = len(before_vals)
            
            if n_cells == 0:
                ax.text(0.5, 0.5, f'No {ct} cells', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{ct} Cells (n=0)')
                continue
            
            # Plot individual cell changes
            for i in range(n_cells):
                color = '#4CAF50' if after_vals[i] >= before_vals[i] else '#f44336'
                ax.plot([0, 1], [before_vals[i], after_vals[i]], 
                       color=color, alpha=0.3, linewidth=0.5)
            
            # Plot mean with error bars
            mean_before = np.mean(before_vals)
            mean_after = np.mean(after_vals)
            sem_before = np.std(before_vals) / np.sqrt(n_cells)
            sem_after = np.std(after_vals) / np.sqrt(n_cells)
            
            ax.errorbar([0], [mean_before], yerr=[sem_before], 
                       fmt='o', markersize=10, color='#2196F3', capsize=5, capthick=2)
            ax.errorbar([1], [mean_after], yerr=[sem_after], 
                       fmt='o', markersize=10, color='#FF5722', capsize=5, capthick=2)
            ax.plot([0, 1], [mean_before, mean_after], 'k-', linewidth=2)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Before', 'After'])
            ax.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)' if col == 0 else 'Firing Rate (Hz)')
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(bottom=0)
            
            # Add stats
            from scipy import stats as scipy_stats
            if n_cells >= 3:
                try:
                    t_stat, p_val = scipy_stats.ttest_rel(before_vals, after_vals)
                    pct_change = (mean_after - mean_before) / mean_before * 100 if mean_before > 0 else 0
                    ax.set_title(f'{ct} Cells (n={n_cells})\n{pct_change:+.1f}%, p={p_val:.3f}')
                except:
                    ax.set_title(f'{ct} Cells (n={n_cells})')
            else:
                ax.set_title(f'{ct} Cells (n={n_cells})')
    
    plt.tight_layout()
    
    output_path = output_dir / 'genotype_paired_changes.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_summary_table(
    stats_by_genotype: Dict[str, List[AlignmentStats]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create a summary statistics table as an image.
    Shows ON and OFF responses separately for ON_OFF cells.
    """
    from scipy import stats as scipy_stats
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(stats_by_genotype.keys())
    
    # Collect summary data
    summary_data = []
    
    for genotype in genotypes:
        stats_list = stats_by_genotype[genotype]
        
        # ON cells - ON response
        before_on, after_on = [], []
        for stats in stats_list:
            for r in stats.responses:
                if r.cell_type == 'ON':
                    before_on.append(r.before_on_response)
                    after_on.append(r.after_on_response)
        
        if len(before_on) > 0:
            n = len(before_on)
            m_b, m_a = np.mean(before_on), np.mean(after_on)
            sem_b, sem_a = np.std(before_on)/np.sqrt(n), np.std(after_on)/np.sqrt(n)
            pct = (m_a - m_b) / m_b * 100 if m_b > 0 else 0
            _, p = scipy_stats.ttest_rel(before_on, after_on) if n >= 3 else (0, np.nan)
            summary_data.append({
                'Genotype': genotype, 'Cell Type': 'ON', 'Response': 'ON',
                'N': n, 'Before (Hz)': f'{m_b:.2f} +/- {sem_b:.2f}',
                'After (Hz)': f'{m_a:.2f} +/- {sem_a:.2f}',
                '% Change': f'{pct:+.1f}%',
                'p-value': f'{p:.4f}' if not np.isnan(p) else 'N/A',
                'Sig': '*' if p < 0.05 else ''
            })
        
        # OFF cells - OFF response
        before_off, after_off = [], []
        for stats in stats_list:
            for r in stats.responses:
                if r.cell_type == 'OFF':
                    before_off.append(r.before_off_response)
                    after_off.append(r.after_off_response)
        
        if len(before_off) > 0:
            n = len(before_off)
            m_b, m_a = np.mean(before_off), np.mean(after_off)
            sem_b, sem_a = np.std(before_off)/np.sqrt(n), np.std(after_off)/np.sqrt(n)
            pct = (m_a - m_b) / m_b * 100 if m_b > 0 else 0
            _, p = scipy_stats.ttest_rel(before_off, after_off) if n >= 3 else (0, np.nan)
            summary_data.append({
                'Genotype': genotype, 'Cell Type': 'OFF', 'Response': 'OFF',
                'N': n, 'Before (Hz)': f'{m_b:.2f} +/- {sem_b:.2f}',
                'After (Hz)': f'{m_a:.2f} +/- {sem_a:.2f}',
                '% Change': f'{pct:+.1f}%',
                'p-value': f'{p:.4f}' if not np.isnan(p) else 'N/A',
                'Sig': '*' if p < 0.05 else ''
            })
        
        # ON_OFF cells - ON response
        before_on_onoff, after_on_onoff = [], []
        before_off_onoff, after_off_onoff = [], []
        for stats in stats_list:
            for r in stats.responses:
                if r.cell_type == 'ON_OFF':
                    before_on_onoff.append(r.before_on_response)
                    after_on_onoff.append(r.after_on_response)
                    before_off_onoff.append(r.before_off_response)
                    after_off_onoff.append(r.after_off_response)
        
        if len(before_on_onoff) > 0:
            n = len(before_on_onoff)
            # ON response
            m_b, m_a = np.mean(before_on_onoff), np.mean(after_on_onoff)
            sem_b, sem_a = np.std(before_on_onoff)/np.sqrt(n), np.std(after_on_onoff)/np.sqrt(n)
            pct = (m_a - m_b) / m_b * 100 if m_b > 0 else 0
            _, p = scipy_stats.ttest_rel(before_on_onoff, after_on_onoff) if n >= 3 else (0, np.nan)
            summary_data.append({
                'Genotype': genotype, 'Cell Type': 'ON_OFF', 'Response': 'ON',
                'N': n, 'Before (Hz)': f'{m_b:.2f} +/- {sem_b:.2f}',
                'After (Hz)': f'{m_a:.2f} +/- {sem_a:.2f}',
                '% Change': f'{pct:+.1f}%',
                'p-value': f'{p:.4f}' if not np.isnan(p) else 'N/A',
                'Sig': '*' if p < 0.05 else ''
            })
            # OFF response
            m_b, m_a = np.mean(before_off_onoff), np.mean(after_off_onoff)
            sem_b, sem_a = np.std(before_off_onoff)/np.sqrt(n), np.std(after_off_onoff)/np.sqrt(n)
            pct = (m_a - m_b) / m_b * 100 if m_b > 0 else 0
            _, p = scipy_stats.ttest_rel(before_off_onoff, after_off_onoff) if n >= 3 else (0, np.nan)
            summary_data.append({
                'Genotype': genotype, 'Cell Type': 'ON_OFF', 'Response': 'OFF',
                'N': n, 'Before (Hz)': f'{m_b:.2f} +/- {sem_b:.2f}',
                'After (Hz)': f'{m_a:.2f} +/- {sem_a:.2f}',
                '% Change': f'{pct:+.1f}%',
                'p-value': f'{p:.4f}' if not np.isnan(p) else 'N/A',
                'Sig': '*' if p < 0.05 else ''
            })
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(14, 2 + len(summary_data) * 0.4))
    ax.axis('off')
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#E3F2FD'] * len(df.columns)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color significant rows
        for i, row in enumerate(summary_data):
            if row['Sig'] == '*':
                for j in range(len(df.columns)):
                    table[(i + 1, j)].set_facecolor('#C8E6C9')
    
    ax.set_title('Summary Statistics: Before vs After Response', fontsize=12, pad=20)
    
    plt.tight_layout()
    
    output_path = output_dir / 'summary_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_summary_markdown(
    stats_by_genotype: Dict[str, List['AlignmentStats']],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Generate a markdown summary of the statistical findings.
    
    Args:
        stats_by_genotype: Dictionary mapping genotype -> list of AlignmentStats
        output_dir: Directory to save the summary
    
    Returns:
        Path to saved markdown file
    """
    from datetime import datetime
    from scipy import stats as scipy_stats
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# HTR Agonist/Antagonist Alignment Analysis Summary\n")
    lines.append("## Experiment Overview\n")
    lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}  ")
    lines.append("**Analysis Windows:**")
    lines.append("- ON Response: 1-3 seconds after step onset (sustained ON)")
    lines.append("- OFF Response: 6-8 seconds (1-3s after step offset at 5s)\n")
    lines.append("---\n")
    
    # Data Summary Table
    lines.append("## Data Summary\n")
    lines.append("| Genotype | Recordings | Total Aligned Pairs | ON Cells | OFF Cells | ON_OFF Cells |")
    lines.append("|----------|------------|---------------------|----------|-----------|--------------|")
    
    total_recordings = 0
    total_pairs = 0
    total_on = 0
    total_off = 0
    total_on_off = 0
    
    for genotype, stats_list in stats_by_genotype.items():
        n_rec = len(stats_list)
        n_pairs = sum(s.n_pairs for s in stats_list)
        n_on = sum(s.n_on for s in stats_list)
        n_off = sum(s.n_off for s in stats_list)
        n_on_off = sum(s.n_on_off for s in stats_list)
        
        total_recordings += n_rec
        total_pairs += n_pairs
        total_on += n_on
        total_off += n_off
        total_on_off += n_on_off
        
        lines.append(f"| {genotype} | {n_rec} | {n_pairs} | {n_on} | {n_off} | {n_on_off} |")
    
    lines.append(f"| **Total** | **{total_recordings}** | **{total_pairs}** | **{total_on}** | **{total_off}** | **{total_on_off}** |")
    lines.append("\n---\n")
    
    # Statistical Findings by Genotype
    lines.append("## Statistical Findings\n")
    
    key_observations = []
    
    for genotype, stats_list in stats_by_genotype.items():
        lines.append(f"### {genotype}\n")
        
        # Collect all responses
        all_on_cells = []
        all_off_cells = []
        all_on_off_cells = []
        
        for stats in stats_list:
            all_on_cells.extend([r for r in stats.responses if r.cell_type == 'ON'])
            all_off_cells.extend([r for r in stats.responses if r.cell_type == 'OFF'])
            all_on_off_cells.extend([r for r in stats.responses if r.cell_type == 'ON_OFF'])
        
        genotype_obs = []
        
        # ON Cells
        if all_on_cells:
            lines.append(f"#### ON Cells (n={len(all_on_cells)})")
            lines.append("| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |")
            lines.append("|----------|-------------|------------|--------|---------|-------------|")
            
            before_on = [r.before_on_response for r in all_on_cells]
            after_on = [r.after_on_response for r in all_on_cells]
            
            m_b, m_a = np.mean(before_on), np.mean(after_on)
            sem_b = np.std(before_on) / np.sqrt(len(before_on))
            sem_a = np.std(after_on) / np.sqrt(len(after_on))
            pct = ((m_a - m_b) / m_b * 100) if m_b != 0 else 0
            _, p = scipy_stats.ttest_rel(before_on, after_on)
            
            sig_mark = "**Yes**" if p < 0.05 else "No"
            pct_str = f"**{pct:+.1f}%**" if p < 0.05 else f"{pct:+.1f}%"
            p_str = f"<0.0001" if p < 0.0001 else f"{p:.4f}"
            
            lines.append(f"| ON | {m_b:.2f} ± {sem_b:.2f} | {m_a:.2f} ± {sem_a:.2f} | {pct_str} | {p_str} | {sig_mark} |")
            lines.append("")
            
            if p < 0.05:
                change_word = "increase" if pct > 0 else "decrease"
                genotype_obs.append(f"**ON cells:** {change_word.capitalize()} in ON response ({pct:+.0f}%)")
        
        # OFF Cells
        if all_off_cells:
            lines.append(f"#### OFF Cells (n={len(all_off_cells)})")
            lines.append("| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |")
            lines.append("|----------|-------------|------------|--------|---------|-------------|")
            
            before_off = [r.before_off_response for r in all_off_cells]
            after_off = [r.after_off_response for r in all_off_cells]
            
            m_b, m_a = np.mean(before_off), np.mean(after_off)
            sem_b = np.std(before_off) / np.sqrt(len(before_off))
            sem_a = np.std(after_off) / np.sqrt(len(after_off))
            pct = ((m_a - m_b) / m_b * 100) if m_b != 0 else 0
            _, p = scipy_stats.ttest_rel(before_off, after_off)
            
            sig_mark = "**Yes**" if p < 0.05 else "No"
            pct_str = f"**{pct:+.1f}%**" if p < 0.05 else f"{pct:+.1f}%"
            p_str = f"<0.0001" if p < 0.0001 else f"{p:.4f}"
            
            lines.append(f"| OFF | {m_b:.2f} ± {sem_b:.2f} | {m_a:.2f} ± {sem_a:.2f} | {pct_str} | {p_str} | {sig_mark} |")
            lines.append("")
            
            if p < 0.05:
                change_word = "increase" if pct > 0 else "decrease"
                genotype_obs.append(f"**OFF cells:** {change_word.capitalize()} in OFF response ({pct:+.0f}%)")
        
        # ON_OFF Cells
        if all_on_off_cells:
            lines.append(f"#### ON_OFF Cells (n={len(all_on_off_cells)})")
            lines.append("| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |")
            lines.append("|----------|-------------|------------|--------|---------|-------------|")
            
            # ON response
            before_on = [r.before_on_response for r in all_on_off_cells]
            after_on = [r.after_on_response for r in all_on_off_cells]
            
            m_b_on, m_a_on = np.mean(before_on), np.mean(after_on)
            sem_b_on = np.std(before_on) / np.sqrt(len(before_on))
            sem_a_on = np.std(after_on) / np.sqrt(len(after_on))
            pct_on = ((m_a_on - m_b_on) / m_b_on * 100) if m_b_on != 0 else 0
            _, p_on = scipy_stats.ttest_rel(before_on, after_on)
            
            sig_mark_on = "**Yes**" if p_on < 0.05 else "No"
            pct_str_on = f"**{pct_on:+.1f}%**" if p_on < 0.05 else f"{pct_on:+.1f}%"
            p_str_on = f"<0.0001" if p_on < 0.0001 else f"{p_on:.4f}"
            
            lines.append(f"| ON | {m_b_on:.2f} ± {sem_b_on:.2f} | {m_a_on:.2f} ± {sem_a_on:.2f} | {pct_str_on} | {p_str_on} | {sig_mark_on} |")
            
            # OFF response
            before_off = [r.before_off_response for r in all_on_off_cells]
            after_off = [r.after_off_response for r in all_on_off_cells]
            
            m_b_off, m_a_off = np.mean(before_off), np.mean(after_off)
            sem_b_off = np.std(before_off) / np.sqrt(len(before_off))
            sem_a_off = np.std(after_off) / np.sqrt(len(after_off))
            pct_off = ((m_a_off - m_b_off) / m_b_off * 100) if m_b_off != 0 else 0
            _, p_off = scipy_stats.ttest_rel(before_off, after_off)
            
            sig_mark_off = "**Yes**" if p_off < 0.05 else "No"
            pct_str_off = f"**{pct_off:+.1f}%**" if p_off < 0.05 else f"{pct_off:+.1f}%"
            p_str_off = f"<0.0001" if p_off < 0.0001 else f"{p_off:.4f}"
            
            lines.append(f"| OFF | {m_b_off:.2f} ± {sem_b_off:.2f} | {m_a_off:.2f} ± {sem_a_off:.2f} | {pct_str_off} | {p_str_off} | {sig_mark_off} |")
            lines.append("")
            
            obs_parts = []
            if p_on < 0.05:
                change_word = "increased" if pct_on > 0 else "decreased"
                obs_parts.append(f"ON {change_word} ({pct_on:+.0f}%)")
            else:
                obs_parts.append(f"ON unchanged (p={p_on:.2f})")
            if p_off < 0.05:
                change_word = "increased" if pct_off > 0 else "decreased"
                obs_parts.append(f"OFF {change_word} ({pct_off:+.0f}%)")
            else:
                obs_parts.append(f"OFF unchanged (p={p_off:.2f})")
            genotype_obs.append(f"**ON_OFF cells:** {', '.join(obs_parts)}")
        
        lines.append("---\n")
        key_observations.append((genotype, genotype_obs))
    
    # Key Observations
    lines.append("## Key Observations\n")
    for i, (genotype, obs_list) in enumerate(key_observations, 1):
        lines.append(f"### {i}. {genotype} Response Changes")
        for obs in obs_list:
            lines.append(f"- {obs}")
        lines.append("")
    
    lines.append("---\n")
    
    # Figures
    lines.append("## Figures\n")
    lines.append("| Figure | Description |")
    lines.append("|--------|-------------|")
    lines.append("| `summary_table.png` | Tabular statistics summary |")
    lines.append("| `genotype_summary.png` | Bar plots of before/after responses |")
    lines.append("| `genotype_mean_traces.png` | Mean PSTH traces by genotype |")
    lines.append("| `genotype_overlay_traces.png` | Before vs After overlay comparison |")
    lines.append("| `genotype_percent_change.png` | Percent change between genotypes |")
    lines.append("| `genotype_distribution.png` | Response distribution box plots |")
    lines.append("| `genotype_paired_changes.png` | Individual cell paired changes |")
    lines.append("\n---\n")
    
    # Methods
    lines.append("## Methods\n")
    lines.append("### Cell Type Classification")
    lines.append("Cells were classified based on sustained step response (1-3s window):")
    lines.append("- **ON cells:** Significant ON response, no OFF response")
    lines.append("- **OFF cells:** Significant OFF response, no ON response")
    lines.append("- **ON_OFF cells:** Both ON and OFF responses significant\n")
    lines.append("### Statistical Tests")
    lines.append("- Paired t-test for before/after comparisons (same cells tracked across recordings)")
    lines.append("- Significance threshold: p < 0.05")
    lines.append("- Values reported as mean ± SEM\n")
    lines.append("### Alignment")
    lines.append("Units were matched across recordings using:")
    lines.append("- Spatial proximity (electrode position)")
    lines.append("- Waveform similarity")
    lines.append("- Response signature similarity")
    
    # Write file
    output_path = output_dir / 'SUMMARY.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Main visualization pipeline."""
    global ALIGNED_DIR, FIGURES_DIR
    
    parser = argparse.ArgumentParser(
        description="Visualization of Before/After Alignment"
    )
    parser.add_argument(
        "--input", type=Path, default=ALIGNED_DIR,
        help=f"Folder containing aligned H5 files (default: {ALIGNED_DIR})"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for figures (default: input/../figures)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set directories
    ALIGNED_DIR = args.input
    FIGURES_DIR = args.output or (ALIGNED_DIR.parent / "figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("HTR Agonist/Antagonist - Alignment Visualization")
    print("=" * 70)
    
    # Find aligned H5 files
    aligned_files = list(ALIGNED_DIR.glob("*_aligned.h5"))
    
    if not aligned_files:
        print(f"No aligned files found in {ALIGNED_DIR}")
        return
    
    print(f"Found {len(aligned_files)} aligned files")
    
    # Process each file
    all_stats = []
    genotype_warnings = []
    
    for aligned_file in aligned_files:
        print(f"\nLoading: {aligned_file.name}")
        
        # Load statistics
        stats = load_alignment_stats(aligned_file)
        all_stats.append(stats)
        
        print(f"  Pairs: {stats.n_pairs}")
        print(f"  Cell types: ON={stats.n_on}, OFF={stats.n_off}, ON_OFF={stats.n_on_off}, unknown={stats.n_unknown}")
        print(f"  Genotype: before='{stats.before_genotype}', after='{stats.after_genotype}'")
        
        if not stats.genotype_consistent:
            warning_msg = f"WARNING: Genotype mismatch in {stats.pair_name}: before='{stats.before_genotype}', after='{stats.after_genotype}'"
            print(f"  {warning_msg}")
            genotype_warnings.append(warning_msg)
    
    # Group by genotype
    print("\n" + "-" * 70)
    print("Grouping by Genotype")
    print("-" * 70)
    
    stats_by_genotype: Dict[str, List[AlignmentStats]] = {}
    for stats in all_stats:
        genotype = stats.before_genotype
        if genotype not in stats_by_genotype:
            stats_by_genotype[genotype] = []
        stats_by_genotype[genotype].append(stats)
    
    for genotype, stats_list in stats_by_genotype.items():
        total_pairs = sum(s.n_pairs for s in stats_list)
        n_on = sum(s.n_on for s in stats_list)
        n_off = sum(s.n_off for s in stats_list)
        n_on_off = sum(s.n_on_off for s in stats_list)
        print(f"  {genotype}: {len(stats_list)} recordings, {total_pairs} pairs")
        print(f"    Cell types: ON={n_on}, OFF={n_off}, ON_OFF={n_on_off}")
    
    # Create genotype-focused visualizations
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)
    
    if stats_by_genotype:
        # 1. Summary statistics table
        summary_table = create_summary_table(stats_by_genotype, output_dir=FIGURES_DIR)
        if summary_table:
            print(f"  [1/6] Summary table: {summary_table.name}")
        
        # 2. Genotype summary bar plots
        genotype_summary = create_genotype_summary(stats_by_genotype, output_dir=FIGURES_DIR)
        if genotype_summary:
            print(f"  [2/6] Genotype summary: {genotype_summary.name}")
        
        # 3. Mean traces by genotype
        genotype_traces = create_genotype_mean_traces(stats_by_genotype, output_dir=FIGURES_DIR)
        if genotype_traces:
            print(f"  [3/6] Mean traces: {genotype_traces.name}")
        
        # 4. Overlay traces comparing genotypes
        overlay_traces = create_genotype_overlay_traces(stats_by_genotype, output_dir=FIGURES_DIR)
        if overlay_traces:
            print(f"  [4/6] Overlay comparison: {overlay_traces.name}")
        
        # 5. Percent change comparison
        pct_change = create_percent_change_comparison(stats_by_genotype, output_dir=FIGURES_DIR)
        if pct_change:
            print(f"  [5/6] Percent change: {pct_change.name}")
        
        # 6. Response distribution box plots
        distribution = create_response_distribution(stats_by_genotype, output_dir=FIGURES_DIR)
        if distribution:
            print(f"  [6/6] Distribution: {distribution.name}")
        
        # 7. Paired change plots
        paired = create_paired_change_plot(stats_by_genotype, output_dir=FIGURES_DIR)
        if paired:
            print(f"  [7/6] Paired changes: {paired.name}")
        
        # 8. Generate statistical summary markdown
        print("\n" + "-" * 70)
        print("Generating Statistical Summary")
        print("-" * 70)
        summary_md = generate_summary_markdown(stats_by_genotype, output_dir=FIGURES_DIR)
        if summary_md:
            print(f"  Created: {summary_md.name}")
    
    # Print warnings summary
    if genotype_warnings:
        print("\n" + "=" * 70)
        print("GENOTYPE CONSISTENCY WARNINGS")
        print("=" * 70)
        for warning in genotype_warnings:
            print(f"  {warning}")
    
    print("\n" + "=" * 70)
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
