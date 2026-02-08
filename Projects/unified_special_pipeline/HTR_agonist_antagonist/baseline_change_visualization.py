"""
Baseline Change Visualization

Visualizes firing rate changes over time with 30-second binning.
Experiment structure:
- First 5 minutes (0-300s): No agonist (baseline)
- After 5 minutes (300s+): Agonist present until end

Groups data by genotype for comparison.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# =============================================================================
# Configuration
# =============================================================================

# Time binning
BIN_SIZE_SEC = 30  # 30-second bins
AGONIST_START_SEC = 300  # Agonist starts at 5 minutes (300 seconds)

# Directories
try:
    from specific_config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "output"

ALIGNED_BASELINE_DIR = OUTPUT_DIR / "aligned_baseline"
FIGURES_DIR = OUTPUT_DIR / "figures_baseline"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UnitTimeCourse:
    """Time course data for a single unit."""
    unit_id: str
    before_firing_rates: np.ndarray  # Firing rate per bin in "before" recording
    after_firing_rates: np.ndarray   # Firing rate per bin in "after" recording
    mean_before_baseline: float = 0.0  # Mean FR in baseline period (before agonist)
    mean_before_agonist: float = 0.0   # Mean FR in agonist period (before recording)
    mean_after_baseline: float = 0.0   # Mean FR in baseline period (after recording)
    mean_after_agonist: float = 0.0    # Mean FR in agonist period (after recording)


@dataclass
class AlignmentData:
    """Data from an aligned recording pair."""
    pair_name: str
    chip: str
    genotype: str
    n_pairs: int
    units: List[UnitTimeCourse] = field(default_factory=list)
    time_bins: np.ndarray = field(default_factory=lambda: np.array([]))


# =============================================================================
# Data Loading
# =============================================================================

def get_genotype_from_h5(h5_path: Path) -> str:
    """Extract genotype from source H5 file."""
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


def calculate_firing_rate_timecourse(
    spike_times: np.ndarray,
    bin_size_sec: float = BIN_SIZE_SEC,
    max_time_sec: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate firing rate time course from spike times.
    
    Args:
        spike_times: Array of spike times in seconds
        bin_size_sec: Bin size in seconds
        max_time_sec: Maximum time to bin (uses max spike time if None)
    
    Returns:
        (time_bins, firing_rates) - time bins (center) and rates in Hz
    """
    if spike_times is None or len(spike_times) == 0:
        return np.array([]), np.array([])
    
    if max_time_sec is None:
        max_time_sec = np.max(spike_times) + bin_size_sec
    
    # Create bins
    n_bins = int(np.ceil(max_time_sec / bin_size_sec))
    bin_edges = np.arange(0, (n_bins + 1) * bin_size_sec, bin_size_sec)
    
    # Count spikes per bin
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    
    # Convert to firing rate (Hz)
    firing_rates = counts / bin_size_sec
    
    # Time bin centers
    time_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return time_bins, firing_rates


def load_alignment_data(aligned_h5_path: Path) -> AlignmentData:
    """Load alignment data from H5 file."""
    logger = logging.getLogger(__name__)
    
    units = []
    time_bins = np.array([])
    
    with h5py.File(aligned_h5_path, 'r') as f:
        # Get metadata
        chip = f.attrs.get('chip', 'unknown')
        if isinstance(chip, bytes):
            chip = chip.decode('utf-8')
        
        pair_name = aligned_h5_path.stem
        
        # Get genotype from source file
        before_h5_path = f.attrs.get('before_h5', '')
        if isinstance(before_h5_path, bytes):
            before_h5_path = before_h5_path.decode('utf-8')
        genotype = get_genotype_from_h5(Path(before_h5_path)) if before_h5_path else "unknown"
        
        # Process paired units
        if 'paired_units' not in f:
            logger.warning(f"No paired_units in {aligned_h5_path}")
            return AlignmentData(
                pair_name=pair_name, chip=chip, genotype=genotype, n_pairs=0
            )
        
        paired_units = f['paired_units']
        
        # Determine max time from all spike times
        max_time = 0
        for pair_key in paired_units.keys():
            pair_group = paired_units[pair_key]
            if 'before' in pair_group and 'spike_times' in pair_group['before']:
                st = pair_group['before']['spike_times'][:]
                if len(st) > 0:
                    max_time = max(max_time, np.max(st))
            if 'after' in pair_group and 'spike_times' in pair_group['after']:
                st = pair_group['after']['spike_times'][:]
                if len(st) > 0:
                    max_time = max(max_time, np.max(st))
        
        # Round up to nearest bin
        max_time_sec = np.ceil(max_time / BIN_SIZE_SEC) * BIN_SIZE_SEC + BIN_SIZE_SEC
        
        for pair_key in paired_units.keys():
            pair_group = paired_units[pair_key]
            
            before_unit = pair_group.attrs.get('before_unit', 'unknown')
            if isinstance(before_unit, bytes):
                before_unit = before_unit.decode('utf-8')
            
            # Get before spike times and calculate firing rates
            before_fr = np.array([])
            if 'before' in pair_group and 'spike_times' in pair_group['before']:
                before_spikes = pair_group['before']['spike_times'][:]
                time_bins, before_fr = calculate_firing_rate_timecourse(
                    before_spikes, BIN_SIZE_SEC, max_time_sec
                )
            
            # Get after spike times and calculate firing rates
            after_fr = np.array([])
            if 'after' in pair_group and 'spike_times' in pair_group['after']:
                after_spikes = pair_group['after']['spike_times'][:]
                _, after_fr = calculate_firing_rate_timecourse(
                    after_spikes, BIN_SIZE_SEC, max_time_sec
                )
            
            # Pad to same length if needed
            if len(before_fr) > 0 and len(after_fr) > 0:
                max_len = max(len(before_fr), len(after_fr))
                if len(before_fr) < max_len:
                    before_fr = np.pad(before_fr, (0, max_len - len(before_fr)))
                if len(after_fr) < max_len:
                    after_fr = np.pad(after_fr, (0, max_len - len(after_fr)))
                
                # Calculate mean firing rates in baseline and agonist periods
                baseline_bins = time_bins < AGONIST_START_SEC
                agonist_bins = time_bins >= AGONIST_START_SEC
                
                mean_before_baseline = np.mean(before_fr[baseline_bins]) if np.any(baseline_bins) else 0
                mean_before_agonist = np.mean(before_fr[agonist_bins]) if np.any(agonist_bins) else 0
                mean_after_baseline = np.mean(after_fr[baseline_bins]) if np.any(baseline_bins) else 0
                mean_after_agonist = np.mean(after_fr[agonist_bins]) if np.any(agonist_bins) else 0
                
                units.append(UnitTimeCourse(
                    unit_id=before_unit,
                    before_firing_rates=before_fr,
                    after_firing_rates=after_fr,
                    mean_before_baseline=mean_before_baseline,
                    mean_before_agonist=mean_before_agonist,
                    mean_after_baseline=mean_after_baseline,
                    mean_after_agonist=mean_after_agonist,
                ))
    
    return AlignmentData(
        pair_name=pair_name,
        chip=chip,
        genotype=genotype,
        n_pairs=len(units),
        units=units,
        time_bins=time_bins,
    )


# =============================================================================
# Visualization Functions
# =============================================================================

def create_timecourse_by_genotype(
    data_by_genotype: Dict[str, List[AlignmentData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create time course plots showing firing rate over time by genotype.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(data_by_genotype.keys())
    n_genotypes = len(genotypes)
    
    if n_genotypes == 0:
        return None
    
    fig, axes = plt.subplots(n_genotypes, 2, figsize=(16, 5 * n_genotypes))
    if n_genotypes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Firing Rate Time Course (30s bins)\nAgonist starts at {AGONIST_START_SEC}s (5 min)',
                 fontsize=14, y=1.02)
    
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    
    for row, genotype in enumerate(genotypes):
        data_list = data_by_genotype[genotype]
        
        # Collect all time courses
        all_before = []
        all_after = []
        time_bins = None
        
        for data in data_list:
            for unit in data.units:
                if len(unit.before_firing_rates) > 0 and len(unit.after_firing_rates) > 0:
                    all_before.append(unit.before_firing_rates)
                    all_after.append(unit.after_firing_rates)
                    if time_bins is None and len(data.time_bins) > 0:
                        time_bins = data.time_bins
        
        if not all_before or time_bins is None:
            continue
        
        # Pad to same length
        max_len = max(max(len(b) for b in all_before), max(len(a) for a in all_after))
        all_before = [np.pad(b, (0, max_len - len(b))) for b in all_before]
        all_after = [np.pad(a, (0, max_len - len(a))) for a in all_after]
        
        before_arr = np.array(all_before)
        after_arr = np.array(all_after)
        
        # Extend time_bins if needed
        if len(time_bins) < max_len:
            extra_bins = np.arange(len(time_bins), max_len) * BIN_SIZE_SEC + BIN_SIZE_SEC / 2
            time_bins = np.concatenate([time_bins, extra_bins])
        time_bins = time_bins[:max_len]
        
        n_units = len(all_before)
        
        # Left plot: Before recording
        ax_before = axes[row, 0]
        mean_before = np.mean(before_arr, axis=0)
        sem_before = np.std(before_arr, axis=0) / np.sqrt(n_units)
        
        ax_before.plot(time_bins / 60, mean_before, color=colors['before'], linewidth=2)
        ax_before.fill_between(time_bins / 60, mean_before - sem_before, mean_before + sem_before,
                              color=colors['before'], alpha=0.3)
        ax_before.axvline(x=AGONIST_START_SEC / 60, color='red', linestyle='--', 
                         linewidth=2, label='Agonist Start')
        ax_before.axvspan(0, AGONIST_START_SEC / 60, color='gray', alpha=0.1, label='Baseline')
        ax_before.axvspan(AGONIST_START_SEC / 60, time_bins[-1] / 60, color='red', alpha=0.05, label='Agonist')
        
        ax_before.set_xlabel('Time (minutes)')
        ax_before.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
        ax_before.set_title(f'BEFORE Recording (n={n_units})')
        ax_before.legend(loc='upper right', fontsize=8)
        ax_before.set_xlim(0, time_bins[-1] / 60)
        ax_before.set_ylim(bottom=0)
        
        # Right plot: After recording
        ax_after = axes[row, 1]
        mean_after = np.mean(after_arr, axis=0)
        sem_after = np.std(after_arr, axis=0) / np.sqrt(n_units)
        
        ax_after.plot(time_bins / 60, mean_after, color=colors['after'], linewidth=2)
        ax_after.fill_between(time_bins / 60, mean_after - sem_after, mean_after + sem_after,
                             color=colors['after'], alpha=0.3)
        ax_after.axvline(x=AGONIST_START_SEC / 60, color='red', linestyle='--', 
                        linewidth=2, label='Agonist Start')
        ax_after.axvspan(0, AGONIST_START_SEC / 60, color='gray', alpha=0.1)
        ax_after.axvspan(AGONIST_START_SEC / 60, time_bins[-1] / 60, color='red', alpha=0.05)
        
        ax_after.set_xlabel('Time (minutes)')
        ax_after.set_ylabel('Firing Rate (Hz)')
        ax_after.set_title(f'AFTER Recording (n={n_units})')
        ax_after.legend(loc='upper right', fontsize=8)
        ax_after.set_xlim(0, time_bins[-1] / 60)
        ax_after.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'timecourse_by_genotype.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_baseline_vs_agonist_comparison(
    data_by_genotype: Dict[str, List[AlignmentData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create bar plots comparing baseline (0-5min) vs agonist (5min+) periods.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(data_by_genotype.keys())
    n_genotypes = len(genotypes)
    
    if n_genotypes == 0:
        return None
    
    fig, axes = plt.subplots(n_genotypes, 2, figsize=(14, 5 * n_genotypes))
    if n_genotypes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Baseline (0-5min) vs Agonist (5min+) Period Comparison', fontsize=14, y=1.02)
    
    bar_width = 0.35
    
    for row, genotype in enumerate(genotypes):
        data_list = data_by_genotype[genotype]
        
        # Collect firing rates
        before_baseline = []
        before_agonist = []
        after_baseline = []
        after_agonist = []
        
        for data in data_list:
            for unit in data.units:
                before_baseline.append(unit.mean_before_baseline)
                before_agonist.append(unit.mean_before_agonist)
                after_baseline.append(unit.mean_after_baseline)
                after_agonist.append(unit.mean_after_agonist)
        
        n_units = len(before_baseline)
        
        if n_units == 0:
            continue
        
        # Left: Before recording - baseline vs agonist
        ax1 = axes[row, 0]
        x = np.array([0, 1])
        means = [np.mean(before_baseline), np.mean(before_agonist)]
        sems = [np.std(before_baseline) / np.sqrt(n_units), 
                np.std(before_agonist) / np.sqrt(n_units)]
        
        bars1 = ax1.bar(x, means, yerr=sems, capsize=5,
                       color=['#90CAF9', '#EF5350'], alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Baseline\n(0-5 min)', 'Agonist\n(5+ min)'])
        ax1.set_ylabel(f'{genotype}\n\nFiring Rate (Hz)')
        
        # Statistical test
        if n_units >= 3:
            try:
                _, p = scipy_stats.ttest_rel(before_baseline, before_agonist)
                pct = (means[1] - means[0]) / means[0] * 100 if means[0] > 0 else 0
                ax1.set_title(f'BEFORE Recording (n={n_units})\nChange: {pct:+.1f}%, p={p:.4f}')
            except:
                ax1.set_title(f'BEFORE Recording (n={n_units})')
        else:
            ax1.set_title(f'BEFORE Recording (n={n_units})')
        ax1.set_ylim(bottom=0)
        
        # Right: After recording - baseline vs agonist
        ax2 = axes[row, 1]
        means = [np.mean(after_baseline), np.mean(after_agonist)]
        sems = [np.std(after_baseline) / np.sqrt(n_units), 
                np.std(after_agonist) / np.sqrt(n_units)]
        
        bars2 = ax2.bar(x, means, yerr=sems, capsize=5,
                       color=['#90CAF9', '#EF5350'], alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Baseline\n(0-5 min)', 'Agonist\n(5+ min)'])
        ax2.set_ylabel('Firing Rate (Hz)')
        
        if n_units >= 3:
            try:
                _, p = scipy_stats.ttest_rel(after_baseline, after_agonist)
                pct = (means[1] - means[0]) / means[0] * 100 if means[0] > 0 else 0
                ax2.set_title(f'AFTER Recording (n={n_units})\nChange: {pct:+.1f}%, p={p:.4f}')
            except:
                ax2.set_title(f'AFTER Recording (n={n_units})')
        else:
            ax2.set_title(f'AFTER Recording (n={n_units})')
        ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'baseline_vs_agonist.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_agonist_effect_comparison(
    data_by_genotype: Dict[str, List[AlignmentData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create plots comparing agonist effect (change from baseline to agonist) 
    between Before and After recordings for each genotype.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(data_by_genotype.keys())
    n_genotypes = len(genotypes)
    
    if n_genotypes == 0:
        return None
    
    fig, axes = plt.subplots(1, n_genotypes, figsize=(6 * n_genotypes, 6))
    if n_genotypes == 1:
        axes = [axes]
    
    fig.suptitle('Agonist Effect: % Change from Baseline to Agonist Period', fontsize=14)
    
    colors = {'before': '#2196F3', 'after': '#FF5722'}
    
    for col, genotype in enumerate(genotypes):
        ax = axes[col]
        data_list = data_by_genotype[genotype]
        
        # Calculate percent change for each unit
        before_pct_change = []
        after_pct_change = []
        
        for data in data_list:
            for unit in data.units:
                if unit.mean_before_baseline > 0:
                    pct = (unit.mean_before_agonist - unit.mean_before_baseline) / unit.mean_before_baseline * 100
                    before_pct_change.append(pct)
                if unit.mean_after_baseline > 0:
                    pct = (unit.mean_after_agonist - unit.mean_after_baseline) / unit.mean_after_baseline * 100
                    after_pct_change.append(pct)
        
        n_before = len(before_pct_change)
        n_after = len(after_pct_change)
        
        if n_before == 0 or n_after == 0:
            continue
        
        # Bar plot
        x = np.array([0, 1])
        means = [np.mean(before_pct_change), np.mean(after_pct_change)]
        sems = [np.std(before_pct_change) / np.sqrt(n_before),
                np.std(after_pct_change) / np.sqrt(n_after)]
        
        bars = ax.bar(x, means, yerr=sems, capsize=5,
                     color=[colors['before'], colors['after']], alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Before\n(n={n_before})', f'After\n(n={n_after})'])
        ax.set_ylabel('Agonist Effect (% Change)')
        
        # Statistical comparison
        if n_before >= 3 and n_after >= 3:
            try:
                _, p = scipy_stats.ttest_ind(before_pct_change, after_pct_change)
                ax.set_title(f'{genotype}\np={p:.4f} (Before vs After)')
            except:
                ax.set_title(genotype)
        else:
            ax.set_title(genotype)
    
    plt.tight_layout()
    
    output_path = output_dir / 'agonist_effect_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_genotype_overlay_timecourse(
    data_by_genotype: Dict[str, List[AlignmentData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Create overlay plots comparing genotypes on same axes for before and after.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genotypes = list(data_by_genotype.keys())
    
    if len(genotypes) == 0:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Genotype Comparison - Firing Rate Time Course\nAgonist starts at {AGONIST_START_SEC}s',
                 fontsize=14)
    
    genotype_colors = ['#1976D2', '#D32F2F', '#388E3C', '#7B1FA2']
    
    for plot_idx, (ax, title, data_key) in enumerate(zip(
        axes, 
        ['BEFORE Recording', 'AFTER Recording'],
        ['before', 'after']
    )):
        for i, genotype in enumerate(genotypes):
            data_list = data_by_genotype[genotype]
            
            # Collect time courses
            all_traces = []
            time_bins = None
            
            for data in data_list:
                for unit in data.units:
                    trace = unit.before_firing_rates if data_key == 'before' else unit.after_firing_rates
                    if len(trace) > 0:
                        all_traces.append(trace)
                        if time_bins is None and len(data.time_bins) > 0:
                            time_bins = data.time_bins
            
            if not all_traces or time_bins is None:
                continue
            
            # Pad to same length
            max_len = max(len(t) for t in all_traces)
            all_traces = [np.pad(t, (0, max_len - len(t))) for t in all_traces]
            trace_arr = np.array(all_traces)
            
            # Extend time_bins if needed
            if len(time_bins) < max_len:
                extra_bins = np.arange(len(time_bins), max_len) * BIN_SIZE_SEC + BIN_SIZE_SEC / 2
                time_bins = np.concatenate([time_bins, extra_bins])
            time_bins = time_bins[:max_len]
            
            n_units = len(all_traces)
            mean_trace = np.mean(trace_arr, axis=0)
            sem_trace = np.std(trace_arr, axis=0) / np.sqrt(n_units)
            
            color = genotype_colors[i % len(genotype_colors)]
            ax.plot(time_bins / 60, mean_trace, color=color, linewidth=2,
                   label=f'{genotype} (n={n_units})')
            ax.fill_between(time_bins / 60, mean_trace - sem_trace, mean_trace + sem_trace,
                           color=color, alpha=0.2)
        
        ax.axvline(x=AGONIST_START_SEC / 60, color='red', linestyle='--', 
                  linewidth=2, alpha=0.7)
        ax.axvspan(0, AGONIST_START_SEC / 60, color='gray', alpha=0.05)
        ax.axvspan(AGONIST_START_SEC / 60, ax.get_xlim()[1], color='red', alpha=0.02)
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'genotype_overlay_timecourse.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_summary_statistics(
    data_by_genotype: Dict[str, List[AlignmentData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create summary statistics table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    
    for genotype in data_by_genotype.keys():
        data_list = data_by_genotype[genotype]
        
        before_baseline = []
        before_agonist = []
        after_baseline = []
        after_agonist = []
        
        for data in data_list:
            for unit in data.units:
                before_baseline.append(unit.mean_before_baseline)
                before_agonist.append(unit.mean_before_agonist)
                after_baseline.append(unit.mean_after_baseline)
                after_agonist.append(unit.mean_after_agonist)
        
        n = len(before_baseline)
        if n == 0:
            continue
        
        # Before recording stats
        mb_base = np.mean(before_baseline)
        mb_agon = np.mean(before_agonist)
        pct_before = (mb_agon - mb_base) / mb_base * 100 if mb_base > 0 else 0
        _, p_before = scipy_stats.ttest_rel(before_baseline, before_agonist) if n >= 3 else (0, np.nan)
        
        # After recording stats
        ma_base = np.mean(after_baseline)
        ma_agon = np.mean(after_agonist)
        pct_after = (ma_agon - ma_base) / ma_base * 100 if ma_base > 0 else 0
        _, p_after = scipy_stats.ttest_rel(after_baseline, after_agonist) if n >= 3 else (0, np.nan)
        
        summary_data.append({
            'Genotype': genotype,
            'N': n,
            'Before Baseline (Hz)': f'{mb_base:.2f}',
            'Before Agonist (Hz)': f'{mb_agon:.2f}',
            'Before Change': f'{pct_before:+.1f}%',
            'Before p': f'{p_before:.4f}' if not np.isnan(p_before) else 'N/A',
            'After Baseline (Hz)': f'{ma_base:.2f}',
            'After Agonist (Hz)': f'{ma_agon:.2f}',
            'After Change': f'{pct_after:+.1f}%',
            'After p': f'{p_after:.4f}' if not np.isnan(p_after) else 'N/A',
        })
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(16, 2 + len(summary_data) * 0.6))
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
    
    ax.set_title('Summary: Baseline (0-5min) vs Agonist (5min+) Period', fontsize=12, pad=20)
    
    plt.tight_layout()
    
    output_path = output_dir / 'summary_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Main visualization pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 70)
    print("Baseline Change Visualization")
    print("=" * 70)
    print(f"Bin size: {BIN_SIZE_SEC} seconds")
    print(f"Agonist start: {AGONIST_START_SEC} seconds (5 minutes)")
    print("=" * 70)
    
    # Find aligned baseline files
    aligned_files = list(ALIGNED_BASELINE_DIR.glob("*_baseline.h5"))
    
    if not aligned_files:
        print(f"No aligned baseline files found in {ALIGNED_BASELINE_DIR}")
        return
    
    print(f"\nFound {len(aligned_files)} aligned baseline files")
    
    # Load data
    all_data = []
    
    for aligned_file in aligned_files:
        print(f"\nLoading: {aligned_file.name}")
        data = load_alignment_data(aligned_file)
        all_data.append(data)
        print(f"  Pairs: {data.n_pairs}, Genotype: {data.genotype}")
    
    # Group by genotype
    print("\n" + "-" * 70)
    print("Grouping by Genotype")
    print("-" * 70)
    
    data_by_genotype: Dict[str, List[AlignmentData]] = {}
    for data in all_data:
        if data.genotype not in data_by_genotype:
            data_by_genotype[data.genotype] = []
        data_by_genotype[data.genotype].append(data)
    
    for genotype, data_list in data_by_genotype.items():
        total_pairs = sum(d.n_pairs for d in data_list)
        print(f"  {genotype}: {len(data_list)} recordings, {total_pairs} pairs")
    
    # Generate visualizations
    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary statistics
    summary = create_summary_statistics(data_by_genotype)
    if summary:
        print(f"  [1/5] Summary statistics: {summary.name}")
    
    # 2. Time course by genotype
    timecourse = create_timecourse_by_genotype(data_by_genotype)
    if timecourse:
        print(f"  [2/5] Time course: {timecourse.name}")
    
    # 3. Baseline vs agonist comparison
    baseline_agonist = create_baseline_vs_agonist_comparison(data_by_genotype)
    if baseline_agonist:
        print(f"  [3/5] Baseline vs Agonist: {baseline_agonist.name}")
    
    # 4. Agonist effect comparison
    agonist_effect = create_agonist_effect_comparison(data_by_genotype)
    if agonist_effect:
        print(f"  [4/5] Agonist effect: {agonist_effect.name}")
    
    # 5. Genotype overlay
    overlay = create_genotype_overlay_timecourse(data_by_genotype)
    if overlay:
        print(f"  [5/5] Genotype overlay: {overlay.name}")
    
    print("\n" + "=" * 70)
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
