"""
Middle Recordings Visualization

Visualizes the aligned middle recordings time course.
- Spike times are in frame numbers at 20kHz
- First 5 minutes of first recording: no agonist
- After 5 minutes: agonist present
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Import config
try:
    from .specific_config import OUTPUT_DIR
except ImportError:
    from specific_config import OUTPUT_DIR

# =============================================================================
# Constants
# =============================================================================

SAMPLE_RATE = 20000  # 20 kHz
AGONIST_START_MIN = 5.0  # Agonist starts at 5 minutes in the first recording
BIN_SIZE_SEC = 30  # 30 second bins

ALIGNED_MIDDLE_DIR = OUTPUT_DIR / "aligned_middle"
FIGURES_DIR = OUTPUT_DIR / "figures_middle"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RecordingData:
    """Data for a single recording."""
    file_name: str
    duration_min: float
    spike_times_sec: np.ndarray  # Spike times in seconds
    zero_filled: bool = False  # True if this recording was zero-filled (missing match)


@dataclass
class UnitTimeCourse:
    """Time course data for an aligned unit chain."""
    chain_id: str
    ref_unit: str
    cell_type: str = "unknown"
    recordings: List[RecordingData] = field(default_factory=list)
    has_zerofill: bool = False  # True if any recording is zero-filled
    
    # Concatenated time course (in minutes)
    time_bins_min: Optional[np.ndarray] = None
    firing_rates: Optional[np.ndarray] = None  # Hz


# Cell types to analyze
CELL_TYPES = ['ON', 'OFF', 'ON_OFF', 'unknown']
CELL_TYPE_COLORS = {
    'ON': '#2ecc71',      # Green
    'OFF': '#e74c3c',     # Red
    'ON_OFF': '#9b59b6',  # Purple
    'unknown': '#95a5a6', # Gray
}

# Genotype order (WT first, KO second)
GENOTYPE_ORDER = ['Httr WT', 'HttrB KO']
GENOTYPE_COLORS = {
    'Httr WT': '#2ecc71',   # Green for WT
    'HttrB KO': '#e74c3c',  # Red for KO
}


def sort_genotypes(chips_by_genotype: Dict[str, List]) -> Dict[str, List]:
    """Sort genotypes dictionary with WT first, KO second."""
    sorted_dict = {}
    for genotype in GENOTYPE_ORDER:
        if genotype in chips_by_genotype:
            sorted_dict[genotype] = chips_by_genotype[genotype]
    # Add any remaining genotypes
    for genotype in chips_by_genotype:
        if genotype not in sorted_dict:
            sorted_dict[genotype] = chips_by_genotype[genotype]
    return sorted_dict


@dataclass
class ChipData:
    """Data for a chip."""
    chip: str
    genotype: str
    middle_files: List[str]
    units: List[UnitTimeCourse] = field(default_factory=list)
    
    # Time info
    total_duration_min: float = 0.0
    recording_boundaries_min: List[float] = field(default_factory=list)


# =============================================================================
# Data Loading
# =============================================================================

def load_chip_data(h5_path: Path) -> Optional[ChipData]:
    """Load aligned middle data from H5 file."""
    logger = logging.getLogger(__name__)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            chip = str(f.attrs['chip'])
            genotype = str(f.attrs['genotype'])
            middle_files = list(f.attrs['middle_files'])
            
            chip_data = ChipData(
                chip=chip,
                genotype=genotype,
                middle_files=middle_files,
            )
            
            # Load each chain
            for chain_key in f['aligned_units'].keys():
                chain_group = f['aligned_units'][chain_key]
                ref_unit = chain_group.attrs['reference_unit']
                cell_type = chain_group.attrs.get('cell_type', 'unknown')
                
                unit_tc = UnitTimeCourse(chain_id=chain_key, ref_unit=ref_unit, cell_type=cell_type)
                
                # Load each recording
                rec_idx = 0
                has_zerofill = False
                while f'recording_{rec_idx}' in chain_group:
                    rec_group = chain_group[f'recording_{rec_idx}']
                    file_name = rec_group.attrs['file']
                    
                    # Check if this recording was zero-filled
                    zero_filled = bool(rec_group.attrs.get('zero_filled', False))
                    if zero_filled:
                        has_zerofill = True
                    
                    if 'spike_times' in rec_group:
                        spike_times_frames = rec_group['spike_times'][:]
                        spike_times_sec = spike_times_frames / SAMPLE_RATE
                        # For zero-filled or empty, use standard 10 min duration
                        if len(spike_times_sec) > 0:
                            duration_min = spike_times_sec.max() / 60
                        else:
                            duration_min = 10.0  # Standard recording duration
                    else:
                        spike_times_sec = np.array([])
                        duration_min = 10.0
                    
                    unit_tc.recordings.append(RecordingData(
                        file_name=file_name,
                        duration_min=duration_min,
                        spike_times_sec=spike_times_sec,
                        zero_filled=zero_filled,
                    ))
                    rec_idx += 1
                
                unit_tc.has_zerofill = has_zerofill
                
                if unit_tc.recordings:
                    chip_data.units.append(unit_tc)
            
            # Calculate recording boundaries
            cumulative_time = 0.0
            chip_data.recording_boundaries_min = [0.0]
            for unit_tc in chip_data.units[:1]:  # Use first unit to get durations
                for rec in unit_tc.recordings:
                    # Use actual recording duration (estimate from max spike time, or ~10 min)
                    rec_duration = max(rec.duration_min, 10.0)  # At least 10 min per recording
                    cumulative_time += rec_duration
                    chip_data.recording_boundaries_min.append(cumulative_time)
            
            chip_data.total_duration_min = cumulative_time
            
            logger.info(f"Loaded {len(chip_data.units)} unit chains from {h5_path.name}")
            return chip_data
            
    except Exception as e:
        logger.error(f"Error loading {h5_path}: {e}")
        return None


def calculate_time_course(chip_data: ChipData, bin_size_sec: float = BIN_SIZE_SEC) -> None:
    """Calculate firing rate time course for each unit."""
    
    for unit_tc in chip_data.units:
        # Concatenate spike times across recordings
        all_spikes_min = []
        cumulative_time = 0.0
        
        for rec in unit_tc.recordings:
            # Shift spike times by cumulative recording time
            shifted_spikes = rec.spike_times_sec / 60 + cumulative_time
            all_spikes_min.extend(shifted_spikes)
            
            # Add recording duration (use 10 min as standard)
            rec_duration = 10.0
            cumulative_time += rec_duration
        
        all_spikes_min = np.array(all_spikes_min)
        
        # Create time bins (in minutes)
        bin_size_min = bin_size_sec / 60
        n_bins = int(np.ceil(cumulative_time / bin_size_min))
        time_bins = np.arange(n_bins) * bin_size_min
        
        # Calculate firing rate (Hz) for each bin
        firing_rates = np.zeros(n_bins)
        for i in range(n_bins):
            bin_start = i * bin_size_min
            bin_end = (i + 1) * bin_size_min
            spikes_in_bin = np.sum((all_spikes_min >= bin_start) & (all_spikes_min < bin_end))
            firing_rates[i] = spikes_in_bin / bin_size_sec  # Convert to Hz
        
        unit_tc.time_bins_min = time_bins
        unit_tc.firing_rates = firing_rates


# =============================================================================
# Visualization Functions
# =============================================================================

def create_timecourse_by_genotype(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Create time course plot grouped by genotype."""
    
    genotypes = list(chips_by_genotype.keys())
    n_genotypes = len(genotypes)
    
    if n_genotypes == 0:
        print("  Skipped: No data to plot")
        return None
    
    fig, axes = plt.subplots(n_genotypes, 1, figsize=(14, 5 * n_genotypes), squeeze=False)
    
    colors = {'baseline': '#3498db', 'agonist': '#e74c3c'}
    
    for row, genotype in enumerate(genotypes):
        ax = axes[row, 0]
        chips = chips_by_genotype[genotype]
        
        # Collect all firing rates
        all_rates = []
        max_bins = 0
        
        for chip_data in chips:
            for unit_tc in chip_data.units:
                if unit_tc.firing_rates is not None:
                    all_rates.append(unit_tc.firing_rates)
                    max_bins = max(max_bins, len(unit_tc.firing_rates))
        
        if not all_rates:
            continue
        
        # Pad to same length
        padded_rates = []
        for rates in all_rates:
            if len(rates) < max_bins:
                padded = np.pad(rates, (0, max_bins - len(rates)), constant_values=np.nan)
            else:
                padded = rates[:max_bins]
            padded_rates.append(padded)
        
        padded_rates = np.array(padded_rates)
        
        # Calculate mean and SEM
        mean_rate = np.nanmean(padded_rates, axis=0)
        sem_rate = np.nanstd(padded_rates, axis=0) / np.sqrt(np.sum(~np.isnan(padded_rates), axis=0))
        
        # Time axis
        time_min = np.arange(max_bins) * (BIN_SIZE_SEC / 60)
        
        # Plot
        ax.plot(time_min, mean_rate, 'k-', linewidth=2, label='Mean firing rate')
        ax.fill_between(time_min, mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.3, color='gray')
        
        # Mark agonist start (at 5 min in first recording)
        ax.axvline(x=AGONIST_START_MIN, color='red', linestyle='--', linewidth=2, label='Agonist start')
        
        # Mark recording boundaries
        for i in range(1, len(chips[0].recording_boundaries_min) - 1):
            boundary = 10.0 * i  # Each recording is ~10 min
            ax.axvline(x=boundary, color='blue', linestyle=':', alpha=0.5)
        
        # Shade agonist period
        ax.axvspan(AGONIST_START_MIN, time_min.max(), alpha=0.1, color='red', label='Agonist present')
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax.set_title(f'{genotype} (n={len(all_rates)} units from {len(chips)} chips)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_min.max())
    
    plt.tight_layout()
    output_path = output_dir / 'timecourse_by_genotype.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_genotype_overlay_timecourse(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create overlay time course plot comparing genotypes."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette for genotypes
    genotype_colors = {
        'Httr WT': '#2ecc71',  # Green
        'HttrB KO': '#e74c3c',  # Red
    }
    
    for genotype, chips in chips_by_genotype.items():
        # Collect all firing rates
        all_rates = []
        max_bins = 0
        
        for chip_data in chips:
            for unit_tc in chip_data.units:
                if unit_tc.firing_rates is not None:
                    all_rates.append(unit_tc.firing_rates)
                    max_bins = max(max_bins, len(unit_tc.firing_rates))
        
        if not all_rates:
            continue
        
        # Pad to same length
        padded_rates = []
        for rates in all_rates:
            if len(rates) < max_bins:
                padded = np.pad(rates, (0, max_bins - len(rates)), constant_values=np.nan)
            else:
                padded = rates[:max_bins]
            padded_rates.append(padded)
        
        padded_rates = np.array(padded_rates)
        
        # Calculate mean and SEM
        mean_rate = np.nanmean(padded_rates, axis=0)
        sem_rate = np.nanstd(padded_rates, axis=0) / np.sqrt(np.sum(~np.isnan(padded_rates), axis=0))
        
        # Time axis
        time_min = np.arange(max_bins) * (BIN_SIZE_SEC / 60)
        
        # Get color
        color = genotype_colors.get(genotype, '#3498db')
        
        # Plot
        ax.plot(time_min, mean_rate, linewidth=2.5, color=color, 
                label=f'{genotype} (n={len(all_rates)})')
        ax.fill_between(time_min, mean_rate - sem_rate, mean_rate + sem_rate, 
                       alpha=0.25, color=color)
    
    # Mark agonist start
    ax.axvline(x=AGONIST_START_MIN, color='red', linestyle='--', linewidth=2.5, 
               label='Agonist start (5 min)')
    
    # Mark recording boundaries
    for i in range(1, 3):
        boundary = 10.0 * i
        ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.5)
    
    # Shade regions
    ax.axvspan(0, AGONIST_START_MIN, alpha=0.08, color='blue', label='No agonist')
    ax.axvspan(AGONIST_START_MIN, 30, alpha=0.08, color='red', label='Agonist present')
    
    ax.set_xlabel('Time (minutes)', fontsize=14)
    ax.set_ylabel('Mean Firing Rate (Hz)', fontsize=14)
    ax.set_title('Firing Rate Time Course: Genotype Comparison', fontsize=16)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    
    plt.tight_layout()
    output_path = output_dir / 'genotype_overlay_timecourse.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_baseline_vs_agonist_comparison(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create bar plot comparing baseline vs agonist firing rates."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    genotypes = list(chips_by_genotype.keys())
    x = np.arange(len(genotypes))
    width = 0.35
    
    # Calculate mean rates for each genotype
    baseline_means = []
    baseline_sems = []
    agonist_means = []
    agonist_sems = []
    n_units = []
    
    for genotype in genotypes:
        chips = chips_by_genotype[genotype]
        
        baseline_rates = []
        agonist_rates = []
        
        for chip_data in chips:
            for unit_tc in chip_data.units:
                if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                    continue
                
                time_min = unit_tc.time_bins_min
                rates = unit_tc.firing_rates
                
                # Baseline: 0-5 min
                baseline_mask = time_min < AGONIST_START_MIN
                if np.any(baseline_mask):
                    baseline_rates.append(np.mean(rates[baseline_mask]))
                
                # Agonist: 5-30 min
                agonist_mask = time_min >= AGONIST_START_MIN
                if np.any(agonist_mask):
                    agonist_rates.append(np.mean(rates[agonist_mask]))
        
        baseline_rates = np.array(baseline_rates)
        agonist_rates = np.array(agonist_rates)
        
        baseline_means.append(np.mean(baseline_rates) if len(baseline_rates) > 0 else 0)
        baseline_sems.append(np.std(baseline_rates) / np.sqrt(len(baseline_rates)) if len(baseline_rates) > 0 else 0)
        agonist_means.append(np.mean(agonist_rates) if len(agonist_rates) > 0 else 0)
        agonist_sems.append(np.std(agonist_rates) / np.sqrt(len(agonist_rates)) if len(agonist_rates) > 0 else 0)
        n_units.append(len(baseline_rates))
    
    # Plot 1: Grouped bar plot
    ax = axes[0]
    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_sems, 
                   label='Baseline (0-5 min)', color='#3498db', capsize=5)
    bars2 = ax.bar(x + width/2, agonist_means, width, yerr=agonist_sems,
                   label='Agonist (5-30 min)', color='#e74c3c', capsize=5)
    
    ax.set_xlabel('Genotype', fontsize=12)
    ax.set_ylabel('Mean Firing Rate (Hz)', fontsize=12)
    ax.set_title('Baseline vs Agonist Period', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}\n(n={n})' for g, n in zip(genotypes, n_units)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Percent change
    ax = axes[1]
    percent_changes = []
    for bm, am in zip(baseline_means, agonist_means):
        if bm > 0:
            percent_changes.append(((am - bm) / bm) * 100)
        else:
            percent_changes.append(0)
    
    colors = ['#2ecc71' if pc > 0 else '#e74c3c' for pc in percent_changes]
    bars = ax.bar(x, percent_changes, color=colors)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Genotype', fontsize=12)
    ax.set_ylabel('% Change from Baseline', fontsize=12)
    ax.set_title('Effect of Agonist on Firing Rate', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}\n(n={n})' for g, n in zip(genotypes, n_units)])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, pc in zip(bars, percent_changes):
        height = bar.get_height()
        ax.annotate(f'{pc:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'baseline_vs_agonist.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_individual_chip_traces(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create individual chip traces."""
    
    all_chips = []
    for chips in chips_by_genotype.values():
        all_chips.extend(chips)
    
    n_chips = len(all_chips)
    fig, axes = plt.subplots(n_chips, 1, figsize=(14, 4 * n_chips), squeeze=False)
    
    for i, chip_data in enumerate(all_chips):
        ax = axes[i, 0]
        
        # Collect all unit traces
        all_rates = []
        max_bins = 0
        
        for unit_tc in chip_data.units:
            if unit_tc.firing_rates is not None:
                all_rates.append(unit_tc.firing_rates)
                max_bins = max(max_bins, len(unit_tc.firing_rates))
        
        if not all_rates:
            continue
        
        # Pad and compute mean
        padded_rates = []
        for rates in all_rates:
            if len(rates) < max_bins:
                padded = np.pad(rates, (0, max_bins - len(rates)), constant_values=np.nan)
            else:
                padded = rates[:max_bins]
            padded_rates.append(padded)
        
        padded_rates = np.array(padded_rates)
        mean_rate = np.nanmean(padded_rates, axis=0)
        sem_rate = np.nanstd(padded_rates, axis=0) / np.sqrt(np.sum(~np.isnan(padded_rates), axis=0))
        
        time_min = np.arange(max_bins) * (BIN_SIZE_SEC / 60)
        
        # Plot individual traces (light gray)
        for rates in padded_rates:
            ax.plot(time_min, rates, 'gray', alpha=0.2, linewidth=0.5)
        
        # Plot mean
        ax.plot(time_min, mean_rate, 'k-', linewidth=2, label='Mean')
        ax.fill_between(time_min, mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.3, color='blue')
        
        # Mark agonist
        ax.axvline(x=AGONIST_START_MIN, color='red', linestyle='--', linewidth=2)
        ax.axvspan(AGONIST_START_MIN, time_min.max(), alpha=0.1, color='red')
        
        ax.set_xlabel('Time (minutes)', fontsize=10)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
        ax.set_title(f'Chip {chip_data.chip} - {chip_data.genotype} (n={len(all_rates)} units)', fontsize=12)
        ax.set_xlim(0, time_min.max())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'individual_chip_traces.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_summary_table(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create summary statistics table."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Collect statistics
    table_data = []
    columns = ['Genotype', 'N Chips', 'N Units', 
               'Baseline FR (Hz)', 'Agonist FR (Hz)', '% Change', 'p-value']
    
    from scipy import stats
    
    for genotype, chips in chips_by_genotype.items():
        baseline_rates = []
        agonist_rates = []
        
        for chip_data in chips:
            for unit_tc in chip_data.units:
                if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                    continue
                
                time_min = unit_tc.time_bins_min
                rates = unit_tc.firing_rates
                
                baseline_mask = time_min < AGONIST_START_MIN
                agonist_mask = time_min >= AGONIST_START_MIN
                
                if np.any(baseline_mask) and np.any(agonist_mask):
                    baseline_rates.append(np.mean(rates[baseline_mask]))
                    agonist_rates.append(np.mean(rates[agonist_mask]))
        
        baseline_rates = np.array(baseline_rates)
        agonist_rates = np.array(agonist_rates)
        
        if len(baseline_rates) > 0:
            baseline_mean = np.mean(baseline_rates)
            baseline_sem = np.std(baseline_rates) / np.sqrt(len(baseline_rates))
            agonist_mean = np.mean(agonist_rates)
            agonist_sem = np.std(agonist_rates) / np.sqrt(len(agonist_rates))
            
            if baseline_mean > 0:
                pct_change = ((agonist_mean - baseline_mean) / baseline_mean) * 100
            else:
                pct_change = 0
            
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(baseline_rates, agonist_rates)
            
            table_data.append([
                genotype,
                len(chips),
                len(baseline_rates),
                f'{baseline_mean:.2f} ± {baseline_sem:.2f}',
                f'{agonist_mean:.2f} ± {agonist_sem:.2f}',
                f'{pct_change:+.1f}%',
                f'{p_val:.4f}' if p_val >= 0.0001 else '<0.0001'
            ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4a90d9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Summary Statistics: Middle Recordings Agonist Effect', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'summary_table.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


# =============================================================================
# Cell Type Separated Plots
# =============================================================================

def create_timecourse_by_cell_type(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create time course plot with subplots for each cell type."""
    
    genotypes = list(chips_by_genotype.keys())
    n_genotypes = len(genotypes)
    cell_types = ['ON', 'OFF', 'ON_OFF']  # Exclude unknown for clarity
    n_cell_types = len(cell_types)
    
    fig, axes = plt.subplots(n_genotypes, n_cell_types, figsize=(6 * n_cell_types, 5 * n_genotypes))
    if n_genotypes == 1:
        axes = axes.reshape(1, -1)
    
    genotype_colors = {
        'Httr WT': '#2ecc71',
        'HttrB KO': '#e74c3c',
    }
    
    for row, genotype in enumerate(genotypes):
        chips = chips_by_genotype[genotype]
        color = genotype_colors.get(genotype, '#3498db')
        
        for col, cell_type in enumerate(cell_types):
            ax = axes[row, col]
            
            # Collect firing rates for this cell type
            all_rates = []
            max_bins = 0
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type == cell_type and unit_tc.firing_rates is not None:
                        all_rates.append(unit_tc.firing_rates)
                        max_bins = max(max_bins, len(unit_tc.firing_rates))
            
            if not all_rates or max_bins == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{genotype} - {cell_type} (n=0)', fontsize=12)
                continue
            
            # Pad to same length
            padded_rates = []
            for rates in all_rates:
                if len(rates) < max_bins:
                    padded = np.pad(rates, (0, max_bins - len(rates)), constant_values=np.nan)
                else:
                    padded = rates[:max_bins]
                padded_rates.append(padded)
            
            padded_rates = np.array(padded_rates)
            
            # Calculate mean and SEM
            mean_rate = np.nanmean(padded_rates, axis=0)
            sem_rate = np.nanstd(padded_rates, axis=0) / np.sqrt(np.sum(~np.isnan(padded_rates), axis=0))
            
            # Time axis
            time_min = np.arange(max_bins) * (BIN_SIZE_SEC / 60)
            
            # Plot
            ax.plot(time_min, mean_rate, color=color, linewidth=2)
            ax.fill_between(time_min, mean_rate - sem_rate, mean_rate + sem_rate, 
                           alpha=0.3, color=color)
            
            # Mark agonist start
            ax.axvline(x=AGONIST_START_MIN, color='red', linestyle='--', linewidth=1.5)
            
            # Shade regions
            ax.axvspan(0, AGONIST_START_MIN, alpha=0.05, color='blue')
            ax.axvspan(AGONIST_START_MIN, time_min.max() if len(time_min) > 0 else 30, 
                      alpha=0.05, color='red')
            
            ax.set_xlabel('Time (min)', fontsize=10)
            ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
            ax.set_title(f'{genotype} - {cell_type} (n={len(all_rates)})', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 30)
    
    plt.tight_layout()
    output_path = output_dir / 'timecourse_by_cell_type.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_cell_type_overlay(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create overlay plot comparing cell types within each genotype."""
    
    genotypes = list(chips_by_genotype.keys())
    n_genotypes = len(genotypes)
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    fig, axes = plt.subplots(1, n_genotypes, figsize=(8 * n_genotypes, 6))
    if n_genotypes == 1:
        axes = [axes]
    
    for ax_idx, genotype in enumerate(genotypes):
        ax = axes[ax_idx]
        chips = chips_by_genotype[genotype]
        
        for cell_type in cell_types:
            color = CELL_TYPE_COLORS.get(cell_type, '#95a5a6')
            
            # Collect firing rates
            all_rates = []
            max_bins = 0
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type == cell_type and unit_tc.firing_rates is not None:
                        all_rates.append(unit_tc.firing_rates)
                        max_bins = max(max_bins, len(unit_tc.firing_rates))
            
            if not all_rates or max_bins == 0:
                continue
            
            # Pad and calculate stats
            padded_rates = []
            for rates in all_rates:
                if len(rates) < max_bins:
                    padded = np.pad(rates, (0, max_bins - len(rates)), constant_values=np.nan)
                else:
                    padded = rates[:max_bins]
                padded_rates.append(padded)
            
            padded_rates = np.array(padded_rates)
            mean_rate = np.nanmean(padded_rates, axis=0)
            sem_rate = np.nanstd(padded_rates, axis=0) / np.sqrt(np.sum(~np.isnan(padded_rates), axis=0))
            
            time_min = np.arange(max_bins) * (BIN_SIZE_SEC / 60)
            
            ax.plot(time_min, mean_rate, color=color, linewidth=2, 
                   label=f'{cell_type} (n={len(all_rates)})')
            ax.fill_between(time_min, mean_rate - sem_rate, mean_rate + sem_rate, 
                           alpha=0.2, color=color)
        
        # Mark agonist
        ax.axvline(x=AGONIST_START_MIN, color='black', linestyle='--', linewidth=2)
        ax.axvspan(0, AGONIST_START_MIN, alpha=0.05, color='blue')
        ax.axvspan(AGONIST_START_MIN, 30, alpha=0.05, color='red')
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax.set_title(f'{genotype}: Cell Type Comparison', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)
    
    plt.tight_layout()
    output_path = output_dir / 'cell_type_overlay.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_cell_type_bar_comparison(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create bar plot comparing baseline vs agonist by cell type."""
    from scipy import stats
    
    genotypes = list(chips_by_genotype.keys())
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    fig, axes = plt.subplots(1, len(genotypes), figsize=(8 * len(genotypes), 6))
    if len(genotypes) == 1:
        axes = [axes]
    
    for ax_idx, genotype in enumerate(genotypes):
        ax = axes[ax_idx]
        chips = chips_by_genotype[genotype]
        
        x = np.arange(len(cell_types))
        width = 0.35
        
        baseline_means = []
        baseline_sems = []
        agonist_means = []
        agonist_sems = []
        n_units_list = []
        p_values = []
        
        for cell_type in cell_types:
            baseline_rates = []
            agonist_rates = []
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type != cell_type:
                        continue
                    if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                        continue
                    
                    time_min = unit_tc.time_bins_min
                    rates = unit_tc.firing_rates
                    
                    baseline_mask = time_min < AGONIST_START_MIN
                    agonist_mask = time_min >= AGONIST_START_MIN
                    
                    if np.any(baseline_mask) and np.any(agonist_mask):
                        baseline_rates.append(np.mean(rates[baseline_mask]))
                        agonist_rates.append(np.mean(rates[agonist_mask]))
            
            baseline_rates = np.array(baseline_rates)
            agonist_rates = np.array(agonist_rates)
            
            if len(baseline_rates) > 0:
                baseline_means.append(np.mean(baseline_rates))
                baseline_sems.append(np.std(baseline_rates) / np.sqrt(len(baseline_rates)))
                agonist_means.append(np.mean(agonist_rates))
                agonist_sems.append(np.std(agonist_rates) / np.sqrt(len(agonist_rates)))
                n_units_list.append(len(baseline_rates))
                
                if len(baseline_rates) > 1:
                    _, p_val = stats.ttest_rel(baseline_rates, agonist_rates)
                    p_values.append(p_val)
                else:
                    p_values.append(1.0)
            else:
                baseline_means.append(0)
                baseline_sems.append(0)
                agonist_means.append(0)
                agonist_sems.append(0)
                n_units_list.append(0)
                p_values.append(1.0)
        
        # Plot bars
        bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_sems,
                      label='Baseline (0-5 min)', color='#3498db', capsize=4)
        bars2 = ax.bar(x + width/2, agonist_means, width, yerr=agonist_sems,
                      label='Agonist (5-30 min)', color='#e74c3c', capsize=4)
        
        # Add significance markers
        for i, p_val in enumerate(p_values):
            max_y = max(baseline_means[i] + baseline_sems[i], 
                       agonist_means[i] + agonist_sems[i])
            if p_val < 0.001:
                ax.text(i, max_y + 1, '***', ha='center', fontsize=12)
            elif p_val < 0.01:
                ax.text(i, max_y + 1, '**', ha='center', fontsize=12)
            elif p_val < 0.05:
                ax.text(i, max_y + 1, '*', ha='center', fontsize=12)
        
        ax.set_xlabel('Cell Type', fontsize=12)
        ax.set_ylabel('Mean Firing Rate (Hz)', fontsize=12)
        ax.set_title(f'{genotype}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{ct}\n(n={n})' for ct, n in zip(cell_types, n_units_list)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Baseline vs Agonist: Cell Type Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'cell_type_bar_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_cell_type_summary_table(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create summary table by cell type."""
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    
    columns = ['Genotype', 'Cell Type', 'N Units', 'Baseline (Hz)', 'Agonist (Hz)', '% Change', 'p-value']
    table_data = []
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    for genotype, chips in chips_by_genotype.items():
        for cell_type in cell_types:
            baseline_rates = []
            agonist_rates = []
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type != cell_type:
                        continue
                    if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                        continue
                    
                    time_min = unit_tc.time_bins_min
                    rates = unit_tc.firing_rates
                    
                    baseline_mask = time_min < AGONIST_START_MIN
                    agonist_mask = time_min >= AGONIST_START_MIN
                    
                    if np.any(baseline_mask) and np.any(agonist_mask):
                        baseline_rates.append(np.mean(rates[baseline_mask]))
                        agonist_rates.append(np.mean(rates[agonist_mask]))
            
            if len(baseline_rates) > 0:
                baseline_rates = np.array(baseline_rates)
                agonist_rates = np.array(agonist_rates)
                
                baseline_mean = np.mean(baseline_rates)
                baseline_sem = np.std(baseline_rates) / np.sqrt(len(baseline_rates))
                agonist_mean = np.mean(agonist_rates)
                agonist_sem = np.std(agonist_rates) / np.sqrt(len(agonist_rates))
                
                pct_change = ((agonist_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
                
                if len(baseline_rates) > 1:
                    _, p_val = stats.ttest_rel(baseline_rates, agonist_rates)
                    p_str = f'{p_val:.4f}' if p_val >= 0.0001 else '<0.0001'
                else:
                    p_str = 'N/A'
                
                table_data.append([
                    genotype, cell_type, len(baseline_rates),
                    f'{baseline_mean:.2f} ± {baseline_sem:.2f}',
                    f'{agonist_mean:.2f} ± {agonist_sem:.2f}',
                    f'{pct_change:+.1f}%', p_str
                ])
    
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        for j in range(len(columns)):
            table[(0, j)].set_facecolor('#4a90d9')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('Summary Statistics by Cell Type', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'cell_type_summary_table.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


# =============================================================================
# Genotype Comparison Plots
# =============================================================================

def create_genotype_change_comparison(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create bar plot comparing WT vs KO percent change by cell type.
    
    Uses group mean calculation (mean_agonist - mean_baseline) / mean_baseline
    to avoid extreme values from individual cells with low baseline.
    """
    from scipy import stats
    
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax_idx, cell_type in enumerate(cell_types):
        ax = axes[ax_idx]
        
        genotype_data = {}
        
        for genotype in GENOTYPE_ORDER:
            if genotype not in chips_by_genotype:
                continue
            chips = chips_by_genotype[genotype]
            
            baseline_rates = []
            agonist_rates = []
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type != cell_type:
                        continue
                    if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                        continue
                    
                    time_min = unit_tc.time_bins_min
                    rates = unit_tc.firing_rates
                    
                    baseline_mask = time_min < AGONIST_START_MIN
                    agonist_mask = time_min >= AGONIST_START_MIN
                    
                    if np.any(baseline_mask) and np.any(agonist_mask):
                        base_rate = np.mean(rates[baseline_mask])
                        ago_rate = np.mean(rates[agonist_mask])
                        baseline_rates.append(base_rate)
                        agonist_rates.append(ago_rate)
            
            genotype_data[genotype] = {
                'baseline': np.array(baseline_rates),
                'agonist': np.array(agonist_rates),
            }
        
        # Plot bars - use group mean calculation
        x = np.arange(len(GENOTYPE_ORDER))
        width = 0.6
        
        pct_changes = []
        sems = []
        n_units = []
        colors = []
        
        for genotype in GENOTYPE_ORDER:
            if genotype in genotype_data:
                base = genotype_data[genotype]['baseline']
                ago = genotype_data[genotype]['agonist']
                
                if len(base) > 0 and np.mean(base) > 0:
                    # Group mean calculation
                    pct_change = ((np.mean(ago) - np.mean(base)) / np.mean(base)) * 100
                    
                    # Bootstrap SEM for percent change
                    n_boot = 1000
                    boot_pcts = []
                    for _ in range(n_boot):
                        idx = np.random.choice(len(base), len(base), replace=True)
                        boot_base = base[idx]
                        boot_ago = ago[idx]
                        if np.mean(boot_base) > 0:
                            boot_pcts.append(((np.mean(boot_ago) - np.mean(boot_base)) / np.mean(boot_base)) * 100)
                    sem = np.std(boot_pcts) if boot_pcts else 0
                else:
                    pct_change = 0
                    sem = 0
                
                pct_changes.append(pct_change)
                sems.append(sem)
                n_units.append(len(base))
                colors.append(GENOTYPE_COLORS.get(genotype, '#3498db'))
            else:
                pct_changes.append(0)
                sems.append(0)
                n_units.append(0)
                colors.append('#3498db')
        
        bars = ax.bar(x, pct_changes, width, yerr=sems, color=colors, capsize=5, edgecolor='black')
        
        # Add significance test between genotypes (using individual differences)
        if len(GENOTYPE_ORDER) >= 2:
            g1, g2 = GENOTYPE_ORDER[0], GENOTYPE_ORDER[1]
            if g1 in genotype_data and g2 in genotype_data:
                base1, ago1 = genotype_data[g1]['baseline'], genotype_data[g1]['agonist']
                base2, ago2 = genotype_data[g2]['baseline'], genotype_data[g2]['agonist']
                
                # Calculate per-unit changes
                diff1 = ago1 - base1
                diff2 = ago2 - base2
                
                if len(diff1) > 1 and len(diff2) > 1:
                    _, p_val = stats.ttest_ind(diff1, diff2)
                    max_y = max(pct_changes[0] + sems[0], pct_changes[1] + sems[1])
                    if p_val < 0.001:
                        sig_text = '***'
                    elif p_val < 0.01:
                        sig_text = '**'
                    elif p_val < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    # Draw significance bracket
                    y_bracket = max_y * 1.1 + 5
                    ax.plot([0, 0, 1, 1], [y_bracket-2, y_bracket, y_bracket, y_bracket-2], 'k-', lw=1)
                    ax.text(0.5, y_bracket + 2, sig_text, ha='center', fontsize=12)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('')
        ax.set_ylabel('% Change from Baseline', fontsize=11)
        ax.set_title(f'{cell_type} Cells', fontsize=14, fontweight='bold', 
                    color=CELL_TYPE_COLORS.get(cell_type, 'black'))
        ax.set_xticks(x)
        ax.set_xticklabels([f'{g}\n(n={n})' for g, n in zip(GENOTYPE_ORDER, n_units)], fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, pct in zip(bars, pct_changes):
            height = bar.get_height()
            ax.annotate(f'{pct:+.0f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height >= 0 else -12),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=11, fontweight='bold')
    
    plt.suptitle('Agonist Effect: WT vs KO Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'genotype_change_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_genotype_difference_heatmap(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create heatmap showing WT vs KO differences across metrics."""
    from scipy import stats
    
    cell_types = ['ON', 'OFF', 'ON_OFF']
    metrics = ['Baseline (Hz)', 'Agonist (Hz)', '% Change']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect data
    data_matrix = np.zeros((len(cell_types), len(metrics) * 2 + 1))  # WT, KO, diff for each metric
    labels_matrix = []
    
    for i, cell_type in enumerate(cell_types):
        row_labels = []
        for j, metric in enumerate(metrics):
            for g_idx, genotype in enumerate(GENOTYPE_ORDER):
                if genotype not in chips_by_genotype:
                    continue
                chips = chips_by_genotype[genotype]
                
                baseline_rates = []
                agonist_rates = []
                
                for chip_data in chips:
                    for unit_tc in chip_data.units:
                        if unit_tc.cell_type != cell_type:
                            continue
                        if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                            continue
                        
                        time_min = unit_tc.time_bins_min
                        rates = unit_tc.firing_rates
                        
                        baseline_mask = time_min < AGONIST_START_MIN
                        agonist_mask = time_min >= AGONIST_START_MIN
                        
                        if np.any(baseline_mask) and np.any(agonist_mask):
                            baseline_rates.append(np.mean(rates[baseline_mask]))
                            agonist_rates.append(np.mean(rates[agonist_mask]))
                
                if len(baseline_rates) > 0:
                    if metric == 'Baseline (Hz)':
                        val = np.mean(baseline_rates)
                    elif metric == 'Agonist (Hz)':
                        val = np.mean(agonist_rates)
                    else:  # % Change
                        base_mean = np.mean(baseline_rates)
                        ago_mean = np.mean(agonist_rates)
                        val = ((ago_mean - base_mean) / base_mean * 100) if base_mean > 0 else 0
                    
                    data_matrix[i, j * 2 + g_idx] = val
        
        labels_matrix.append(row_labels)
    
    # Create summary table instead of heatmap for clearer comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    columns = ['Cell Type', 'WT Baseline', 'KO Baseline', 'WT Agonist', 'KO Agonist', 
               'WT % Change', 'KO % Change', 'p-value (WT vs KO)']
    table_data = []
    
    for cell_type in cell_types:
        row = [cell_type]
        
        genotype_results = {}
        for genotype in GENOTYPE_ORDER:
            if genotype not in chips_by_genotype:
                continue
            chips = chips_by_genotype[genotype]
            
            baseline_rates = []
            agonist_rates = []
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type != cell_type:
                        continue
                    if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                        continue
                    
                    time_min = unit_tc.time_bins_min
                    rates = unit_tc.firing_rates
                    
                    baseline_mask = time_min < AGONIST_START_MIN
                    agonist_mask = time_min >= AGONIST_START_MIN
                    
                    if np.any(baseline_mask) and np.any(agonist_mask):
                        baseline_rates.append(np.mean(rates[baseline_mask]))
                        agonist_rates.append(np.mean(rates[agonist_mask]))
            
            genotype_results[genotype] = {
                'baseline': np.array(baseline_rates),
                'agonist': np.array(agonist_rates),
            }
        
        # Add values to row
        for genotype in GENOTYPE_ORDER:
            if genotype in genotype_results:
                base = genotype_results[genotype]['baseline']
                row.append(f'{np.mean(base):.1f}' if len(base) > 0 else 'N/A')
            else:
                row.append('N/A')
        
        for genotype in GENOTYPE_ORDER:
            if genotype in genotype_results:
                ago = genotype_results[genotype]['agonist']
                row.append(f'{np.mean(ago):.1f}' if len(ago) > 0 else 'N/A')
            else:
                row.append('N/A')
        
        for genotype in GENOTYPE_ORDER:
            if genotype in genotype_results:
                base = genotype_results[genotype]['baseline']
                ago = genotype_results[genotype]['agonist']
                if len(base) > 0 and np.mean(base) > 0:
                    pct = ((np.mean(ago) - np.mean(base)) / np.mean(base)) * 100
                    row.append(f'{pct:+.0f}%')
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        
        # Calculate p-value for WT vs KO percent change
        if len(GENOTYPE_ORDER) >= 2:
            g1, g2 = GENOTYPE_ORDER[0], GENOTYPE_ORDER[1]
            if g1 in genotype_results and g2 in genotype_results:
                base1 = genotype_results[g1]['baseline']
                ago1 = genotype_results[g1]['agonist']
                base2 = genotype_results[g2]['baseline']
                ago2 = genotype_results[g2]['agonist']
                
                if len(base1) > 1 and len(base2) > 1:
                    pct1 = [(a - b) / b * 100 for a, b in zip(ago1, base1) if b > 0]
                    pct2 = [(a - b) / b * 100 for a, b in zip(ago2, base2) if b > 0]
                    if len(pct1) > 1 and len(pct2) > 1:
                        _, p_val = stats.ttest_ind(pct1, pct2)
                        row.append(f'{p_val:.4f}' if p_val >= 0.0001 else '<0.0001')
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            else:
                row.append('N/A')
        else:
            row.append('N/A')
        
        table_data.append(row)
    
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4a90d9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color cell type column
    for i in range(len(cell_types)):
        table[(i+1, 0)].set_text_props(fontweight='bold', 
                                       color=CELL_TYPE_COLORS.get(cell_types[i], 'black'))
    
    ax.set_title('WT vs KO Direct Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'genotype_comparison_table.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_combined_timecourse_comparison(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create combined time course with WT and KO overlaid for each cell type."""
    
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax_idx, cell_type in enumerate(cell_types):
        ax = axes[ax_idx]
        
        for genotype in GENOTYPE_ORDER:
            if genotype not in chips_by_genotype:
                continue
            chips = chips_by_genotype[genotype]
            color = GENOTYPE_COLORS.get(genotype, '#3498db')
            
            # Collect firing rates
            all_rates = []
            max_bins = 0
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type == cell_type and unit_tc.firing_rates is not None:
                        all_rates.append(unit_tc.firing_rates)
                        max_bins = max(max_bins, len(unit_tc.firing_rates))
            
            if not all_rates or max_bins == 0:
                continue
            
            # Pad and calculate stats
            padded_rates = []
            for rates in all_rates:
                if len(rates) < max_bins:
                    padded = np.pad(rates, (0, max_bins - len(rates)), constant_values=np.nan)
                else:
                    padded = rates[:max_bins]
                padded_rates.append(padded)
            
            padded_rates = np.array(padded_rates)
            mean_rate = np.nanmean(padded_rates, axis=0)
            sem_rate = np.nanstd(padded_rates, axis=0) / np.sqrt(np.sum(~np.isnan(padded_rates), axis=0))
            
            time_min = np.arange(max_bins) * (BIN_SIZE_SEC / 60)
            
            ax.plot(time_min, mean_rate, color=color, linewidth=2.5, 
                   label=f'{genotype} (n={len(all_rates)})')
            ax.fill_between(time_min, mean_rate - sem_rate, mean_rate + sem_rate, 
                           alpha=0.2, color=color)
        
        # Mark agonist
        ax.axvline(x=AGONIST_START_MIN, color='black', linestyle='--', linewidth=2)
        ax.axvspan(0, AGONIST_START_MIN, alpha=0.05, color='blue')
        ax.axvspan(AGONIST_START_MIN, 30, alpha=0.05, color='red')
        
        ax.set_xlabel('Time (minutes)', fontsize=12)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
        ax.set_title(f'{cell_type} Cells', fontsize=14, fontweight='bold',
                    color=CELL_TYPE_COLORS.get(cell_type, 'black'))
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)
    
    plt.suptitle('WT vs KO Time Course Comparison by Cell Type', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'wt_ko_timecourse_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_response_magnitude_scatter(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create scatter plot of baseline vs agonist response colored by genotype."""
    
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax_idx, cell_type in enumerate(cell_types):
        ax = axes[ax_idx]
        
        for genotype in GENOTYPE_ORDER:
            if genotype not in chips_by_genotype:
                continue
            chips = chips_by_genotype[genotype]
            color = GENOTYPE_COLORS.get(genotype, '#3498db')
            
            baseline_all = []
            agonist_all = []
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type != cell_type:
                        continue
                    if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                        continue
                    
                    time_min = unit_tc.time_bins_min
                    rates = unit_tc.firing_rates
                    
                    baseline_mask = time_min < AGONIST_START_MIN
                    agonist_mask = time_min >= AGONIST_START_MIN
                    
                    if np.any(baseline_mask) and np.any(agonist_mask):
                        baseline_all.append(np.mean(rates[baseline_mask]))
                        agonist_all.append(np.mean(rates[agonist_mask]))
            
            if baseline_all:
                ax.scatter(baseline_all, agonist_all, c=color, alpha=0.5, s=30, 
                          label=f'{genotype} (n={len(baseline_all)})', edgecolors='white', linewidths=0.5)
        
        # Add diagonal line (no change)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='No change')
        
        ax.set_xlabel('Baseline Firing Rate (Hz)', fontsize=11)
        ax.set_ylabel('Agonist Firing Rate (Hz)', fontsize=11)
        ax.set_title(f'{cell_type} Cells', fontsize=14, fontweight='bold',
                    color=CELL_TYPE_COLORS.get(cell_type, 'black'))
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle('Baseline vs Agonist Response (Each Dot = 1 Unit)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'baseline_vs_agonist_scatter.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def create_effect_size_comparison(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Create Cohen's d effect size comparison between genotypes."""
    from scipy import stats
    
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(cell_types))
    width = 0.35
    
    effect_sizes = {g: [] for g in GENOTYPE_ORDER}
    
    for cell_type in cell_types:
        for genotype in GENOTYPE_ORDER:
            if genotype not in chips_by_genotype:
                effect_sizes[genotype].append(0)
                continue
            chips = chips_by_genotype[genotype]
            
            baseline_rates = []
            agonist_rates = []
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type != cell_type:
                        continue
                    if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                        continue
                    
                    time_min = unit_tc.time_bins_min
                    rates = unit_tc.firing_rates
                    
                    baseline_mask = time_min < AGONIST_START_MIN
                    agonist_mask = time_min >= AGONIST_START_MIN
                    
                    if np.any(baseline_mask) and np.any(agonist_mask):
                        baseline_rates.append(np.mean(rates[baseline_mask]))
                        agonist_rates.append(np.mean(rates[agonist_mask]))
            
            if len(baseline_rates) > 1:
                # Cohen's d
                diff = np.mean(agonist_rates) - np.mean(baseline_rates)
                pooled_std = np.sqrt((np.var(baseline_rates) + np.var(agonist_rates)) / 2)
                if pooled_std > 0:
                    cohens_d = diff / pooled_std
                else:
                    cohens_d = 0
                effect_sizes[genotype].append(cohens_d)
            else:
                effect_sizes[genotype].append(0)
    
    # Plot grouped bars
    for i, genotype in enumerate(GENOTYPE_ORDER):
        color = GENOTYPE_COLORS.get(genotype, '#3498db')
        bars = ax.bar(x + (i - 0.5) * width, effect_sizes[genotype], width, 
                     label=genotype, color=color, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=-0.8, color='gray', linestyle=':', alpha=0.5)
    ax.text(2.6, 0.85, 'Large effect', fontsize=9, color='gray')
    
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title('Effect Size of Agonist on Firing Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types, fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'effect_size_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


# =============================================================================
# Statistical Summary
# =============================================================================

def generate_statistical_summary(
    chips_by_genotype: Dict[str, List[ChipData]],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generate a markdown file summarizing all statistical results."""
    from scipy import stats
    from datetime import datetime
    
    cell_types = ['ON', 'OFF', 'ON_OFF']
    
    # Collect all statistics
    stats_data = {}
    
    for cell_type in cell_types:
        stats_data[cell_type] = {}
        
        for genotype in GENOTYPE_ORDER:
            if genotype not in chips_by_genotype:
                continue
            chips = chips_by_genotype[genotype]
            
            baseline_rates = []
            agonist_rates = []
            
            for chip_data in chips:
                for unit_tc in chip_data.units:
                    if unit_tc.cell_type != cell_type:
                        continue
                    if unit_tc.firing_rates is None or unit_tc.time_bins_min is None:
                        continue
                    
                    time_min = unit_tc.time_bins_min
                    rates = unit_tc.firing_rates
                    
                    baseline_mask = time_min < AGONIST_START_MIN
                    agonist_mask = time_min >= AGONIST_START_MIN
                    
                    if np.any(baseline_mask) and np.any(agonist_mask):
                        baseline_rates.append(np.mean(rates[baseline_mask]))
                        agonist_rates.append(np.mean(rates[agonist_mask]))
            
            baseline_rates = np.array(baseline_rates)
            agonist_rates = np.array(agonist_rates)
            
            if len(baseline_rates) > 0:
                # Paired t-test for within-genotype effect
                if len(baseline_rates) > 1:
                    t_stat, p_paired = stats.ttest_rel(baseline_rates, agonist_rates)
                else:
                    t_stat, p_paired = np.nan, np.nan
                
                # Cohen's d effect size
                diff = np.mean(agonist_rates) - np.mean(baseline_rates)
                pooled_std = np.sqrt((np.var(baseline_rates) + np.var(agonist_rates)) / 2)
                cohens_d = diff / pooled_std if pooled_std > 0 else 0
                
                # Percent change
                pct_change = (diff / np.mean(baseline_rates) * 100) if np.mean(baseline_rates) > 0 else 0
                
                stats_data[cell_type][genotype] = {
                    'n': len(baseline_rates),
                    'baseline_mean': np.mean(baseline_rates),
                    'baseline_std': np.std(baseline_rates),
                    'baseline_sem': np.std(baseline_rates) / np.sqrt(len(baseline_rates)),
                    'agonist_mean': np.mean(agonist_rates),
                    'agonist_std': np.std(agonist_rates),
                    'agonist_sem': np.std(agonist_rates) / np.sqrt(len(agonist_rates)),
                    'pct_change': pct_change,
                    'cohens_d': cohens_d,
                    'p_paired': p_paired,
                    'baseline_rates': baseline_rates,
                    'agonist_rates': agonist_rates,
                }
    
    # Calculate WT vs KO comparisons
    wt_ko_comparisons = {}
    for cell_type in cell_types:
        if len(GENOTYPE_ORDER) >= 2:
            g1, g2 = GENOTYPE_ORDER[0], GENOTYPE_ORDER[1]
            if g1 in stats_data[cell_type] and g2 in stats_data[cell_type]:
                # Compare absolute change (agonist - baseline) between genotypes
                diff1 = stats_data[cell_type][g1]['agonist_rates'] - stats_data[cell_type][g1]['baseline_rates']
                diff2 = stats_data[cell_type][g2]['agonist_rates'] - stats_data[cell_type][g2]['baseline_rates']
                
                if len(diff1) > 1 and len(diff2) > 1:
                    _, p_ttest = stats.ttest_ind(diff1, diff2)
                    _, p_mannwhitney = stats.mannwhitneyu(diff1, diff2, alternative='two-sided')
                else:
                    p_ttest, p_mannwhitney = np.nan, np.nan
                
                wt_ko_comparisons[cell_type] = {
                    'p_ttest': p_ttest,
                    'p_mannwhitney': p_mannwhitney,
                }
    
    # Count totals
    total_units = sum(len(c.units) for chips in chips_by_genotype.values() for c in chips)
    zerofill_count = sum(1 for chips in chips_by_genotype.values() 
                        for c in chips for u in c.units if u.has_zerofill)
    
    # Generate markdown
    md_lines = [
        "# HTR Agonist Effect: Statistical Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Dataset Overview",
        "",
        "| Genotype | Chips | Total Units |",
        "|----------|-------|-------------|",
    ]
    
    for genotype in GENOTYPE_ORDER:
        if genotype in chips_by_genotype:
            chips = chips_by_genotype[genotype]
            n_units = sum(len(c.units) for c in chips)
            md_lines.append(f"| {genotype} | {len(chips)} | {n_units} |")
    
    md_lines.extend([
        "",
        f"**Total units analyzed:** {total_units}",
        f"**Units with zero-filled recordings:** {zerofill_count} ({zerofill_count/total_units*100:.1f}%)",
        "",
        "### Cell Type Distribution",
        "",
        "| Cell Type | Count | Percentage |",
        "|-----------|-------|------------|",
    ])
    
    cell_type_counts = {}
    for chips in chips_by_genotype.values():
        for chip_data in chips:
            for unit_tc in chip_data.units:
                ct = unit_tc.cell_type
                cell_type_counts[ct] = cell_type_counts.get(ct, 0) + 1
    
    for ct in ['ON', 'OFF', 'ON_OFF', 'unknown']:
        count = cell_type_counts.get(ct, 0)
        pct = count / total_units * 100 if total_units > 0 else 0
        md_lines.append(f"| {ct} | {count} | {pct:.1f}% |")
    
    md_lines.extend([
        "",
        "---",
        "",
        "## Agonist Effect by Cell Type",
        "",
    ])
    
    for cell_type in cell_types:
        md_lines.extend([
            f"### {cell_type} Cells",
            "",
            "| Genotype | n | Baseline (Hz) | Agonist (Hz) | % Change | Cohen's d | p-value (paired) |",
            "|----------|---|---------------|--------------|----------|-----------|------------------|",
        ])
        
        for genotype in GENOTYPE_ORDER:
            if genotype in stats_data[cell_type]:
                d = stats_data[cell_type][genotype]
                p_str = f"{d['p_paired']:.4f}" if not np.isnan(d['p_paired']) else "N/A"
                if d['p_paired'] < 0.001:
                    p_str = "<0.001 ***"
                elif d['p_paired'] < 0.01:
                    p_str = f"{d['p_paired']:.4f} **"
                elif d['p_paired'] < 0.05:
                    p_str = f"{d['p_paired']:.4f} *"
                
                md_lines.append(
                    f"| {genotype} | {d['n']} | "
                    f"{d['baseline_mean']:.1f} ± {d['baseline_sem']:.1f} | "
                    f"{d['agonist_mean']:.1f} ± {d['agonist_sem']:.1f} | "
                    f"{d['pct_change']:+.0f}% | "
                    f"{d['cohens_d']:.2f} | {p_str} |"
                )
        
        md_lines.append("")
    
    md_lines.extend([
        "---",
        "",
        "## WT vs KO Comparison",
        "",
        "Comparing the magnitude of agonist effect (firing rate change) between genotypes.",
        "",
        "| Cell Type | WT % Change | KO % Change | p-value (t-test) | p-value (Mann-Whitney) | Significance |",
        "|-----------|-------------|-------------|------------------|------------------------|--------------|",
    ])
    
    for cell_type in cell_types:
        g1, g2 = GENOTYPE_ORDER[0], GENOTYPE_ORDER[1]
        
        wt_pct = stats_data[cell_type].get(g1, {}).get('pct_change', 0)
        ko_pct = stats_data[cell_type].get(g2, {}).get('pct_change', 0)
        
        if cell_type in wt_ko_comparisons:
            p_t = wt_ko_comparisons[cell_type]['p_ttest']
            p_mw = wt_ko_comparisons[cell_type]['p_mannwhitney']
            
            if p_t < 0.001:
                sig = "*** (p<0.001)"
            elif p_t < 0.01:
                sig = "** (p<0.01)"
            elif p_t < 0.05:
                sig = "* (p<0.05)"
            else:
                sig = "ns"
            
            md_lines.append(
                f"| {cell_type} | {wt_pct:+.0f}% | {ko_pct:+.0f}% | "
                f"{p_t:.4f} | {p_mw:.4f} | {sig} |"
            )
        else:
            md_lines.append(f"| {cell_type} | {wt_pct:+.0f}% | {ko_pct:+.0f}% | N/A | N/A | N/A |")
    
    md_lines.extend([
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ])
    
    # Generate key findings based on the data
    findings = []
    
    # Check for significant WT vs KO differences
    for cell_type in cell_types:
        if cell_type in wt_ko_comparisons:
            p = wt_ko_comparisons[cell_type]['p_ttest']
            if p < 0.05:
                g1, g2 = GENOTYPE_ORDER[0], GENOTYPE_ORDER[1]
                wt_pct = stats_data[cell_type].get(g1, {}).get('pct_change', 0)
                ko_pct = stats_data[cell_type].get(g2, {}).get('pct_change', 0)
                
                if abs(ko_pct) > abs(wt_pct):
                    findings.append(
                        f"- **{cell_type} cells** show a significantly larger agonist response in KO "
                        f"({ko_pct:+.0f}%) compared to WT ({wt_pct:+.0f}%), p={p:.4f}"
                    )
                else:
                    findings.append(
                        f"- **{cell_type} cells** show a significantly larger agonist response in WT "
                        f"({wt_pct:+.0f}%) compared to KO ({ko_pct:+.0f}%), p={p:.4f}"
                    )
    
    # Check for significant within-genotype effects
    for cell_type in cell_types:
        for genotype in GENOTYPE_ORDER:
            if genotype in stats_data[cell_type]:
                d = stats_data[cell_type][genotype]
                if d['p_paired'] < 0.001:
                    direction = "increase" if d['pct_change'] > 0 else "decrease"
                    findings.append(
                        f"- **{genotype} {cell_type} cells** show a highly significant {direction} "
                        f"in firing rate ({d['pct_change']:+.0f}%, Cohen's d={d['cohens_d']:.2f}, p<0.001)"
                    )
    
    # Check effect sizes
    for cell_type in cell_types:
        for genotype in GENOTYPE_ORDER:
            if genotype in stats_data[cell_type]:
                d = stats_data[cell_type][genotype]
                if abs(d['cohens_d']) >= 0.8:
                    effect = "large"
                elif abs(d['cohens_d']) >= 0.5:
                    effect = "medium"
                else:
                    continue  # Skip small effects
                findings.append(
                    f"- {genotype} {cell_type} cells: {effect} effect size (Cohen's d = {d['cohens_d']:.2f})"
                )
    
    if findings:
        md_lines.extend(findings)
    else:
        md_lines.append("- No statistically significant differences found between genotypes.")
    
    md_lines.extend([
        "",
        "---",
        "",
        "## Methods",
        "",
        "- **Baseline period:** 0-5 minutes (before agonist application)",
        "- **Agonist period:** 5-30 minutes (after agonist application)",
        "- **Bin size:** 30 seconds",
        "- **Statistical tests:**",
        "  - Paired t-test: Within-genotype baseline vs agonist comparison",
        "  - Independent t-test: WT vs KO comparison of agonist effect magnitude",
        "  - Mann-Whitney U test: Non-parametric alternative for WT vs KO comparison",
        "- **Effect size:** Cohen's d (small: 0.2, medium: 0.5, large: 0.8)",
        "",
        "---",
        "",
        "*Report generated automatically by middle_visualization.py*",
    ])
    
    # Write to file
    output_path = output_dir / 'STATISTICAL_SUMMARY.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Middle Recordings Visualization"
    )
    parser.add_argument(
        "--input", type=Path, default=ALIGNED_MIDDLE_DIR,
        help=f"Folder containing aligned middle H5 files (default: {ALIGNED_MIDDLE_DIR})"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for figures (default: input/../figures_middle)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set directories
    input_dir = args.input
    output_dir = args.output or (input_dir.parent / "figures_middle")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all chip data
    chips_by_genotype: Dict[str, List[ChipData]] = {}
    
    for h5_file in sorted(input_dir.glob('*.h5')):
        chip_data = load_chip_data(h5_file)
        if chip_data:
            calculate_time_course(chip_data)
            
            genotype = chip_data.genotype
            if genotype not in chips_by_genotype:
                chips_by_genotype[genotype] = []
            chips_by_genotype[genotype].append(chip_data)
    
    # Sort genotypes: WT first, KO second
    chips_by_genotype = sort_genotypes(chips_by_genotype)
    
    logger.info(f"Loaded data from {len(chips_by_genotype)} genotypes")
    for genotype, chips in chips_by_genotype.items():
        total_units = sum(len(c.units) for c in chips)
        logger.info(f"  {genotype}: {len(chips)} chips, {total_units} unit chains")
    
    # Count cell types and zero-fill statistics
    cell_type_counts = {}
    zerofill_count = 0
    total_units = 0
    for chips in chips_by_genotype.values():
        for chip_data in chips:
            for unit_tc in chip_data.units:
                ct = unit_tc.cell_type
                cell_type_counts[ct] = cell_type_counts.get(ct, 0) + 1
                total_units += 1
                if unit_tc.has_zerofill:
                    zerofill_count += 1
    logger.info(f"Cell types: {cell_type_counts}")
    logger.info(f"Units with zero-filled recordings: {zerofill_count}/{total_units}")
    
    # Check for empty data
    if not chips_by_genotype or total_units == 0:
        print("\nNo data to visualize. Check that:")
        print("  1. Aligned middle files exist in the input folder")
        print("  2. Middle recordings alignment was successful")
        print(f"\nInput folder: {input_dir}")
        return
    
    # Helper to print result
    def print_result(fig_path):
        if fig_path:
            print(f"  Created: {fig_path.name}")
        else:
            print("  Skipped: No data")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    fig_path = create_timecourse_by_genotype(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_genotype_overlay_timecourse(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_baseline_vs_agonist_comparison(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_individual_chip_traces(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_summary_table(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    # Cell type specific plots
    print("\nGenerating cell type plots...")
    
    fig_path = create_timecourse_by_cell_type(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_cell_type_overlay(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_cell_type_bar_comparison(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_cell_type_summary_table(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    # WT vs KO comparison plots
    print("\nGenerating WT vs KO comparison plots...")
    
    fig_path = create_genotype_change_comparison(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_genotype_difference_heatmap(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_combined_timecourse_comparison(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_response_magnitude_scatter(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    fig_path = create_effect_size_comparison(chips_by_genotype, output_dir)
    print_result(fig_path)
    
    # Generate statistical summary markdown
    print("\nGenerating statistical summary...")
    summary_path = generate_statistical_summary(chips_by_genotype, output_dir)
    print_result(summary_path)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
