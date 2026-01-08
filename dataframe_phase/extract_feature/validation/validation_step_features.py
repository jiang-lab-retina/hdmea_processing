"""
Validation plots for step response feature extraction.

Shows example traces with highlighted time windows and extracted feature values.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from step_config import (
    PRE_MARGIN,
    TRANSIENT_START,
    TRANSIENT_END,
    SUSTAINED_START,
    SUSTAINED_END,
    OFF_TRANSIENT_START,
    OFF_TRANSIENT_END,
    OFF_SUSTAINED_START,
    OFF_SUSTAINED_END,
    STEP_TRACE_COLUMN,
    SAMPLING_RATE,
    VALID_MAX_FIRING_RATE,
    LOWPASS_CUTOFF_FREQ,
    LOWPASS_FILTER_ORDER,
)

from extract_feature_step import bessel_lowpass_filter


def compute_mean_trace_filtered(
    trials_data,
    apply_filter: bool = True,
    cutoff_freq: float = LOWPASS_CUTOFF_FREQ,
    filter_order: int = LOWPASS_FILTER_ORDER,
    sampling_rate: float = SAMPLING_RATE,
) -> np.ndarray | None:
    """Compute mean trace from multiple trials with optional low-pass filtering."""
    if trials_data is None:
        return None
    
    if isinstance(trials_data, float) and np.isnan(trials_data):
        return None
    
    try:
        valid_trials = [np.array(trial) for trial in trials_data if trial is not None]
        if len(valid_trials) == 0:
            return None
        trials_array = np.vstack(valid_trials)
        
        # Apply low-pass filter to each trial
        if apply_filter:
            for i in range(trials_array.shape[0]):
                try:
                    trials_array[i] = bessel_lowpass_filter(
                        trials_array[i],
                        cutoff_freq=cutoff_freq,
                        order=filter_order,
                        sampling_rate=sampling_rate,
                    )
                except Exception:
                    pass
        
        return trials_array.mean(axis=0)
    except Exception:
        return None


def plot_step_features_validation(
    df: pd.DataFrame,
    trace_column: str = STEP_TRACE_COLUMN,
    qi_column: str = "step_up_QI",
    output_path: Path | str | None = None,
    n_units: int = 10,
    top_percentile: float = 0.60,
    figsize: tuple = (24, 20),
    random_seed: int = 42,
):
    """
    Create validation plot showing step response features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trace data and extracted features
    trace_column : str
        Column containing the step response trials
    qi_column : str
        Column for quality index to filter top units
    output_path : Path or str, optional
        Path to save the figure
    n_units : int
        Number of units to plot
    top_percentile : float
        Top fraction of units by QI to sample from (e.g., 0.10 for top 10%)
    figsize : tuple
        Figure size
    random_seed : int
        Random seed for reproducibility
    """
    # Filter to top percentile by QI
    qi_threshold = df[qi_column].quantile(1 - top_percentile)
    df_top = df[df[qi_column] >= qi_threshold].copy()
    print(f"Top {top_percentile*100:.0f}% QI threshold: {qi_threshold:.4f}")
    print(f"Units in top {top_percentile*100:.0f}%: {len(df_top)}")
    
    # Random sample
    np.random.seed(random_seed)
    sample_size = min(n_units, len(df_top))
    sample_indices = df_top.sample(n=sample_size, random_state=random_seed).index
    
    # Create figure with 2 rows x 5 columns
    n_cols = 5
    n_rows = (n_units + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Color scheme for time windows
    colors = {
        'baseline': '#E3F2FD',      # Light blue
        'on_transient': '#FFCDD2',   # Light red
        'on_sustained': '#F8BBD9',   # Light pink
        'off_transient': '#C8E6C9',  # Light green
        'off_sustained': '#B2DFDB',  # Light teal
    }
    
    for plot_idx, unit_idx in enumerate(sample_indices):
        ax = axes[plot_idx]
        row = df.loc[unit_idx]
        
        # Get trace data (with same filter as feature extraction)
        trials_data = row[trace_column]
        mean_trace = compute_mean_trace_filtered(trials_data, apply_filter=True)
        
        if mean_trace is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            continue
        
        n_frames = len(mean_trace)
        frames = np.arange(n_frames)
        
        # Get extracted features
        on_peak_extreme = row['on_peak_extreme']
        on_sustained = row['on_sustained']
        off_peak_extreme = row['off_peak_extreme']
        off_sustained = row['off_sustained']
        base_mean = row['base_mean']
        base_std = row['base_std']
        time_to_on_peak_extreme = row['time_to_on_peak_extreme']
        time_to_off_peak_extreme = row['time_to_off_peak_extreme']
        on_off_ratio = row['on_off_ratio']
        on_trans_sus_ratio = row['on_trans_sus_ratio']
        off_trans_sus_ratio = row['off_trans_sus_ratio']
        on_off_sus_ratio = row['on_off_sus_ratio']
        qi_val = row[qi_column]
        
        # Plot colored background regions
        y_min, y_max = mean_trace.min() - 10, mean_trace.max() + 10
        
        # Baseline
        ax.axvspan(0, PRE_MARGIN, alpha=0.5, color=colors['baseline'], 
                   label='Baseline')
        
        # ON transient
        ax.axvspan(TRANSIENT_START, TRANSIENT_END, alpha=0.5, 
                   color=colors['on_transient'], label='ON transient')
        
        # ON sustained
        ax.axvspan(SUSTAINED_START, SUSTAINED_END, alpha=0.5, 
                   color=colors['on_sustained'], label='ON sustained')
        
        # OFF transient
        ax.axvspan(OFF_TRANSIENT_START, OFF_TRANSIENT_END, alpha=0.5, 
                   color=colors['off_transient'], label='OFF transient')
        
        # OFF sustained
        ax.axvspan(OFF_SUSTAINED_START, min(OFF_SUSTAINED_END, n_frames), alpha=0.5, 
                   color=colors['off_sustained'], label='OFF sustained')
        
        # Plot mean trace
        ax.plot(frames, mean_trace, color='black', linewidth=1.5, zorder=10)
        
        # Plot baseline reference line
        ax.axhline(base_mean, color='blue', linestyle='--', linewidth=1, 
                   alpha=0.7, label=f'Baseline: {base_mean:.1f} Hz')
        
        # Mark ON peak extreme (only if valid peak was found)
        if not pd.isna(time_to_on_peak_extreme):
            on_peak_frame = TRANSIENT_START + int(time_to_on_peak_extreme)
            on_peak_value = base_mean + on_peak_extreme
            ax.scatter([on_peak_frame], [on_peak_value], color='red', s=100, 
                       zorder=20, marker='v', edgecolors='darkred', linewidths=1.5)
            ax.annotate(f'{on_peak_extreme:.1f}', (on_peak_frame, on_peak_value),
                       xytext=(5, 10), textcoords='offset points', fontsize=8,
                       color='darkred', fontweight='bold')
        
        # Mark OFF peak extreme (only if valid peak was found)
        if not pd.isna(time_to_off_peak_extreme):
            off_peak_frame = OFF_TRANSIENT_START + int(time_to_off_peak_extreme)
            off_peak_value = base_mean + off_peak_extreme
            ax.scatter([off_peak_frame], [off_peak_value], color='green', s=100, 
                       zorder=20, marker='v', edgecolors='darkgreen', linewidths=1.5)
            ax.annotate(f'{off_peak_extreme:.1f}', (off_peak_frame, off_peak_value),
                       xytext=(5, 10), textcoords='offset points', fontsize=8,
                       color='darkgreen', fontweight='bold')
        
        # Mark ON sustained level
        on_sus_level = base_mean + on_sustained
        ax.hlines(on_sus_level, SUSTAINED_START, SUSTAINED_END, 
                  color='#C2185B', linewidth=2, linestyle='-', zorder=15)
        ax.annotate(f'{on_sustained:.1f}', ((SUSTAINED_START + SUSTAINED_END) / 2, on_sus_level),
                   xytext=(0, -15), textcoords='offset points', fontsize=8,
                   color='#C2185B', fontweight='bold', ha='center')
        
        # Mark OFF sustained level
        off_sus_level = base_mean + off_sustained
        ax.hlines(off_sus_level, OFF_SUSTAINED_START, min(OFF_SUSTAINED_END, n_frames), 
                  color='#00695C', linewidth=2, linestyle='-', zorder=15)
        ax.annotate(f'{off_sustained:.1f}', 
                   ((OFF_SUSTAINED_START + min(OFF_SUSTAINED_END, n_frames)) / 2, off_sus_level),
                   xytext=(0, -15), textcoords='offset points', fontsize=8,
                   color='#00695C', fontweight='bold', ha='center')
        
        # Title with QI
        ax.set_title(f'QI={qi_val:.4f}', fontsize=11, fontweight='bold')
        
        # Feature text box
        on_peak_str = f"{on_peak_extreme:.1f}" if not pd.isna(on_peak_extreme) else "NaN"
        off_peak_str = f"{off_peak_extreme:.1f}" if not pd.isna(off_peak_extreme) else "NaN"
        feature_text = (
            f"on_peak_extreme: {on_peak_str}\n"
            f"on_sustained: {on_sustained:.1f}\n"
            f"off_peak_extreme: {off_peak_str}\n"
            f"off_sustained: {off_sustained:.1f}\n"
            f"base_mean: {base_mean:.1f}\n"
            f"───────────\n"
            f"on_off_ratio: {on_off_ratio:.2f}\n"
            f"on_trans_sus: {on_trans_sus_ratio:.2f}\n"
            f"off_trans_sus: {off_trans_sus_ratio:.2f}\n"
            f"on_off_sus: {on_off_sus_ratio:.2f}"
        )
        ax.text(0.98, 0.98, feature_text, transform=ax.transAxes, fontsize=7,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor='gray'),
               family='monospace')
        
        # Axis settings
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xlim(0, n_frames)
        ax.grid(True, alpha=0.3, linestyle=':')
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axes)):
        axes[i].set_visible(False)
    
    # Create legend
    legend_patches = [
        mpatches.Patch(color=colors['baseline'], alpha=0.5, label=f'Baseline [0:{PRE_MARGIN}]'),
        mpatches.Patch(color=colors['on_transient'], alpha=0.5, 
                      label=f'ON transient [{TRANSIENT_START}:{TRANSIENT_END}]'),
        mpatches.Patch(color=colors['on_sustained'], alpha=0.5, 
                      label=f'ON sustained [{SUSTAINED_START}:{SUSTAINED_END}]'),
        mpatches.Patch(color=colors['off_transient'], alpha=0.5, 
                      label=f'OFF transient [{OFF_TRANSIENT_START}:{OFF_TRANSIENT_END}]'),
        mpatches.Patch(color=colors['off_sustained'], alpha=0.5, 
                      label=f'OFF sustained [{OFF_SUSTAINED_START}:{OFF_SUSTAINED_END}]'),
    ]
    fig.legend(handles=legend_patches, loc='upper center', ncol=5, 
               fontsize=10, bbox_to_anchor=(0.5, 0.02))
    
    # Overall title
    fig.suptitle(
        f'Step Response Feature Validation - Top {top_percentile*100:.0f}% by QI\n'
        f'(10 randomly selected units, QI >= {qi_threshold:.4f}, {LOWPASS_CUTOFF_FREQ} Hz Bessel filtered)',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved validation plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_distributions(
    df_original: pd.DataFrame,
    df_filtered: pd.DataFrame,
    output_path: Path | str | None = None,
    figsize: tuple = (28, 28),
    qi_column: str = "step_up_QI",
):
    """
    Plot feature distributions, NaN counts, and QC rejection statistics.
    
    Parameters
    ----------
    df_original : pd.DataFrame
        Original DataFrame before QC filtering (with step features extracted)
    df_filtered : pd.DataFrame
        DataFrame after QC filtering
    output_path : Path or str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    qi_column : str
        Column name for quality index
    """
    # Feature columns
    magnitude_features = ['on_peak_extreme', 'on_sustained', 'off_peak_extreme', 'off_sustained', 'base_mean', 'base_std']
    timing_features = ['time_to_on_peak_extreme', 'time_to_off_peak_extreme']
    ratio_features = ['on_off_ratio', 'on_trans_sus_ratio', 'off_trans_sus_ratio', 'on_off_sus_ratio']
    all_features = magnitude_features + timing_features + ratio_features
    
    # Create figure: 5 rows
    # Row 1-2: Feature histograms (6 per row)
    # Row 3: QC summary (NaN counts, rejected counts, summary)
    # Row 4: Rejection by QI range
    # Row 5: Feature statistics table
    fig = plt.figure(figsize=figsize)
    
    # =========================================================================
    # Row 1: Magnitude features (first 3)
    # =========================================================================
    for i, feat in enumerate(magnitude_features[:3]):
        ax = fig.add_subplot(5, 6, i + 1)
        values = df_filtered[feat].dropna()
        
        ax.hist(values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(values.mean(), color='darkred', linestyle='--', linewidth=2, 
                   label=f'Mean: {values.mean():.2f}')
        ax.axvline(values.median(), color='darkorange', linestyle='-', linewidth=2, 
                   label=f'Median: {values.median():.2f}')
        
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value (Hz)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 1 continued: Magnitude features (last 3)
    # =========================================================================
    for i, feat in enumerate(magnitude_features[3:]):
        ax = fig.add_subplot(5, 6, i + 4)
        values = df_filtered[feat].dropna()
        
        ax.hist(values, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(values.mean(), color='darkred', linestyle='--', linewidth=2, 
                   label=f'Mean: {values.mean():.2f}')
        ax.axvline(values.median(), color='darkorange', linestyle='-', linewidth=2, 
                   label=f'Median: {values.median():.2f}')
        
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value (Hz)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 2: Timing features and Ratio features
    # =========================================================================
    for i, feat in enumerate(timing_features):
        ax = fig.add_subplot(5, 6, 7 + i)
        values = df_filtered[feat].dropna()
        
        ax.hist(values, bins=30, color='#8E24AA', edgecolor='white', alpha=0.8)
        ax.axvline(values.mean(), color='darkred', linestyle='--', linewidth=2, 
                   label=f'Mean: {values.mean():.1f}')
        
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame index', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    for i, feat in enumerate(ratio_features):
        ax = fig.add_subplot(5, 6, 9 + i)
        values = df_filtered[feat].dropna()
        
        ax.hist(values, bins=50, color='#00897B', edgecolor='white', alpha=0.8)
        ax.axvline(values.mean(), color='darkred', linestyle='--', linewidth=2, 
                   label=f'Mean: {values.mean():.2f}')
        ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('Ratio (tanh)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_xlim(-1.1, 1.1)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 3: NaN counts bar chart
    # =========================================================================
    ax_nan = fig.add_subplot(5, 3, 7)
    
    nan_counts = {}
    for feat in all_features:
        if feat in df_original.columns:
            nan_counts[feat] = df_original[feat].isna().sum()
    
    features = list(nan_counts.keys())
    counts = list(nan_counts.values())
    colors_nan = ['#E57373' if c > 0 else '#81C784' for c in counts]
    
    bars = ax_nan.barh(features, counts, color=colors_nan, edgecolor='white')
    ax_nan.set_xlabel('NaN Count', fontsize=10)
    ax_nan.set_title('NaN Values per Feature\n(Before QC Filtering)', fontsize=11, fontweight='bold')
    ax_nan.tick_params(labelsize=8)
    ax_nan.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        if count > 0:
            ax_nan.text(count + max(counts) * 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{count:,}', va='center', fontsize=8, fontweight='bold')
    
    # =========================================================================
    # Row 3: Rejected units by max firing rate
    # =========================================================================
    ax_reject = fig.add_subplot(5, 3, 8)
    
    qc_features = ['on_peak_extreme', 'on_sustained', 'off_peak_extreme', 'off_sustained', 'base_mean']
    rejection_counts = {}
    
    for feat in qc_features:
        if feat in df_original.columns:
            over_threshold = (df_original[feat].abs() > VALID_MAX_FIRING_RATE).sum()
            rejection_counts[feat] = over_threshold
    
    features_reject = list(rejection_counts.keys())
    counts_reject = list(rejection_counts.values())
    
    bars_reject = ax_reject.barh(features_reject, counts_reject, color='#FF7043', edgecolor='white')
    ax_reject.set_xlabel('Rejected Count', fontsize=10)
    ax_reject.set_title(f'Units Rejected by Max Firing Rate\n(Threshold: ±{VALID_MAX_FIRING_RATE} Hz)', 
                        fontsize=11, fontweight='bold')
    ax_reject.tick_params(labelsize=8)
    ax_reject.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    max_count = max(counts_reject) if counts_reject else 1
    for bar, count in zip(bars_reject, counts_reject):
        if count > 0:
            ax_reject.text(count + max_count * 0.02, bar.get_y() + bar.get_height()/2, 
                          f'{count:,}', va='center', fontsize=8, fontweight='bold')
    
    # =========================================================================
    # Row 3: Summary statistics
    # =========================================================================
    ax_summary = fig.add_subplot(5, 3, 9)
    ax_summary.axis('off')
    
    total_original = len(df_original)
    total_filtered = len(df_filtered)
    total_rejected = total_original - total_filtered
    
    summary_text = (
        f"SUMMARY STATISTICS\n"
        f"{'='*40}\n\n"
        f"Original units:     {total_original:,}\n"
        f"After QC filtering: {total_filtered:,}\n"
        f"Total rejected:     {total_rejected:,} ({100*total_rejected/total_original:.1f}%)\n\n"
        f"{'='*40}\n"
        f"Max firing rate threshold: ±{VALID_MAX_FIRING_RATE} Hz\n\n"
        f"Rejection breakdown:\n"
    )
    
    for feat, count in rejection_counts.items():
        if count > 0:
            summary_text += f"  {feat}: {count:,}\n"
    
    ax_summary.text(0.1, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=11, verticalalignment='top', 
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='gray'))
    
    # =========================================================================
    # Row 4: Rejection by QI range
    # =========================================================================
    ax_qi_range = fig.add_subplot(5, 2, 7)
    
    # Define QI ranges (quintiles: 0-20%, 20-40%, etc.)
    qi_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    qi_range_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    # Check if QI column exists and has valid values
    if qi_column in df_original.columns:
        qi_values = df_original[qi_column].dropna()
        qi_min, qi_max = qi_values.min(), qi_values.max()
        
        # Calculate rejection counts by QI range
        qc_features = ['on_peak_extreme', 'on_sustained', 'off_peak_extreme', 'off_sustained', 'base_mean']
        
        rejection_by_range = {label: {'total': 0, 'rejected': 0} for label in qi_range_labels}
        
        for idx, row in df_original.iterrows():
            qi_val = row[qi_column]
            if pd.isna(qi_val):
                continue
            
            # Normalize QI to 0-1 range based on actual data range
            qi_normalized = (qi_val - qi_min) / (qi_max - qi_min) if qi_max > qi_min else 0.5
            
            # Find which range this unit belongs to
            for (low, high), label in zip(qi_ranges, qi_range_labels):
                if low <= qi_normalized < high or (high == 1.0 and qi_normalized == 1.0):
                    rejection_by_range[label]['total'] += 1
                    
                    # Check if rejected
                    is_rejected = False
                    for feat in qc_features:
                        if feat in df_original.columns:
                            val = row[feat]
                            if pd.notna(val) and abs(val) > VALID_MAX_FIRING_RATE:
                                is_rejected = True
                                break
                    
                    if is_rejected:
                        rejection_by_range[label]['rejected'] += 1
                    break
        
        # Plot stacked bar chart
        x = np.arange(len(qi_range_labels))
        width = 0.6
        
        totals = [rejection_by_range[label]['total'] for label in qi_range_labels]
        rejected = [rejection_by_range[label]['rejected'] for label in qi_range_labels]
        retained = [t - r for t, r in zip(totals, rejected)]
        
        bars1 = ax_qi_range.bar(x, retained, width, label='Retained', color='#81C784')
        bars2 = ax_qi_range.bar(x, rejected, width, bottom=retained, label='Rejected', color='#E57373')
        
        ax_qi_range.set_xlabel('QI Range (percentile)', fontsize=10)
        ax_qi_range.set_ylabel('Unit Count', fontsize=10)
        ax_qi_range.set_title(f'Units by Step QI Range\n(QI range: {qi_min:.3f} - {qi_max:.3f})', 
                              fontsize=11, fontweight='bold')
        ax_qi_range.set_xticks(x)
        ax_qi_range.set_xticklabels(qi_range_labels)
        ax_qi_range.legend(fontsize=9)
        ax_qi_range.tick_params(labelsize=8)
        ax_qi_range.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, (ret, rej) in enumerate(zip(retained, rejected)):
            if ret > 0:
                ax_qi_range.text(i, ret/2, f'{ret}', ha='center', va='center', fontsize=8, fontweight='bold')
            if rej > 0:
                ax_qi_range.text(i, ret + rej/2, f'{rej}', ha='center', va='center', fontsize=8, 
                                fontweight='bold', color='white')
    
    # =========================================================================
    # Row 4: Rejection percentage by QI range
    # =========================================================================
    ax_qi_pct = fig.add_subplot(5, 2, 8)
    
    if qi_column in df_original.columns:
        rejection_pcts = []
        for label in qi_range_labels:
            total = rejection_by_range[label]['total']
            rejected = rejection_by_range[label]['rejected']
            pct = (rejected / total * 100) if total > 0 else 0
            rejection_pcts.append(pct)
        
        bars = ax_qi_pct.bar(x, rejection_pcts, width, color='#FF7043')
        ax_qi_pct.set_xlabel('QI Range (percentile)', fontsize=10)
        ax_qi_pct.set_ylabel('Rejection Rate (%)', fontsize=10)
        ax_qi_pct.set_title('Rejection Rate by Step QI Range', fontsize=11, fontweight='bold')
        ax_qi_pct.set_xticks(x)
        ax_qi_pct.set_xticklabels(qi_range_labels)
        ax_qi_pct.tick_params(labelsize=8)
        ax_qi_pct.grid(True, alpha=0.3, axis='y')
        ax_qi_pct.set_ylim(0, max(rejection_pcts) * 1.2 if max(rejection_pcts) > 0 else 10)
        
        # Add percentage labels on bars
        for i, pct in enumerate(rejection_pcts):
            ax_qi_pct.text(i, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =========================================================================
    # Row 5: Feature statistics table
    # =========================================================================
    ax_table = fig.add_subplot(5, 1, 5)
    ax_table.axis('off')
    
    # Create statistics table
    table_data = []
    headers = ['Feature', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    
    for feat in all_features:
        if feat in df_filtered.columns:
            values = df_filtered[feat].dropna()
            if len(values) > 0:
                stats = values.describe()
                row = [
                    feat,
                    f'{int(stats["count"]):,}',
                    f'{stats["mean"]:.2f}',
                    f'{stats["std"]:.2f}',
                    f'{stats["min"]:.2f}',
                    f'{stats["25%"]:.2f}',
                    f'{stats["50%"]:.2f}',
                    f'{stats["75%"]:.2f}',
                    f'{stats["max"]:.2f}',
                ]
                table_data.append(row)
    
    table = ax_table.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#E3F2FD'] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(fontweight='bold')
    
    ax_table.set_title('Feature Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle(
        'Step Response Feature Distributions and QC Summary',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Load data and generate validation plots."""
    # Paths
    input_path_original = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected.parquet")
    input_path_filtered = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected_step.parquet")
    output_dir = Path("dataframe_phase/extract_feature/validation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Step Feature Validation Plots")
    print("=" * 60)
    
    # Load filtered data (with step features, after QC)
    print(f"\nLoading filtered data from: {input_path_filtered}")
    df_filtered = pd.read_parquet(input_path_filtered)
    print(f"Filtered DataFrame shape: {df_filtered.shape}")
    
    # Load original data to compute features for QC comparison
    print(f"\nLoading original data from: {input_path_original}")
    df_original = pd.read_parquet(input_path_original)
    print(f"Original DataFrame shape: {df_original.shape}")
    
    # We need to extract features from original data to get NaN/rejection stats
    # Import the extraction function
    from extract_feature_step_gb_freq import extract_step_features_from_trace, compute_mean_trace
    
    print("\nExtracting features from original data for QC comparison...")
    feature_names = [
        "on_peak_extreme", "on_sustained", "off_peak_extreme", "off_sustained",
        "base_mean", "base_std", "time_to_on_peak_extreme", "time_to_off_peak_extreme",
        "on_off_ratio", "on_trans_sus_ratio", "off_trans_sus_ratio", "on_off_sus_ratio"
    ]
    
    # Initialize feature columns with NaN
    for name in feature_names:
        df_original[name] = np.nan
    
    # Extract features for each unit (without filtering)
    from tqdm import tqdm
    valid_count = 0
    for idx in tqdm(df_original.index, desc="Extracting features"):
        trials_data = df_original.loc[idx, STEP_TRACE_COLUMN]
        mean_trace = compute_mean_trace(trials_data)
        if mean_trace is None:
            continue
        
        try:
            features = extract_step_features_from_trace(mean_trace)
            for name, value in features.items():
                df_original.loc[idx, name] = value
            valid_count += 1
        except Exception:
            continue
    
    print(f"Extracted features for {valid_count} / {len(df_original)} units")
    
    # Print config being used
    print("\nTime windows from config:")
    print(f"  Baseline: [0:{PRE_MARGIN}]")
    print(f"  ON transient: [{TRANSIENT_START}:{TRANSIENT_END}]")
    print(f"  ON sustained: [{SUSTAINED_START}:{SUSTAINED_END}]")
    print(f"  OFF transient: [{OFF_TRANSIENT_START}:{OFF_TRANSIENT_END}]")
    print(f"  OFF sustained: [{OFF_SUSTAINED_START}:{OFF_SUSTAINED_END}]")
    print(f"  Max firing rate: {VALID_MAX_FIRING_RATE} Hz")
    
    # Generate validation plot (example traces)
    print("\nGenerating example traces plot...")
    plot_step_features_validation(
        df_filtered,
        trace_column=STEP_TRACE_COLUMN,
        qi_column="step_up_QI",
        output_path=output_dir / "step_features_validation.png",
        n_units=10,
        top_percentile=0.10,
        figsize=(24, 12),
        random_seed=42,
    )
    
    # Generate distribution plot
    print("\nGenerating feature distribution plot...")
    plot_feature_distributions(
        df_original=df_original,
        df_filtered=df_filtered,
        output_path=output_dir / "step_features_distributions.png",
        figsize=(28, 24),
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

