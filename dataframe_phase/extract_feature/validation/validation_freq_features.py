"""
Validation plots for frequency step response feature extraction.

Shows example traces with fitted sine waves and extracted feature values.
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
    FREQ_PRE_MARGIN,
    FREQ_GAP_DURATION,
    FREQ_STEP_DURATION,
    FREQ_STEP_FREQUENCIES,
    FREQ_STEP_BOUNDS,
    FREQ_TRACE_COLUMN,
    FREQ_AMP_GUESS,
    FREQ_AMP_UPPER_LIM,
    FREQ_OFFSET_UPPER_LIM,
    FREQ_OFFSET_LOWER_LIM,
    FREQ_R_SQUARED_THRESHOLD,
    FREQ_FIT_SKIP_FRAMES,
    SAMPLING_RATE,
    LOWPASS_CUTOFF_FREQ,
    LOWPASS_FILTER_ORDER,
)

from extract_feature_freq import (
    bessel_lowpass_filter,
    compute_mean_trace,
    fit_sine_wave_fixed_freq,
    freq_to_column_str,
)


# =============================================================================
# Sine Fit Visualization
# =============================================================================

def plot_freq_step_sine_fits(
    df: pd.DataFrame,
    trace_column: str = FREQ_TRACE_COLUMN,
    qi_column: str = "step_up_QI",
    output_path: Path | str | None = None,
    n_units: int = 6,
    top_percentile: float = 0.20,
    figsize: tuple = (28, 24),
    random_seed: int = 42,
):
    """
    Create validation plot showing sine wave fits for each frequency.
    
    Each row is a unit, each column is a frequency.
    Shows raw trace segment with fitted sine overlay.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trace data and extracted features
    trace_column : str
        Column containing the frequency step response trials
    qi_column : str
        Column for quality index to filter top units
    output_path : Path or str, optional
        Path to save the figure
    n_units : int
        Number of units to plot
    top_percentile : float
        Top fraction of units by QI to sample from
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
    
    n_freqs = len(FREQ_STEP_FREQUENCIES)
    fig, axes = plt.subplots(n_units, n_freqs, figsize=figsize)
    
    # Color scheme for frequencies
    freq_colors = {
        0.5: '#E57373',   # Red
        1: '#FFB74D',     # Orange
        2: '#81C784',     # Green
        4: '#64B5F6',     # Blue
        10: '#BA68C8',    # Purple
    }
    
    for row_idx, unit_idx in enumerate(sample_indices):
        row = df.loc[unit_idx]
        
        # Get trace data
        trials_data = row[trace_column]
        mean_trace = compute_mean_trace(trials_data, apply_filter=False)
        
        if mean_trace is None:
            for col_idx in range(n_freqs):
                axes[row_idx, col_idx].text(
                    0.5, 0.5, 'No data', ha='center', va='center',
                    transform=axes[row_idx, col_idx].transAxes, fontsize=12
                )
            continue
        
        qi_val = row[qi_column]
        
        for col_idx, freq in enumerate(FREQ_STEP_FREQUENCIES):
            ax = axes[row_idx, col_idx]
            freq_str = freq_to_column_str(freq)
            start, end = FREQ_STEP_BOUNDS[freq]
            
            # Get segment
            if end <= len(mean_trace):
                segment = mean_trace[start:end]
            else:
                ax.text(0.5, 0.5, 'Trace too short', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                continue
            
            n_samples = len(segment)
            time_array = np.arange(n_samples) / SAMPLING_RATE
            frames = np.arange(start, end)
            
            # Get fitted parameters from DataFrame
            amp = row.get(f"freq_step_{freq_str}hz_amp", np.nan)
            phase = row.get(f"freq_step_{freq_str}hz_phase", np.nan)
            r_squared = row.get(f"freq_step_{freq_str}hz_r_squared", np.nan)
            offset = row.get(f"freq_step_{freq_str}hz_offset", np.nan)
            std_val = row.get(f"freq_step_{freq_str}hz_std", np.nan)
            
            # Plot raw trace
            ax.plot(frames, segment, color='black', linewidth=1.5, 
                   label='Data', alpha=0.8)
            
            # Plot fitted sine if valid
            # Note: fit was done on data starting from skip_frames for non-0.5Hz
            if not pd.isna(phase) and not pd.isna(amp):
                if freq != 0.5:
                    fit_start_offset = FREQ_FIT_SKIP_FRAMES
                else:
                    fit_start_offset = 0
                
                fit_n_samples = n_samples - fit_start_offset
                fit_time_array = np.arange(fit_n_samples) / SAMPLING_RATE
                fit_frames = np.arange(start + fit_start_offset, end)
                fitted_sine = amp * np.sin(2 * np.pi * freq * fit_time_array + phase) + offset
                ax.plot(fit_frames, fitted_sine, color=freq_colors[freq], 
                       linewidth=2, linestyle='--', label='Fit', alpha=0.9)
                
                # Draw vertical line to show where fitting starts
                if fit_start_offset > 0:
                    ax.axvline(start + fit_start_offset, color='gray', 
                              linestyle=':', linewidth=1, alpha=0.7)
            
            # Plot offset line
            if not pd.isna(offset):
                ax.axhline(offset, color='gray', linestyle=':', linewidth=1, alpha=0.7)
            
            # Feature text box
            amp_str = f"{amp:.1f}" if not pd.isna(amp) else "N/A"
            phase_str = f"{phase:.2f}" if not pd.isna(phase) else "N/A"
            r2_str = f"{r_squared:.3f}" if not pd.isna(r_squared) else "N/A"
            offset_str = f"{offset:.1f}" if not pd.isna(offset) else "N/A"
            std_str = f"{std_val:.1f}" if not pd.isna(std_val) else "N/A"
            
            # Color R^2 based on threshold
            r2_color = 'darkgreen' if (not pd.isna(r_squared) and r_squared >= FREQ_R_SQUARED_THRESHOLD) else 'darkred'
            
            feature_text = (
                f"amp: {amp_str}\n"
                f"phase: {phase_str}\n"
                f"R²: {r2_str}\n"
                f"offset: {offset_str}\n"
                f"std: {std_str}"
            )
            ax.text(0.98, 0.98, feature_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                            edgecolor='gray'),
                   family='monospace')
            
            # Axis settings
            ax.set_xlim(start, end)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.tick_params(axis='both', labelsize=7)
            
            # Column title (frequency) - only on first row
            if row_idx == 0:
                ax.set_title(f'{freq} Hz', fontsize=12, fontweight='bold',
                           color=freq_colors[freq])
            
            # Row label (unit QI) - only on first column
            if col_idx == 0:
                ax.set_ylabel(f'QI={qi_val:.3f}\nFiring Rate (Hz)', fontsize=9)
            
            # X label - only on last row
            if row_idx == n_units - 1:
                ax.set_xlabel('Frame', fontsize=9)
    
    # Legend
    legend_patches = [
        plt.Line2D([0], [0], color='black', linewidth=1.5, label='Mean Trace'),
        plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Fitted Sine'),
    ]
    fig.legend(handles=legend_patches, loc='upper center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, 0.02))
    
    # Overall title
    fig.suptitle(
        f'Frequency Step Sine Wave Fits - Top {top_percentile*100:.0f}% by QI\n'
        f'({n_units} randomly selected units, no filtering)',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved sine fit plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_freq_step_full_trace_with_fits(
    df: pd.DataFrame,
    trace_column: str = FREQ_TRACE_COLUMN,
    qi_column: str = "step_up_QI",
    output_path: Path | str | None = None,
    n_units: int = 4,
    top_percentile: float = 0.20,
    figsize: tuple = (24, 16),
    random_seed: int = 42,
):
    """
    Plot full trace with all frequency segments highlighted and fits overlaid.
    
    Each subplot shows one unit's complete trace with colored regions
    for each frequency and fitted sine waves.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trace data and extracted features
    trace_column : str
        Column containing the frequency step response trials
    qi_column : str
        Column for quality index to filter top units
    output_path : Path or str, optional
        Path to save the figure
    n_units : int
        Number of units to plot
    top_percentile : float
        Top fraction of units by QI to sample from
    figsize : tuple
        Figure size
    random_seed : int
        Random seed for reproducibility
    """
    # Filter to top percentile by QI
    qi_threshold = df[qi_column].quantile(1 - top_percentile)
    df_top = df[df[qi_column] >= qi_threshold].copy()
    
    # Random sample
    np.random.seed(random_seed)
    sample_size = min(n_units, len(df_top))
    sample_indices = df_top.sample(n=sample_size, random_state=random_seed).index
    
    n_cols = 2
    n_rows = (n_units + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Color scheme
    freq_colors = {
        0.5: '#FFCDD2',   # Light red
        1: '#FFE0B2',     # Light orange
        2: '#C8E6C9',     # Light green
        4: '#BBDEFB',     # Light blue
        10: '#E1BEE7',    # Light purple
    }
    freq_dark_colors = {
        0.5: '#C62828',   # Dark red
        1: '#EF6C00',     # Dark orange
        2: '#2E7D32',     # Dark green
        4: '#1565C0',     # Dark blue
        10: '#7B1FA2',    # Dark purple
    }
    
    for plot_idx, unit_idx in enumerate(sample_indices):
        ax = axes[plot_idx]
        row = df.loc[unit_idx]
        
        # Get trace data
        trials_data = row[trace_column]
        mean_trace = compute_mean_trace(trials_data, apply_filter=False)
        
        if mean_trace is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            continue
        
        n_frames = len(mean_trace)
        frames = np.arange(n_frames)
        qi_val = row[qi_column]
        
        # Plot background regions for each frequency
        y_min, y_max = mean_trace.min() - 10, mean_trace.max() + 10
        for freq in FREQ_STEP_FREQUENCIES:
            start, end = FREQ_STEP_BOUNDS[freq]
            if end <= n_frames:
                ax.axvspan(start, end, alpha=0.3, color=freq_colors[freq])
        
        # Plot full trace
        ax.plot(frames, mean_trace, color='black', linewidth=1, alpha=0.8)
        
        # Overlay fitted sine waves for each frequency
        for freq in FREQ_STEP_FREQUENCIES:
            freq_str = freq_to_column_str(freq)
            start, end = FREQ_STEP_BOUNDS[freq]
            
            if end > n_frames:
                continue
            
            amp = row.get(f"freq_step_{freq_str}hz_amp", np.nan)
            phase = row.get(f"freq_step_{freq_str}hz_phase", np.nan)
            offset = row.get(f"freq_step_{freq_str}hz_offset", np.nan)
            r_squared = row.get(f"freq_step_{freq_str}hz_r_squared", np.nan)
            
            if not pd.isna(phase) and not pd.isna(amp) and amp > 0:
                # Fit was done starting from skip_frames for non-0.5Hz
                if freq != 0.5:
                    fit_start_offset = FREQ_FIT_SKIP_FRAMES
                else:
                    fit_start_offset = 0
                
                fit_start = start + fit_start_offset
                segment_frames = np.arange(fit_start, end)
                time_array = np.arange(end - fit_start) / SAMPLING_RATE
                fitted_sine = amp * np.sin(2 * np.pi * freq * time_array + phase) + offset
                ax.plot(segment_frames, fitted_sine, color=freq_dark_colors[freq],
                       linewidth=2, linestyle='--')
        
        # Title
        ax.set_title(f'Unit QI = {qi_val:.4f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
        ax.set_xlim(0, n_frames)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(labelsize=8)
        
        # Feature summary text
        feature_lines = []
        for freq in FREQ_STEP_FREQUENCIES:
            freq_str = freq_to_column_str(freq)
            amp = row.get(f"freq_step_{freq_str}hz_amp", np.nan)
            r2 = row.get(f"freq_step_{freq_str}hz_r_squared", np.nan)
            std_val = row.get(f"freq_step_{freq_str}hz_std", np.nan)
            amp_str = f"{amp:.1f}" if not pd.isna(amp) else "N/A"
            r2_str = f"{r2:.2f}" if not pd.isna(r2) else "N/A"
            std_str = f"{std_val:.1f}" if not pd.isna(std_val) else "N/A"
            feature_lines.append(f"{freq}Hz: amp={amp_str}, R²={r2_str}, std={std_str}")
        
        feature_text = "\n".join(feature_lines)
        ax.text(0.02, 0.98, feature_text, transform=ax.transAxes, fontsize=7,
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                        edgecolor='gray'),
               family='monospace')
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axes)):
        axes[i].set_visible(False)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=freq_colors[freq], alpha=0.5, label=f'{freq} Hz')
        for freq in FREQ_STEP_FREQUENCIES
    ]
    fig.legend(handles=legend_patches, loc='upper center', ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, 0.02))
    
    fig.suptitle(
        f'Frequency Step Full Traces with Sine Fits\n'
        f'(Top {top_percentile*100:.0f}% by QI, dashed lines = fitted sine)',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved full trace plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_freq_feature_distributions(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    figsize: tuple = (28, 20),
):
    """
    Plot feature distributions for all frequencies.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with extracted frequency features
    output_path : Path or str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    n_freqs = len(FREQ_STEP_FREQUENCIES)
    feature_types = ['amp', 'phase', 'r_squared', 'offset', 'std']
    
    fig, axes = plt.subplots(len(feature_types), n_freqs, figsize=figsize)
    
    # Colors for feature types
    feature_colors = {
        'amp': '#1976D2',
        'phase': '#388E3C',
        'r_squared': '#F57C00',
        'offset': '#7B1FA2',
        'std': '#00ACC1',
    }
    
    for row_idx, feat_type in enumerate(feature_types):
        for col_idx, freq in enumerate(FREQ_STEP_FREQUENCIES):
            ax = axes[row_idx, col_idx]
            freq_str = freq_to_column_str(freq)
            col_name = f"freq_step_{freq_str}hz_{feat_type}"
            
            if col_name not in df.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
                continue
            
            values = df[col_name].dropna()
            # Filter out infinite values
            values = values[np.isfinite(values)]
            
            if len(values) == 0:
                ax.text(0.5, 0.5, 'All NaN/Inf', ha='center', va='center',
                       transform=ax.transAxes)
                continue
            
            # Plot histogram
            ax.hist(values, bins=50, color=feature_colors[feat_type],
                   edgecolor='white', alpha=0.8)
            
            # Add mean line
            mean_val = values.mean()
            ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2f}')
            
            # For r_squared, add threshold line
            if feat_type == 'r_squared':
                ax.axvline(FREQ_R_SQUARED_THRESHOLD, color='black', 
                          linestyle=':', linewidth=2,
                          label=f'Threshold: {FREQ_R_SQUARED_THRESHOLD}')
                low_count = (values < FREQ_R_SQUARED_THRESHOLD).sum()
                ax.text(0.98, 0.85, f'Low R²: {low_count}\n({100*low_count/len(values):.1f}%)',
                       transform=ax.transAxes, fontsize=8, ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Column title (frequency) - only on first row
            if row_idx == 0:
                ax.set_title(f'{freq} Hz', fontsize=11, fontweight='bold')
            
            # Row label - only on first column
            if col_idx == 0:
                ax.set_ylabel(f'{feat_type}\nCount', fontsize=9)
            
            # X label - only on last row
            if row_idx == len(feature_types) - 1:
                ax.set_xlabel('Value', fontsize=9)
    
    fig.suptitle(
        'Frequency Step Feature Distributions by Frequency',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_r_squared_analysis(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    figsize: tuple = (16, 10),
):
    """
    Plot R-squared quality analysis across frequencies.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with extracted frequency features
    output_path : Path or str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: R^2 distribution by frequency (overlaid)
    ax1 = axes[0, 0]
    freq_colors = {
        0.5: '#E57373', 1: '#FFB74D', 2: '#81C784', 4: '#64B5F6', 10: '#BA68C8'
    }
    for freq in FREQ_STEP_FREQUENCIES:
        freq_str = freq_to_column_str(freq)
        col_name = f"freq_step_{freq_str}hz_r_squared"
        if col_name in df.columns:
            values = df[col_name].dropna()
            values = values[np.isfinite(values)]  # Filter out inf values
            if len(values) > 0:
                ax1.hist(values, bins=50, alpha=0.5, color=freq_colors[freq],
                        label=f'{freq} Hz', edgecolor='none')
    
    ax1.axvline(FREQ_R_SQUARED_THRESHOLD, color='black', linestyle='--', 
               linewidth=2, label=f'Threshold ({FREQ_R_SQUARED_THRESHOLD})')
    ax1.set_xlabel('R²', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('R² Distribution by Frequency', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Low R^2 count by frequency
    ax2 = axes[0, 1]
    low_r2_counts = []
    total_counts = []
    for freq in FREQ_STEP_FREQUENCIES:
        freq_str = freq_to_column_str(freq)
        col_name = f"freq_step_{freq_str}hz_r_squared"
        if col_name in df.columns:
            values = df[col_name].dropna()
            values = values[np.isfinite(values)]  # Filter out inf values
            low_count = (values < FREQ_R_SQUARED_THRESHOLD).sum()
            low_r2_counts.append(low_count)
            total_counts.append(len(values))
        else:
            low_r2_counts.append(0)
            total_counts.append(0)
    
    x = np.arange(len(FREQ_STEP_FREQUENCIES))
    bars = ax2.bar(x, low_r2_counts, color=[freq_colors[f] for f in FREQ_STEP_FREQUENCIES])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{f} Hz' for f in FREQ_STEP_FREQUENCIES])
    ax2.set_ylabel('Count with R² < 0.1', fontsize=10)
    ax2.set_title('Low Quality Fits by Frequency', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (bar, low, total) in enumerate(zip(bars, low_r2_counts, total_counts)):
        if total > 0:
            pct = 100 * low / total
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    # Plot 3: Mean R^2 by frequency
    ax3 = axes[1, 0]
    mean_r2 = []
    std_r2 = []
    for freq in FREQ_STEP_FREQUENCIES:
        freq_str = freq_to_column_str(freq)
        col_name = f"freq_step_{freq_str}hz_r_squared"
        if col_name in df.columns:
            values = df[col_name].dropna()
            values = values[np.isfinite(values)]  # Filter out inf values
            mean_r2.append(values.mean() if len(values) > 0 else 0)
            std_r2.append(values.std() if len(values) > 0 else 0)
        else:
            mean_r2.append(0)
            std_r2.append(0)
    
    bars = ax3.bar(x, mean_r2, yerr=std_r2, capsize=5,
                   color=[freq_colors[f] for f in FREQ_STEP_FREQUENCIES],
                   edgecolor='black', linewidth=1)
    ax3.axhline(FREQ_R_SQUARED_THRESHOLD, color='black', linestyle='--',
               linewidth=2, label=f'Threshold ({FREQ_R_SQUARED_THRESHOLD})')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{f} Hz' for f in FREQ_STEP_FREQUENCIES])
    ax3.set_ylabel('Mean R² (± std)', fontsize=10)
    ax3.set_title('Mean R² by Frequency', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1)
    
    # Plot 4: Mean amplitude by frequency
    ax4 = axes[1, 1]
    mean_amp = []
    std_amp = []
    for freq in FREQ_STEP_FREQUENCIES:
        freq_str = freq_to_column_str(freq)
        col_name = f"freq_step_{freq_str}hz_amp"
        if col_name in df.columns:
            values = df[col_name].dropna()
            values = values[np.isfinite(values)]  # Filter out inf values
            mean_amp.append(values.mean() if len(values) > 0 else 0)
            std_amp.append(values.std() if len(values) > 0 else 0)
        else:
            mean_amp.append(0)
            std_amp.append(0)
    
    bars = ax4.bar(x, mean_amp, yerr=std_amp, capsize=5,
                   color=[freq_colors[f] for f in FREQ_STEP_FREQUENCIES],
                   edgecolor='black', linewidth=1)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{f} Hz' for f in FREQ_STEP_FREQUENCIES])
    ax4.set_ylabel('Mean Amplitude (± std)', fontsize=10)
    ax4.set_title('Mean Fitted Amplitude by Frequency', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(
        'Frequency Step R² Quality Analysis',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved R² analysis plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_statistics_table(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    figsize: tuple = (20, 12),
):
    """
    Create a statistics summary table for all frequency features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with extracted frequency features
    output_path : Path or str, optional
        Path to save the figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Build table data
    headers = ['Frequency', 'Feature', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    table_data = []
    
    for freq in FREQ_STEP_FREQUENCIES:
        freq_str = freq_to_column_str(freq)
        for feat_type in ['amp', 'phase', 'r_squared', 'offset', 'std']:
            col_name = f"freq_step_{freq_str}hz_{feat_type}"
            if col_name in df.columns:
                values = df[col_name].dropna()
                values = values[np.isfinite(values)]  # Filter out inf values
                if len(values) > 0:
                    stats = values.describe()
                    row = [
                        f'{freq} Hz',
                        feat_type,
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
    
    table = ax.table(
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
    
    ax.set_title('Frequency Step Feature Statistics Summary', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistics table to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_valid_invalid_comparison(
    df: pd.DataFrame,
    trace_column: str = FREQ_TRACE_COLUMN,
    output_path: Path | str | None = None,
    n_valid: int = 5,
    n_invalid: int = 5,
    figsize: tuple = (28, 32),
    random_seed: int = 42,
):
    """
    Plot comparison of units with valid 10 Hz fits vs invalid 0.5 Hz fits.
    
    Shows all 5 frequencies for each unit in a row.
    Top rows: units with valid R² for 10 Hz
    Bottom rows: units with invalid R² for 0.5 Hz
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trace data and extracted features
    trace_column : str
        Column containing the frequency step response trials
    output_path : Path or str, optional
        Path to save the figure
    n_valid : int
        Number of units with valid 10 Hz to show
    n_invalid : int
        Number of units with invalid 0.5 Hz to show
    figsize : tuple
        Figure size
    random_seed : int
        Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    # Find units with valid 10 Hz R² - select from top 10% of R² values
    r2_10hz_col = "freq_step_10hz_r_squared"
    if r2_10hz_col in df.columns:
        valid_10hz_mask = df[r2_10hz_col] >= FREQ_R_SQUARED_THRESHOLD
        df_valid_10hz_all = df[valid_10hz_mask].copy()
        
        # Filter to top 10% of R² values among valid fits
        if len(df_valid_10hz_all) > 0:
            r2_90th_percentile = df_valid_10hz_all[r2_10hz_col].quantile(0.90)
            df_valid_10hz = df_valid_10hz_all[df_valid_10hz_all[r2_10hz_col] >= r2_90th_percentile].copy()
            print(f"Units with valid 10 Hz R²: {len(df_valid_10hz_all)}")
            print(f"  Top 10% R² threshold: {r2_90th_percentile:.3f}")
            print(f"  Units in top 10%: {len(df_valid_10hz)}")
        else:
            df_valid_10hz = pd.DataFrame()
    else:
        df_valid_10hz = pd.DataFrame()
        print("Warning: 10 Hz R² column not found")
    
    # Find units with invalid 0.5 Hz R² (< threshold)
    r2_05hz_col = "freq_step_05hz_r_squared"
    if r2_05hz_col in df.columns:
        invalid_05hz_mask = df[r2_05hz_col] < FREQ_R_SQUARED_THRESHOLD
        df_invalid_05hz = df[invalid_05hz_mask].copy()
        print(f"Units with invalid 0.5 Hz R²: {len(df_invalid_05hz)}")
    else:
        df_invalid_05hz = pd.DataFrame()
        print("Warning: 0.5 Hz R² column not found")
    
    # Sample from each group
    n_valid_sample = min(n_valid, len(df_valid_10hz))
    n_invalid_sample = min(n_invalid, len(df_invalid_05hz))
    
    valid_indices = df_valid_10hz.sample(n=n_valid_sample, random_state=random_seed).index if n_valid_sample > 0 else []
    invalid_indices = df_invalid_05hz.sample(n=n_invalid_sample, random_state=random_seed).index if n_invalid_sample > 0 else []
    
    total_rows = n_valid_sample + n_invalid_sample
    n_freqs = len(FREQ_STEP_FREQUENCIES)
    
    if total_rows == 0:
        print("No units to plot!")
        return
    
    fig, axes = plt.subplots(total_rows, n_freqs, figsize=figsize)
    if total_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Color scheme for frequencies
    freq_colors = {
        0.5: '#E57373',   # Red
        1: '#FFB74D',     # Orange
        2: '#81C784',     # Green
        4: '#64B5F6',     # Blue
        10: '#BA68C8',    # Purple
    }
    
    def plot_unit_row(row_idx, unit_idx, row_label_prefix, highlight_freq=None):
        """Plot all frequencies for a single unit in one row."""
        row = df.loc[unit_idx]
        
        # Get trace data
        trials_data = row[trace_column]
        mean_trace = compute_mean_trace(trials_data, apply_filter=False)
        
        if mean_trace is None:
            for col_idx in range(n_freqs):
                axes[row_idx, col_idx].text(
                    0.5, 0.5, 'No data', ha='center', va='center',
                    transform=axes[row_idx, col_idx].transAxes, fontsize=10
                )
            return
        
        for col_idx, freq in enumerate(FREQ_STEP_FREQUENCIES):
            ax = axes[row_idx, col_idx]
            freq_str = freq_to_column_str(freq)
            start, end = FREQ_STEP_BOUNDS[freq]
            
            # Get segment
            if end <= len(mean_trace):
                segment = mean_trace[start:end]
            else:
                ax.text(0.5, 0.5, 'Trace too short', ha='center', va='center',
                       transform=ax.transAxes, fontsize=9)
                continue
            
            n_samples = len(segment)
            time_array = np.arange(n_samples) / SAMPLING_RATE
            frames = np.arange(start, end)
            
            # Get fitted parameters from DataFrame
            amp = row.get(f"freq_step_{freq_str}hz_amp", np.nan)
            phase = row.get(f"freq_step_{freq_str}hz_phase", np.nan)
            r_squared = row.get(f"freq_step_{freq_str}hz_r_squared", np.nan)
            offset = row.get(f"freq_step_{freq_str}hz_offset", np.nan)
            std_val = row.get(f"freq_step_{freq_str}hz_std", np.nan)
            
            # Highlight background if this is the target frequency
            if highlight_freq is not None and freq == highlight_freq:
                is_valid = not pd.isna(r_squared) and r_squared >= FREQ_R_SQUARED_THRESHOLD
                bg_color = '#C8E6C9' if is_valid else '#FFCDD2'  # Green if valid, red if invalid
                ax.set_facecolor(bg_color)
            
            # Plot raw trace
            ax.plot(frames, segment, color='black', linewidth=1.5, alpha=0.8)
            
            # Plot fitted sine if valid
            # Note: fit was done on data starting from skip_frames for non-0.5Hz
            if not pd.isna(phase) and not pd.isna(amp) and amp > 0:
                if freq != 0.5:
                    fit_start_offset = FREQ_FIT_SKIP_FRAMES
                else:
                    fit_start_offset = 0
                
                fit_n_samples = n_samples - fit_start_offset
                fit_time_array = np.arange(fit_n_samples) / SAMPLING_RATE
                fit_frames = np.arange(start + fit_start_offset, end)
                fitted_sine = amp * np.sin(2 * np.pi * freq * fit_time_array + phase) + offset
                ax.plot(fit_frames, fitted_sine, color=freq_colors[freq], 
                       linewidth=2, linestyle='--', alpha=0.9)
                
                # Draw vertical line to show where fitting starts
                if fit_start_offset > 0:
                    ax.axvline(start + fit_start_offset, color='gray', 
                              linestyle=':', linewidth=1, alpha=0.5)
            
            # Feature text box
            amp_str = f"{amp:.1f}" if not pd.isna(amp) else "N/A"
            r2_str = f"{r_squared:.3f}" if not pd.isna(r_squared) else "N/A"
            std_str = f"{std_val:.1f}" if not pd.isna(std_val) else "N/A"
            
            # Color R^2 text based on validity
            r2_valid = not pd.isna(r_squared) and r_squared >= FREQ_R_SQUARED_THRESHOLD
            r2_text_color = 'darkgreen' if r2_valid else 'darkred'
            
            feature_text = f"amp: {amp_str}\nR²: {r2_str}\nstd: {std_str}"
            ax.text(0.98, 0.98, feature_text, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                            edgecolor=r2_text_color, linewidth=1.5),
                   family='monospace', color=r2_text_color)
            
            # Axis settings
            ax.set_xlim(start, end)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.tick_params(axis='both', labelsize=6)
            
            # Column title (frequency) - only on first row
            if row_idx == 0:
                ax.set_title(f'{freq} Hz', fontsize=11, fontweight='bold',
                           color=freq_colors[freq])
            
            # Row label - only on first column
            if col_idx == 0:
                ax.set_ylabel(f'{row_label_prefix}\nFR (Hz)', fontsize=8)
    
    # Plot valid 10 Hz units (top rows)
    for i, unit_idx in enumerate(valid_indices):
        plot_unit_row(i, unit_idx, f"Valid 10Hz #{i+1}", highlight_freq=10)
    
    # Add separator line
    if n_valid_sample > 0 and n_invalid_sample > 0:
        # Add a visual separator
        for col_idx in range(n_freqs):
            axes[n_valid_sample - 1, col_idx].spines['bottom'].set_linewidth(3)
            axes[n_valid_sample - 1, col_idx].spines['bottom'].set_color('black')
    
    # Plot invalid 0.5 Hz units (bottom rows)
    for i, unit_idx in enumerate(invalid_indices):
        row_idx = n_valid_sample + i
        plot_unit_row(row_idx, unit_idx, f"Invalid 0.5Hz #{i+1}", highlight_freq=0.5)
    
    # Overall title
    fig.suptitle(
        f'Frequency Step Validation: Top 10% R² for 10 Hz (top {n_valid_sample}) vs Invalid 0.5 Hz (bottom {n_invalid_sample})\n'
        f'(Green/Red background = highlighted frequency, dashed = fitted sine)',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved valid/invalid comparison plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    """Load data and generate validation plots."""
    # Paths
    input_path = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected_step_gb_freq.parquet")
    output_dir = Path("dataframe_phase/extract_feature/validation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Frequency Step Feature Validation Plots")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded DataFrame shape: {df.shape}")
    
    # Print config
    print("\nFrame indices from config:")
    for freq, (start, end) in FREQ_STEP_BOUNDS.items():
        print(f"  {freq} Hz: [{start}:{end}]")
    print(f"  R² threshold: {FREQ_R_SQUARED_THRESHOLD}")
    
    # 1. Sine fit visualization (main validation plot)
    print("\n" + "-" * 40)
    print("Generating sine fit validation plot...")
    plot_freq_step_sine_fits(
        df,
        trace_column=FREQ_TRACE_COLUMN,
        qi_column="step_up_QI",
        output_path=output_dir / "freq_features_sine_fits.png",
        n_units=6,
        top_percentile=0.20,
        figsize=(28, 24),
        random_seed=42,
    )
    
    # 2. Full trace with fits
    print("\n" + "-" * 40)
    print("Generating full trace plot with fits...")
    plot_freq_step_full_trace_with_fits(
        df,
        trace_column=FREQ_TRACE_COLUMN,
        qi_column="step_up_QI",
        output_path=output_dir / "freq_features_full_traces.png",
        n_units=4,
        top_percentile=0.20,
        figsize=(24, 16),
        random_seed=42,
    )
    
    # 3. Feature distributions
    print("\n" + "-" * 40)
    print("Generating feature distribution plot...")
    plot_freq_feature_distributions(
        df,
        output_path=output_dir / "freq_features_distributions.png",
        figsize=(28, 20),
    )
    
    # 4. R-squared analysis
    print("\n" + "-" * 40)
    print("Generating R² analysis plot...")
    plot_r_squared_analysis(
        df,
        output_path=output_dir / "freq_features_r_squared_analysis.png",
        figsize=(16, 10),
    )
    
    # 5. Statistics table
    print("\n" + "-" * 40)
    print("Generating statistics table...")
    plot_feature_statistics_table(
        df,
        output_path=output_dir / "freq_features_statistics.png",
        figsize=(20, 12),
    )
    
    # 6. Valid/Invalid comparison plot
    print("\n" + "-" * 40)
    print("Generating valid 10Hz vs invalid 0.5Hz comparison plot...")
    plot_valid_invalid_comparison(
        df,
        trace_column=FREQ_TRACE_COLUMN,
        output_path=output_dir / "freq_features_valid_invalid_comparison.png",
        n_valid=5,
        n_invalid=5,
        figsize=(28, 32),
        random_seed=42,
    )
    
    print("\n" + "=" * 60)
    print("Done! All plots saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

