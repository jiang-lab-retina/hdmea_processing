"""
Validation plots for extracted features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path


def bessel_lowpass_filter(
    trace: np.ndarray,
    cutoff_freq: float = 10.0,
    order: int = 5,
    sampling_rate: float = 60.0,
) -> np.ndarray:
    """
    Apply Bessel low-pass filter to a trace.
    """
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1.0:
        return trace  # Can't filter, return original
    
    b, a = signal.bessel(order, normalized_cutoff, btype='low', analog=False, output='ba')
    filtered_trace = signal.filtfilt(b, a, trace)
    return filtered_trace


def plot_qi_example_traces(
    df: pd.DataFrame,
    trace_column: str = "step_up_5s_5i_b0_3x",
    qi_column: str = "step_up_QI",
    output_path: Path | str | None = None,
    n_examples: int = 5,
    n_ranges: int = 10,
    figsize: tuple = (16, 20),
    max_samples: int | None = None,
    sampling_rate: float = 60.0,
    filter_cutoff: float | None = None,
    filter_order: int = 5,
    qi_min: float = 0.0,
    qi_max: float = 1.0,
):
    """
    Plot example traces for different QI ranges.
    
    Creates a grid of subplots with:
    - Rows: QI ranges (0-0.1, 0.1-0.2, ..., 0.9-1.0)
    - Columns: Example units from each range
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with trace data and QI column
    trace_column : str
        Column containing the firing rate traces
    qi_column : str
        Column containing quality index values
    output_path : Path or str, optional
        Path to save the figure
    n_examples : int
        Number of examples per QI range
    n_ranges : int
        Number of QI ranges (default 10 for 0.1 increments)
    figsize : tuple
        Figure size
    max_samples : int, optional
        Maximum number of samples to display. If None, shows full trace.
    sampling_rate : float
        Sampling rate in Hz for time axis calculation
    filter_cutoff : float, optional
        Lowpass filter cutoff frequency in Hz. If None, no filtering.
    filter_order : int
        Bessel filter order (default 5)
    qi_min : float
        Minimum QI value for range (default 0.0)
    qi_max : float
        Maximum QI value for range (default 1.0)
    """
    fig, axes = plt.subplots(n_ranges, n_examples, figsize=figsize)
    
    # Ensure axes is 2D even with single row
    if n_ranges == 1:
        axes = axes.reshape(1, -1)
    
    # Define QI ranges
    qi_bins = np.linspace(qi_min, qi_max, n_ranges + 1)
    
    for row_idx in range(n_ranges):
        qi_low = qi_bins[row_idx]
        qi_high = qi_bins[row_idx + 1]
        
        # Filter units in this QI range
        mask = (df[qi_column] >= qi_low) & (df[qi_column] < qi_high)
        if row_idx == n_ranges - 1:  # Include 1.0 in the last bin
            mask = (df[qi_column] >= qi_low) & (df[qi_column] <= qi_high)
        
        df_range = df[mask]
        
        # Sample examples (or take all if fewer than n_examples)
        n_available = min(n_examples, len(df_range))
        if n_available > 0:
            sample_indices = df_range.sample(n=n_available, random_state=42).index
        else:
            sample_indices = []
        
        for col_idx in range(n_examples):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(sample_indices):
                idx = sample_indices[col_idx]
                data = df.loc[idx, trace_column]
                qi_val = df.loc[idx, qi_column]
                
                # Stack trials into 2D array
                try:
                    trials_array = np.vstack([np.array(trial) for trial in data])
                    n_trials, n_timepoints = trials_array.shape
                    
                    # Apply lowpass filter if specified
                    if filter_cutoff is not None:
                        for trial_idx in range(n_trials):
                            try:
                                trials_array[trial_idx] = bessel_lowpass_filter(
                                    trials_array[trial_idx],
                                    cutoff_freq=filter_cutoff,
                                    order=filter_order,
                                    sampling_rate=sampling_rate,
                                )
                            except Exception:
                                pass  # Keep original if filtering fails
                    
                    # Trim to max_samples if specified
                    if max_samples is not None and n_timepoints > max_samples:
                        trials_array = trials_array[:, :max_samples]
                        n_timepoints = max_samples
                    
                    # Time axis
                    time = np.arange(n_timepoints) / sampling_rate
                    
                    # Plot individual trials (transparent)
                    for trial_idx in range(n_trials):
                        ax.plot(time, trials_array[trial_idx], 
                               color='steelblue', alpha=0.5, linewidth=0.8)
                    
                    # Plot mean (solid, on top)
                    mean_trace = trials_array.mean(axis=0)
                    ax.plot(time, mean_trace, color='darkred', linewidth=1.5)
                    
                    # Title with QI value
                    ax.set_title(f'QI={qi_val:.3f}', fontsize=8)
                    
                except Exception:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=8)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_facecolor('#f0f0f0')
            
            # Row labels (QI range) on leftmost column
            if col_idx == 0:
                # Determine decimal places based on interval size
                interval = qi_high - qi_low
                if interval < 0.1:
                    ax.set_ylabel(f'{qi_low:.2f}-{qi_high:.2f}', fontsize=9, fontweight='bold')
                else:
                    ax.set_ylabel(f'{qi_low:.1f}-{qi_high:.1f}', fontsize=9, fontweight='bold')
            
            # Remove x ticks except for bottom row
            if row_idx < n_ranges - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=8)
            
            # Clean up ticks
            ax.tick_params(axis='both', labelsize=7)
    
    # Add column headers
    for col_idx in range(n_examples):
        axes[0, col_idx].text(0.5, 1.15, f'Example {col_idx + 1}', 
                              ha='center', va='bottom', 
                              transform=axes[0, col_idx].transAxes,
                              fontsize=10, fontweight='bold')
    
    # Overall title
    filter_info = f", {filter_cutoff} Hz lowpass" if filter_cutoff else ""
    fig.suptitle(f'{qi_column} - Response Traces by Quality Index Range\n'
                 f'(Blue: individual trials, Red: mean{filter_info})', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved example traces plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_qi_threshold_curve(
    df: pd.DataFrame,
    qi_column: str = "step_up_QI",
    output_path: Path | str | None = None,
    figsize: tuple = (10, 6),
    n_thresholds: int = 100,
):
    """
    Plot qualified cell percentage vs QI threshold.
    
    Shows what percentage of cells would pass (have QI >= threshold)
    for different threshold values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the quality index column
    qi_column : str
        Name of the quality index column
    output_path : Path or str, optional
        Path to save the figure. If None, displays interactively.
    figsize : tuple
        Figure size (width, height)
    n_thresholds : int
        Number of threshold points to evaluate
    """
    qi_values = df[qi_column].dropna()
    total_cells = len(qi_values)
    
    # Create threshold range
    thresholds = np.linspace(0, 1, n_thresholds)
    
    # Calculate percentage passing each threshold
    percentages = []
    for thresh in thresholds:
        n_passing = (qi_values >= thresh).sum()
        pct = 100.0 * n_passing / total_cells
        percentages.append(pct)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the curve
    ax.plot(thresholds, percentages, color='steelblue', linewidth=2)
    ax.fill_between(thresholds, percentages, alpha=0.3, color='steelblue')
    
    # Add reference lines at common thresholds
    common_thresholds = [0.3, 0.5, 0.7]
    for thresh in common_thresholds:
        pct_at_thresh = 100.0 * (qi_values >= thresh).sum() / total_cells
        ax.axvline(thresh, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(pct_at_thresh, color='gray', linestyle=':', alpha=0.3)
        ax.scatter([thresh], [pct_at_thresh], color='darkred', s=50, zorder=5)
        ax.annotate(
            f'{pct_at_thresh:.1f}%',
            xy=(thresh, pct_at_thresh),
            xytext=(thresh + 0.05, pct_at_thresh + 3),
            fontsize=9,
            color='darkred',
        )
    
    # Labels and title
    ax.set_xlabel('QI Threshold', fontsize=12)
    ax.set_ylabel('Qualified Cells (%)', fontsize=12)
    ax.set_title(f'{qi_column} - Qualified Cell Percentage vs Threshold\n(n={total_cells:,} total cells)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Add summary text
    summary_lines = []
    for thresh in common_thresholds:
        n_pass = (qi_values >= thresh).sum()
        pct = 100.0 * n_pass / total_cells
        summary_lines.append(f'QI ≥ {thresh}: {n_pass:,} ({pct:.1f}%)')
    
    ax.text(
        0.98, 0.95, '\n'.join(summary_lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved threshold curve to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_quality_index_histogram(
    df: pd.DataFrame,
    qi_column: str = "step_up_QI",
    output_path: Path | str | None = None,
    bins: int = 50,
    figsize: tuple = (10, 6),
):
    """
    Plot histogram of quality index values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the quality index column
    qi_column : str
        Name of the quality index column
    output_path : Path or str, optional
        Path to save the figure. If None, displays interactively.
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size (width, height)
    """
    qi_values = df[qi_column].dropna()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    counts, bin_edges, patches = ax.hist(
        qi_values, 
        bins=bins, 
        color='steelblue', 
        edgecolor='white',
        alpha=0.8,
    )
    
    # Add statistics lines
    mean_val = qi_values.mean()
    median_val = qi_values.median()
    
    ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='darkorange', linestyle='-', linewidth=2, label=f'Median: {median_val:.3f}')
    
    # Labels and title
    ax.set_xlabel('Quality Index', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {qi_column}\n(n={len(qi_values):,} units)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add statistics text box
    stats_text = (
        f'Mean: {mean_val:.4f}\n'
        f'Median: {median_val:.4f}\n'
        f'Std: {qi_values.std():.4f}\n'
        f'Min: {qi_values.min():.4f}\n'
        f'Max: {qi_values.max():.4f}\n'
        f'Valid: {len(qi_values):,} / {len(df):,}'
    )
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Load data and generate validation plots."""
    # Paths
    input_path = Path("dataframe_phase/extract_feature/firing_rate_with_features.parquet")
    output_dir = Path("dataframe_phase/extract_feature/validation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded DataFrame shape: {df.shape}")
    
    # Plot quality index histogram
    print("\nGenerating quality index histogram...")
    plot_quality_index_histogram(
        df,
        qi_column="step_up_QI",
        output_path=output_dir / "step_up_QI_histogram.png",
        bins=50,
    )
    
    # Plot example traces by QI range
    print("\nGenerating example traces plot (10 QI ranges x 5 examples, 10 Hz filter)...")
    plot_qi_example_traces(
        df,
        trace_column="step_up_5s_5i_b0_3x",
        qi_column="step_up_QI",
        output_path=output_dir / "step_up_QI_example_traces.png",
        n_examples=5,
        n_ranges=10,
        filter_cutoff=10.0,  # 10 Hz lowpass filter
        sampling_rate=60.0,
    )
    
    # =========================================================================
    # ipRGC 2Hz QI validation plots
    # =========================================================================
    print("\nGenerating iprgc_2hz_QI histogram...")
    plot_quality_index_histogram(
        df,
        qi_column="iprgc_2hz_QI",
        output_path=output_dir / "iprgc_2hz_QI_histogram.png",
        bins=50,
    )
    
    print("\nGenerating iprgc_2hz_QI threshold curve...")
    plot_qi_threshold_curve(
        df,
        qi_column="iprgc_2hz_QI",
        output_path=output_dir / "iprgc_2hz_QI_threshold_curve.png",
    )
    
    print("\nGenerating iprgc_2hz_QI example traces plot (10 QI ranges x 10 examples, 2 Hz filter)...")
    plot_qi_example_traces(
        df,
        trace_column="iprgc_test",
        qi_column="iprgc_2hz_QI",
        output_path=output_dir / "iprgc_2hz_QI_example_traces.png",
        n_examples=10,
        n_ranges=10,
        figsize=(28, 20),
        filter_cutoff=2.0,  # 2 Hz lowpass filter
        sampling_rate=60.0,
    )
    
    # =========================================================================
    # ipRGC 20Hz QI validation plots
    # =========================================================================
    print("\nGenerating iprgc_20hz_QI histogram...")
    plot_quality_index_histogram(
        df,
        qi_column="iprgc_20hz_QI",
        output_path=output_dir / "iprgc_20hz_QI_histogram.png",
        bins=50,
    )
    
    print("\nGenerating iprgc_20hz_QI example traces plot (10 QI ranges x 10 examples, first 10s, 20 Hz filter)...")
    plot_qi_example_traces(
        df,
        trace_column="iprgc_test",
        qi_column="iprgc_20hz_QI",
        output_path=output_dir / "iprgc_20hz_QI_example_traces.png",
        n_examples=10,
        n_ranges=10,
        figsize=(28, 20),
        max_samples=600,  # First 10 seconds at 60 Hz
        sampling_rate=60.0,
        filter_cutoff=20.0,  # 20 Hz lowpass filter
    )
    
    # =========================================================================
    # Detailed ipRGC QI plots for high QI range (0.7-1.0)
    # =========================================================================
    print("\nGenerating iprgc_2hz_QI detail traces plot (QI 0.5-1.0, 0.02 interval, 20 examples)...")
    plot_qi_example_traces(
        df,
        trace_column="iprgc_test",
        qi_column="iprgc_2hz_QI",
        output_path=output_dir / "iprgc_2hz_QI_example_traces_detail_range.png",
        n_examples=20,
        n_ranges=25,  # 0.02 interval from 0.5 to 1.0
        figsize=(50, 50),
        filter_cutoff=2.0,
        sampling_rate=60.0,
        qi_min=0.5,
        qi_max=1.0,
    )
    
    print("\nGenerating iprgc_20hz_QI detail traces plot (QI 0.5-1.0, 0.02 interval, 20 examples, first 10s)...")
    plot_qi_example_traces(
        df,
        trace_column="iprgc_test",
        qi_column="iprgc_20hz_QI",
        output_path=output_dir / "iprgc_20hz_QI_example_traces_detail_range.png",
        n_examples=20,
        n_ranges=25,  # 0.02 interval from 0.5 to 1.0
        figsize=(50, 50),
        max_samples=600,
        filter_cutoff=20.0,
        sampling_rate=60.0,
        qi_min=0.5,
        qi_max=1.0,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

