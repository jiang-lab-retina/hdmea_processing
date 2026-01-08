"""
Plot preprocessed traces for visual validation.

This script loads data, applies the preprocessing pipeline, and plots
20 random units' traces to visually verify the preprocessing is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
_THIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataframe_phase.classification_v2.Baden_method import config, preprocessing


def plot_validation_traces(
    n_units: int = 20,
    output_dir: str = None,
    random_seed: int = 42,
):
    """
    Plot preprocessed traces for n random units.
    
    Args:
        n_units: Number of random units to plot.
        output_dir: Output directory for plots.
        random_seed: Random seed for reproducibility.
    """
    if output_dir is None:
        # Use config paths (already absolute)
        output_dir = config.PLOTS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load and filter data
    print("Loading data...")
    # Use config path (already absolute)
    df = preprocessing.load_data(config.INPUT_PATH)
    print(f"Loaded {len(df)} cells")
    
    print("Filtering cells...")
    df_filtered = preprocessing.filter_rows(df)
    print(f"After filtering: {len(df_filtered)} cells")
    
    # Select random units
    np.random.seed(random_seed)
    n_units = min(n_units, len(df_filtered))
    selected_indices = np.random.choice(df_filtered.index, size=n_units, replace=False)
    df_sample = df_filtered.loc[selected_indices].copy()
    
    print(f"Selected {n_units} random units for visualization")
    
    # Store raw traces before preprocessing
    raw_traces = {}
    for col in [config.CHIRP_COL, config.COLOR_COL, config.BASELINE_TRACE_COL] + config.BAR_COLS[:2]:
        raw_traces[col] = df_sample[col].apply(preprocessing.average_trials).values
    
    # Apply preprocessing
    print("Preprocessing traces...")
    df_preprocessed = preprocessing.preprocess_traces(df_sample)
    
    # Plot each unit
    print("Generating plots...")
    
    # Create figure with subplots for each stimulus type
    fig, axes = plt.subplots(n_units, 5, figsize=(20, 3 * n_units))
    if n_units == 1:
        axes = axes.reshape(1, -1)
    
    # Column titles
    col_titles = ['Chirp (freq_step)', 'Color (green_blue)', 'Step-Up (baseline src)', 
                  'Bar Dir 0°', 'Bar Dir 45°']
    trace_cols = [config.CHIRP_COL, config.COLOR_COL, config.BASELINE_TRACE_COL,
                  config.BAR_COLS[0], config.BAR_COLS[1]]
    
    for row_idx, (df_idx, row) in enumerate(df_preprocessed.iterrows()):
        for col_idx, (title, col) in enumerate(zip(col_titles, trace_cols)):
            ax = axes[row_idx, col_idx]
            
            # Get trace - special handling for baseline trace (not preprocessed)
            if col == config.BASELINE_TRACE_COL:
                # Baseline trace is not preprocessed, use raw averaged trace
                trace = raw_traces[col][row_idx]
                try:
                    trace = np.asarray(trace, dtype=np.float64).flatten()
                    # Normalize for display
                    trace = trace - np.median(trace[:8])
                    max_abs = np.max(np.abs(trace)) + 1e-8
                    trace = trace / max_abs
                except (ValueError, TypeError):
                    ax.text(0.5, 0.5, 'Invalid trace', ha='center', va='center', transform=ax.transAxes)
                    continue
            else:
                # Get preprocessed trace and ensure it's a flat float array
                trace = row[col]
                try:
                    trace = np.asarray(trace, dtype=np.float64).flatten()
                except (ValueError, TypeError):
                    # If conversion fails, skip this trace
                    ax.text(0.5, 0.5, 'Invalid trace', ha='center', va='center', transform=ax.transAxes)
                    continue
            
            if len(trace) > 0:
                # Use appropriate sampling rate for time axis
                if col == config.BASELINE_TRACE_COL:
                    t = np.arange(len(trace)) / config.SAMPLING_RATE  # 60 Hz (raw)
                    ax.plot(t, trace, 'g-', linewidth=1, label='Raw (normalized)')
                else:
                    t = np.arange(len(trace)) / config.TARGET_SAMPLING_RATE  # 10 Hz (preprocessed)
                    ax.plot(t, trace, 'b-', linewidth=1, label='Preprocessed')
                
                # Also plot raw trace (scaled for comparison) - skip for baseline col
                if col in raw_traces and col != config.BASELINE_TRACE_COL:
                    raw = raw_traces[col][row_idx]
                    try:
                        raw = np.asarray(raw, dtype=np.float64).flatten()
                        t_raw = np.arange(len(raw)) / config.SAMPLING_RATE  # Time at 60 Hz
                        # Normalize raw for display
                        raw_norm = raw - np.median(raw[:8])
                        max_abs = np.max(np.abs(raw_norm)) + 1e-8
                        raw_norm = raw_norm / max_abs
                        ax.plot(t_raw, raw_norm, 'r-', linewidth=0.5, alpha=0.5, label='Raw (norm)')
                    except (ValueError, TypeError):
                        pass  # Skip raw trace if conversion fails
            
            # Set title for first row
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            
            # Set ylabel for first column
            if col_idx == 0:
                ax.set_ylabel(f'Unit {df_idx}', fontsize=8)
            
            ax.set_xlim(0, max(t) if len(trace) > 0 else 1)
            ax.set_ylim(-1.2, 1.2)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            
            # Add legend only for first plot
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=6, loc='upper right')
    
    # Add common labels
    fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Normalized Response', va='center', rotation='vertical', fontsize=12)
    
    plt.suptitle(f'Preprocessed Traces - {n_units} Random Units\n'
                 f'(10 Hz low-pass, baseline from step_up, max-abs normalized, downsampled to 10 Hz)',
                 fontsize=14)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    
    # Save figure
    output_path = output_dir / 'preprocessed_traces_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Also create a summary statistics plot
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot baseline values distribution
    ax = axes2[0, 0]
    baselines = []
    for idx in df_preprocessed.index:
        trace = df_sample.loc[idx, config.BASELINE_TRACE_COL]
        trace = preprocessing.average_trials(trace)
        trace = preprocessing.lowpass_filter(trace)
        trace = preprocessing.downsample(trace)
        baseline = np.median(trace[:config.BASELINE_N_SAMPLES])
        baselines.append(baseline)
    
    ax.hist(baselines, bins=30, edgecolor='black')
    ax.set_xlabel('Baseline Value (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Baseline Values\n(from step_up trace)')
    ax.axvline(np.median(baselines), color='r', linestyle='--', label=f'Median: {np.median(baselines):.2f}')
    ax.legend()
    
    # Plot trace lengths after preprocessing
    ax = axes2[0, 1]
    trace_lengths = {col: [] for col in [config.CHIRP_COL, config.COLOR_COL]}
    for col in trace_lengths:
        for trace in df_preprocessed[col]:
            try:
                trace_arr = np.asarray(trace, dtype=np.float64).flatten()
                trace_lengths[col].append(len(trace_arr))
            except (ValueError, TypeError):
                trace_lengths[col].append(0)
    
    x_pos = np.arange(len(trace_lengths))
    means = [np.mean(v) for v in trace_lengths.values()]
    ax.bar(x_pos, means)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Chirp', 'Color'])
    ax.set_ylabel('Trace Length (samples @ 10 Hz)')
    ax.set_title('Preprocessed Trace Lengths')
    
    # Plot example preprocessed chirp trace with annotations
    ax = axes2[0, 2]
    example_trace = np.asarray(df_preprocessed.iloc[0][config.CHIRP_COL], dtype=np.float64).flatten()
    t = np.arange(len(example_trace)) / config.TARGET_SAMPLING_RATE
    ax.plot(t, example_trace, 'b-', linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(1, color='r', linestyle=':', linewidth=0.5, label='Max norm = 1')
    ax.axhline(-1, color='r', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Response')
    ax.set_title('Example Preprocessed Chirp Trace')
    ax.legend()
    
    # Plot raw vs preprocessed comparison for one unit
    ax = axes2[1, 0]
    raw_chirp = np.asarray(raw_traces[config.CHIRP_COL][0], dtype=np.float64).flatten()
    t_raw = np.arange(len(raw_chirp)) / config.SAMPLING_RATE
    ax.plot(t_raw, raw_chirp, 'r-', linewidth=0.8, label='Raw (60 Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Raw Chirp Trace (before preprocessing)')
    ax.legend()
    
    ax = axes2[1, 1]
    proc_chirp = np.asarray(df_preprocessed.iloc[0][config.CHIRP_COL], dtype=np.float64).flatten()
    t_proc = np.arange(len(proc_chirp)) / config.TARGET_SAMPLING_RATE
    ax.plot(t_proc, proc_chirp, 'b-', linewidth=1, label='Preprocessed (10 Hz)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized Response')
    ax.set_title('Preprocessed Chirp Trace (after pipeline)')
    ax.legend()
    
    # Show preprocessing parameters
    ax = axes2[1, 2]
    ax.axis('off')
    params_text = f"""
    Preprocessing Parameters:
    ─────────────────────────
    Sampling Rate (original): {config.SAMPLING_RATE} Hz
    Target Sampling Rate: {config.TARGET_SAMPLING_RATE} Hz
    Downsample Factor: {config.DOWNSAMPLE_FACTOR}x
    
    Low-pass Filter:
      Cutoff: {config.LOWPASS_CUTOFF} Hz
      Order: {config.FILTER_ORDER}
    
    Baseline:
      Source: {config.BASELINE_TRACE_COL}
      Samples: {config.BASELINE_N_SAMPLES} (at 10 Hz)
      Method: Median
    
    Normalization:
      Method: Max-absolute value
      Epsilon: {config.NORMALIZE_EPS}
    
    Units plotted: {n_units}
    Total filtered cells: {len(df_filtered)}
    """
    ax.text(0.1, 0.9, params_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Preprocessing Validation Summary', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path2 = output_dir / 'preprocessing_summary.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")
    plt.close()
    
    print("\nValidation plots complete!")
    return output_dir


if __name__ == "__main__":
    plot_validation_traces()

