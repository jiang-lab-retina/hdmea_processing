"""
Investigation: Visualize step responses of units with highest step QI.

Plots the step responses (individual trials + mean) for the 10 units
with the highest step_up_QI values.
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
    """Apply Bessel low-pass filter to a trace."""
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1.0:
        return trace
    
    b, a = signal.bessel(order, normalized_cutoff, btype='low', analog=False, output='ba')
    filtered_trace = signal.filtfilt(b, a, trace)
    return filtered_trace


def plot_top_qi_responses(
    df: pd.DataFrame,
    trace_column: str = "step_up_5s_5i_b0_3x",
    qi_column: str = "step_up_QI",
    output_path: Path | str | None = None,
    n_units: int = 10,
    figsize: tuple = (20, 12),
    sampling_rate: float = 60.0,
    filter_cutoff: float = 10.0,
    filter_order: int = 5,
    title: str | None = None,
    x_tick_interval: int = 10,
    n_cols: int = 5,
):
    """
    Plot responses for units with highest QI values.
    
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
    n_units : int
        Number of top units to plot
    figsize : tuple
        Figure size
    sampling_rate : float
        Sampling rate in Hz
    filter_cutoff : float
        Lowpass filter cutoff frequency in Hz
    filter_order : int
        Bessel filter order
    title : str, optional
        Custom title for the plot
    x_tick_interval : int
        Interval for x-axis ticks (default 10)
    n_cols : int
        Number of columns in subplot grid (default 5)
    """
    # Filter out NaN QI values and sort by QI descending
    df_valid = df[df[qi_column].notna()].copy()
    df_sorted = df_valid.sort_values(qi_column, ascending=False)
    
    # Take top n_units
    top_units = df_sorted.head(n_units)
    
    print(f"Top {n_units} units by {qi_column}:")
    for i, (idx, row) in enumerate(top_units.iterrows(), 1):
        print(f"  {i}. Index {idx}: QI = {row[qi_column]:.4f}")
    
    # Create subplot grid
    n_rows = (n_units + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for plot_idx, (unit_idx, row) in enumerate(top_units.iterrows()):
        ax = axes[plot_idx]
        data = row[trace_column]
        qi_val = row[qi_column]
        
        try:
            # Stack trials into 2D array
            valid_trials = [np.array(trial) for trial in data if trial is not None]
            trials_array = np.vstack(valid_trials)
            n_trials, n_timepoints = trials_array.shape
            
            # Apply lowpass filter
            for trial_idx in range(n_trials):
                try:
                    trials_array[trial_idx] = bessel_lowpass_filter(
                        trials_array[trial_idx],
                        cutoff_freq=filter_cutoff,
                        order=filter_order,
                        sampling_rate=sampling_rate,
                    )
                except Exception:
                    pass
            
            # Time axis in frames
            time_frames = np.arange(n_timepoints)
            
            # Plot individual trials (transparent)
            for trial_idx in range(n_trials):
                ax.plot(time_frames, trials_array[trial_idx],
                       color='steelblue', alpha=0.4, linewidth=0.8)
            
            # Plot mean (solid, on top)
            mean_trace = trials_array.mean(axis=0)
            ax.plot(time_frames, mean_trace, color='darkred', linewidth=2,
                   label='Mean')
            
            # Title with rank and QI value
            ax.set_title(f'Rank #{plot_idx + 1}\nQI = {qi_val:.4f}', 
                        fontsize=11, fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9)
        
        # Enable grid
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        
        # Set x ticks as multiples of x_tick_interval
        max_frame = n_timepoints if 'n_timepoints' in dir() else 600
        ax.set_xticks(np.arange(0, max_frame + 1, x_tick_interval))
        ax.tick_params(axis='x', labelsize=6, rotation=45)
        ax.tick_params(axis='y', labelsize=8)
        
        # Labels
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_units, len(axes)):
        axes[i].set_visible(False)
    
    # Overall title
    if title is None:
        title = f'Top {n_units} Units by {qi_column} - {trace_column} Traces'
    fig.suptitle(
        f'{title}\n'
        f'(Blue: individual trials, Red: mean, {filter_cutoff} Hz lowpass filter)',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Load data and generate visualization."""
    # Paths
    input_path = Path("dataframe_phase/extract_feature/firing_rate_with_dsgc_features_typed20251230_dsgc_corrected.parquet")
    output_dir = Path("dataframe_phase/extract_feature/investigation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check step_up_QI statistics
    qi_col = "step_up_QI"
    if qi_col in df.columns:
        valid_qi = df[qi_col].dropna()
        print(f"\n{qi_col} statistics:")
        print(f"  Valid values: {len(valid_qi)} / {len(df)}")
        print(f"  Mean: {valid_qi.mean():.4f}")
        print(f"  Max: {valid_qi.max():.4f}")
        print(f"  Min: {valid_qi.min():.4f}")
    else:
        print(f"\nWarning: {qi_col} column not found!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # ==========================================================================
    # Plot 1: Step Up responses
    # ==========================================================================
    print("\n" + "="*60)
    print("Generating top QI step_up response plot...")
    plot_top_qi_responses(
        df,
        trace_column="step_up_5s_5i_b0_3x",
        qi_column="step_up_QI",
        output_path=output_dir / "top_10_step_qi_step_up_responses.png",
        n_units=10,
        figsize=(22, 10),
        sampling_rate=60.0,
        filter_cutoff=10.0,
        title="Top 10 Units by Step QI - Step Up Responses",
        x_tick_interval=10,
    )
    
    # ==========================================================================
    # Plot 2: Green-Blue responses
    # ==========================================================================
    print("\n" + "="*60)
    print("Generating top QI green_blue response plot...")
    plot_top_qi_responses(
        df,
        trace_column="green_blue_3s_3i_3x",
        qi_column="step_up_QI",
        output_path=output_dir / "top_10_step_qi_green_blue_responses.png",
        n_units=10,
        figsize=(22, 10),
        sampling_rate=60.0,
        filter_cutoff=10.0,
        title="Top 10 Units by Step QI - Green-Blue Responses",
        x_tick_interval=10,
    )
    
    # ==========================================================================
    # Plot 3: Freq Step responses
    # ==========================================================================
    print("\n" + "="*60)
    print("Generating top QI freq_step response plot...")
    plot_top_qi_responses(
        df,
        trace_column="freq_step_5st_3x",
        qi_column="step_up_QI",
        output_path=output_dir / "top_10_step_qi_freq_step_responses.png",
        n_units=10,
        figsize=(22, 10),
        sampling_rate=60.0,
        filter_cutoff=10.0,
        title="Top 10 Units by Step QI - Freq Step Responses",
        x_tick_interval=10,
    )
    
    # ==========================================================================
    # Plot 4: Freq Step responses (3 examples, 3 rows vertical layout)
    # ==========================================================================
    print("\n" + "="*60)
    print("Generating top 3 freq_step response plot (3 rows)...")
    plot_top_qi_responses(
        df,
        trace_column="freq_step_5st_3x",
        qi_column="step_up_QI",
        output_path=output_dir / "top_3_step_qi_freq_step_responses_3rows.png",
        n_units=3,
        figsize=(48, 6),
        sampling_rate=60.0,
        filter_cutoff=10.0,
        title="Top 3 Units by Step QI - Freq Step Responses",
        x_tick_interval=10,
        n_cols=1,
    )
    
    print("\n" + "="*60)
    print("Done! All plots saved.")


if __name__ == "__main__":
    main()

