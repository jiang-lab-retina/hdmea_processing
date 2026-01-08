"""
DSGC Validation Plots.

Visualizes direction and orientation selectivity with:
- Polar plots showing response amplitude at each direction
- Trace grids showing raw traces for each direction
- Histograms for DSI/OSI and their p-values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from pathlib import Path
from typing import List, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths
INPUT_PARQUET = Path(__file__).parent.parent / "firing_rate_with_dsgc_features20251230.parquet"
OUTPUT_DIR = Path(__file__).parent / "plots"

# Direction configuration
DIRECTION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
DIRECTION_COLUMNS = [f"moving_h_bar_s5_d8_3x_{angle}" for angle in DIRECTION_ANGLES]

# Plot settings
N_EXAMPLE_UNITS = 20
RANDOM_SEED = 42

# Colors
MEAN_COLOR = "steelblue"
TRIAL_COLOR = "lightblue"
POLAR_COLOR = "coral"

# =============================================================================
# POLAR PLOT FUNCTION
# =============================================================================


def get_total_firing_rate_per_trial(trial_traces: np.ndarray) -> np.ndarray:
    """Compute total firing rate for each trial by summing the trace."""
    totals = []
    for trace in trial_traces:
        if trace is not None and len(trace) > 0:
            totals.append(np.sum(trace))
        else:
            totals.append(0.0)
    return np.array(totals)


def plot_polar_tuning(
    ax: plt.Axes,
    row: pd.Series,
    direction_columns: List[str],
    directions: np.ndarray,
    title: str = "",
    show_preferred: bool = True,
    index_value: Optional[float] = None,
):
    """
    Plot polar tuning curve showing response amplitude at each direction.
    
    Parameters
    ----------
    ax : plt.Axes
        Polar axis to plot on
    row : pd.Series
        DataFrame row containing direction data
    direction_columns : List[str]
        Column names for each direction
    directions : np.ndarray
        Direction angles in degrees
    title : str
        Plot title
    show_preferred : bool
        Whether to show preferred direction arrow
    index_value : float, optional
        DSI or OSI value to scale the arrow length (0-1 range expected)
    """
    # Compute mean response per direction
    mean_responses = []
    for col in direction_columns:
        trial_traces = row.get(col)
        if trial_traces is None:
            mean_responses.append(0)
            continue
        
        if isinstance(trial_traces, list):
            trial_traces = np.array(trial_traces, dtype=object)
        
        totals = get_total_firing_rate_per_trial(trial_traces)
        mean_responses.append(np.mean(totals))
    
    mean_responses = np.array(mean_responses)
    
    # Convert to radians for polar plot
    angles_rad = np.deg2rad(directions)
    
    # Close the polar plot by appending first value
    angles_closed = np.append(angles_rad, angles_rad[0])
    responses_closed = np.append(mean_responses, mean_responses[0])
    
    # Plot
    ax.plot(angles_closed, responses_closed, color=POLAR_COLOR, linewidth=2)
    ax.fill(angles_closed, responses_closed, color=POLAR_COLOR, alpha=0.3)
    ax.scatter(angles_rad, mean_responses, color=POLAR_COLOR, s=30, zorder=5)
    
    # Show preferred direction arrow with length proportional to DSI/OSI
    if show_preferred and "preferred_direction" in row.index:
        pref_dir = row["preferred_direction"]
        if not np.isnan(pref_dir):
            pref_rad = np.deg2rad(pref_dir)
            max_r = np.max(mean_responses) if np.max(mean_responses) > 0 else 1
            
            # Scale arrow length by index value (DSI or OSI)
            # index_value ranges from 0 to ~1, so arrow length = max_r * index_value
            if index_value is not None and not np.isnan(index_value):
                # Clamp index_value to reasonable range and scale
                arrow_scale = min(max(index_value, 0.1), 1.0)  # At least 10% visible
                arrow_length = max_r * arrow_scale
            else:
                arrow_length = max_r * 0.5  # Default to 50% if no index
            
            ax.annotate(
                "",
                xy=(pref_rad, arrow_length),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=2),
            )
    
    # Set labels
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(angles_rad)
    ax.set_xticklabels([f"{int(d)}°" for d in directions], fontsize=7)
    
    if title:
        ax.set_title(title, fontsize=8, pad=10)


# =============================================================================
# TRACE PLOT FUNCTION
# =============================================================================


def plot_direction_trace(
    ax: plt.Axes,
    trial_traces: np.ndarray,
    direction: int,
    sampling_rate: float = 60.0,
):
    """
    Plot traces for a single direction with trials as transparent background.
    
    Parameters
    ----------
    ax : plt.Axes
        Axis to plot on
    trial_traces : np.ndarray
        Array of trial traces
    direction : int
        Direction angle in degrees (for title)
    sampling_rate : float
        Sampling rate in Hz
    """
    if trial_traces is None or len(trial_traces) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{direction}°", fontsize=8)
        return
    
    # Convert to list of arrays
    if isinstance(trial_traces, list):
        trial_traces = np.array(trial_traces, dtype=object)
    
    # Stack trials
    try:
        trials_array = np.vstack([np.array(trial) for trial in trial_traces])
    except ValueError:
        ax.text(0.5, 0.5, "Shape error", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{direction}°", fontsize=8)
        return
    
    n_trials, n_samples = trials_array.shape
    time_axis = np.arange(n_samples) / sampling_rate
    
    # Plot individual trials (transparent)
    for trial in trials_array:
        ax.plot(time_axis, trial, color=TRIAL_COLOR, alpha=0.4, linewidth=0.5)
    
    # Plot mean trace
    mean_trace = np.mean(trials_array, axis=0)
    ax.plot(time_axis, mean_trace, color=MEAN_COLOR, linewidth=1.5)
    
    # Style
    ax.set_title(f"{direction}°", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlim(0, time_axis[-1])


# =============================================================================
# EXAMPLE GRID FUNCTION
# =============================================================================


def plot_selectivity_examples(
    df: pd.DataFrame,
    filter_column: str,
    significant: bool,
    index_column: str,
    output_path: Optional[Path] = None,
    n_units: int = 20,
    figsize: Tuple[int, int] = (24, 40),
    p_threshold: float = 0.05,
):
    """
    Plot example grid showing polar plot and traces for each direction.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DSGC features
    filter_column : str
        Column to filter by (ds_p_value or os_p_value)
    significant : bool
        If True, filter p < threshold; if False, p >= threshold
    index_column : str
        Column name for the index value (dsi or osi)
    output_path : Path, optional
        Path to save figure
    n_units : int
        Number of units to display
    figsize : tuple
        Figure size
    p_threshold : float
        P-value threshold (default 0.05)
    """
    # Filter units
    if significant:
        mask = df[filter_column] < p_threshold
        title_suffix = f"(p < {p_threshold})"
    else:
        mask = df[filter_column] >= p_threshold
        title_suffix = f"(p >= {p_threshold})"
    
    df_filtered = df[mask].dropna(subset=[filter_column, index_column])
    
    if len(df_filtered) == 0:
        print(f"No units found with {filter_column} {'<' if significant else '>='} {p_threshold}")
        return
    
    # Sample units
    n_sample = min(n_units, len(df_filtered))
    df_sample = df_filtered.sample(n=n_sample, random_state=RANDOM_SEED)
    
    # Sort by index value for better visualization
    df_sample = df_sample.sort_values(index_column, ascending=False)
    
    # Create figure: n_units rows x 9 columns (1 polar + 8 directions)
    fig = plt.figure(figsize=figsize)
    
    # Create grid spec for different subplot sizes
    n_cols = 9
    directions = np.array(DIRECTION_ANGLES)
    
    for row_idx, (unit_idx, row) in enumerate(df_sample.iterrows()):
        # Column 0: Polar plot
        ax_polar = fig.add_subplot(
            n_sample, n_cols, row_idx * n_cols + 1,
            projection="polar"
        )
        
        # Get values for title
        p_val = row[filter_column]
        idx_val = row[index_column]
        pref_dir = row.get("preferred_direction", np.nan)
        
        title = f"{index_column.upper()}={idx_val:.2f}\np={p_val:.3f}"
        plot_polar_tuning(ax_polar, row, DIRECTION_COLUMNS, directions, title=title, index_value=idx_val)
        
        # Add unit ID on the left
        ax_polar.annotate(
            unit_idx[:30] + "..." if len(unit_idx) > 30 else unit_idx,
            xy=(-0.3, 0.5),
            xycoords="axes fraction",
            fontsize=6,
            rotation=90,
            va="center",
            ha="center",
        )
        
        # Columns 1-8: Direction traces
        for col_idx, (direction, col_name) in enumerate(zip(DIRECTION_ANGLES, DIRECTION_COLUMNS)):
            ax_trace = fig.add_subplot(n_sample, n_cols, row_idx * n_cols + col_idx + 2)
            
            trial_traces = row.get(col_name)
            plot_direction_trace(ax_trace, trial_traces, direction)
            
            # Only show x-label on bottom row
            if row_idx < n_sample - 1:
                ax_trace.set_xticklabels([])
            else:
                ax_trace.set_xlabel("Time (s)", fontsize=6)
    
    # Add overall title
    index_name = "Direction Selectivity" if index_column == "dsi" else "Orientation Selectivity"
    sig_label = "Significant" if significant else "Non-significant"
    fig.suptitle(
        f"{sig_label} {index_name} Examples {title_suffix}\n(n={n_sample} units)",
        fontsize=14,
        y=0.995,
    )
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.99])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# TOP INDEX EXAMPLES FUNCTION
# =============================================================================


def plot_top_index_examples(
    df: pd.DataFrame,
    index_column: str,
    output_path: Optional[Path] = None,
    n_units: int = 20,
    figsize: Tuple[int, int] = (24, 40),
    min_index: float = 0.0,
):
    """
    Plot example grid showing units with highest DSI or OSI values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DSGC features
    index_column : str
        Column name for the index value (dsi or osi)
    output_path : Path, optional
        Path to save figure
    n_units : int
        Number of units to display
    figsize : tuple
        Figure size
    min_index : float
        Minimum index value to include (default 0.0)
    """
    # Get p-value column
    p_column = "ds_p_value" if index_column == "dsi" else "os_p_value"
    
    # Filter by minimum index value
    df_filtered = df[df[index_column] >= min_index].dropna(subset=[index_column, p_column])
    
    if len(df_filtered) == 0:
        print(f"No units found with {index_column} >= {min_index}")
        return
    
    # Sort by index value (descending) and take top n_units
    df_sample = df_filtered.nlargest(n_units, index_column)
    
    # Create figure: n_units rows x 9 columns (1 polar + 8 directions)
    fig = plt.figure(figsize=figsize)
    
    n_cols = 9
    directions = np.array(DIRECTION_ANGLES)
    n_sample = len(df_sample)
    
    for row_idx, (unit_idx, row) in enumerate(df_sample.iterrows()):
        # Column 0: Polar plot
        ax_polar = fig.add_subplot(
            n_sample, n_cols, row_idx * n_cols + 1,
            projection="polar"
        )
        
        # Get values for title
        p_val = row[p_column]
        idx_val = row[index_column]
        
        title = f"{index_column.upper()}={idx_val:.2f}\np={p_val:.3f}"
        plot_polar_tuning(ax_polar, row, DIRECTION_COLUMNS, directions, title=title, index_value=idx_val)
        
        # Add unit ID on the left
        ax_polar.annotate(
            unit_idx[:30] + "..." if len(unit_idx) > 30 else unit_idx,
            xy=(-0.3, 0.5),
            xycoords="axes fraction",
            fontsize=6,
            rotation=90,
            va="center",
            ha="center",
        )
        
        # Columns 1-8: Direction traces
        for col_idx, (direction, col_name) in enumerate(zip(DIRECTION_ANGLES, DIRECTION_COLUMNS)):
            ax_trace = fig.add_subplot(n_sample, n_cols, row_idx * n_cols + col_idx + 2)
            
            trial_traces = row.get(col_name)
            plot_direction_trace(ax_trace, trial_traces, direction)
            
            # Only show x-label on bottom row
            if row_idx < n_sample - 1:
                ax_trace.set_xticklabels([])
            else:
                ax_trace.set_xlabel("Time (s)", fontsize=6)
    
    # Add overall title
    index_name = "Direction Selectivity" if index_column == "dsi" else "Orientation Selectivity"
    fig.suptitle(
        f"Top {n_sample} Units by {index_name} Index ({index_column.upper()})\n"
        f"Sorted by {index_column.upper()} descending",
        fontsize=14,
        y=0.995,
    )
    
    plt.tight_layout(rect=[0.02, 0, 1, 0.99])
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# HISTOGRAM FUNCTION
# =============================================================================


def plot_dsgc_histograms(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
    bins: int = 50,
):
    """
    Plot 2x2 grid of histograms for DSI, OSI, and their p-values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DSGC features
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    bins : int
        Number of histogram bins
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot configurations
    plot_configs = [
        ("dsi", "Direction Selectivity Index (DSI)", axes[0, 0]),
        ("osi", "Orientation Selectivity Index (OSI)", axes[0, 1]),
        ("ds_p_value", "DS P-Value", axes[1, 0]),
        ("os_p_value", "OS P-Value", axes[1, 1]),
    ]
    
    for col, title, ax in plot_configs:
        values = df[col].dropna()
        
        if len(values) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Plot histogram
        ax.hist(values, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        
        # Statistics
        mean_val = values.mean()
        median_val = values.median()
        
        ax.axvline(mean_val, color="darkred", linestyle="--", linewidth=2, 
                   label=f"Mean: {mean_val:.3f}")
        ax.axvline(median_val, color="darkorange", linestyle="-", linewidth=2,
                   label=f"Median: {median_val:.3f}")
        
        # For p-values, add significance threshold line
        if "p_value" in col:
            ax.axvline(0.05, color="green", linestyle=":", linewidth=2,
                       label=f"p=0.05")
            n_sig = (values < 0.05).sum()
            pct_sig = 100.0 * n_sig / len(values)
            ax.text(
                0.98, 0.75,
                f"Significant (p<0.05):\n{n_sig:,} ({pct_sig:.1f}%)",
                transform=ax.transAxes,
                fontsize=9,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            )
        
        # Labels
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {title}\n(n={len(values):,})", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        
        # Statistics text box
        stats_text = (
            f"Mean: {mean_val:.4f}\n"
            f"Median: {median_val:.4f}\n"
            f"Std: {values.std():.4f}\n"
            f"Range: [{values.min():.3f}, {values.max():.3f}]"
        )
        ax.text(
            0.98, 0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Generate all DSGC validation plots."""
    print("=" * 80)
    print("DSGC Validation Plots")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded DataFrame: {df.shape} (units x columns)")
    
    # Check required columns
    required_cols = ["dsi", "osi", "ds_p_value", "os_p_value", "preferred_direction"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return
    
    # Summary
    print(f"\nSummary:")
    print(f"  DSGCs (ds_p < 0.05): {(df['ds_p_value'] < 0.05).sum():,}")
    print(f"  Non-DSGCs (ds_p >= 0.05): {(df['ds_p_value'] >= 0.05).sum():,}")
    print(f"  OS cells (os_p < 0.05): {(df['os_p_value'] < 0.05).sum():,}")
    print(f"  Non-OS cells (os_p >= 0.05): {(df['os_p_value'] >= 0.05).sum():,}")
    
    # Figure 1: Significant DSGCs
    print("\n[1/5] Generating significant DSGC examples...")
    plot_selectivity_examples(
        df,
        filter_column="ds_p_value",
        significant=True,
        index_column="dsi",
        output_path=OUTPUT_DIR / "dsgc_significant_examples.png",
    )
    
    # Figure 2: Non-significant DSGCs
    print("[2/5] Generating non-significant DSGC examples...")
    plot_selectivity_examples(
        df,
        filter_column="ds_p_value",
        significant=False,
        index_column="dsi",
        output_path=OUTPUT_DIR / "dsgc_nonsignificant_examples.png",
    )
    
    # Figure 3: Significant orientation-selective cells
    print("[3/5] Generating significant orientation-selective examples...")
    plot_selectivity_examples(
        df,
        filter_column="os_p_value",
        significant=True,
        index_column="osi",
        output_path=OUTPUT_DIR / "os_significant_examples.png",
    )
    
    # Figure 4: Non-significant orientation-selective cells
    print("[4/5] Generating non-significant orientation-selective examples...")
    plot_selectivity_examples(
        df,
        filter_column="os_p_value",
        significant=False,
        index_column="osi",
        output_path=OUTPUT_DIR / "os_nonsignificant_examples.png",
    )
    
    # Figure 5: Histograms
    print("[5/7] Generating DSI/OSI histograms...")
    plot_dsgc_histograms(
        df,
        output_path=OUTPUT_DIR / "dsgc_histograms.png",
    )
    
    # Figure 6: Top DSI units
    print("[6/7] Generating top DSI examples...")
    plot_top_index_examples(
        df,
        index_column="dsi",
        output_path=OUTPUT_DIR / "dsgc_top_dsi_examples.png",
        n_units=20,
    )
    
    # Figure 7: Top OSI units
    print("[7/7] Generating top OSI examples...")
    plot_top_index_examples(
        df,
        index_column="osi",
        output_path=OUTPUT_DIR / "osgc_top_osi_examples.png",
        n_units=20,
    )
    
    print("\n" + "=" * 80)
    print("Done! All plots saved to:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()

