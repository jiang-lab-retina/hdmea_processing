"""
Visualization for Step Change Analysis Pipeline

This module provides plotting functions for visualizing step responses,
response timecourses, and unit alignment chains.

Ported from: Legacy_code/.../low_glucose/A04_step_analysis_v2.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .specific_config import (
    VisualizationConfig,
    PipelineConfig,
    default_config,
    FIGURES_DIR,
)
from .response_analysis import (
    normalize_features,
    compute_binned_statistics,
    compare_groups_statistics,
    summarize_response_timecourse,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Step Response Grid Plots
# =============================================================================

def plot_step_responses_grid(
    data: Dict[str, Any],
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot grid of mean step responses for all units.
    
    Args:
        data: Recording data dictionary with units
        config: Visualization configuration
        save_path: Path to save figure (optional)
        title: Figure title
    
    Returns:
        Matplotlib Figure object
    """
    if config is None:
        config = default_config.visualization
    
    units = data.get("units", {})
    if not units:
        logger.warning("No units to plot")
        return plt.figure()
    
    unit_ids = sorted(units.keys(), key=lambda x: int(x) if x.isdigit() else x)
    n_units = len(unit_ids)
    
    n_rows = config.grid_rows
    n_cols = n_units // n_rows + (1 if n_units % n_rows else 0)
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=config.grid_figure_size,
    )
    axes = axes.flatten()
    
    for i, unit_id in enumerate(unit_ids):
        unit_data = units[unit_id]
        ax = axes[i]
        
        if "step_responses" in unit_data:
            responses = np.array(unit_data["step_responses"])
            if responses.size > 0:
                mean_response = responses.mean(axis=0)
                ax.plot(mean_response)
                
                qi = unit_data.get("quality_index", 0)
                ax.set_title(f"{unit_id}\nQI: {qi:.2f}", fontsize=8)
        
        ax.set_xlim(0, None)
    
    # Hide unused axes
    for i in range(n_units, len(axes)):
        axes[i].set_visible(False)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure: {save_path}")
    
    return fig


# =============================================================================
# Response Timecourse Plots
# =============================================================================

def plot_response_timecourse(
    summary: Dict[str, Any],
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_treatment_line: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot response magnitude over time with error bars.
    
    Args:
        summary: Summary from summarize_response_timecourse()
        config: Visualization configuration
        save_path: Path to save figure
        show_treatment_line: Show vertical line at treatment time
        ax: Existing axes to plot on (creates new figure if None)
    
    Returns:
        Matplotlib Figure object
    """
    if config is None:
        config = default_config.visualization
    
    if not summary:
        logger.warning("No summary data to plot")
        return plt.figure()
    
    stats = summary.get("binned_stats", {})
    time_points = stats.get("time_points", np.array([]))
    means = stats.get("mean", np.array([]))
    sems = stats.get("sem", np.array([]))
    
    if len(time_points) == 0:
        logger.warning("No data points to plot")
        return plt.figure()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.timecourse_figure_size)
    else:
        fig = ax.figure
    
    # Plot mean with error bars
    ax.errorbar(
        time_points,
        means,
        yerr=sems,
        fmt="o-",
        capsize=3,
        color=config.control_color,
        label=f"n = {summary.get('n_chains', 0)}",
    )
    
    # Treatment line
    if show_treatment_line:
        treatment_time = summary.get("treatment_time_s", 0)
        if treatment_time > 0:
            ax.axvline(
                treatment_time,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Agonist",
            )
    
    # Labels
    peak_type = summary.get("peak_type", "")
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Normalized Response", fontsize=12)
    ax.set_title(f"{peak_type} Response Timecourse (n = {summary.get('n_chains', 0)})")
    
    if config.y_lim[0] is not None or config.y_lim[1] is not None:
        ax.set_ylim(config.y_lim)
    if config.x_lim[0] is not None or config.x_lim[1] is not None:
        ax.set_xlim(config.x_lim)
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure: {save_path}")
    
    return fig


def plot_compare_timecourses(
    summary1: Dict[str, Any],
    summary2: Dict[str, Any],
    label1: str = "Control",
    label2: str = "Treatment",
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_significance: bool = True,
) -> plt.Figure:
    """
    Plot comparison of two response timecourses.
    
    Args:
        summary1: First group summary
        summary2: Second group summary
        label1: Label for first group
        label2: Label for second group
        config: Visualization configuration
        save_path: Path to save figure
        show_significance: Show significance markers
    
    Returns:
        Matplotlib Figure object
    """
    if config is None:
        config = default_config.visualization
    
    fig, ax = plt.subplots(figsize=config.timecourse_figure_size)
    
    stats1 = summary1.get("binned_stats", {})
    stats2 = summary2.get("binned_stats", {})
    
    time1 = stats1.get("time_points", np.array([]))
    means1 = stats1.get("mean", np.array([]))
    sems1 = stats1.get("sem", np.array([]))
    
    time2 = stats2.get("time_points", np.array([]))
    means2 = stats2.get("mean", np.array([]))
    sems2 = stats2.get("sem", np.array([]))
    
    # Plot both groups
    ax.errorbar(
        time1, means1, yerr=sems1,
        fmt="o-", capsize=3,
        color=config.control_color,
        label=f"{label1} (n={summary1.get('n_chains', 0)})",
    )
    
    ax.errorbar(
        time2, means2, yerr=sems2,
        fmt="o-", capsize=3,
        color=config.treatment_color,
        label=f"{label2} (n={summary2.get('n_chains', 0)})",
    )
    
    # Significance markers
    if show_significance:
        comparison = compare_groups_statistics(stats1, stats2)
        sig_points = comparison.get("significant", np.array([]))
        
        for i, (is_sig, t) in enumerate(zip(sig_points, time1)):
            if is_sig:
                y_pos = max(means1[i] + sems1[i], means2[i] + sems2[i]) + 0.02
                ax.text(t, y_pos, "*", ha="center", va="bottom", fontsize=12)
    
    # Labels
    peak_type = summary1.get("peak_type", "")
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Normalized Response", fontsize=12)
    ax.set_title(f"{peak_type} Response Comparison")
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure: {save_path}")
    
    return fig


# =============================================================================
# Alignment Chain Visualization
# =============================================================================

def plot_alignment_chains(
    grouped_data: Dict[str, Any],
    max_chains: int = 50,
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    use_fixed_chains: bool = True,
) -> plt.Figure:
    """
    Visualize unit alignment chains across recordings.
    
    Args:
        grouped_data: Grouped aligned data
        max_chains: Maximum number of chains to display
        config: Visualization configuration
        save_path: Path to save figure
        use_fixed_chains: Use fixed reference alignment
    
    Returns:
        Matplotlib Figure object
    """
    if config is None:
        config = default_config.visualization
    
    # Get chains
    if use_fixed_chains and "fixed_alignment_chains" in grouped_data:
        chains_df = grouped_data["fixed_alignment_chains"]
    else:
        chains_df = grouped_data.get("alignment_chains", pd.DataFrame())
    
    if chains_df.empty:
        logger.warning("No chains to visualize")
        return plt.figure()
    
    # Limit number of chains
    if len(chains_df) > max_chains:
        chains_df = chains_df.iloc[:max_chains]
    
    n_chains = len(chains_df)
    n_recordings = len(chains_df.columns)
    
    fig, ax = plt.subplots(figsize=(n_recordings * 2, n_chains * 0.3))
    
    # Create grid
    for i, (idx, chain) in enumerate(chains_df.iterrows()):
        for j, (rec_name, unit_id) in enumerate(chain.items()):
            if pd.notna(unit_id) and str(unit_id).strip():
                ax.plot(j, i, "o", markersize=10, color="blue")
                ax.text(
                    j, i, str(unit_id).strip(),
                    ha="center", va="center",
                    fontsize=6, color="white",
                )
                
                # Draw connection to next
                if j < n_recordings - 1:
                    next_id = chain.iloc[j + 1]
                    if pd.notna(next_id) and str(next_id).strip():
                        ax.plot([j, j + 1], [i, i], "b-", alpha=0.5)
    
    # Labels
    ax.set_xticks(range(n_recordings))
    ax.set_xticklabels(chains_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(n_chains))
    ax.set_yticklabels([f"Chain {i}" for i in range(n_chains)])
    
    ax.set_xlabel("Recording")
    ax.set_ylabel("Alignment Chain")
    ax.set_title(f"Unit Alignment Chains (n = {n_chains})")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure: {save_path}")
    
    return fig


# =============================================================================
# Response Heatmap
# =============================================================================

def plot_response_heatmap(
    features_df: pd.DataFrame,
    treatment_time_s: float = 0,
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Response Heatmap",
) -> plt.Figure:
    """
    Plot heatmap of responses over time for all chains.
    
    Args:
        features_df: Features DataFrame (time x chains)
        treatment_time_s: Treatment time for vertical line
        config: Visualization configuration
        save_path: Path to save figure
        title: Figure title
    
    Returns:
        Matplotlib Figure object
    """
    if features_df.empty:
        logger.warning("No data for heatmap")
        return plt.figure()
    
    if config is None:
        config = default_config.visualization
    
    # Normalize
    norm_df = normalize_features(features_df, mode="first")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Transpose so chains are rows
    data = norm_df.T.values
    time_points = norm_df.index.astype(float)
    
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=0.5,
        vmax=1.5,
        extent=[time_points.min(), time_points.max(), 0, len(data)],
    )
    
    # Treatment line
    if treatment_time_s > 0:
        ax.axvline(treatment_time_s, color="white", linestyle="--", linewidth=2)
    
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Unit Chain", fontsize=12)
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Response", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure: {save_path}")
    
    return fig


# =============================================================================
# Summary Plot
# =============================================================================

def plot_analysis_summary(
    grouped_data: Dict[str, Any],
    config: Optional[PipelineConfig] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> List[plt.Figure]:
    """
    Generate all summary plots for analysis.
    
    Args:
        grouped_data: Grouped aligned data
        config: Pipeline configuration
        save_dir: Directory to save figures
    
    Returns:
        List of Figure objects
    """
    if config is None:
        config = default_config
    
    if save_dir is None:
        save_dir = config.figures_dir
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    figures = []
    
    # 1. ON Response Timecourse
    on_summary = summarize_response_timecourse(grouped_data, "ON", config)
    if on_summary:
        fig = plot_response_timecourse(
            on_summary,
            config.visualization,
            save_path=save_dir / "on_response_timecourse.png",
        )
        figures.append(fig)
        plt.close(fig)
    
    # 2. OFF Response Timecourse
    off_summary = summarize_response_timecourse(grouped_data, "OFF", config)
    if off_summary:
        fig = plot_response_timecourse(
            off_summary,
            config.visualization,
            save_path=save_dir / "off_response_timecourse.png",
        )
        figures.append(fig)
        plt.close(fig)
    
    # 3. Alignment Chains
    fig = plot_alignment_chains(
        grouped_data,
        config=config.visualization,
        save_path=save_dir / "alignment_chains.png",
    )
    figures.append(fig)
    plt.close(fig)
    
    # 4. ON Response Heatmap
    if on_summary:
        fig = plot_response_heatmap(
            on_summary.get("normalized_df", pd.DataFrame()),
            treatment_time_s=config.agonist_start_time_s,
            config=config.visualization,
            save_path=save_dir / "on_response_heatmap.png",
            title="ON Response Heatmap",
        )
        figures.append(fig)
        plt.close(fig)
    
    # 5. OFF Response Heatmap
    if off_summary:
        fig = plot_response_heatmap(
            off_summary.get("normalized_df", pd.DataFrame()),
            treatment_time_s=config.agonist_start_time_s,
            config=config.visualization,
            save_path=save_dir / "off_response_heatmap.png",
            title="OFF Response Heatmap",
        )
        figures.append(fig)
        plt.close(fig)
    
    logger.info(f"Generated {len(figures)} summary figures in {save_dir}")
    
    return figures


# =============================================================================
# Individual Recording Plots
# =============================================================================

def plot_recording_summary(
    data: Dict[str, Any],
    recording_name: str,
    save_dir: Optional[Union[str, Path]] = None,
    config: Optional[VisualizationConfig] = None,
) -> plt.Figure:
    """
    Generate summary plot for a single recording.
    
    Args:
        data: Recording data dictionary
        recording_name: Name of the recording
        save_dir: Directory to save figure
        config: Visualization configuration
    
    Returns:
        Matplotlib Figure object
    """
    if config is None:
        config = default_config.visualization
    
    units = data.get("units", {})
    n_units = len(units)
    
    # Count high quality units
    high_quality = sum(
        1 for u in units.values()
        if u.get("quality_index", 0) >= default_config.quality.quality_threshold
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Quality index distribution
    ax = axes[0, 0]
    qi_values = [u.get("quality_index", 0) for u in units.values()]
    ax.hist(qi_values, bins=30, edgecolor="black")
    ax.axvline(
        default_config.quality.quality_threshold,
        color="red",
        linestyle="--",
        label=f"Threshold ({default_config.quality.quality_threshold})",
    )
    ax.set_xlabel("Quality Index")
    ax.set_ylabel("Count")
    ax.set_title(f"Quality Distribution (n={n_units}, HQ={high_quality})")
    ax.legend()
    
    # 2. Mean response across all units
    ax = axes[0, 1]
    all_means = []
    for unit_data in units.values():
        if "step_responses" in unit_data:
            responses = np.array(unit_data["step_responses"])
            if responses.size > 0:
                all_means.append(responses.mean(axis=0))
    
    if all_means:
        all_means = np.array(all_means)
        mean_trace = all_means.mean(axis=0)
        std_trace = all_means.std(axis=0)
        
        x = np.arange(len(mean_trace))
        ax.plot(x, mean_trace, color="blue")
        ax.fill_between(
            x,
            mean_trace - std_trace,
            mean_trace + std_trace,
            alpha=0.3,
        )
    
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Firing Rate")
    ax.set_title("Mean Step Response (all units)")
    
    # 3. Light reference
    ax = axes[1, 0]
    light_ref = data.get("stimulus", {}).get("light_reference_10hz", np.array([]))
    if len(light_ref) > 0:
        x = np.arange(len(light_ref)) / 10  # Convert to seconds
        ax.plot(x, light_ref)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Light Intensity")
        ax.set_title("Light Reference")
    
    # 4. Step times
    ax = axes[1, 1]
    on_times = data.get("stimulus", {}).get("step_on_times", np.array([]))
    off_times = data.get("stimulus", {}).get("step_off_times", np.array([]))
    
    if len(on_times) > 0:
        # Plot step intervals
        intervals = np.diff(on_times) / 10  # Convert to seconds
        ax.bar(range(len(intervals)), intervals)
        ax.set_xlabel("Step Number")
        ax.set_ylabel("Interval (seconds)")
        ax.set_title(f"Step Intervals (n={len(on_times)} steps)")
    
    fig.suptitle(f"Recording: {recording_name}", fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{recording_name}_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure: {save_path}")
    
    return fig
