"""
Cell Type Comparison Validation Plots.

Classifies and visualizes RGC subtypes (DSGC, OSGC, ipRGC, Other)
with pie charts and overlap diagrams.

Classification Criteria:
    - DSGC: ds_p_value < 0.05
    - OSGC: os_p_value < 0.05
    - ipRGC: iprgc_2hz_QI > 0.8
    - Other: None of the above
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from pathlib import Path
from typing import Dict, List, Set, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths
INPUT_PARQUET = Path(__file__).parent.parent / "firing_rate_with_dsgc_features20251230.parquet"
OUTPUT_DIR = Path(__file__).parent / "plots"

# Classification thresholds
DS_P_THRESHOLD = 0.05
OS_P_THRESHOLD = 0.05
IPRGC_QI_THRESHOLD = 0.8

# Colors for cell types
COLORS = {
    "DSGC": "#E74C3C",      # Red
    "OSGC": "#3498DB",      # Blue
    "ipRGC": "#2ECC71",     # Green
    "Other": "#95A5A6",     # Gray
    "DSGC+OSGC": "#9B59B6",  # Purple
    "DSGC+ipRGC": "#E67E22", # Orange
    "OSGC+ipRGC": "#1ABC9C", # Teal
    "All three": "#F1C40F",  # Yellow
}

# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================


def classify_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify cells into types based on thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ds_p_value, os_p_value, and iprgc_2hz_QI columns
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added boolean columns for each cell type
    """
    df = df.copy()
    
    # Boolean flags for each cell type
    df["is_DSGC"] = df["ds_p_value"] < DS_P_THRESHOLD
    df["is_OSGC"] = df["os_p_value"] < OS_P_THRESHOLD
    df["is_ipRGC"] = df["iprgc_2hz_QI"] > IPRGC_QI_THRESHOLD
    
    # Other = not any of the above
    df["is_Other"] = ~(df["is_DSGC"] | df["is_OSGC"] | df["is_ipRGC"])
    
    return df


def get_cell_type_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get counts for each cell type (non-exclusive).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with is_DSGC, is_OSGC, is_ipRGC columns
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping cell type to count
    """
    return {
        "DSGC": df["is_DSGC"].sum(),
        "OSGC": df["is_OSGC"].sum(),
        "ipRGC": df["is_ipRGC"].sum(),
        "Other": df["is_Other"].sum(),
    }


def get_combination_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get counts for all possible combinations of cell types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with is_DSGC, is_OSGC, is_ipRGC columns
        
    Returns
    -------
    Dict[str, int]
        Dictionary mapping combination to count
    """
    combinations = {}
    
    # All possible combinations (excluding "none of the three")
    dsgc = df["is_DSGC"]
    osgc = df["is_OSGC"]
    iprgc = df["is_ipRGC"]
    
    # Single types only
    combinations["DSGC only"] = (dsgc & ~osgc & ~iprgc).sum()
    combinations["OSGC only"] = (~dsgc & osgc & ~iprgc).sum()
    combinations["ipRGC only"] = (~dsgc & ~osgc & iprgc).sum()
    
    # Two types
    combinations["DSGC + OSGC"] = (dsgc & osgc & ~iprgc).sum()
    combinations["DSGC + ipRGC"] = (dsgc & ~osgc & iprgc).sum()
    combinations["OSGC + ipRGC"] = (~dsgc & osgc & iprgc).sum()
    
    # All three
    combinations["All three"] = (dsgc & osgc & iprgc).sum()
    
    # None (Other)
    combinations["Other"] = (~dsgc & ~osgc & ~iprgc).sum()
    
    return combinations


def get_venn_counts(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get counts for Venn diagram regions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with is_DSGC, is_OSGC, is_ipRGC columns
        
    Returns
    -------
    Dict[str, int]
        Counts for each Venn region (100, 010, 001, 110, 101, 011, 111)
    """
    dsgc = df["is_DSGC"]
    osgc = df["is_OSGC"]
    iprgc = df["is_ipRGC"]
    
    return {
        "100": (dsgc & ~osgc & ~iprgc).sum(),   # DSGC only
        "010": (~dsgc & osgc & ~iprgc).sum(),   # OSGC only
        "001": (~dsgc & ~osgc & iprgc).sum(),   # ipRGC only
        "110": (dsgc & osgc & ~iprgc).sum(),    # DSGC + OSGC
        "101": (dsgc & ~osgc & iprgc).sum(),    # DSGC + ipRGC
        "011": (~dsgc & osgc & iprgc).sum(),    # OSGC + ipRGC
        "111": (dsgc & osgc & iprgc).sum(),     # All three
    }


# =============================================================================
# PIE CHART FUNCTION
# =============================================================================


def plot_cell_type_pie(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Plot pie charts for cell type distribution.
    
    Creates two pie charts:
    - Left: Non-exclusive counts (cells can be in multiple categories)
    - Right: Exclusive combinations (mutually exclusive assignment)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cell type boolean columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Non-exclusive counts
    ax = axes[0]
    type_counts = get_cell_type_counts(df)
    
    labels = list(type_counts.keys())
    sizes = list(type_counts.values())
    colors = [COLORS.get(label, "#999999") for label in labels]
    
    # Calculate percentages
    total = len(df)
    percentages = [100 * s / total for s in sizes]
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(labels),
    )
    
    # Legend with counts
    legend_labels = [f"{label}: {count:,} ({100*count/total:.1f}%)" 
                     for label, count in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Cell Type Distribution (Non-exclusive)\nn={total:,} total cells", fontsize=12)
    
    # Right: Exclusive combinations
    ax = axes[1]
    combo_counts = get_combination_counts(df)
    
    # Filter to non-zero combinations
    combo_counts = {k: v for k, v in combo_counts.items() if v > 0}
    
    labels = list(combo_counts.keys())
    sizes = list(combo_counts.values())
    
    # Color mapping for combinations
    combo_colors = {
        "DSGC only": COLORS["DSGC"],
        "OSGC only": COLORS["OSGC"],
        "ipRGC only": COLORS["ipRGC"],
        "Other": COLORS["Other"],
        "DSGC + OSGC": "#9B59B6",
        "DSGC + ipRGC": "#E67E22",
        "OSGC + ipRGC": "#1ABC9C",
        "All three": "#F1C40F",
    }
    colors = [combo_colors.get(label, "#999999") for label in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(labels),
    )
    
    # Legend with counts
    legend_labels = [f"{label}: {count:,} ({100*count/total:.1f}%)" 
                     for label, count in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Cell Type Combinations (Exclusive)\nn={total:,} total cells", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# VENN DIAGRAM FUNCTION
# =============================================================================


def plot_venn_diagram(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (12, 9),
):
    """
    Plot 3-way Venn diagram for DSGC, OSGC, and ipRGC overlap with Other as background.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cell type boolean columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get counts for each region
    venn_counts = get_venn_counts(df)
    total = len(df)
    other_count = df["is_Other"].sum()
    other_pct = 100 * other_count / total
    
    # Draw "Other" as a background rectangle (representing all cells)
    from matplotlib.patches import Rectangle, FancyBboxPatch
    other_rect = FancyBboxPatch(
        (-3.2, -3.0), 6.4, 5.5,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        facecolor=COLORS["Other"], alpha=0.15,
        edgecolor=COLORS["Other"], linewidth=2, linestyle='--'
    )
    ax.add_patch(other_rect)
    
    # Circle parameters (positioned for nice overlap)
    r = 1.5  # radius
    centers = {
        "DSGC": (-0.8, 0.3),
        "OSGC": (0.8, 0.3),
        "ipRGC": (0, -1.0),
    }
    
    # Draw circles
    for label, center in centers.items():
        circle = plt.Circle(center, r, fill=True, alpha=0.35, 
                           color=COLORS[label], linewidth=2, edgecolor=COLORS[label])
        ax.add_patch(circle)
    
    # Add labels for circles
    ax.text(-2.0, 1.5, f"DSGC\n(n={df['is_DSGC'].sum():,})", 
            fontsize=11, fontweight='bold', color=COLORS["DSGC"], ha='center')
    ax.text(2.0, 1.5, f"OSGC\n(n={df['is_OSGC'].sum():,})", 
            fontsize=11, fontweight='bold', color=COLORS["OSGC"], ha='center')
    ax.text(0, -2.8, f"ipRGC\n(n={df['is_ipRGC'].sum():,})", 
            fontsize=11, fontweight='bold', color=COLORS["ipRGC"], ha='center')
    
    # Add "Other" label in top-right corner
    ax.text(2.8, 2.0, f"Other\n(n={other_count:,}, {other_pct:.1f}%)", 
            fontsize=11, fontweight='bold', color=COLORS["Other"], ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS["Other"], alpha=0.9))
    
    # Add counts in each region
    # Positions for each region (approximate)
    region_positions = {
        "100": (-1.6, 0.5),    # DSGC only
        "010": (1.6, 0.5),     # OSGC only
        "001": (0, -1.8),      # ipRGC only
        "110": (0, 0.9),       # DSGC + OSGC
        "101": (-0.6, -0.5),   # DSGC + ipRGC
        "011": (0.6, -0.5),    # OSGC + ipRGC
        "111": (0, 0.0),       # All three
    }
    
    for region, pos in region_positions.items():
        count = venn_counts[region]
        pct = 100 * count / total
        if count > 0:
            ax.text(pos[0], pos[1], f"{count:,}\n({pct:.1f}%)", 
                   fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Set axis properties
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.3, 2.7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title(f"Cell Type Overlap (n={total:,} cells)\n"
                 f"DSGC: p<{DS_P_THRESHOLD} | OSGC: p<{OS_P_THRESHOLD} | ipRGC: QI>{IPRGC_QI_THRESHOLD} | Other: none",
                 fontsize=13, pad=20)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# OVERLAP MATRIX FUNCTION
# =============================================================================


def plot_overlap_matrix(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot pairwise overlap heatmap between cell types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cell type boolean columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    cell_types = ["DSGC", "OSGC", "ipRGC", "Other"]
    n_types = len(cell_types)
    
    # Calculate overlap matrix
    overlap_matrix = np.zeros((n_types, n_types))
    overlap_counts = np.zeros((n_types, n_types), dtype=int)
    
    for i, type1 in enumerate(cell_types):
        for j, type2 in enumerate(cell_types):
            mask1 = df[f"is_{type1}"]
            mask2 = df[f"is_{type2}"]
            
            overlap_count = (mask1 & mask2).sum()
            overlap_counts[i, j] = overlap_count
            
            # Calculate Jaccard index (overlap / union)
            union_count = (mask1 | mask2).sum()
            if union_count > 0:
                overlap_matrix[i, j] = overlap_count / union_count
            else:
                overlap_matrix[i, j] = 0
    
    # Plot heatmap
    im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Jaccard Index (Overlap / Union)")
    
    # Add text annotations
    for i in range(n_types):
        for j in range(n_types):
            count = overlap_counts[i, j]
            jaccard = overlap_matrix[i, j]
            text = f"{count:,}\n({jaccard:.2f})"
            color = "white" if jaccard > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
    
    # Labels
    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_types))
    ax.set_xticklabels(cell_types, fontsize=11)
    ax.set_yticklabels(cell_types, fontsize=11)
    
    ax.set_title("Pairwise Cell Type Overlap\n(count and Jaccard index)", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# UPSET-STYLE COMBINATION PLOT
# =============================================================================


def plot_upset_style(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot UpSet-style combination bar chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cell type boolean columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Get combination counts
    combo_counts = get_combination_counts(df)
    
    # Sort by count (descending)
    combo_counts = dict(sorted(combo_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Define the membership for each combination (DSGC, OSGC, ipRGC, Other)
    memberships = {
        "DSGC only": (True, False, False, False),
        "OSGC only": (False, True, False, False),
        "ipRGC only": (False, False, True, False),
        "DSGC + OSGC": (True, True, False, False),
        "DSGC + ipRGC": (True, False, True, False),
        "OSGC + ipRGC": (False, True, True, False),
        "All three": (True, True, True, False),
        "Other": (False, False, False, True),
    }
    
    labels = list(combo_counts.keys())
    counts = list(combo_counts.values())
    total = len(df)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, 
                             gridspec_kw={'height_ratios': [2.5, 1.2]}, sharex=True)
    
    # Top: Bar chart
    ax_bar = axes[0]
    x = np.arange(len(labels))
    bars = ax_bar.bar(x, counts, color=[
        "#E74C3C" if "DSGC" in l and "OSGC" not in l and "ipRGC" not in l and "All" not in l else
        "#3498DB" if "OSGC" in l and "DSGC" not in l and "ipRGC" not in l and "All" not in l else
        "#2ECC71" if "ipRGC" in l and "DSGC" not in l and "OSGC" not in l and "All" not in l else
        "#9B59B6" if l == "DSGC + OSGC" else
        "#E67E22" if l == "DSGC + ipRGC" else
        "#1ABC9C" if l == "OSGC + ipRGC" else
        "#F1C40F" if l == "All three" else
        "#95A5A6"
        for l in labels
    ], edgecolor="white")
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        pct = 100 * count / total
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    ax_bar.set_ylabel("Number of cells", fontsize=11)
    ax_bar.set_title(f"Cell Type Combinations (n={total:,} total cells)", fontsize=14)
    ax_bar.set_ylim(0, max(counts) * 1.15)
    
    # Bottom: Membership matrix (now includes "Other" as 4th row)
    ax_mat = axes[1]
    cell_types = ["DSGC", "OSGC", "ipRGC", "Other"]
    
    for i, label in enumerate(labels):
        membership = memberships.get(label, (False, False, False, False))
        for j, (is_member, cell_type) in enumerate(zip(membership, cell_types)):
            if is_member:
                ax_mat.scatter(i, j, s=200, c=COLORS[cell_type], marker='o', zorder=3)
            else:
                ax_mat.scatter(i, j, s=100, c='lightgray', marker='o', zorder=2)
        
        # Connect filled dots with a line (only for non-Other combinations)
        filled_indices = [j for j, m in enumerate(membership[:3]) if m]  # Exclude "Other" from lines
        if len(filled_indices) > 1:
            ax_mat.plot([i] * len(filled_indices), filled_indices, 
                       c='black', linewidth=2, zorder=1)
    
    ax_mat.set_yticks(range(len(cell_types)))
    ax_mat.set_yticklabels(cell_types, fontsize=10)
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels([])  # Hide x labels
    ax_mat.set_xlim(-0.5, len(labels) - 0.5)
    ax_mat.set_ylim(-0.5, len(cell_types) - 0.5)
    ax_mat.invert_yaxis()
    
    # Add grid
    for j in range(len(cell_types)):
        ax_mat.axhline(j, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    
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
    """Generate all cell type comparison plots."""
    print("=" * 80)
    print("Cell Type Comparison Plots")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded DataFrame: {df.shape} (units x columns)")
    
    # Check required columns
    required_cols = ["ds_p_value", "os_p_value", "iprgc_2hz_QI"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return
    
    # Classify cells
    print("\nClassifying cells...")
    df = classify_cells(df)
    
    # Print summary
    print(f"\nClassification Summary (thresholds: DS p<{DS_P_THRESHOLD}, "
          f"OS p<{OS_P_THRESHOLD}, ipRGC QI>{IPRGC_QI_THRESHOLD}):")
    type_counts = get_cell_type_counts(df)
    total = len(df)
    for cell_type, count in type_counts.items():
        print(f"  {cell_type}: {count:,} ({100*count/total:.1f}%)")
    
    print("\nCombination counts:")
    combo_counts = get_combination_counts(df)
    for combo, count in sorted(combo_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {combo}: {count:,} ({100*count/total:.1f}%)")
    
    # Generate plots
    print("\n[1/4] Generating pie charts...")
    plot_cell_type_pie(df, output_path=OUTPUT_DIR / "cell_type_pie_chart.png")
    
    print("[2/4] Generating Venn diagram...")
    plot_venn_diagram(df, output_path=OUTPUT_DIR / "cell_type_venn.png")
    
    print("[3/4] Generating overlap matrix...")
    plot_overlap_matrix(df, output_path=OUTPUT_DIR / "cell_type_overlap_matrix.png")
    
    print("[4/4] Generating UpSet-style plot...")
    plot_upset_style(df, output_path=OUTPUT_DIR / "cell_type_combinations.png")
    
    print("\n" + "=" * 80)
    print("Done! All plots saved to:", OUTPUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()

