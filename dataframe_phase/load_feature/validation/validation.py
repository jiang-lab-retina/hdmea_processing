"""
Axon Type Feature Validation Plots.

Visualizes the loaded axon_type labels and their relationship with
computed cell type classifications (DSGC, OSGC, ipRGC).

Axon Types (from manual labeling):
    - rgc: Retinal Ganglion Cell
    - ac: Amacrine Cell
    - other: Other cell types
    - unknown: Unclassified
    - no_label: No manual label available
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output paths
INPUT_PARQUET = Path(__file__).parent.parent.parent / "extract_feature" / "firing_rate_with_all_features_loaded_extracted20260102.parquet"
OUTPUT_DIR = Path(__file__).parent / "plots"

# Classification thresholds (matching validation_cell_types.py)
DS_P_THRESHOLD = 0.05
OS_P_THRESHOLD = 0.05
IPRGC_QI_THRESHOLD = 0.8

# Axon type display names and colors
AXON_TYPE_LABELS = {
    "rgc": "RGC",
    "ac": "AC",
    "other": "Other",
    "unknown": "Unknown",
    "no_label": "No Label",
}

AXON_TYPE_COLORS = {
    "rgc": "#2ECC71",       # Green
    "ac": "#E74C3C",        # Red
    "other": "#9B59B6",     # Purple
    "unknown": "#95A5A6",   # Gray
    "no_label": "#BDC3C7",  # Light gray
}

# Subtype colors
SUBTYPE_COLORS = {
    "DSGC": "#E74C3C",      # Red
    "OSGC": "#3498DB",      # Blue
    "ipRGC": "#F39C12",     # Orange
    "Other RGC": "#27AE60", # Green
}


# =============================================================================
# DATA PREPARATION
# =============================================================================


def load_and_prepare_data(parquet_path: Path) -> pd.DataFrame:
    """
    Load parquet and add classification columns.
    
    Parameters
    ----------
    parquet_path : Path
        Path to input parquet file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added classification columns
    """
    df = pd.read_parquet(parquet_path)
    
    # Add subtype classifications
    df["is_DSGC"] = df["ds_p_value"] < DS_P_THRESHOLD
    df["is_OSGC"] = df["os_p_value"] < OS_P_THRESHOLD
    df["is_ipRGC"] = df["iprgc_2hz_QI"] > IPRGC_QI_THRESHOLD
    
    # Normalize axon_type to lowercase
    df["axon_type_clean"] = df["axon_type"].str.lower().fillna("no_label")
    
    # Create display labels
    df["axon_type_display"] = df["axon_type_clean"].map(AXON_TYPE_LABELS).fillna("Unknown")
    
    return df


# =============================================================================
# PLOT 1: AXON TYPE DISTRIBUTION
# =============================================================================


def plot_axon_type_distribution(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 5),
):
    """
    Plot axon type distribution as pie chart and bar chart.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with axon_type_clean column
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get counts
    type_counts = df["axon_type_clean"].value_counts()
    total = len(df)
    
    # Order for display
    order = ["rgc", "ac", "other", "unknown", "no_label"]
    type_counts = type_counts.reindex([t for t in order if t in type_counts.index])
    
    labels = [AXON_TYPE_LABELS.get(t, t) for t in type_counts.index]
    sizes = type_counts.values
    colors = [AXON_TYPE_COLORS.get(t, "#999999") for t in type_counts.index]
    
    # Left: Pie chart
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(labels),
    )
    
    legend_labels = [f"{label}: {count:,} ({100*count/total:.1f}%)" 
                     for label, count in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Axon Type Distribution\nn={total:,} total cells", fontsize=12)
    
    # Right: Bar chart
    ax = axes[1]
    x = np.arange(len(labels))
    bars = ax.bar(x, sizes, color=colors, edgecolor="white", linewidth=1.5)
    
    # Add count labels
    for bar, count in zip(bars, sizes):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.02,
               f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Number of Cells", fontsize=11)
    ax.set_title(f"Axon Type Counts\nn={total:,} total cells", fontsize=12)
    ax.set_ylim(0, max(sizes) * 1.2)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 2: AXON TYPE VS SUBTYPES CROSSTAB
# =============================================================================


def plot_axon_subtype_crosstab(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot heatmap showing relationship between axon type and RGC subtypes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with axon_type_clean and subtype columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create subtype assignment (exclusive, prioritizing DSGC > OSGC > ipRGC)
    def assign_primary_subtype(row):
        if row["is_DSGC"]:
            return "DSGC"
        elif row["is_OSGC"]:
            return "OSGC"
        elif row["is_ipRGC"]:
            return "ipRGC"
        else:
            return "Other RGC"
    
    df["primary_subtype"] = df.apply(assign_primary_subtype, axis=1)
    
    # Create crosstab
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    subtype_order = ["DSGC", "OSGC", "ipRGC", "Other RGC"]
    
    crosstab = pd.crosstab(
        df["axon_type_clean"], 
        df["primary_subtype"]
    )
    
    # Reindex to ensure order
    crosstab = crosstab.reindex(
        index=[t for t in axon_order if t in crosstab.index],
        columns=[s for s in subtype_order if s in crosstab.columns],
        fill_value=0
    )
    
    # Calculate row percentages
    row_pcts = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    # Plot heatmap
    im = ax.imshow(crosstab.values, cmap="Blues", aspect="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Cell Count")
    
    # Add text annotations (count and percentage)
    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            count = crosstab.iloc[i, j]
            pct = row_pcts.iloc[i, j]
            text = f"{count:,}\n({pct:.1f}%)"
            color = "white" if count > crosstab.values.max() * 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    
    # Labels
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_xticklabels(crosstab.columns, fontsize=11)
    ax.set_yticklabels([AXON_TYPE_LABELS.get(t, t) for t in crosstab.index], fontsize=11)
    
    ax.set_xlabel("Functional Subtype (Computed)", fontsize=12)
    ax.set_ylabel("Axon Type (Manual Label)", fontsize=12)
    ax.set_title("Axon Type vs Functional Subtype\n(row percentages in parentheses)", fontsize=13)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 3: STACKED BAR - SUBTYPES PER AXON TYPE
# =============================================================================


def plot_stacked_subtypes(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (12, 7),
):
    """
    Plot stacked bar chart showing subtype composition per axon type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with axon_type_clean and subtype columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Assign primary subtype
    def assign_primary_subtype(row):
        if row["is_DSGC"]:
            return "DSGC"
        elif row["is_OSGC"]:
            return "OSGC"
        elif row["is_ipRGC"]:
            return "ipRGC"
        else:
            return "Other RGC"
    
    df["primary_subtype"] = df.apply(assign_primary_subtype, axis=1)
    
    # Create crosstab
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    subtype_order = ["DSGC", "OSGC", "ipRGC", "Other RGC"]
    
    crosstab = pd.crosstab(
        df["axon_type_clean"], 
        df["primary_subtype"]
    )
    
    crosstab = crosstab.reindex(
        index=[t for t in axon_order if t in crosstab.index],
        columns=[s for s in subtype_order if s in crosstab.columns],
        fill_value=0
    )
    
    # Normalize to percentages
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    # Plot stacked bars
    x = np.arange(len(crosstab.index))
    width = 0.7
    
    bottom = np.zeros(len(crosstab.index))
    for subtype in subtype_order:
        if subtype in crosstab_pct.columns:
            values = crosstab_pct[subtype].values
            ax.bar(x, values, width, label=subtype, bottom=bottom, 
                   color=SUBTYPE_COLORS[subtype], edgecolor="white", linewidth=0.5)
            
            # Add percentage labels for segments > 5%
            for i, (val, bot) in enumerate(zip(values, bottom)):
                if val > 5:
                    ax.text(i, bot + val/2, f'{val:.0f}%', 
                           ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            
            bottom += values
    
    # Add count labels on top
    totals = crosstab.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, 102, f'n={total:,}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels([AXON_TYPE_LABELS.get(t, t) for t in crosstab.index], fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xlabel("Axon Type", fontsize=11)
    ax.set_title("Functional Subtype Composition by Axon Type", fontsize=13)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", title="Subtype")
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 4: DSI/OSI DISTRIBUTIONS BY AXON TYPE
# =============================================================================


def plot_dsi_osi_by_axon_type(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot DSI and OSI distributions for each axon type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dsi, osi, and axon_type_clean columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    axon_types = [t for t in axon_order if t in df["axon_type_clean"].unique()]
    
    # Filter to valid DSI/OSI values
    df_valid = df.dropna(subset=["dsi", "osi"])
    
    # Top Left: DSI histogram by axon type
    ax = axes[0, 0]
    for axon_type in axon_types:
        mask = df_valid["axon_type_clean"] == axon_type
        data = df_valid.loc[mask, "dsi"]
        ax.hist(data, bins=50, alpha=0.6, label=AXON_TYPE_LABELS.get(axon_type, axon_type),
               color=AXON_TYPE_COLORS.get(axon_type, "#999999"), density=True)
    
    ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.7, label="DSI=0.3 threshold")
    ax.set_xlabel("Direction Selectivity Index (DSI)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("DSI Distribution by Axon Type", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1.5)
    
    # Top Right: OSI histogram by axon type
    ax = axes[0, 1]
    for axon_type in axon_types:
        mask = df_valid["axon_type_clean"] == axon_type
        data = df_valid.loc[mask, "osi"]
        ax.hist(data, bins=50, alpha=0.6, label=AXON_TYPE_LABELS.get(axon_type, axon_type),
               color=AXON_TYPE_COLORS.get(axon_type, "#999999"), density=True)
    
    ax.axvline(x=0.3, color="blue", linestyle="--", alpha=0.7, label="OSI=0.3 threshold")
    ax.set_xlabel("Orientation Selectivity Index (OSI)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("OSI Distribution by Axon Type", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1.5)
    
    # Bottom Left: DSI boxplot
    ax = axes[1, 0]
    data_for_box = [df_valid.loc[df_valid["axon_type_clean"] == t, "dsi"].dropna() 
                    for t in axon_types]
    bp = ax.boxplot(data_for_box, tick_labels=[AXON_TYPE_LABELS.get(t, t) for t in axon_types],
                    patch_artist=True)
    
    for patch, axon_type in zip(bp['boxes'], axon_types):
        patch.set_facecolor(AXON_TYPE_COLORS.get(axon_type, "#999999"))
        patch.set_alpha(0.7)
    
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("DSI", fontsize=11)
    ax.set_title("DSI by Axon Type (boxplot)", fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    
    # Bottom Right: OSI boxplot
    ax = axes[1, 1]
    data_for_box = [df_valid.loc[df_valid["axon_type_clean"] == t, "osi"].dropna() 
                    for t in axon_types]
    bp = ax.boxplot(data_for_box, tick_labels=[AXON_TYPE_LABELS.get(t, t) for t in axon_types],
                    patch_artist=True)
    
    for patch, axon_type in zip(bp['boxes'], axon_types):
        patch.set_facecolor(AXON_TYPE_COLORS.get(axon_type, "#999999"))
        patch.set_alpha(0.7)
    
    ax.axhline(y=0.3, color="blue", linestyle="--", alpha=0.7)
    ax.set_ylabel("OSI", fontsize=11)
    ax.set_title("OSI by Axon Type (boxplot)", fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 5: RGC SUBTYPE BREAKDOWN
# =============================================================================


def plot_rgc_subtype_breakdown(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Plot detailed breakdown of RGC subtypes (only for cells labeled as RGC).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with axon_type_clean and subtype columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Filter to RGC only
    df_rgc = df[df["axon_type_clean"] == "rgc"].copy()
    total_rgc = len(df_rgc)
    
    if total_rgc == 0:
        print("No RGC cells found!")
        plt.close()
        return
    
    # Create combination labels for RGC
    def get_subtype_combo(row):
        types = []
        if row["is_DSGC"]:
            types.append("DS")
        if row["is_OSGC"]:
            types.append("OS")
        if row["is_ipRGC"]:
            types.append("ip")
        if not types:
            return "Non-selective"
        return "+".join(types)
    
    df_rgc["subtype_combo"] = df_rgc.apply(get_subtype_combo, axis=1)
    
    # Left: Pie chart of RGC subtypes
    ax = axes[0]
    combo_counts = df_rgc["subtype_combo"].value_counts()
    
    # Color mapping
    combo_colors = {
        "DS": "#E74C3C",
        "OS": "#3498DB",
        "ip": "#F39C12",
        "DS+OS": "#9B59B6",
        "DS+ip": "#E67E22",
        "OS+ip": "#1ABC9C",
        "DS+OS+ip": "#F1C40F",
        "Non-selective": "#95A5A6",
    }
    
    colors = [combo_colors.get(c, "#999999") for c in combo_counts.index]
    
    wedges, texts, autotexts = ax.pie(
        combo_counts.values,
        labels=None,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(combo_counts),
    )
    
    legend_labels = [f"{label}: {count:,} ({100*count/total_rgc:.1f}%)" 
                     for label, count in combo_counts.items()]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_title(f"RGC Subtype Composition\nn={total_rgc:,} RGC cells", fontsize=12)
    
    # Right: Bar chart
    ax = axes[1]
    x = np.arange(len(combo_counts))
    bars = ax.bar(x, combo_counts.values, 
                  color=[combo_colors.get(c, "#999999") for c in combo_counts.index],
                  edgecolor="white", linewidth=1.5)
    
    for bar, count in zip(bars, combo_counts.values):
        pct = 100 * count / total_rgc
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(combo_counts.values)*0.02,
               f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(combo_counts.index, fontsize=10, rotation=30, ha='right')
    ax.set_ylabel("Number of RGC Cells", fontsize=11)
    ax.set_title(f"RGC Functional Subtype Counts\nn={total_rgc:,} RGC cells", fontsize=12)
    ax.set_ylim(0, max(combo_counts.values) * 1.2)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 6: IPRGC QUALITY INDEX BY AXON TYPE
# =============================================================================


def plot_iprgc_qi_by_axon_type(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot ipRGC Quality Index distribution by axon type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with iprgc_2hz_QI and axon_type_clean columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    axon_types = [t for t in axon_order if t in df["axon_type_clean"].unique()]
    
    # Filter to valid QI values
    df_valid = df.dropna(subset=["iprgc_2hz_QI"])
    
    # Left: Histogram
    ax = axes[0]
    for axon_type in axon_types:
        mask = df_valid["axon_type_clean"] == axon_type
        data = df_valid.loc[mask, "iprgc_2hz_QI"]
        ax.hist(data, bins=50, alpha=0.6, label=AXON_TYPE_LABELS.get(axon_type, axon_type),
               color=AXON_TYPE_COLORS.get(axon_type, "#999999"), density=True)
    
    ax.axvline(x=IPRGC_QI_THRESHOLD, color="orange", linestyle="--", alpha=0.7, 
               label=f"QI={IPRGC_QI_THRESHOLD} threshold")
    ax.set_xlabel("ipRGC 2Hz Quality Index", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("ipRGC QI Distribution by Axon Type", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    
    # Right: Boxplot
    ax = axes[1]
    data_for_box = [df_valid.loc[df_valid["axon_type_clean"] == t, "iprgc_2hz_QI"].dropna() 
                    for t in axon_types]
    bp = ax.boxplot(data_for_box, tick_labels=[AXON_TYPE_LABELS.get(t, t) for t in axon_types],
                    patch_artist=True)
    
    for patch, axon_type in zip(bp['boxes'], axon_types):
        patch.set_facecolor(AXON_TYPE_COLORS.get(axon_type, "#999999"))
        patch.set_alpha(0.7)
    
    ax.axhline(y=IPRGC_QI_THRESHOLD, color="orange", linestyle="--", alpha=0.7)
    ax.set_ylabel("ipRGC 2Hz QI", fontsize=11)
    ax.set_title("ipRGC QI by Axon Type", fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 7: P-VALUE DISTRIBUTIONS BY AXON TYPE
# =============================================================================


def plot_pvalue_distributions(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 5),
):
    """
    Plot p-value distributions (DS and OS) by axon type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ds_p_value, os_p_value, and axon_type_clean columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    axon_types = [t for t in axon_order if t in df["axon_type_clean"].unique()]
    
    # Filter to valid p-values
    df_valid = df.dropna(subset=["ds_p_value", "os_p_value"])
    
    # Left: DS p-value histogram (log scale)
    ax = axes[0]
    for axon_type in axon_types:
        mask = df_valid["axon_type_clean"] == axon_type
        data = df_valid.loc[mask, "ds_p_value"]
        # Clip to avoid log(0)
        data = data.clip(lower=1e-10)
        ax.hist(np.log10(data), bins=50, alpha=0.6, 
                label=AXON_TYPE_LABELS.get(axon_type, axon_type),
                color=AXON_TYPE_COLORS.get(axon_type, "#999999"), density=True)
    
    ax.axvline(x=np.log10(DS_P_THRESHOLD), color="red", linestyle="--", alpha=0.7,
               label=f"p={DS_P_THRESHOLD}")
    ax.set_xlabel("log10(DS p-value)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("DS p-value Distribution by Axon Type", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    
    # Right: OS p-value histogram (log scale)
    ax = axes[1]
    for axon_type in axon_types:
        mask = df_valid["axon_type_clean"] == axon_type
        data = df_valid.loc[mask, "os_p_value"]
        data = data.clip(lower=1e-10)
        ax.hist(np.log10(data), bins=50, alpha=0.6, 
                label=AXON_TYPE_LABELS.get(axon_type, axon_type),
                color=AXON_TYPE_COLORS.get(axon_type, "#999999"), density=True)
    
    ax.axvline(x=np.log10(OS_P_THRESHOLD), color="blue", linestyle="--", alpha=0.7,
               label=f"p={OS_P_THRESHOLD}")
    ax.set_xlabel("log10(OS p-value)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("OS p-value Distribution by Axon Type", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 8: DSI VS OSI SCATTER BY AXON TYPE
# =============================================================================


def plot_dsi_osi_scatter(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 12),
):
    """
    Plot DSI vs OSI scatter plots faceted by axon type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dsi, osi, and axon_type_clean columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    axon_types = [t for t in axon_order if t in df["axon_type_clean"].unique()]
    
    n_types = len(axon_types)
    n_cols = 3
    n_rows = (n_types + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_types > 1 else [axes]
    
    # Filter to valid DSI/OSI
    df_valid = df.dropna(subset=["dsi", "osi"])
    
    for i, axon_type in enumerate(axon_types):
        ax = axes[i]
        mask = df_valid["axon_type_clean"] == axon_type
        data = df_valid[mask]
        
        # Scatter plot
        ax.scatter(data["dsi"], data["osi"], alpha=0.3, s=10,
                  color=AXON_TYPE_COLORS.get(axon_type, "#999999"))
        
        # Add threshold lines
        ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=0.3, color="blue", linestyle="--", alpha=0.5, linewidth=1)
        
        # Count cells in each quadrant
        n_dsgc_only = ((data["dsi"] > 0.3) & (data["osi"] <= 0.3)).sum()
        n_osgc_only = ((data["dsi"] <= 0.3) & (data["osi"] > 0.3)).sum()
        n_both = ((data["dsi"] > 0.3) & (data["osi"] > 0.3)).sum()
        n_neither = ((data["dsi"] <= 0.3) & (data["osi"] <= 0.3)).sum()
        
        ax.set_xlabel("DSI", fontsize=10)
        ax.set_ylabel("OSI", fontsize=10)
        ax.set_title(f"{AXON_TYPE_LABELS.get(axon_type, axon_type)}\n(n={len(data):,})", fontsize=11)
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.5)
        
        # Add quadrant counts
        ax.text(0.95, 0.95, f"Both: {n_both}", transform=ax.transAxes, 
               ha='right', va='top', fontsize=8)
        ax.text(0.95, 0.05, f"DS only: {n_dsgc_only}", transform=ax.transAxes, 
               ha='right', va='bottom', fontsize=8)
        ax.text(0.05, 0.95, f"OS only: {n_osgc_only}", transform=ax.transAxes, 
               ha='left', va='top', fontsize=8)
        ax.text(0.05, 0.05, f"Neither: {n_neither}", transform=ax.transAxes, 
               ha='left', va='bottom', fontsize=8)
    
    # Hide unused axes
    for i in range(len(axon_types), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("DSI vs OSI by Axon Type", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 9: NAN VALUE DISTRIBUTION HEATMAP
# =============================================================================


def plot_nan_distribution_heatmap(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (16, 10),
):
    """
    Plot heatmap showing NaN distribution across feature columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Define feature groups
    feature_groups = {
        "DSGC Features": ["dsi", "osi", "preferred_direction", "ds_p_value", "os_p_value"],
        "Step Response": ["on_peak_extreme", "on_sustained", "off_peak_extreme", "off_sustained", 
                          "base_mean", "time_to_on_peak_extreme", "time_to_off_peak_extreme"],
        "Green-Blue": ["green_on_peak_extreme", "blue_on_peak_extreme", "green_off_peak_extreme",
                       "blue_off_peak_extreme", "time_to_green_on_peak", "time_to_blue_on_peak"],
        "Frequency Response": ["freq_step_05hz_amp", "freq_step_1hz_amp", "freq_step_2hz_amp",
                               "freq_step_4hz_amp", "freq_step_10hz_amp", "freq_step_05hz_phase",
                               "freq_step_1hz_phase", "freq_step_2hz_phase"],
        "Quality Indices": ["step_up_QI", "iprgc_2hz_QI", "iprgc_20hz_QI"],
        "Spatial Features": ["transformed_x", "transformed_y", "polar_radius", "polar_theta_deg",
                             "gaussian_sigma_x", "gaussian_amp", "dog_sigma_exc", "dog_r2"],
    }
    
    # Left: Bar chart of NaN counts by feature group
    ax = axes[0]
    group_nan_counts = {}
    group_total_counts = {}
    
    for group, cols in feature_groups.items():
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            nan_count = df[valid_cols].isna().sum().sum()
            total_count = len(df) * len(valid_cols)
            group_nan_counts[group] = nan_count
            group_total_counts[group] = total_count
    
    groups = list(group_nan_counts.keys())
    nan_pcts = [100 * group_nan_counts[g] / group_total_counts[g] for g in groups]
    
    colors = plt.cm.RdYlGn_r(np.array(nan_pcts) / max(nan_pcts) * 0.8)
    bars = ax.barh(groups, nan_pcts, color=colors, edgecolor="white")
    
    for bar, pct, count in zip(bars, nan_pcts, [group_nan_counts[g] for g in groups]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}% ({count:,})', va='center', fontsize=9)
    
    ax.set_xlabel("NaN Percentage (%)", fontsize=11)
    ax.set_title("Missing Data by Feature Group", fontsize=12)
    ax.set_xlim(0, max(nan_pcts) * 1.3)
    ax.invert_yaxis()
    
    # Right: NaN counts by axon type for key features
    ax = axes[1]
    key_features = ["dsi", "step_up_QI", "iprgc_2hz_QI", "on_peak_extreme", "freq_step_1hz_amp"]
    key_features = [f for f in key_features if f in df.columns]
    
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    axon_types = [t for t in axon_order if t in df["axon_type_clean"].unique()]
    
    x = np.arange(len(key_features))
    width = 0.15
    
    for i, axon_type in enumerate(axon_types):
        mask = df["axon_type_clean"] == axon_type
        nan_pcts = [100 * df.loc[mask, f].isna().sum() / mask.sum() for f in key_features]
        ax.bar(x + i * width, nan_pcts, width, 
               label=AXON_TYPE_LABELS.get(axon_type, axon_type),
               color=AXON_TYPE_COLORS.get(axon_type, "#999999"))
    
    ax.set_xticks(x + width * (len(axon_types) - 1) / 2)
    ax.set_xticklabels([f.replace("_", "\n") for f in key_features], fontsize=9)
    ax.set_ylabel("NaN Percentage (%)", fontsize=11)
    ax.set_title("Missing Data by Axon Type (Key Features)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 10: PER-RECORDING QUALITY STATISTICS
# =============================================================================


def plot_recording_quality_stats(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (16, 12),
):
    """
    Plot per-recording quality statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with recording information
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract recording ID from index
    df = df.copy()
    df["recording_id"] = df.index.str.rsplit("_unit_", n=1).str[0]
    
    # Group by recording
    recording_stats = df.groupby("recording_id").agg({
        "step_up_QI": ["mean", "count"],
        "dsi": lambda x: x.notna().sum() / len(x) * 100,  # Valid DSI %
        "axon_type_clean": lambda x: (x == "rgc").sum() / len(x) * 100,  # RGC %
    }).reset_index()
    recording_stats.columns = ["recording_id", "mean_step_QI", "unit_count", "valid_dsi_pct", "rgc_pct"]
    
    # Extract date from recording_id for temporal analysis
    recording_stats["date"] = pd.to_datetime(
        recording_stats["recording_id"].str[:10], 
        format="%Y.%m.%d", 
        errors="coerce"
    )
    recording_stats = recording_stats.sort_values("date")
    
    # Top Left: Unit count distribution
    ax = axes[0, 0]
    ax.hist(recording_stats["unit_count"], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(recording_stats["unit_count"].median(), color="red", linestyle="--", 
               label=f'Median: {recording_stats["unit_count"].median():.0f}')
    ax.set_xlabel("Units per Recording", fontsize=11)
    ax.set_ylabel("Number of Recordings", fontsize=11)
    ax.set_title(f"Unit Count Distribution\n(n={len(recording_stats)} recordings)", fontsize=12)
    ax.legend()
    
    # Top Right: Mean step_up_QI distribution
    ax = axes[0, 1]
    valid_qi = recording_stats["mean_step_QI"].dropna()
    ax.hist(valid_qi, bins=30, color="forestgreen", edgecolor="white", alpha=0.8)
    ax.axvline(valid_qi.median(), color="red", linestyle="--",
               label=f'Median: {valid_qi.median():.3f}')
    ax.axvline(0.5, color="orange", linestyle=":", label="QI=0.5 threshold")
    ax.set_xlabel("Mean step_up_QI", fontsize=11)
    ax.set_ylabel("Number of Recordings", fontsize=11)
    ax.set_title("Mean Quality Index per Recording", fontsize=12)
    ax.legend()
    
    # Bottom Left: Valid DSI percentage per recording
    ax = axes[1, 0]
    ax.hist(recording_stats["valid_dsi_pct"], bins=30, color="purple", edgecolor="white", alpha=0.8)
    ax.axvline(recording_stats["valid_dsi_pct"].median(), color="red", linestyle="--",
               label=f'Median: {recording_stats["valid_dsi_pct"].median():.1f}%')
    ax.set_xlabel("Valid DSI Percentage (%)", fontsize=11)
    ax.set_ylabel("Number of Recordings", fontsize=11)
    ax.set_title("Proportion of Units with Valid DSI", fontsize=12)
    ax.legend()
    
    # Bottom Right: RGC percentage per recording
    ax = axes[1, 1]
    ax.hist(recording_stats["rgc_pct"], bins=30, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(recording_stats["rgc_pct"].median(), color="red", linestyle="--",
               label=f'Median: {recording_stats["rgc_pct"].median():.1f}%')
    ax.set_xlabel("RGC Percentage (%)", fontsize=11)
    ax.set_ylabel("Number of Recordings", fontsize=11)
    ax.set_title("Proportion of RGC Cells per Recording", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 11: TEMPORAL TRENDS ACROSS RECORDINGS
# =============================================================================


def plot_temporal_trends(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (16, 10),
):
    """
    Plot temporal trends in recording quality over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with recording information
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract recording ID and date
    df = df.copy()
    df["recording_id"] = df.index.str.rsplit("_unit_", n=1).str[0]
    df["date"] = pd.to_datetime(
        df["recording_id"].str[:10], 
        format="%Y.%m.%d", 
        errors="coerce"
    )
    
    # Group by date (weekly rolling)
    daily_stats = df.groupby("date").agg({
        "step_up_QI": "mean",
        "dsi": lambda x: x.notna().mean() * 100,
        "axon_type_clean": lambda x: (x == "rgc").mean() * 100,
        "recording_id": "count",
    }).reset_index()
    daily_stats.columns = ["date", "mean_step_QI", "valid_dsi_pct", "rgc_pct", "unit_count"]
    daily_stats = daily_stats.sort_values("date")
    
    # Calculate monthly rolling average
    daily_stats["month"] = daily_stats["date"].dt.to_period("M")
    monthly = daily_stats.groupby("month").agg({
        "mean_step_QI": "mean",
        "valid_dsi_pct": "mean",
        "rgc_pct": "mean",
        "unit_count": "sum",
    }).reset_index()
    monthly["date"] = monthly["month"].dt.to_timestamp()
    
    # Top Left: Units per month
    ax = axes[0, 0]
    ax.bar(monthly["date"], monthly["unit_count"], width=25, color="steelblue", alpha=0.8)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Total Units", fontsize=11)
    ax.set_title("Units Recorded per Month", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Top Right: Mean QI over time
    ax = axes[0, 1]
    ax.plot(monthly["date"], monthly["mean_step_QI"], "o-", color="forestgreen", markersize=6)
    ax.axhline(0.5, color="orange", linestyle=":", alpha=0.7, label="QI=0.5")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Mean step_up_QI", fontsize=11)
    ax.set_title("Quality Index Trend", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # Bottom Left: Valid DSI % over time
    ax = axes[1, 0]
    ax.plot(monthly["date"], monthly["valid_dsi_pct"], "o-", color="purple", markersize=6)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Valid DSI (%)", fontsize=11)
    ax.set_title("Valid Direction Selectivity Data Trend", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Bottom Right: RGC % over time
    ax = axes[1, 1]
    ax.plot(monthly["date"], monthly["rgc_pct"], "o-", color="coral", markersize=6)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("RGC (%)", fontsize=11)
    ax.set_title("RGC Proportion Trend", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 12: FEATURE CORRELATION HEATMAP
# =============================================================================


def plot_feature_correlation(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 12),
):
    """
    Plot correlation heatmap for key numeric features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select key numeric features
    key_features = [
        "dsi", "osi", "ds_p_value", "os_p_value",
        "step_up_QI", "iprgc_2hz_QI", "iprgc_20hz_QI",
        "on_peak_extreme", "off_peak_extreme", "on_off_ratio",
        "green_on_peak_extreme", "blue_on_peak_extreme", "green_blue_on_ratio",
        "freq_step_1hz_amp", "freq_step_2hz_amp", "freq_step_4hz_amp",
        "polar_radius", "gaussian_amp",
    ]
    
    available_features = [f for f in key_features if f in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[available_features].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Pearson Correlation", shrink=0.8)
    
    # Add text annotations for strong correlations
    for i in range(len(available_features)):
        for j in range(len(available_features)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.3:
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                       color=color, fontsize=7)
    
    # Labels
    ax.set_xticks(range(len(available_features)))
    ax.set_yticks(range(len(available_features)))
    feature_labels = [f.replace("_", "\n") for f in available_features]
    ax.set_xticklabels(feature_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(feature_labels, fontsize=8)
    
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 13: STEP RESPONSE FEATURE DISTRIBUTIONS
# =============================================================================


def plot_step_response_features(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (16, 12),
):
    """
    Plot step response feature distributions by axon type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with step response features
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    axon_types = [t for t in axon_order if t in df["axon_type_clean"].unique()]
    
    features = [
        ("on_peak_extreme", "ON Peak Response (Hz)"),
        ("off_peak_extreme", "OFF Peak Response (Hz)"),
        ("on_off_ratio", "ON/OFF Ratio"),
        ("on_trans_sus_ratio", "ON Trans/Sus Ratio"),
        ("off_trans_sus_ratio", "OFF Trans/Sus Ratio"),
        ("step_up_QI", "Quality Index"),
    ]
    
    for idx, (feature, title) in enumerate(features):
        ax = axes.flatten()[idx]
        
        if feature not in df.columns:
            ax.text(0.5, 0.5, f"Column '{feature}' not found", ha="center", va="center")
            continue
        
        data_for_box = []
        labels = []
        for axon_type in axon_types:
            data = df.loc[df["axon_type_clean"] == axon_type, feature].dropna()
            if len(data) > 0:
                # Clip outliers for visualization
                q1, q3 = data.quantile([0.01, 0.99])
                data = data.clip(q1, q3)
                data_for_box.append(data)
                labels.append(AXON_TYPE_LABELS.get(axon_type, axon_type))
        
        if data_for_box:
            bp = ax.boxplot(data_for_box, tick_labels=labels, patch_artist=True)
            
            for patch, axon_type in zip(bp['boxes'], axon_types):
                patch.set_facecolor(AXON_TYPE_COLORS.get(axon_type, "#999999"))
                patch.set_alpha(0.7)
        
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.yaxis.grid(True, alpha=0.3)
    
    plt.suptitle("Step Response Features by Axon Type", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# PLOT 14: FREQUENCY RESPONSE PROFILES
# =============================================================================


def plot_frequency_response_profiles(
    df: pd.DataFrame,
    output_path: Path = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot frequency response amplitude profiles by axon type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with frequency step features
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    frequencies = [0.5, 1, 2, 4, 10]
    freq_cols = [f"freq_step_{str(f).replace('.', '')}hz_amp" for f in frequencies]
    freq_cols = [c for c in freq_cols if c in df.columns]
    
    axon_order = ["rgc", "ac", "other", "unknown", "no_label"]
    axon_types = [t for t in axon_order if t in df["axon_type_clean"].unique()]
    
    # Top Left: Mean amplitude profile per axon type
    ax = axes[0, 0]
    x = np.arange(len(freq_cols))
    
    for axon_type in axon_types:
        mask = df["axon_type_clean"] == axon_type
        means = [df.loc[mask, col].mean() for col in freq_cols]
        ax.plot(x, means, "o-", label=AXON_TYPE_LABELS.get(axon_type, axon_type),
                color=AXON_TYPE_COLORS.get(axon_type, "#999999"), markersize=8, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in frequencies])
    ax.set_xlabel("Stimulus Frequency", fontsize=11)
    ax.set_ylabel("Mean Amplitude (Hz)", fontsize=11)
    ax.set_title("Frequency Response Profile by Axon Type", fontsize=12)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.3)
    
    # Top Right: Boxplot for 1 Hz response
    ax = axes[0, 1]
    if "freq_step_1hz_amp" in df.columns:
        data_for_box = []
        labels = []
        for axon_type in axon_types:
            data = df.loc[df["axon_type_clean"] == axon_type, "freq_step_1hz_amp"].dropna()
            if len(data) > 0:
                data = data.clip(0, data.quantile(0.99))
                data_for_box.append(data)
                labels.append(AXON_TYPE_LABELS.get(axon_type, axon_type))
        
        if data_for_box:
            bp = ax.boxplot(data_for_box, tick_labels=labels, patch_artist=True)
            for patch, axon_type in zip(bp['boxes'], axon_types):
                patch.set_facecolor(AXON_TYPE_COLORS.get(axon_type, "#999999"))
                patch.set_alpha(0.7)
    
    ax.set_ylabel("1 Hz Amplitude (Hz)", fontsize=11)
    ax.set_title("1 Hz Response by Axon Type", fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    
    # Bottom Left: R-squared distribution for 1 Hz
    ax = axes[1, 0]
    if "freq_step_1hz_r_squared" in df.columns:
        for axon_type in axon_types:
            mask = df["axon_type_clean"] == axon_type
            data = df.loc[mask, "freq_step_1hz_r_squared"].dropna()
            ax.hist(data, bins=30, alpha=0.5, label=AXON_TYPE_LABELS.get(axon_type, axon_type),
                    color=AXON_TYPE_COLORS.get(axon_type, "#999999"), density=True)
    
    ax.axvline(0.1, color="red", linestyle="--", alpha=0.7, label="R²=0.1 threshold")
    ax.set_xlabel("R² (1 Hz fit)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Fit Quality Distribution (1 Hz)", fontsize=12)
    ax.legend(fontsize=8)
    
    # Bottom Right: NaN phase distribution by frequency
    ax = axes[1, 1]
    phase_cols = [f"freq_step_{str(f).replace('.', '')}hz_phase" for f in frequencies]
    phase_cols = [c for c in phase_cols if c in df.columns]
    
    nan_pcts = [100 * df[col].isna().sum() / len(df) for col in phase_cols]
    x = np.arange(len(phase_cols))
    ax.bar(x, nan_pcts, color="salmon", edgecolor="white")
    
    for i, pct in enumerate(nan_pcts):
        ax.text(i, pct + 1, f"{pct:.1f}%", ha="center", fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in frequencies])
    ax.set_xlabel("Stimulus Frequency", fontsize=11)
    ax.set_ylabel("NaN Phase (%)", fontsize=11)
    ax.set_title("Missing Phase Data (Poor Fit R² < 0.1)", fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    
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
    """Generate all axon type validation plots."""
    print("=" * 80)
    print("Axon Type Feature Validation Plots")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print(f"\nLoading data from: {INPUT_PARQUET}")
    df = load_and_prepare_data(INPUT_PARQUET)
    print(f"Loaded DataFrame: {df.shape} (units x columns)")
    
    # Print summary
    print("\nAxon Type Summary:")
    type_counts = df["axon_type_clean"].value_counts()
    total = len(df)
    for axon_type, count in type_counts.items():
        label = AXON_TYPE_LABELS.get(axon_type, axon_type)
        print(f"  {label}: {count:,} ({100*count/total:.1f}%)")
    
    # Generate plots
    print("\n[1/14] Generating axon type distribution plot...")
    plot_axon_type_distribution(df, output_path=OUTPUT_DIR / "axon_type_distribution.png")
    
    print("[2/14] Generating axon type vs subtype crosstab...")
    plot_axon_subtype_crosstab(df, output_path=OUTPUT_DIR / "axon_subtype_crosstab.png")
    
    print("[3/14] Generating stacked subtypes plot...")
    plot_stacked_subtypes(df, output_path=OUTPUT_DIR / "stacked_subtypes_by_axon.png")
    
    print("[4/14] Generating DSI/OSI distributions by axon type...")
    plot_dsi_osi_by_axon_type(df, output_path=OUTPUT_DIR / "dsi_osi_by_axon_type.png")
    
    print("[5/14] Generating RGC subtype breakdown...")
    plot_rgc_subtype_breakdown(df, output_path=OUTPUT_DIR / "rgc_subtype_breakdown.png")
    
    print("[6/14] Generating ipRGC QI by axon type...")
    plot_iprgc_qi_by_axon_type(df, output_path=OUTPUT_DIR / "iprgc_qi_by_axon_type.png")
    
    print("[7/14] Generating p-value distributions...")
    plot_pvalue_distributions(df, output_path=OUTPUT_DIR / "pvalue_distributions.png")
    
    print("[8/14] Generating DSI vs OSI scatter plots...")
    plot_dsi_osi_scatter(df, output_path=OUTPUT_DIR / "dsi_osi_scatter_by_axon.png")
    
    print("[9/14] Generating NaN distribution heatmap...")
    plot_nan_distribution_heatmap(df, output_path=OUTPUT_DIR / "nan_distribution_heatmap.png")
    
    print("[10/14] Generating per-recording quality statistics...")
    plot_recording_quality_stats(df, output_path=OUTPUT_DIR / "recording_quality_stats.png")
    
    print("[11/14] Generating temporal trends...")
    plot_temporal_trends(df, output_path=OUTPUT_DIR / "temporal_trends.png")
    
    print("[12/14] Generating feature correlation heatmap...")
    plot_feature_correlation(df, output_path=OUTPUT_DIR / "feature_correlation.png")
    
    print("[13/14] Generating step response feature distributions...")
    plot_step_response_features(df, output_path=OUTPUT_DIR / "step_response_features.png")
    
    print("[14/14] Generating frequency response profiles...")
    plot_frequency_response_profiles(df, output_path=OUTPUT_DIR / "frequency_response_profiles.png")
    
    print("\n" + "=" * 80)
    print(f"Done! All plots saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

