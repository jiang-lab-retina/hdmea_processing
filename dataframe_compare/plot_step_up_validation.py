"""
Plot step-up response comparison: Reference vs Before vs After blocker.

One figure per group (ipRGC, DSGC, OSGC, Other).
Each row   = one cluster/subtype within the group.
Columns    = Reference | Before Blocker | After Blocker.
Each panel = mean trace (solid line) + shaded mean +/- std.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent

from compare_config import OUTPUT_DIR, FIG_DIR_BASE

BLOCKER_PATH = OUTPUT_DIR / "compared_dataframe_v2_labeled.parquet"
REF_PATH = (
    PROJECT_ROOT
    / "dataframe_phase"
    / "classification_v2"
    / "divide_conquer_method"
    / "results"
    / "labeled_dataframe.parquet"
)
FIG_DIR = FIG_DIR_BASE / "validation"

TRACE_COL = "step_up_5s_5i_b0_3x"
GROUP_ORDER = ["ipRGC", "DSGC", "OSGC", "Other"]
SAMPLING_RATE = 60.0  # Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trials_to_mean_trace(cell_data):
    """Average across trials for a single cell, returning a 1-D array."""
    if cell_data is None:
        return None
    if isinstance(cell_data, np.ndarray) and cell_data.dtype == object:
        arrs = [np.asarray(a, dtype=np.float64) for a in cell_data if a is not None]
    elif isinstance(cell_data, list):
        arrs = [np.asarray(a, dtype=np.float64) for a in cell_data if a is not None]
    else:
        return np.asarray(cell_data, dtype=np.float64)
    if len(arrs) == 0:
        return None
    min_len = min(len(a) for a in arrs)
    stacked = np.vstack([a[:min_len] for a in arrs])
    return stacked.mean(axis=0)


def collect_traces(df, col):
    """Return (n_cells, trace_len) array of trial-averaged traces."""
    traces = []
    for val in df[col]:
        mean_t = trials_to_mean_trace(val)
        if mean_t is not None and len(mean_t) > 0:
            traces.append(mean_t)
    if len(traces) == 0:
        return None
    min_len = min(len(t) for t in traces)
    return np.vstack([t[:min_len] for t in traces])


def plot_mean_std(ax, traces_2d, color, label, alpha_fill=0.2):
    """Plot mean line + shaded std band."""
    if traces_2d is None or len(traces_2d) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray")
        return
    mean = traces_2d.mean(axis=0)
    std = traces_2d.std(axis=0)
    t = np.arange(len(mean)) / SAMPLING_RATE
    ax.fill_between(t, mean - std, mean + std, color="gray", alpha=alpha_fill)
    ax.plot(t, mean, color=color, linewidth=1.2, label=label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading blocker data from {BLOCKER_PATH} ...")
    df_blocker = pd.read_parquet(BLOCKER_PATH)
    df_labeled = df_blocker[df_blocker["group"] != ""].copy()
    print(f"  {len(df_labeled)} labeled cells")

    print(f"Loading reference data from {REF_PATH} ...")
    df_ref = pd.read_parquet(REF_PATH)
    # Exclude _invalid subtypes
    df_ref = df_ref[~df_ref["subtype"].str.contains("invalid", na=False)].copy()
    print(f"  {len(df_ref)} reference cells")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for group in GROUP_ORDER:
        # Subtypes present in the blocker data for this group
        grp_mask = df_labeled["group"] == group
        subtypes = sorted(
            df_labeled.loc[grp_mask, "subtype"].unique(),
            key=lambda s: int(s.rsplit("_", 1)[-1]),
        )
        n_subtypes = len(subtypes)
        if n_subtypes == 0:
            continue

        fig, axes = plt.subplots(
            n_subtypes, 3,
            figsize=(14, 2.6 * n_subtypes),
            squeeze=False,
            sharex=True,
        )

        col_titles = ["Reference", "Before Blocker", "After Blocker"]
        col_colors = ["#1f77b4", "#2ca02c", "#d62728"]  # blue, green, red

        for row_idx, subtype in enumerate(subtypes):
            # ----- Reference -----
            ref_sub = df_ref[df_ref["subtype"] == subtype]
            ref_traces = collect_traces(ref_sub, TRACE_COL) if len(ref_sub) > 0 else None

            # ----- Before -----
            blk_sub = df_labeled[df_labeled["subtype"] == subtype]
            before_traces = collect_traces(blk_sub, f"before_{TRACE_COL}")

            # ----- After -----
            after_traces = collect_traces(blk_sub, f"after_{TRACE_COL}")

            all_row_traces = [ref_traces, before_traces, after_traces]

            # Compute shared y-limits across all three columns in this row
            y_min, y_max = np.inf, -np.inf
            for traces in all_row_traces:
                if traces is not None and len(traces) > 0:
                    mean = traces.mean(axis=0)
                    std = traces.std(axis=0)
                    y_min = min(y_min, (mean - std).min())
                    y_max = max(y_max, (mean + std).max())
            if np.isinf(y_min):
                y_min, y_max = 0, 1
            margin = (y_max - y_min) * 0.05
            row_ylim = (y_min - margin, y_max + margin)

            for col_idx, (traces, color, title) in enumerate(
                zip(
                    all_row_traces,
                    col_colors,
                    col_titles,
                )
            ):
                ax = axes[row_idx, col_idx]
                plot_mean_std(ax, traces, color, title)
                ax.set_ylim(row_ylim)

                n = 0 if traces is None else traces.shape[0]
                # Row label on left column
                if col_idx == 0:
                    ax.set_ylabel(f"{subtype}\n(n={n})", fontsize=9)
                else:
                    # Show n in top-right
                    ax.text(
                        0.97, 0.93, f"n={n}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="gray",
                    )

                # Column title on first row
                if row_idx == 0:
                    ax.set_title(title, fontsize=10, fontweight="bold")

                ax.tick_params(labelsize=7)

        # Shared x-label
        for ax in axes[-1, :]:
            ax.set_xlabel("Time (s)", fontsize=9)

        fig.suptitle(
            f"{group} -- Step-Up Response Comparison",
            fontsize=13, fontweight="bold", y=1.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        out_path = FIG_DIR / f"step_up_validation_{group}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}  ({n_subtypes} subtypes)")

    print("Done.")


if __name__ == "__main__":
    main()
