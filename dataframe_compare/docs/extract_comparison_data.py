"""
Extract pipeline statistics organised by literature-review themes.

Reads spatial_quant_combined.parquet, radial_center_combined.parquet,
spatial_metrics_compare.parquet, and the labeled-spatial parquet for each
experiment, then writes per-theme CSV tables and a cross-experiment summary
into docs/tables/.

Usage:
    python extract_comparison_data.py          # all 3 experiments
    python extract_comparison_data.py --exp _ptx_str
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent          # dataframe_compare/
TABLE_DIR = Path(__file__).resolve().parent / "tables"  # docs/tables/

EXPERIMENTS = ["_ptx_str", "_ptx", "_str"]

# ── theme-to-feature mapping ────────────────────────────────────────────────

THEMES: Dict[str, Dict] = {
    "opsin_gradient": {
        "label": "Opsin gradient / color opponency",
        "features": [
            "green_blue_on_ratio",
            "green_blue_off_ratio",
            "green_blue_on_ratio_high",
            "green_blue_off_ratio_high",
            "gb_base_mean",
            "gb_base_mean_high",
        ],
    },
    "alpha_sustained": {
        "label": "Alpha-like ON sustained topography",
        "features": ["on_sustained", "on_peak_extreme"],
    },
    "direction_selectivity": {
        "label": "Direction selectivity spatial bias",
        "features": ["dsi"],
    },
    "on_off_pathway": {
        "label": "ON/OFF pathway modulation by inhibition",
        "features": [
            "on_off_ratio",
            "on_off_sus_ratio",
            "on_peak_extreme",
            "off_peak_extreme",
        ],
    },
    "temporal_filtering": {
        "label": "Transient/sustained temporal filtering",
        "features": ["on_trans_sus_ratio", "off_trans_sus_ratio"],
    },
    "iprgc": {
        "label": "ipRGC spatial organisation",
        "features": ["step_up_QI"],
    },
    "orientation": {
        "label": "Orientation selectivity",
        "features": ["osi"],
    },
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_read(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    print(f"  [WARN] missing: {path}")
    return None


def _exp_label(exp: str) -> str:
    return exp.lstrip("_")


# ── per-experiment extraction ────────────────────────────────────────────────

def extract_theme_table(
    sq: pd.DataFrame,
    rc: pd.DataFrame | None,
    features: List[str],
    exp: str,
) -> pd.DataFrame:
    """Build a before / after / delta comparison for *features* from the
    spatial-quant and radial-center dataframes (all_cells scope only)."""

    rows = []
    for feat in features:
        sq_feat = sq[
            (sq["feature"] == feat) & (sq["scope"] == "all_cells")
        ]
        for cond in ("before", "after", "delta"):
            sq_row = sq_feat[sq_feat["condition"] == cond]
            if sq_row.empty:
                continue
            s = sq_row.iloc[0]
            entry = {
                "experiment": _exp_label(exp),
                "feature": feat,
                "condition": cond,
                "grad_mag": s.get("grad_mag", np.nan),
                "grad_dir_deg": s.get("grad_dir_deg", np.nan),
                "plane_r2": s.get("plane_r2", np.nan),
                "moran_i": s.get("moran_i", np.nan),
                "radial_r": s.get("radial_r", np.nan),
                "radial_p": s.get("radial_p", np.nan),
                "fdr_q_plane_r2": s.get("fdr_q_plane_r2", np.nan),
                "fdr_q_moran": s.get("fdr_q_moran", np.nan),
                "n_bins": s.get("n_bins", np.nan),
            }

            # radial centre info (raw_mean only)
            if rc is not None:
                rc_row = rc[
                    (rc["feature"] == feat)
                    & (rc["condition"] == cond)
                    & (rc["scope"] == "all_cells")
                    & (rc["data_type"] == "raw_mean")
                ]
                if not rc_row.empty:
                    r = rc_row.iloc[0]
                    entry["best_cx"] = r["best_center_x"]
                    entry["best_cy"] = r["best_center_y"]
                    entry["best_r"] = r["best_r"]
                    entry["best_p"] = r["best_p"]
                    entry["origin_r"] = r["origin_r"]
                    entry["abs_r_improvement"] = r["abs_r_improvement"]

            rows.append(entry)
    return pd.DataFrame(rows)


def extract_before_after_delta(
    sq: pd.DataFrame,
    features: List[str],
    exp: str,
) -> pd.DataFrame:
    """Side-by-side before vs after comparison for Delta-R2, Delta-Moran."""
    rows = []
    ac = sq[sq["scope"] == "all_cells"]
    for feat in features:
        bf = ac[(ac["feature"] == feat) & (ac["condition"] == "before")]
        af = ac[(ac["feature"] == feat) & (ac["condition"] == "after")]
        if bf.empty or af.empty:
            continue
        b, a = bf.iloc[0], af.iloc[0]
        rows.append({
            "experiment": _exp_label(exp),
            "feature": feat,
            "before_R2": b["plane_r2"],
            "after_R2": a["plane_r2"],
            "delta_R2": a["plane_r2"] - b["plane_r2"],
            "before_moran": b["moran_i"],
            "after_moran": a["moran_i"],
            "delta_moran": a["moran_i"] - b["moran_i"],
            "before_grad_dir": b["grad_dir_deg"],
            "after_grad_dir": a["grad_dir_deg"],
            "before_grad_mag": b["grad_mag"],
            "after_grad_mag": a["grad_mag"],
        })
    return pd.DataFrame(rows)


def extract_radial_shifts(
    rc: pd.DataFrame,
    features: List[str],
    exp: str,
) -> pd.DataFrame:
    """Radial centre displacement between before and after."""
    rows = []
    raw = rc[(rc["scope"] == "all_cells") & (rc["data_type"] == "raw_mean")]
    for feat in features:
        bf = raw[(raw["feature"] == feat) & (raw["condition"] == "before")]
        af = raw[(raw["feature"] == feat) & (raw["condition"] == "after")]
        if bf.empty or af.empty:
            continue
        b, a = bf.iloc[0], af.iloc[0]
        shift = np.hypot(
            a["best_center_x"] - b["best_center_x"],
            a["best_center_y"] - b["best_center_y"],
        )
        rows.append({
            "experiment": _exp_label(exp),
            "feature": feat,
            "before_cx": b["best_center_x"],
            "before_cy": b["best_center_y"],
            "after_cx": a["best_center_x"],
            "after_cy": a["best_center_y"],
            "shift_um": shift,
            "before_abs_r": abs(b["best_r"]),
            "after_abs_r": abs(a["best_r"]),
            "delta_abs_r": abs(a["best_r"]) - abs(b["best_r"]),
        })
    return pd.DataFrame(rows)


def extract_per_group_stats(
    sq: pd.DataFrame,
    features: List[str],
    exp: str,
) -> pd.DataFrame:
    """Per-group (DSGC / ipRGC / OSGC / Other) spatial stats."""
    rows = []
    groups = [g for g in sq["scope"].unique() if g != "all_cells"]
    for feat in features:
        for grp in groups:
            for cond in ("before", "after", "delta"):
                sel = sq[
                    (sq["feature"] == feat)
                    & (sq["scope"] == grp)
                    & (sq["condition"] == cond)
                ]
                if sel.empty:
                    continue
                s = sel.iloc[0]
                rows.append({
                    "experiment": _exp_label(exp),
                    "feature": feat,
                    "group": grp,
                    "condition": cond,
                    "plane_r2": s["plane_r2"],
                    "moran_i": s["moran_i"],
                    "grad_mag": s["grad_mag"],
                    "grad_dir_deg": s["grad_dir_deg"],
                    "radial_r": s["radial_r"],
                    "n_bins": s["n_bins"],
                })
    return pd.DataFrame(rows)


def extract_cell_counts(labeled_path: Path, exp: str) -> pd.DataFrame:
    if not labeled_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(labeled_path, columns=["group"])
    vc = df["group"].value_counts()
    rows = [
        {"experiment": _exp_label(exp), "group": g, "count": int(c)}
        for g, c in vc.items()
    ]
    rows.append({
        "experiment": _exp_label(exp),
        "group": "TOTAL",
        "count": len(df),
    })
    return pd.DataFrame(rows)


# ── cross-experiment summary ─────────────────────────────────────────────────

def build_cross_experiment_summary(all_ba: pd.DataFrame) -> pd.DataFrame:
    """For each feature, show delta-R2 and delta-Moran across experiments."""
    if all_ba.empty:
        return pd.DataFrame()
    pivot = all_ba.pivot_table(
        index="feature",
        columns="experiment",
        values=["delta_R2", "delta_moran"],
    )
    pivot.columns = [f"{v}_{e}" for v, e in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=None,
                        help="Experiments to process (default: all 3)")
    args = parser.parse_args()
    exps = args.exp or EXPERIMENTS

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_ba_frames = []
    all_cell_count_frames = []

    for exp in exps:
        tag = _exp_label(exp)
        out_dir = BASE / f"output{exp}"
        print(f"\n{'='*60}")
        print(f"  Experiment: {tag}  ({out_dir})")
        print(f"{'='*60}")

        sq = _safe_read(out_dir / "spatial_quant_combined.parquet")
        rc = _safe_read(out_dir / "radial_center_combined.parquet")
        labeled_path = out_dir / "compared_dataframe_v2_labeled_spatial.parquet"

        if sq is None:
            print(f"  SKIP {tag}: no spatial_quant_combined.parquet")
            continue

        # cell counts
        cc = extract_cell_counts(labeled_path, exp)
        if not cc.empty:
            all_cell_count_frames.append(cc)

        # per-theme tables
        for theme_key, theme_info in THEMES.items():
            feats = theme_info["features"]

            detail = extract_theme_table(sq, rc, feats, exp)
            ba = extract_before_after_delta(sq, feats, exp)
            all_ba_frames.append(ba)

            grp = extract_per_group_stats(sq, feats, exp)
            shift = extract_radial_shifts(rc, feats, exp) if rc is not None else pd.DataFrame()

            csv_path = TABLE_DIR / f"theme_{theme_key}_{tag}.csv"
            combined = pd.concat([detail], ignore_index=True)
            combined.to_csv(csv_path, index=False)
            print(f"  wrote {csv_path.name}  ({len(combined)} rows)")

            if not ba.empty:
                ba_path = TABLE_DIR / f"theme_{theme_key}_{tag}_ba.csv"
                ba.to_csv(ba_path, index=False)

            if not shift.empty:
                shift_path = TABLE_DIR / f"theme_{theme_key}_{tag}_shift.csv"
                shift.to_csv(shift_path, index=False)

            if not grp.empty:
                grp_path = TABLE_DIR / f"theme_{theme_key}_{tag}_pergroup.csv"
                grp.to_csv(grp_path, index=False)

    # cross-experiment summary
    if all_ba_frames:
        all_ba = pd.concat(all_ba_frames, ignore_index=True)
        xexp = build_cross_experiment_summary(all_ba)
        xexp_path = TABLE_DIR / "cross_experiment_summary.csv"
        xexp.to_csv(xexp_path, index=False)
        print(f"\n  wrote {xexp_path.name}  ({len(xexp)} rows)")

        # also save the full before-after table
        all_ba.to_csv(TABLE_DIR / "all_before_after.csv", index=False)

    if all_cell_count_frames:
        cc_all = pd.concat(all_cell_count_frames, ignore_index=True)
        cc_all.to_csv(TABLE_DIR / "cell_counts.csv", index=False)
        print(f"  wrote cell_counts.csv")

    # JSON summary for easy consumption by the markdown document
    summary = {}
    for csv in sorted(TABLE_DIR.glob("*.csv")):
        key = csv.stem
        df = pd.read_csv(csv)
        summary[key] = df.to_dict(orient="records")

    json_path = TABLE_DIR / "comparison_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  wrote {json_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
