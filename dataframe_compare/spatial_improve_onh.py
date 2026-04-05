"""
Step 1: Improved ONH Detection for Blocker Comparison
=====================================================
Extracts AP pathway data from H5 files, applies robust ONH detection
per recording, recomputes improved coordinates, and saves an augmented
parquet with improved_tx / improved_ty.

Reuses logic from improve_onh_v6.py.
"""

import importlib.util
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from compare_config import OUTPUT_DIR, EXPORT_DIR as H5_DIR

INPUT_PARQUET = OUTPUT_DIR / "compared_dataframe_v2_labeled.parquet"
OUTPUT_PARQUET = OUTPUT_DIR / "compared_dataframe_v2_labeled_spatial.parquet"

COORD_SCALE = 16
COORD_LIMIT = 100

# ------------------------------------------------------------------
# Import AP tracking modules (same approach as improve_onh_v6.py)
# ------------------------------------------------------------------

def _import_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pa = _import_mod(
    "pathway_analysis",
    PROJECT_ROOT / "src" / "hdmea" / "features" / "ap_tracking" / "pathway_analysis.py",
)
_dvnt = _import_mod(
    "dvnt_parser",
    PROJECT_ROOT / "src" / "hdmea" / "features" / "ap_tracking" / "dvnt_parser.py",
)

APPathway = _pa.APPathway
APIntersection = _pa.APIntersection
calculate_optimal_intersection = _pa.calculate_optimal_intersection
calculate_soma_polar_coordinates = _pa.calculate_soma_polar_coordinates

DVNTPosition = _dvnt.DVNTPosition
parse_dvnt_from_center_xy = _dvnt.parse_dvnt_from_center_xy

logger.info("AP tracking imports done")


# =====================================================================
# ONH detection (from improve_onh_v6.py)
# =====================================================================

def robust_onh(pathways, r2_min=0.7, mad_factor=3.0, max_dist=80.0):
    """Median + MAD outlier rejection, high R^2 threshold."""
    valid = {
        k: v for k, v in pathways.items()
        if v is not None and v.r_value ** 2 >= r2_min
    }
    if len(valid) < 2:
        return None
    ids = list(valid.keys())
    pts = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            f1, f2 = valid[ids[i]], valid[ids[j]]
            dm = f1.slope - f2.slope
            if abs(dm) < 0.05:
                continue
            xi = (f2.intercept - f1.intercept) / dm
            yi = f1.slope * xi + f1.intercept
            if abs(xi - 33) > max_dist or abs(yi - 33) > max_dist:
                continue
            pts.append((xi, yi))
    if len(pts) == 0:
        return None
    arr = np.array(pts)
    med_x, med_y = np.median(arr[:, 0]), np.median(arr[:, 1])
    if len(pts) >= 5:
        dists = np.sqrt((arr[:, 0] - med_x) ** 2 + (arr[:, 1] - med_y) ** 2)
        mad = np.median(dists)
        if mad > 0:
            keep = dists < mad_factor * mad
            if keep.sum() >= 3:
                arr = arr[keep]
    fx = float(np.median(arr[:, 0]))
    fy = float(np.median(arr[:, 1]))
    errors = [
        ((fit.slope * fx - fy + fit.intercept) ** 2 / (1 + fit.slope ** 2))
        for fit in valid.values()
    ]
    return APIntersection(x=fx, y=fy, mse=float(np.mean(errors)))


def robust_onh_with_fallback(pathways):
    onh = robust_onh(pathways, r2_min=0.7)
    if onh is None:
        onh = robust_onh(pathways, r2_min=0.5)
    return onh


# =====================================================================
# H5 data extraction
# =====================================================================

def extract_recording_data(h5_path):
    """
    Extract AP pathway, soma position, and DVNT data from one H5 file.

    Returns
    -------
    pathways : dict[str, APPathway]
    soma_positions : dict[str, tuple[int, int]]  (row, col)
    center_xy_str : str
    legacy_onh : APIntersection or None
    """
    pathways = {}
    soma_positions = {}
    center_xy_str = ""
    legacy_onh = None

    with h5py.File(str(h5_path), "r") as f:
        # DVNT from gsheet_row
        meta = f.get("metadata")
        if meta is not None:
            gs = meta.get("gsheet_row")
            if gs is not None and "Center_xy" in gs:
                val = gs["Center_xy"][()]
                if isinstance(val, bytes):
                    val = val.decode()
                center_xy_str = val

        # Legacy ONH
        if meta is not None:
            at = meta.get("ap_tracking")
            if at is not None and "all_ap_intersection" in at:
                aint = at["all_ap_intersection"]
                ox = float(aint["x"][()])
                oy = float(aint["y"][()])
                if not (np.isnan(ox) or np.isnan(oy)):
                    rmse_val = 0.0
                    if "rmse" in aint:
                        rmse_val = float(aint["rmse"][()])
                    elif "mse" in aint:
                        rmse_val = float(aint["mse"][()])
                    legacy_onh = APIntersection(x=ox, y=oy, mse=rmse_val)

        # Per-unit data
        units_grp = f.get("units")
        if units_grp is None:
            return pathways, soma_positions, center_xy_str, legacy_onh

        for uid in units_grp:
            u = units_grp[uid]

            # Soma position
            um = u.get("unit_meta")
            if um is not None:
                row_val = um.get("row")
                col_val = um.get("column")
                if row_val is not None and col_val is not None:
                    soma_positions[uid] = (int(row_val[()]), int(col_val[()]))

            # AP pathway
            ap_track = u.get("features", {}).get("ap_tracking")
            if ap_track is None:
                continue
            ap_pw = ap_track.get("ap_pathway")
            if ap_pw is None:
                continue

            slope_ds = ap_pw.get("slope")
            intercept_ds = ap_pw.get("intercept")
            r_value_ds = ap_pw.get("r_value")
            if slope_ds is None or intercept_ds is None or r_value_ds is None:
                continue

            s = float(slope_ds[()])
            i = float(intercept_ds[()])
            r = float(r_value_ds[()])
            if np.isnan(s) or np.isnan(i) or np.isnan(r):
                continue

            pathways[uid] = APPathway(
                slope=s, intercept=i, r_value=r,
                p_value=0.0, std_err=0.0, num_points=0,
            )

    return pathways, soma_positions, center_xy_str, legacy_onh


def compute_polar_coords(onh, soma_positions, center_xy_str):
    """
    Given an ONH intersection, compute transformed coordinates for each unit.

    Returns dict[uid] -> (transformed_x, transformed_y)
    """
    if onh is None:
        return {}

    dvnt = DVNTPosition(dv_position=None, nt_position=None, lr_position=None)
    if isinstance(center_xy_str, str) and center_xy_str.strip():
        try:
            dvnt = parse_dvnt_from_center_xy(center_xy_str)
        except Exception:
            pass

    result = {}
    for uid, (sr, sc) in soma_positions.items():
        try:
            polar = calculate_soma_polar_coordinates(
                soma_xy=(sr, sc),
                intersection=onh,
                dv_position=dvnt.dv_position,
                nt_position=dvnt.nt_position,
            )
            result[uid] = (polar.transformed_x, polar.transformed_y)
        except Exception:
            continue
    return result


# =====================================================================
# Gradient evaluation (from improve_onh_v6.py)
# =====================================================================

def evaluate_gradient(tx_series, ty_series, feat_series):
    """Pearson r of feat vs Y coordinate, with coord/feature filtering."""
    tmp = pd.DataFrame({"tx": tx_series, "ty": ty_series, "feat": feat_series})
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    mask = (tmp["tx"].abs() < COORD_LIMIT) & (tmp["ty"].abs() < COORD_LIMIT)
    tmp = tmp[mask]
    if len(tmp) < 30:
        return np.nan, np.nan, len(tmp)
    y = tmp["ty"].values * COORD_SCALE
    c = tmp["feat"].values
    ry, py = pearsonr(y, c)
    return float(ry), float(py), len(tmp)


# =====================================================================
# Main
# =====================================================================

def main():
    logger.info("Loading compared DataFrame ...")
    df = pd.read_parquet(INPUT_PARQUET)
    logger.info(f"  Shape: {df.shape}")

    # Need before_dataset_id, before_unit_id to map to H5 files
    if "before_dataset_id" not in df.columns:
        logger.error("before_dataset_id column not found -- cannot proceed")
        return
    if "before_unit_id" not in df.columns:
        logger.error("before_unit_id column not found -- cannot proceed")
        return

    # Group by before_dataset_id (each = one before recording)
    unique_recs = df["before_dataset_id"].unique()
    logger.info(f"  Unique before recordings: {len(unique_recs)}")

    # For each recording, extract AP data and compute coords
    # with both legacy and robust ONH
    legacy_coords = {}  # pair_key -> (tx, ty)
    robust_coords = {}  # pair_key -> (tx, ty)

    for rec_name in sorted(unique_recs):
        h5_path = H5_DIR / f"{rec_name}.h5"
        if not h5_path.exists():
            logger.warning(f"  H5 not found: {h5_path}")
            continue

        pathways, soma_positions, center_xy_str, legacy_onh = extract_recording_data(h5_path)

        # Robust ONH
        robust_onh_result = robust_onh_with_fallback(pathways)

        # Legacy ONH
        legacy_onh_result = legacy_onh
        if legacy_onh_result is None:
            legacy_onh_result = calculate_optimal_intersection(pathways)

        # Compute coords with both methods
        leg_coords = compute_polar_coords(legacy_onh_result, soma_positions, center_xy_str)
        rob_coords = compute_polar_coords(robust_onh_result, soma_positions, center_xy_str)

        n_leg = len(leg_coords)
        n_rob = len(rob_coords)
        n_pw = len(pathways)

        # Map back to DataFrame rows
        rec_rows = df[df["before_dataset_id"] == rec_name]
        for pair_key, row in rec_rows.iterrows():
            uid = row["before_unit_id"]
            if uid in leg_coords:
                legacy_coords[pair_key] = leg_coords[uid]
            if uid in rob_coords:
                robust_coords[pair_key] = rob_coords[uid]

        logger.info(
            f"  {rec_name}: {n_pw} AP pathways, "
            f"legacy={n_leg} coords, robust={n_rob} coords"
        )

    # Build coordinate series
    leg_tx = pd.Series(np.nan, index=df.index, dtype=float)
    leg_ty = pd.Series(np.nan, index=df.index, dtype=float)
    rob_tx = pd.Series(np.nan, index=df.index, dtype=float)
    rob_ty = pd.Series(np.nan, index=df.index, dtype=float)

    for pk, (tx, ty) in legacy_coords.items():
        leg_tx.at[pk] = tx
        leg_ty.at[pk] = ty
    for pk, (tx, ty) in robust_coords.items():
        rob_tx.at[pk] = tx
        rob_ty.at[pk] = ty

    leg_valid = leg_tx.notna().sum()
    rob_valid = rob_tx.notna().sum()
    logger.info(f"  Legacy coords: {leg_valid} valid")
    logger.info(f"  Robust coords: {rob_valid} valid")

    # Evaluate gradient quality using green_blue_on_ratio
    # Use before_green_blue_on_ratio as the feature
    feat_col = None
    for candidate in ["before_green_blue_on_ratio", "before_green_blue_on_ratio_high"]:
        if candidate in df.columns:
            feat_col = candidate
            break

    if feat_col is None:
        logger.warning("No green_blue_on_ratio found; using robust coords by default")
        best_tx, best_ty = rob_tx, rob_ty
        best_label = "Robust"
    else:
        feat = df[feat_col]

        ry_leg, py_leg, n_leg = evaluate_gradient(leg_tx, leg_ty, feat)
        ry_rob, py_rob, n_rob = evaluate_gradient(rob_tx, rob_ty, feat)

        # Also evaluate the existing before_transformed coords
        ry_exist, py_exist, n_exist = evaluate_gradient(
            df.get("before_transformed_x", pd.Series(dtype=float)),
            df.get("before_transformed_y", pd.Series(dtype=float)),
            feat,
        )

        logger.info(f"  Gradient ({feat_col} vs Y):")
        logger.info(f"    Existing coords: r={ry_exist:.6f}, p={py_exist:.2e}, n={n_exist}")
        logger.info(f"    Legacy ONH:      r={ry_leg:.6f}, p={py_leg:.2e}, n={n_leg}")
        logger.info(f"    Robust ONH:      r={ry_rob:.6f}, p={py_rob:.2e}, n={n_rob}")

        # Pick the best (highest |r| with Y)
        candidates = [
            (abs(ry_exist) if not np.isnan(ry_exist) else -1,
             df.get("before_transformed_x", pd.Series(dtype=float)),
             df.get("before_transformed_y", pd.Series(dtype=float)),
             "Existing"),
            (abs(ry_leg) if not np.isnan(ry_leg) else -1, leg_tx, leg_ty, "Legacy"),
            (abs(ry_rob) if not np.isnan(ry_rob) else -1, rob_tx, rob_ty, "Robust"),
        ]
        best = max(candidates, key=lambda t: t[0])
        best_tx, best_ty = best[1], best[2]
        best_label = best[3]
        logger.info(f"  Best method: {best_label} (|r|={best[0]:.6f})")

    # Add improved coordinates to DataFrame
    df["improved_tx"] = best_tx
    df["improved_ty"] = best_ty

    # Also store the alternatives for reference
    df["legacy_onh_tx"] = leg_tx
    df["legacy_onh_ty"] = leg_ty
    df["robust_onh_tx"] = rob_tx
    df["robust_onh_ty"] = rob_ty

    # Save
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=True)
    logger.info(f"  Saved -> {OUTPUT_PARQUET}")

    n_improved = df["improved_tx"].notna().sum()
    logger.info(f"  Cells with improved coords: {n_improved}/{len(df)}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
