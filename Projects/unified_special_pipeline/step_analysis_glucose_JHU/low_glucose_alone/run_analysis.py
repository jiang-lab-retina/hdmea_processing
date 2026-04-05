"""
Low-glucose-alone analysis pipeline.

Reproduces Figure X from legacy visualize_chains_5.py:
  1. Convert CMCR/CMTR to HDF5  (reuses unified data_loader)
  2. Align recording pairs       (reuses unified unit_alignment)
  3. Extract ON/OFF features     (mirrors legacy get_trace_feature_list)
  4. Export pooled feature arrays for plotting

Usage:
  python run_analysis.py                  # full pipeline
  python run_analysis.py --skip-convert   # skip CMCR->HDF5 if already done
  python run_analysis.py --skip-align     # skip alignment if already done
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_GLUCOSE_JHU_DIR = _THIS_DIR.parent
_USP_DIR = _GLUCOSE_JHU_DIR.parent

for _p in (str(_GLUCOSE_JHU_DIR), str(_USP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step_change_analysis.data_loader import (
    load_cmcr_cmtr_data,
    save_recording_to_hdf5,
    load_recording_from_hdf5,
)
from step_change_analysis.specific_config import (
    PipelineConfig,
    StepDetectionConfig,
    QualityConfig,
    AlignmentConfig,
)
from step_change_analysis.unit_alignment import (
    create_aligned_group,
    load_aligned_group_from_hdf5,
)

_local_cfg_spec = importlib.util.spec_from_file_location(
    "low_glucose_alone_config", _THIS_DIR / "specific_config.py")
_local_cfg = importlib.util.module_from_spec(_local_cfg_spec)
_local_cfg_spec.loader.exec_module(_local_cfg)
LowGlucoseAloneConfig = _local_cfg.LowGlucoseAloneConfig
default_config = _local_cfg.default_config
find_data_folder = _local_cfg.find_data_folder
cmcr_path = _local_cfg.cmcr_path
cmtr_path = _local_cfg.cmtr_path

_parent_ra_path = _GLUCOSE_JHU_DIR / "high-glucose-alone" / "run_analysis.py"
_spec_ra = importlib.util.spec_from_file_location("parent_run_analysis", _parent_ra_path)
_parent_ra = importlib.util.module_from_spec(_spec_ra)
_spec_ra.loader.exec_module(_parent_ra)
median_mean_smooth = _parent_ra.median_mean_smooth
classify_unit_on_off = _parent_ra.classify_unit_on_off

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---- Step 1: Convert CMCR/CMTR to HDF5 ----

def convert_all_recordings(config: LowGlucoseAloneConfig):
    """Convert every CMCR/CMTR pair to HDF5."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ctrl_folder = find_data_folder(config.control_folder_name)
    low_folder = find_data_folder(config.low_glucose_folder_name)

    pipeline_cfg = PipelineConfig()

    all_stems = set()
    for pair in config.control_pairs + config.low_glucose_pairs:
        all_stems.update(pair)

    for stem in sorted(all_stems):
        h5_out = config.output_dir / f"{stem}.h5"
        if h5_out.exists():
            logger.info("HDF5 exists, skipping: %s", h5_out.name)
            continue

        if stem.startswith("2025.10.07"):
            folder = ctrl_folder
        else:
            folder = low_folder

        cmcr = cmcr_path(folder, stem)
        cmtr_p = cmtr_path(folder, stem)
        if not cmcr.exists() or not cmtr_p.exists():
            logger.warning("Missing file for %s, skipping", stem)
            continue

        logger.info("Converting %s ...", stem)
        data = load_cmcr_cmtr_data(cmcr, cmtr_p)
        save_recording_to_hdf5(
            data, h5_out,
            step_config=pipeline_cfg.step_detection,
            quality_config=pipeline_cfg.quality,
            overwrite=True,
        )
        logger.info("  -> %s  (%d units)",
                     h5_out.name,
                     len(data.get("units", {})))


# ---- Step 2: Align recording pairs ----

def align_all_pairs(config: LowGlucoseAloneConfig):
    """Run unit alignment on each recording pair and save grouped HDF5."""
    pipeline_cfg = PipelineConfig()

    for pair in config.control_pairs + config.low_glucose_pairs:
        h5_paths = [config.output_dir / f"{s}.h5" for s in pair]
        missing = [p for p in h5_paths if not p.exists()]
        if missing:
            logger.warning("Missing HDF5 for pair %s, skipping", pair)
            continue

        group_name = f"{pair[0]}_{pair[1]}_aligned.h5"
        group_path = config.output_dir / group_name
        if group_path.exists():
            logger.info("Aligned group exists, skipping: %s", group_name)
            continue

        logger.info("Aligning pair: %s", pair)
        create_aligned_group(
            h5_paths,
            output_path=group_path,
            config=pipeline_cfg,
            use_fixed_ref=False,
        )
        logger.info("  -> %s", group_name)


# ---- Step 3: Extract features (legacy-compatible) ----

def _load_alignment_chains(group_path: Path):
    """Load alignment chains from a grouped HDF5, handling numpy dtype issues."""
    import h5py
    with h5py.File(group_path, "r") as f:
        if "alignment" not in f:
            return None
        columns = list(f["alignment"].attrs["columns"])
        raw = f["alignment/chains"][:]
        str_data = raw.astype(str)
        df = pd.DataFrame(str_data, columns=columns)
        df = df.replace("", np.nan)
        return df


def _extract_feature_from_pair(
    pair: Tuple[str, str],
    config: LowGlucoseAloneConfig,
    peak_range: Tuple[int, int],
) -> Tuple[np.ndarray, list]:
    """Extract per-chain feature traces from an aligned pair.

    Uses BOTH recordings in the pair and concatenates their features,
    matching legacy get_trace_feature_list with group_file_selection_range=[0,2].
    Each recording is clipped to repeat_num_clip trials, so the output
    has up to 2 * repeat_num_clip trials per chain.

    Returns:
      feature_traces: (n_chains, n_total_trials) after smoothing + per-unit norm
      raw_trace_arrays: list of (n_total_trials, n_bins_per_trial) arrays
    """
    group_name = f"{pair[0]}_{pair[1]}_aligned.h5"
    group_path = config.output_dir / group_name

    if not group_path.exists():
        logger.warning("Aligned group not found: %s", group_name)
        return np.empty((0, config.repeat_num_clip * 2)), []

    chains_df = _load_alignment_chains(group_path)
    if chains_df is None or chains_df.empty:
        logger.warning("No alignment chains in: %s", group_name)
        return np.empty((0, config.repeat_num_clip * 2)), []

    rec_names = sorted(chains_df.columns)

    rec_units = {}
    for rec_stem in pair:
        h5_path = config.output_dir / f"{rec_stem}.h5"
        if not h5_path.exists():
            logger.warning("HDF5 not found: %s", h5_path)
            return np.empty((0, config.repeat_num_clip * 2)), []
        rec_data = load_recording_from_hdf5(h5_path)
        rec_units[rec_stem] = rec_data.get("units", {})

    bl = config.baseline_range
    pk = peak_range
    clip = config.repeat_num_clip

    feature_traces = []
    raw_trace_arrays = []

    complete_chains = chains_df.dropna(how="any")
    for _, row in complete_chains.iterrows():
        per_rec_feats = []
        chain_raw = []
        valid = True

        for rec_name in rec_names:
            uid_raw = row[rec_name]
            units = rec_units.get(rec_name, {})
            uid = str(int(float(uid_raw))) if uid_raw not in units else str(uid_raw)

            udata = units.get(uid)
            if udata is None:
                valid = False
                break
            sr = udata.get("step_responses")
            if sr is None:
                valid = False
                break
            sr = np.array(sr)
            n_avail = min(sr.shape[0], clip)
            if n_avail < 10:
                valid = False
                break

            sr_clipped = sr[:n_avail]
            feat = np.abs(
                sr_clipped[:, pk[0]:pk[1]].max(axis=1)
                - sr_clipped[:, bl[0]:bl[1]].mean(axis=1)
            )
            per_rec_feats.append(feat)
            chain_raw.extend(sr_clipped)

        if not valid or len(per_rec_feats) == 0:
            continue

        feature_traces.append(per_rec_feats)
        raw_trace_arrays.append(np.array(chain_raw))

    if not feature_traces:
        return np.empty((0, clip * 2)), []

    n_recs = len(feature_traces[0])

    smoothed_segs = []
    for per_rec in feature_traces:
        unit_segs = []
        for seg in per_rec:
            s = seg.astype(float).reshape(1, -1)
            s = median_mean_smooth(s, window=config.smoothing_window).ravel()
            mx = s.max()
            if mx > 0:
                s = s / mx
            unit_segs.append(s)
        smoothed_segs.append(unit_segs)

    if n_recs > 1:
        for k in range(1, n_recs):
            seg_k_all = np.array([u[k] for u in smoothed_segs
                                  if len(u[k]) == len(smoothed_segs[0][k])])
            pop_mean_first = np.nanmean(seg_k_all[:, 0]) if seg_k_all.size else 1.0
            if pop_mean_first > 0:
                for u_segs in smoothed_segs:
                    u_segs[k] = u_segs[k] / pop_mean_first

            seg_prev_all = np.array([u[k - 1] for u in smoothed_segs
                                     if len(u[k - 1]) == len(smoothed_segs[0][k - 1])])
            pop_mean_last = np.nanmean(seg_prev_all[:, -1]) if seg_prev_all.size else 1.0

            for u_segs in smoothed_segs:
                u_segs[k] = u_segs[k] * pop_mean_last

    max_len = max(sum(len(s) for s in u) for u in smoothed_segs)
    rows = []
    for u_segs in smoothed_segs:
        ft_arr = np.concatenate(u_segs)
        if len(ft_arr) < max_len:
            pad = np.full(max_len - len(ft_arr), np.nan)
            ft_arr = np.concatenate([ft_arr, pad])
        rows.append(ft_arr)

    arr = np.array(rows)

    return arr, raw_trace_arrays


def extract_all_features(
    config: LowGlucoseAloneConfig,
) -> Dict[str, dict]:
    """Extract ON and OFF features for control and low-glucose groups.

    Returns dict with keys like 'off_control', 'off_low', 'on_control', 'on_low',
    each containing 'features' (array) and 'raw' (list of arrays).
    """
    results = {}

    for feature_name, peak_range in [
        ("off", config.off_peak_range),
        ("on", config.on_peak_range),
    ]:
        for group_label, pairs in [
            ("control", config.control_pairs),
            ("low", config.low_glucose_pairs),
        ]:
            all_feat = []
            all_raw = []
            for pair in pairs:
                feat, raw = _extract_feature_from_pair(pair, config, peak_range)
                if feat.shape[0] > 0:
                    all_feat.append(feat)
                    all_raw.extend(raw)

            if all_feat:
                max_trials = max(f.shape[1] for f in all_feat)
                padded_feat = []
                for f in all_feat:
                    if f.shape[1] < max_trials:
                        pad = np.full((f.shape[0], max_trials - f.shape[1]), np.nan)
                        padded_feat.append(np.hstack([f, pad]))
                    else:
                        padded_feat.append(f)
                combined = np.concatenate(padded_feat, axis=0)
            else:
                combined = np.empty((0, config.repeat_num_clip))

            key = f"{feature_name}_{group_label}"
            results[key] = {"features": combined, "raw": all_raw}
            logger.info("  %s: %d units", key, combined.shape[0])

    return results


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-align", action="store_true")
    args = parser.parse_args()

    config = default_config

    if not args.skip_convert:
        logger.info("=== Step 1: Convert CMCR/CMTR to HDF5 ===")
        convert_all_recordings(config)

    if not args.skip_align:
        logger.info("=== Step 2: Align recording pairs ===")
        align_all_pairs(config)

    logger.info("=== Step 3: Extract features ===")
    results = extract_all_features(config)

    out_path = config.output_dir / "extracted_features.npz"
    save_dict = {}
    for key, val in results.items():
        save_dict[key] = val["features"]
    np.savez(out_path, **save_dict)
    logger.info("Saved features to %s", out_path)

    for key, val in results.items():
        logger.info("  %s: %s", key, val["features"].shape)

    logger.info("Done.")


if __name__ == "__main__":
    main()
