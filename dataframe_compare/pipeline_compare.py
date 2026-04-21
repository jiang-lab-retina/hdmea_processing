"""
Blocker Before/After Comparison DataFrame Pipeline.

Reads aligned H5 pair mappings, computes firing rates and loads/extracts
features for both before and after units from output_export/ H5 files,
and produces a single comparison DataFrame with before_/after_ column prefixes.
Also computes intensity-specific green-blue features (low/mid/high).

Steps:
  0 - Build pair index from aligned H5 files
  1 - Compute firing rates
  2 - Load HDF5 features (axon type, coordinates, etc.)
  3 - Extract derived features (QI, DSI, step, GB, freq)
  4 - Merge before/after with prefixes -> compared_dataframe.parquet
  5 - Intensity-specific GB features   -> compared_dataframe_v2.parquet

Usage:
    python pipeline_compare.py                     # Process all pairs
    python pipeline_compare.py --end 2             # Process first 2 pairs (test)
    python pipeline_compare.py --start-step 2      # Resume from step 2
    python pipeline_compare.py --start-step 5      # Re-run intensity GB only
    python pipeline_compare.py --no-features        # Skip feature extraction (steps 2-3)
"""

import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# PATH SETUP
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dataframe_phase" / "load_traces"))
sys.path.insert(0, str(PROJECT_ROOT / "dataframe_phase" / "load_feature"))
sys.path.insert(0, str(PROJECT_ROOT / "dataframe_phase" / "extract_feature"))

# Import from existing modules (do NOT modify originals)
from pipeline_firing_rate import (
    get_frame_aligned_firing_rate,
    get_sample_based_firing_rate,
    parse_column_groups,
    validate_units,
    reshape_to_movies,
)
from load_feature import load_features_from_hdf5

from extract_feature_gb import compute_mean_trace, extract_gb_features_from_trace

from compare_config import (
    ALIGNED_DIR,
    EXPORT_DIR,
    OUTPUT_DIR,
    MOVIES_BEFORE,
    MOVIES_AFTER,
    MOVIE_DIRECTION_SECTION,
    MOVIE_SAMPLE_BASED,
    IPRGC_TARGET_RATE_HZ,
    IPRGC_EXPECTED_BINS,
    IPRGC_LENGTH_TOLERANCE,
    EXCLUDED_MOVIES,
    MOVING_BAR_PREFIX,
    GB_TRACE_COLUMN,
    STEP_TRACE_COLUMN,
    FREQ_TRACE_COLUMN,
    FEATURE_PATHS,
)


# =============================================================================
# STEP 0: BUILD PAIR INDEX
# =============================================================================

def build_pair_index(
    aligned_dir: Path = ALIGNED_DIR,
    start: int = 0,
    end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read aligned H5 files and build a pair index DataFrame.

    Each row represents one aligned unit pair with metadata.

    Returns:
        DataFrame indexed by pair_key ({pair_id}_{aligned_idx:04d})
        with columns: pair_id, chip, genotype, before_dataset_id,
        before_unit_id, after_dataset_id, after_unit_id, cell_type
    """
    aligned_files = sorted(aligned_dir.glob("*_aligned.h5"))
    if not aligned_files:
        raise FileNotFoundError(f"No aligned H5 files found in {aligned_dir}")

    print(f"Found {len(aligned_files)} aligned H5 files")

    rows = []
    for af in tqdm(aligned_files, desc="Reading aligned files"):
        with h5py.File(af, "r") as f:
            pair_id = str(f.attrs["pair_id"])
            chip = str(f.attrs["chip"])
            genotype = str(f.attrs.get("genotype", ""))

            # Extract dataset IDs from H5 paths stored in attrs
            before_h5_name = Path(str(f.attrs["before_h5"])).stem
            after_h5_name = Path(str(f.attrs["after_h5"])).stem

            # Read paired unit arrays
            if "connections/before_units" not in f or "connections/after_units" not in f:
                continue

            before_units = f["connections/before_units"][:]
            after_units = f["connections/after_units"][:]

            # Read cell types from paired_units group
            paired_units_grp = f.get("paired_units", None)

            for i, (bu, au) in enumerate(zip(before_units, after_units)):
                bu_str = bu.decode("utf-8") if isinstance(bu, bytes) else str(bu)
                au_str = au.decode("utf-8") if isinstance(au, bytes) else str(au)

                cell_type = "unknown"
                if paired_units_grp is not None:
                    pair_key_h5 = f"pair_{i:04d}"
                    if pair_key_h5 in paired_units_grp:
                        cell_type = str(
                            paired_units_grp[pair_key_h5].attrs.get("cell_type", "unknown")
                        )

                rows.append({
                    "pair_id": pair_id,
                    "chip": chip,
                    "genotype": genotype,
                    "before_dataset_id": before_h5_name,
                    "before_unit_id": bu_str,
                    "after_dataset_id": after_h5_name,
                    "after_unit_id": au_str,
                    "cell_type": cell_type,
                })

    df = pd.DataFrame(rows)

    # Create pair_key index: {pair_id}_{sequential_idx:04d}
    pair_counters: Dict[str, int] = {}
    pair_keys = []
    for _, row in df.iterrows():
        pid = row["pair_id"]
        idx = pair_counters.get(pid, 0)
        pair_counters[pid] = idx + 1
        pair_keys.append(f"{pid}_{idx:04d}")

    df.index = pair_keys
    df.index.name = "pair_key"

    # Apply start/end slicing
    if end is not None:
        df = df.iloc[start:end]
    elif start > 0:
        df = df.iloc[start:]

    print(f"Pair index: {len(df)} unit pairs across {df['pair_id'].nunique()} recording pairs")
    return df


# =============================================================================
# STEP 1: COMPUTE FIRING RATES (adapted from pipeline_firing_rate.py)
# =============================================================================

def determine_target_lengths_for_files(
    h5_paths: List[Path],
    movies_frame_aligned: List[str],
    include_iprgc: bool = False,
    include_direction: bool = True,
) -> Dict[str, int]:
    """
    First pass: determine minimum frame counts per trial type.

    Adapted from pipeline_firing_rate.determine_target_lengths()
    with configurable movie lists.
    """
    frame_counts: Dict[str, List[int]] = {}

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            if "metadata/frame_timestamps" not in f:
                continue
            all_frames = f["metadata/frame_timestamps"][:]
            acq_rate = float(f["metadata/acquisition_rate"][()])

            unit_ids = list(f["units"].keys())
            if not unit_ids:
                continue
            sample_unit = unit_ids[0]

            # Regular movies
            for movie_name in movies_frame_aligned:
                trials_path = f"units/{sample_unit}/spike_times_sectioned/{movie_name}/trials_start_end"
                if trials_path not in f:
                    continue
                trials_start_end = f[trials_path][:]
                for trial_idx, (start, end) in enumerate(trials_start_end):
                    mask = (all_frames >= start) & (all_frames < end)
                    n_bins = np.sum(mask) - 1
                    col_key = f"{movie_name}_{trial_idx}"
                    frame_counts.setdefault(col_key, []).append(n_bins)

            # Direction sections
            if include_direction:
                dir_section_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/direction_section"
                if dir_section_path in f:
                    dir_group = f[dir_section_path]
                    for direction in dir_group.keys():
                        if direction == "_attrs":
                            continue
                        dir_data = dir_group[direction]
                        if "section_bounds" not in dir_data:
                            continue
                        bounds = dir_data["section_bounds"][:]
                        for rep_idx, (start_rel, end_rel) in enumerate(bounds):
                            n_frames = end_rel - start_rel
                            if n_frames <= 0:
                                continue
                            col_key = f"{MOVIE_DIRECTION_SECTION}_{direction}_{rep_idx}"
                            frame_counts.setdefault(col_key, []).append(n_frames - 1)

            # iprgc_test (sample-based)
            if include_iprgc:
                iprgc_path = f"units/{sample_unit}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_start_end"
                if iprgc_path in f:
                    trials_start_end = f[iprgc_path][:]
                    samples_per_bin = acq_rate / IPRGC_TARGET_RATE_HZ
                    min_expected = int(IPRGC_EXPECTED_BINS * (1 - IPRGC_LENGTH_TOLERANCE))
                    max_expected = int(IPRGC_EXPECTED_BINS * (1 + IPRGC_LENGTH_TOLERANCE))
                    for trial_idx, (start, end) in enumerate(trials_start_end):
                        n_bins = int(np.ceil((end - start) / samples_per_bin))
                        if min_expected <= n_bins <= max_expected:
                            col_key = f"{MOVIE_SAMPLE_BASED}_{trial_idx}"
                            frame_counts.setdefault(col_key, []).append(n_bins)

    return {k: min(v) for k, v in frame_counts.items()}


def process_h5_files(
    h5_paths: List[Path],
    target_lengths: Dict[str, int],
    movies_frame_aligned: List[str],
    include_iprgc: bool = False,
    include_direction: bool = True,
    unit_filter: Optional[Dict[str, Set[str]]] = None,
) -> pd.DataFrame:
    """
    Process H5 files to compute firing rates.

    Adapted from pipeline_firing_rate.process_all_data() with:
    - Configurable movie lists
    - Scalar dataset handling for trial spike times
    - Optional unit filtering

    Args:
        h5_paths: List of H5 file paths
        target_lengths: Target lengths per column
        movies_frame_aligned: List of frame-aligned movie names
        include_iprgc: Whether to process iprgc_test
        include_direction: Whether to process direction sections
        unit_filter: If provided, dict mapping dataset_id -> set of unit_ids to include
    """
    rows = []

    for h5_path in tqdm(h5_paths, desc="Computing firing rates"):
        with h5py.File(h5_path, "r") as f:
            dataset_id = h5_path.stem
            if "metadata/frame_timestamps" not in f:
                continue
            all_frames = f["metadata/frame_timestamps"][:]
            acq_rate = float(f["metadata/acquisition_rate"][()])

            for unit_id in f["units"].keys():
                # Filter if requested
                if unit_filter is not None:
                    if dataset_id not in unit_filter:
                        continue
                    if unit_id not in unit_filter[dataset_id]:
                        continue

                row_data = {"row_index": f"{dataset_id}_{unit_id}"}

                # Regular movies
                for movie_name in movies_frame_aligned:
                    trials_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/trials_spike_times"
                    starts_path = f"units/{unit_id}/spike_times_sectioned/{movie_name}/trials_start_end"
                    if trials_path not in f or starts_path not in f:
                        continue
                    trials_start_end = f[starts_path][:]
                    trials_group = f[trials_path]

                    for trial_idx_str in trials_group.keys():
                        trial_idx = int(trial_idx_str)
                        trial_ds = trials_group[trial_idx_str]
                        # Handle scalar datasets (single spike stored as scalar)
                        if trial_ds.shape == ():
                            spike_times = np.array([trial_ds[()]])
                        elif trial_ds.size == 0:
                            spike_times = np.array([])
                        else:
                            spike_times = trial_ds[:]
                        trial_start, trial_end = trials_start_end[trial_idx]

                        fr, _ = get_frame_aligned_firing_rate(
                            spike_times, trial_start, trial_end, all_frames, acq_rate
                        )
                        col_key = f"{movie_name}_{trial_idx}"
                        target_len = target_lengths.get(col_key, len(fr))
                        if len(fr) > target_len:
                            fr = fr[:target_len]
                        elif len(fr) < target_len:
                            fr = np.pad(fr, (0, target_len - len(fr)))
                        row_data[col_key] = fr.tolist()

                # Direction sections
                if include_direction:
                    dir_section_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/direction_section"
                    if dir_section_path in f:
                        dir_group = f[dir_section_path]
                        movie_trials_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_DIRECTION_SECTION}/trials_start_end"
                        if movie_trials_path in f:
                            movie_start_end = f[movie_trials_path][:]
                            movie_start_sample = movie_start_end[0, 0]
                            movie_start_frame_idx = np.searchsorted(all_frames, movie_start_sample)

                            for direction in dir_group.keys():
                                if direction == "_attrs":
                                    continue
                                dir_data = dir_group[direction]
                                if "section_bounds" not in dir_data or "trials" not in dir_data:
                                    continue
                                bounds = dir_data["section_bounds"][:]
                                trials_group_d = dir_data["trials"]

                                for rep_idx_str in trials_group_d.keys():
                                    rep_idx = int(rep_idx_str)
                                    start_frame_rel, end_frame_rel = bounds[rep_idx]
                                    if end_frame_rel <= start_frame_rel:
                                        continue

                                    trial_ds = trials_group_d[rep_idx_str]
                                    if trial_ds.shape == ():
                                        spike_frames = np.array([trial_ds[()]])
                                    elif trial_ds.size == 0:
                                        spike_frames = np.array([])
                                    else:
                                        spike_frames = trial_ds[:]

                                    start_frame_abs = movie_start_frame_idx + start_frame_rel
                                    end_frame_abs = movie_start_frame_idx + end_frame_rel
                                    if start_frame_abs >= len(all_frames) or end_frame_abs > len(all_frames):
                                        continue
                                    section_frames = all_frames[int(start_frame_abs):int(end_frame_abs)]
                                    if len(section_frames) < 2:
                                        continue

                                    n_frames = len(section_frames)
                                    counts = np.zeros(n_frames - 1, dtype=np.float32)
                                    for spike_frame in spike_frames:
                                        bin_idx = spike_frame - start_frame_rel
                                        if 0 <= bin_idx < (n_frames - 1):
                                            counts[int(bin_idx)] += 1

                                    frame_intervals = np.diff(section_frames) / acq_rate
                                    fr = np.zeros_like(counts, dtype=np.float32)
                                    valid = frame_intervals > 0
                                    fr[valid] = counts[valid] / frame_intervals[valid]

                                    col_key = f"{MOVIE_DIRECTION_SECTION}_{direction}_{rep_idx}"
                                    target_len = target_lengths.get(col_key, len(fr))
                                    if len(fr) > target_len:
                                        fr = fr[:target_len]
                                    elif len(fr) < target_len:
                                        fr = np.pad(fr, (0, target_len - len(fr)))
                                    row_data[col_key] = fr.tolist()

                # iprgc_test (sample-based)
                if include_iprgc:
                    iprgc_trials_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_spike_times"
                    iprgc_starts_path = f"units/{unit_id}/spike_times_sectioned/{MOVIE_SAMPLE_BASED}/trials_start_end"
                    if iprgc_trials_path in f and iprgc_starts_path in f:
                        trials_start_end = f[iprgc_starts_path][:]
                        trials_group_i = f[iprgc_trials_path]
                        min_expected = int(IPRGC_EXPECTED_BINS * (1 - IPRGC_LENGTH_TOLERANCE))
                        max_expected = int(IPRGC_EXPECTED_BINS * (1 + IPRGC_LENGTH_TOLERANCE))

                        valid_trials = []
                        for trial_idx_str in trials_group_i.keys():
                            trial_idx = int(trial_idx_str)
                            trial_ds = trials_group_i[trial_idx_str]
                            if trial_ds.shape == ():
                                spike_times = np.array([trial_ds[()]])
                            elif trial_ds.size == 0:
                                spike_times = np.array([])
                            else:
                                spike_times = trial_ds[:]
                            trial_start, trial_end = trials_start_end[trial_idx]

                            fr, n_bins = get_sample_based_firing_rate(
                                spike_times, trial_start, trial_end,
                                IPRGC_TARGET_RATE_HZ, acq_rate,
                            )
                            if not (min_expected <= n_bins <= max_expected):
                                continue
                            valid_trials.append((trial_idx, fr))

                        if valid_trials:
                            min_len_unit = min(len(fr) for _, fr in valid_trials)
                            for trial_idx, fr in valid_trials:
                                col_key = f"{MOVIE_SAMPLE_BASED}_{trial_idx}"
                                fr_trimmed = fr[:min_len_unit]
                                row_data[col_key] = fr_trimmed.tolist()

                rows.append(row_data)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.set_index("row_index")


def compute_firing_rates(
    pair_index: pd.DataFrame,
    export_dir: Path = EXPORT_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute firing rates for all before and after units in the pair index.

    Returns:
        (before_movies_df, after_movies_df) -- reshaped movie-based DataFrames
    """
    # Collect unique files and relevant units
    before_files: Dict[str, Set[str]] = defaultdict(set)
    after_files: Dict[str, Set[str]] = defaultdict(set)

    for _, row in pair_index.iterrows():
        before_files[row["before_dataset_id"]].add(row["before_unit_id"])
        after_files[row["after_dataset_id"]].add(row["after_unit_id"])

    before_paths = [export_dir / f"{did}.h5" for did in sorted(before_files.keys())]
    after_paths = [export_dir / f"{did}.h5" for did in sorted(after_files.keys())]

    # Validate files exist
    for p in before_paths + after_paths:
        if not p.exists():
            print(f"  WARNING: Missing file {p}")

    before_paths = [p for p in before_paths if p.exists()]
    after_paths = [p for p in after_paths if p.exists()]

    print(f"\n  Before: {len(before_paths)} files, {sum(len(v) for v in before_files.values())} units")
    print(f"  After:  {len(after_paths)} files, {sum(len(v) for v in after_files.values())} units")

    # --- BEFORE firing rates ---
    print("\n  Computing BEFORE target lengths...")
    before_targets = determine_target_lengths_for_files(
        before_paths, MOVIES_BEFORE, include_iprgc=False,
    )
    print(f"  BEFORE target lengths: {len(before_targets)} columns")

    print("  Processing BEFORE files...")
    before_raw = process_h5_files(
        before_paths, before_targets, MOVIES_BEFORE,
        include_iprgc=False, unit_filter=dict(before_files),
    )
    print(f"  BEFORE raw: {before_raw.shape}")

    # --- AFTER firing rates ---
    print("\n  Computing AFTER target lengths...")
    after_targets = determine_target_lengths_for_files(
        after_paths, MOVIES_AFTER, include_iprgc=True,
    )
    print(f"  AFTER target lengths: {len(after_targets)} columns")

    print("  Processing AFTER files...")
    after_raw = process_h5_files(
        after_paths, after_targets, MOVIES_AFTER,
        include_iprgc=True, unit_filter=dict(after_files),
    )
    print(f"  AFTER raw: {after_raw.shape}")

    # --- Reshape to movie-based format ---
    print("\n  Reshaping BEFORE...")
    before_groups = parse_column_groups(before_raw.columns.tolist())
    # Filter out excluded movies
    before_groups = {k: v for k, v in before_groups.items() if k not in EXCLUDED_MOVIES}
    before_movies = reshape_to_movies(before_raw, before_groups)
    print(f"  BEFORE movies: {before_movies.shape} -- columns: {list(before_movies.columns)}")

    print("  Reshaping AFTER...")
    after_groups = parse_column_groups(after_raw.columns.tolist())
    after_groups = {k: v for k, v in after_groups.items() if k not in EXCLUDED_MOVIES}
    after_movies = reshape_to_movies(after_raw, after_groups)
    print(f"  AFTER movies: {after_movies.shape} -- columns: {list(after_movies.columns)}")

    return before_movies, after_movies


# =============================================================================
# STEP 2: LOAD HDF5 FEATURES
# =============================================================================

def load_hdf5_features(
    df: pd.DataFrame,
    export_dir: Path = EXPORT_DIR,
) -> pd.DataFrame:
    """
    Load HDF5 features for all units in a DataFrame.

    Reuses load_features_from_hdf5 from load_feature.py.
    Index format: {dataset_id}_unit_{unit_id} -> parse to get file + unit.
    """
    # Group indices by source file
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for idx in df.index:
        parts = idx.rsplit("_unit_", 1)
        if len(parts) == 2:
            dataset_id = parts[0]
            unit_id = f"unit_{parts[1]}"
            grouped[dataset_id].append((idx, unit_id))

    # Initialize feature columns
    feature_series = {col: pd.Series(index=df.index, dtype=object) for col in FEATURE_PATHS.keys()}
    total_unit_count_series = pd.Series(index=df.index, dtype=object)

    missing_files = []
    feature_counts = {col: 0 for col in FEATURE_PATHS.keys()}

    for dataset_id, index_unit_pairs in tqdm(grouped.items(), desc="Loading HDF5 features"):
        h5_path = export_dir / f"{dataset_id}.h5"
        if not h5_path.exists():
            missing_files.append(dataset_id)
            continue

        unit_ids = [uid for _, uid in index_unit_pairs]
        unit_features, file_total = load_features_from_hdf5(h5_path, unit_ids, FEATURE_PATHS)

        for idx, unit_id in index_unit_pairs:
            features = unit_features.get(unit_id, {})
            for col_name in FEATURE_PATHS.keys():
                value = features.get(col_name)
                feature_series[col_name][idx] = value
                if value is not None:
                    feature_counts[col_name] += 1
            total_unit_count_series[idx] = file_total

    # Add columns to DataFrame
    for col_name, series in feature_series.items():
        df[col_name] = series
    df["total_unit_count"] = total_unit_count_series

    if missing_files:
        print(f"  Missing files: {len(missing_files)}")

    # Summary
    n_total = len(df)
    for col_name, count in feature_counts.items():
        if count < n_total and col_name in ("axon_type", "transformed_x"):
            print(f"  {col_name}: {count}/{n_total} found")

    return df


# =============================================================================
# STEP 3: EXTRACT DERIVED FEATURES
# =============================================================================

def extract_all_features(
    df: pd.DataFrame,
    label: str = "before",
    has_iprgc: bool = False,
) -> pd.DataFrame:
    """
    Run all feature extraction steps on a DataFrame.

    Imports from extract_feature/ modules. Passes blocker-specific
    column names where needed.
    """
    from extract_feature_step_iprgc import compute_step_up_qi, compute_iprgc_qi, add_good_cell_counts
    from extract_feature_dsgc import (
        remap_direction_columns, process_unit,
        DIRECTION_ANGLES, CORRECTED_DIRECTION_COLUMNS,
        N_PERMUTATIONS, N_TRIALS,
    )
    from extract_feature_step import extract_step_features
    from extract_feature_gb import extract_gb_features
    from extract_feature_freq import extract_freq_step_features, extract_freq_sectioned_traces

    print(f"\n  [{label.upper()}] Step/ipRGC QI...")
    if STEP_TRACE_COLUMN in df.columns:
        step_qi = compute_step_up_qi(df, movie_col=STEP_TRACE_COLUMN)
        df["step_up_QI"] = step_qi
        print(f"    step_up_QI: {step_qi.notna().sum()} valid")
    else:
        df["step_up_QI"] = np.nan
        print(f"    step_up_QI: SKIPPED (no {STEP_TRACE_COLUMN} column)")

    if has_iprgc and MOVIE_SAMPLE_BASED in df.columns:
        iprgc_2hz, iprgc_20hz = compute_iprgc_qi(df, movie_col=MOVIE_SAMPLE_BASED)
        df["iprgc_2hz_QI"] = iprgc_2hz
        df["iprgc_20hz_QI"] = iprgc_20hz
        print(f"    iprgc_2hz_QI: {iprgc_2hz.notna().sum()} valid")
    else:
        df["iprgc_2hz_QI"] = np.nan
        df["iprgc_20hz_QI"] = np.nan
        print(f"    iprgc QI: SKIPPED (no iprgc_test column in {label})")

    # Good cell counts
    if "axon_type" in df.columns and "step_up_QI" in df.columns:
        df = add_good_cell_counts(df)
        print(f"    good_count, good_rgc_count added")

    # DSGC features
    print(f"  [{label.upper()}] DSGC features...")
    # Check if any direction columns exist
    dir_cols = [c for c in df.columns if c.startswith(MOVING_BAR_PREFIX + "_")]
    if dir_cols and "angle_correction_applied" in df.columns:
        corrected_results = []
        for idx in df.index:
            row = df.loc[idx]
            corrected_row = remap_direction_columns(row)
            corrected_results.append(corrected_row)
        corrected_df = pd.DataFrame(corrected_results, index=df.index)
        for col in CORRECTED_DIRECTION_COLUMNS:
            if col in corrected_df.columns:
                df[col] = corrected_df[col]

        directions = np.array(DIRECTION_ANGLES)
        results = []
        for idx in df.index:
            row = df.loc[idx]
            if pd.isna(row.get("angle_correction_applied")):
                results.append({
                    "dsi": np.nan, "osi": np.nan,
                    "preferred_direction": np.nan,
                    "preferred_orientation": np.nan,
                    "ds_p_value": np.nan, "os_p_value": np.nan,
                })
            else:
                result = process_unit(row, directions, CORRECTED_DIRECTION_COLUMNS, N_PERMUTATIONS, N_TRIALS)
                results.append(result)
        results_df = pd.DataFrame(results, index=df.index)
        for col in ["dsi", "osi", "preferred_direction", "preferred_orientation", "ds_p_value", "os_p_value"]:
            df[col] = results_df[col]
        valid_dsi = df["dsi"].notna().sum()
        print(f"    DSI/OSI: {valid_dsi} valid")
    else:
        for col in ["dsi", "osi", "preferred_direction", "preferred_orientation", "ds_p_value", "os_p_value"]:
            df[col] = np.nan
        print(f"    DSI/OSI: SKIPPED (no direction columns)")

    # Step response features
    print(f"  [{label.upper()}] Step response features...")
    if STEP_TRACE_COLUMN in df.columns:
        df = extract_step_features(df, trace_column=STEP_TRACE_COLUMN, skip_filtering=True)
    else:
        print(f"    SKIPPED (no {STEP_TRACE_COLUMN})")

    # Green-Blue features
    print(f"  [{label.upper()}] Green-Blue features...")
    if GB_TRACE_COLUMN in df.columns:
        df = extract_gb_features(df, trace_column=GB_TRACE_COLUMN, skip_filtering=True)
    else:
        print(f"    SKIPPED (no {GB_TRACE_COLUMN})")

    # Frequency features
    print(f"  [{label.upper()}] Frequency features...")
    if FREQ_TRACE_COLUMN in df.columns:
        df = extract_freq_step_features(df, trace_column=FREQ_TRACE_COLUMN)
        df = extract_freq_sectioned_traces(df, trace_column=FREQ_TRACE_COLUMN)
    else:
        print(f"    SKIPPED (no {FREQ_TRACE_COLUMN})")

    return df


# =============================================================================
# STEP 4: MERGE WITH PREFIXES
# =============================================================================

def merge_with_prefixes(
    pair_index: pd.DataFrame,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge before and after DataFrames using the pair index.

    Renames columns with before_/after_ prefixes and joins on
    the pair mapping.
    """
    # Build lookup keys for before and after
    # before_df index: {dataset_id}_unit_{unit_id}
    # pair_index has: before_dataset_id, before_unit_id

    result_rows = []
    for pair_key, pair_row in pair_index.iterrows():
        before_lookup = f"{pair_row['before_dataset_id']}_{pair_row['before_unit_id']}"
        after_lookup = f"{pair_row['after_dataset_id']}_{pair_row['after_unit_id']}"

        row = {
            "pair_key": pair_key,
            "pair_id": pair_row["pair_id"],
            "chip": pair_row["chip"],
            "genotype": pair_row["genotype"],
            "cell_type": pair_row["cell_type"],
            "before_dataset_id": pair_row["before_dataset_id"],
            "after_dataset_id": pair_row["after_dataset_id"],
            "before_unit_id": pair_row["before_unit_id"],
            "after_unit_id": pair_row["after_unit_id"],
        }

        # Add before data
        if before_lookup in before_df.index:
            for col in before_df.columns:
                row[f"before_{col}"] = before_df.loc[before_lookup, col]
        else:
            for col in before_df.columns:
                row[f"before_{col}"] = None

        # Add after data
        if after_lookup in after_df.index:
            for col in after_df.columns:
                row[f"after_{col}"] = after_df.loc[after_lookup, col]
        else:
            for col in after_df.columns:
                row[f"after_{col}"] = None

        result_rows.append(row)

    result = pd.DataFrame(result_rows)
    result = result.set_index("pair_key")
    return result


# =============================================================================
# STEP 5: INTENSITY-SPECIFIC GREEN-BLUE FEATURES
# (previously in pipeline_compare_update.py)
# =============================================================================

GB_SOURCE_COLUMN = "green_blue_3s_3i_3x_64_128_255"

INTENSITY_LEVELS = {
    "low":  (0, 3),   # trials 0-2, intensity 64
    "mid":  (3, 6),   # trials 3-5, intensity 128
    "high": (6, 9),   # trials 6-8, intensity 255
}

GB_KEEP_FEATURES = [
    "green_on_peak_extreme",
    "blue_on_peak_extreme",
    "green_off_peak_extreme",
    "blue_off_peak_extreme",
    "gb_base_mean",
    "gb_base_std",
    "green_blue_on_ratio",
    "green_blue_off_ratio",
]


def add_intensity_gb_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Add intensity-specific GB features for one condition prefix.

    For each intensity level (low/mid/high), computes GB features from
    the corresponding trial subset of the 9-trial green-blue stimulus.
    Also stores high-intensity traces (trials 6-8) as a separate column.
    """
    src_col = f"{prefix}{GB_SOURCE_COLUMN}"
    trace_col = f"{prefix}green_blue_3s_3i_3x"

    if src_col not in df.columns:
        print(f"  SKIPPED: column {src_col} not found")
        return df

    df[trace_col] = None
    for level in INTENSITY_LEVELS:
        for feat in GB_KEEP_FEATURES:
            df[f"{prefix}{feat}_{level}"] = np.nan

    valid_count = 0
    for idx in tqdm(df.index, desc=f"  {prefix}GB intensity features"):
        trials_data = df.at[idx, src_col]

        if trials_data is None:
            continue
        if isinstance(trials_data, float) and np.isnan(trials_data):
            continue

        try:
            trials_list = list(trials_data)
        except (TypeError, ValueError):
            continue

        if len(trials_list) < 9:
            continue

        high_trials = [np.array(t) for t in trials_list[6:9]]
        df.at[idx, trace_col] = high_trials

        for level, (start, end) in INTENSITY_LEVELS.items():
            subset = trials_list[start:end]
            mean_trace = compute_mean_trace(subset)
            if mean_trace is None:
                continue

            try:
                features = extract_gb_features_from_trace(mean_trace)
                for feat in GB_KEEP_FEATURES:
                    df.at[idx, f"{prefix}{feat}_{level}"] = features[feat]
            except Exception:
                continue

        valid_count += 1

    print(f"  Processed {valid_count} / {len(df)} units")
    return df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Blocker Before/After Comparison DataFrame Pipeline",
    )
    parser.add_argument("--start", "-s", type=int, default=0, help="Start pair index")
    parser.add_argument("--end", "-e", type=int, default=None, help="End pair index (exclusive)")
    parser.add_argument("--start-step", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
                        help="Start from step (0=pair index, 1=firing rate, 2=features, 3=extract, 4=merge, 5=intensity GB)")
    parser.add_argument("--no-features", action="store_true",
                        help="Skip feature extraction (steps 2-3)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output filenames")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Blocker Before/After Comparison DataFrame Pipeline")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix
    pair_index_path = OUTPUT_DIR / f"pair_index{suffix}.parquet"
    before_raw_path = OUTPUT_DIR / f"before_firing_rate{suffix}.parquet"
    after_raw_path = OUTPUT_DIR / f"after_firing_rate{suffix}.parquet"
    before_movies_path = OUTPUT_DIR / f"before_movies{suffix}.parquet"
    after_movies_path = OUTPUT_DIR / f"after_movies{suffix}.parquet"
    before_features_path = OUTPUT_DIR / f"before_features{suffix}.parquet"
    after_features_path = OUTPUT_DIR / f"after_features{suffix}.parquet"
    output_path = OUTPUT_DIR / f"compared_dataframe{suffix}.parquet"

    # =========================================================================
    # STEP 0: Build pair index
    # =========================================================================
    if args.start_step <= 0:
        print("\n" + "=" * 80)
        print("STEP 0: Build Pair Index")
        print("=" * 80)

        pair_index = build_pair_index(start=args.start, end=args.end)
        pair_index.to_parquet(pair_index_path)
        print(f"Saved: {pair_index_path}")
    else:
        print("\n  Loading pair index from file...")
        pair_index = pd.read_parquet(pair_index_path)
        print(f"  Loaded {len(pair_index)} pairs")

    # =========================================================================
    # STEP 1: Compute firing rates
    # =========================================================================
    if args.start_step <= 1:
        print("\n" + "=" * 80)
        print("STEP 1: Compute Firing Rates")
        print("=" * 80)

        before_movies, after_movies = compute_firing_rates(pair_index)

        # Save intermediate
        before_save = before_movies.copy()
        after_save = after_movies.copy()
        for col in before_save.columns:
            before_save[col] = before_save[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
        for col in after_save.columns:
            after_save[col] = after_save[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
        before_save.to_parquet(before_movies_path)
        after_save.to_parquet(after_movies_path)
        print(f"\n  Saved: {before_movies_path}")
        print(f"  Saved: {after_movies_path}")
    else:
        print("\n  Loading firing rates from files...")
        before_movies = pd.read_parquet(before_movies_path)
        after_movies = pd.read_parquet(after_movies_path)
        # Convert lists back to arrays
        for col in before_movies.columns:
            before_movies[col] = before_movies[col].apply(
                lambda x: np.array(x) if isinstance(x, list) else x
            )
        for col in after_movies.columns:
            after_movies[col] = after_movies[col].apply(
                lambda x: np.array(x) if isinstance(x, list) else x
            )
        print(f"  BEFORE: {before_movies.shape}, AFTER: {after_movies.shape}")

    if args.no_features:
        print("\n  Skipping feature loading/extraction (--no-features)")
        before_final = before_movies
        after_final = after_movies
    else:
        # =====================================================================
        # STEP 2: Load HDF5 features
        # =====================================================================
        if args.start_step <= 2:
            print("\n" + "=" * 80)
            print("STEP 2: Load HDF5 Features")
            print("=" * 80)

            print("\n  Loading features for BEFORE units...")
            before_movies = load_hdf5_features(before_movies)
            print(f"  BEFORE columns: {len(before_movies.columns)}")

            print("\n  Loading features for AFTER units...")
            after_movies = load_hdf5_features(after_movies)
            print(f"  AFTER columns: {len(after_movies.columns)}")

            # Save intermediate
            before_feat_save = before_movies.copy()
            after_feat_save = after_movies.copy()
            for col in before_feat_save.columns:
                before_feat_save[col] = before_feat_save[col].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
            for col in after_feat_save.columns:
                after_feat_save[col] = after_feat_save[col].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
            before_feat_save.to_parquet(before_features_path)
            after_feat_save.to_parquet(after_features_path)
            print(f"  Saved: {before_features_path}")
            print(f"  Saved: {after_features_path}")
        else:
            print("\n  Loading features from files...")
            before_movies = pd.read_parquet(before_features_path)
            after_movies = pd.read_parquet(after_features_path)
            for col in before_movies.columns:
                before_movies[col] = before_movies[col].apply(
                    lambda x: np.array(x) if isinstance(x, list) else x
                )
            for col in after_movies.columns:
                after_movies[col] = after_movies[col].apply(
                    lambda x: np.array(x) if isinstance(x, list) else x
                )

        # =====================================================================
        # STEP 3: Extract derived features
        # =====================================================================
        if args.start_step <= 3:
            print("\n" + "=" * 80)
            print("STEP 3: Extract Derived Features")
            print("=" * 80)

            before_movies = extract_all_features(before_movies, label="before", has_iprgc=False)
            after_movies = extract_all_features(after_movies, label="after", has_iprgc=True)

        before_final = before_movies
        after_final = after_movies

    # =========================================================================
    # STEP 4: Merge with prefixes
    # =========================================================================
    if args.start_step <= 4:
        print("\n" + "=" * 80)
        print("STEP 4: Merge with Prefixes")
        print("=" * 80)

        result = merge_with_prefixes(pair_index, before_final, after_final)
        print(f"\n  Merged DataFrame: {result.shape}")

        before_cols = [c for c in result.columns if c.startswith("before_")]
        after_cols = [c for c in result.columns if c.startswith("after_")]
        meta_cols = [c for c in result.columns if not c.startswith("before_") and not c.startswith("after_")]
        print(f"  Before columns: {len(before_cols)}")
        print(f"  After columns:  {len(after_cols)}")
        print(f"  Metadata columns: {len(meta_cols)}")

        result_save = result.copy()
        for col in result_save.columns:
            result_save[col] = result_save[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
        result_save.to_parquet(output_path)
        print(f"\n  Saved: {output_path}")
    else:
        print("\n  Loading merged dataframe from file...")
        result = pd.read_parquet(output_path)
        for col in result.columns:
            result[col] = result[col].apply(
                lambda x: np.array(x) if isinstance(x, list) else x
            )
        print(f"  Loaded: {result.shape}")

    # =========================================================================
    # STEP 5: Intensity-specific Green-Blue features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Intensity-Specific GB Features")
    print("=" * 80)

    original_cols = set(result.columns)

    print("\nProcessing BEFORE columns...")
    result = add_intensity_gb_features(result, "before_")

    print("\nProcessing AFTER columns...")
    result = add_intensity_gb_features(result, "after_")

    new_cols = sorted(set(result.columns) - original_cols)
    print(f"\nNew columns added: {len(new_cols)}")
    for c in new_cols:
        print(f"  {c}")

    output_v2_path = OUTPUT_DIR / f"compared_dataframe_v2{suffix}.parquet"
    result_save = result.copy()
    for col in result_save.columns:
        result_save[col] = result_save[col].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
    result_save.to_parquet(output_v2_path)
    print(f"\n  Saved: {output_v2_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Total paired units: {len(result)}")
    print(f"  Recording pairs: {result['pair_id'].nunique()}")
    print(f"  Unique chips: {result['chip'].nunique()}")
    print(f"  Cell types: {dict(result['cell_type'].value_counts())}")
    print(f"\n  Outputs:")
    print(f"    {output_path}")
    print(f"    {output_v2_path}")
    print(f"  Shape: {result.shape}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)

    return result


if __name__ == "__main__":
    main()
