"""
Run Analysis for Low Glucose Experiment

Loads CMCR/CMTR data via step_change_analysis.data_loader, extracts per-trial
response features for every unit in each recording, and generates
visualizations showing how response amplitudes change during a 3-phase
glucose protocol: normal -> high (25mM) -> low (2mM).

No cross-recording unit alignment is performed.

Usage (from the low_glucose directory):
    python run_analysis.py
    python run_analysis.py --overwrite
    python run_analysis.py --combined-only
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# sys.path -- make sibling and parent packages importable
# ---------------------------------------------------------------------------
import importlib.util

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

# Load parent run_analysis via importlib to avoid circular import
_parent_ra_path = _GLUCOSE_JHU_DIR / "high-glucose-alone" / "run_analysis.py"
_spec_ra = importlib.util.spec_from_file_location("parent_run_analysis", _parent_ra_path)
_parent_ra = importlib.util.module_from_spec(_spec_ra)
_spec_ra.loader.exec_module(_parent_ra)
median_mean_smooth = _parent_ra.median_mean_smooth
FEATURE_CONFIGS = _parent_ra.FEATURE_CONFIGS
extract_feature_per_trial = _parent_ra.extract_feature_per_trial
classify_unit_on_off = _parent_ra.classify_unit_on_off
extract_recording_features = _parent_ra.extract_recording_features

# Load local specific_config via importlib to avoid shadowing by parent
_local_cfg_path = _THIS_DIR / "specific_config.py"
_spec_cfg = importlib.util.spec_from_file_location("low_glucose_config", _local_cfg_path)
_local_cfg = importlib.util.module_from_spec(_spec_cfg)
_spec_cfg.loader.exec_module(_local_cfg)
LowGlucosePipelineConfig = _local_cfg.LowGlucosePipelineConfig
LowGlucoseTimingInfo = _local_cfg.LowGlucoseTimingInfo
default_config = _local_cfg.default_config
load_recording_info = _local_cfg.load_recording_info
get_output_hdf5_path = _local_cfg.get_output_hdf5_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_all_recordings(
    config: Optional[LowGlucosePipelineConfig] = None,
    overwrite: bool = False,
) -> List[Tuple[Dict[str, Any], LowGlucoseTimingInfo]]:
    """Load every recording listed in the Google Sheet CSV, save to HDF5."""
    if config is None:
        config = default_config

    recording_infos = load_recording_info(
        config.gsheet_csv_path, config.data_folders,
    )

    results: List[Tuple[Dict[str, Any], LowGlucoseTimingInfo]] = []
    for info in recording_infos:
        cmcr_path = info.data_folder / info.cmcr
        cmtr_path = info.data_folder / info.cmtr
        h5_path = get_output_hdf5_path(info.cmcr, config.output_dir)

        if h5_path.exists() and not overwrite:
            logger.info("Loading existing HDF5: %s", h5_path.name)
            data = load_recording_from_hdf5(h5_path)
        else:
            logger.info("Processing CMCR/CMTR: %s", info.cmcr)
            raw_data = load_cmcr_cmtr_data(cmcr_path, cmtr_path)
            save_recording_to_hdf5(
                raw_data,
                h5_path,
                step_config=config.step_detection,
                quality_config=config.quality,
                overwrite=overwrite,
            )
            data = load_recording_from_hdf5(h5_path)

        n_units = len(data.get("units", {}))
        logger.info("  %s  ->  %d units", info.description, n_units)
        results.append((data, info))

    return results


# =============================================================================
# Visualization Helpers -- 3-phase glucose shading
# =============================================================================

def _add_glucose_shading_3phase(
    ax: plt.Axes,
    info: LowGlucoseTimingInfo,
):
    """Add shaded regions for the 3-phase glucose protocol."""
    ax.axvspan(
        info.high_glucose_min, info.low_glucose_min,
        color="red", alpha=0.08, label="High glucose (25mM)",
    )
    ylim = ax.get_ylim()
    ax.axvspan(
        info.low_glucose_min, ax.get_xlim()[1],
        color="blue", alpha=0.06, label="Low glucose (2mM)",
    )
    ax.set_ylim(ylim)
    ax.axvline(info.high_glucose_min, color="red", ls="--", lw=1, alpha=0.6)
    ax.axvline(info.low_glucose_min, color="blue", ls="--", lw=1, alpha=0.6)


def plot_recording_feature(
    features: np.ndarray,
    info: LowGlucoseTimingInfo,
    feature_name: str,
    save_dir: Optional[Path] = None,
    smooth_window: int = 9,
    normalize: bool = True,
    trial_interval_s: float = 10.0,
) -> plt.Figure:
    """Two-panel plot: individual traces | mean +/- SEM."""
    if features.size == 0:
        return plt.figure()

    fs = features.copy()
    if fs.shape[0] > 1 and smooth_window > 1:
        fs = median_mean_smooth(fs, window=smooth_window)

    if normalize:
        row_max = fs.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        fs = fs / row_max

    n_units, n_trials = fs.shape
    x_min = np.arange(n_trials) * trial_interval_s / 60.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    for trace in fs:
        ax.plot(x_min, trace, alpha=0.3, lw=0.8)
    _add_glucose_shading_3phase(ax, info)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Normalized" if normalize else "Amplitude")
    ax.set_title(f"{info.description} - {feature_name}\n"
                 f"(n={n_units} units, individual)")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    mean_trace = fs.mean(axis=0)
    sem = fs.std(axis=0) / np.sqrt(n_units) if n_units > 1 else np.zeros(n_trials)
    ax.plot(x_min, mean_trace, color="black", lw=2)
    ax.fill_between(
        x_min, mean_trace - sem, mean_trace + sem,
        color="gray", alpha=0.5,
    )
    _add_glucose_shading_3phase(ax, info)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Normalized" if normalize else "Amplitude")
    ax.set_title(f"{info.description} - {feature_name}\n"
                 f"(n={n_units} units, mean +/- SEM)")
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe = info.description.replace(" ", "_")
        path = save_dir / f"{safe}_{feature_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", path)
        plt.close(fig)

    return fig


def plot_cross_recording_summary(
    all_recording_features: List[Tuple[np.ndarray, LowGlucoseTimingInfo]],
    feature_name: str,
    save_dir: Optional[Path] = None,
    smooth_window: int = 9,
    trial_interval_s: float = 10.0,
) -> plt.Figure:
    """Overlay mean traces from every recording on one axes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for i, (features, info) in enumerate(all_recording_features):
        if features.size == 0:
            continue

        fs = features.copy()
        if fs.shape[0] > 1 and smooth_window > 1:
            fs = median_mean_smooth(fs, window=smooth_window)

        row_max = fs.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        fs = fs / row_max

        n_units, n_trials = fs.shape
        x_min = np.arange(n_trials) * trial_interval_s / 60.0

        mean_trace = fs.mean(axis=0)
        sem = fs.std(axis=0) / np.sqrt(n_units) if n_units > 1 else np.zeros(n_trials)
        color = cmap(i % 10)

        ax.plot(x_min, mean_trace, color=color, lw=2, label=info.description)
        ax.fill_between(x_min, mean_trace - sem, mean_trace + sem,
                        color=color, alpha=0.15)

        ax.axvline(info.high_glucose_min, color=color, ls=":", alpha=0.4)
        ax.axvline(info.low_glucose_min, color=color, ls="--", alpha=0.4)

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Normalized Response", fontsize=12)
    ax.set_title(f"Cross-Recording Summary  -  {feature_name}")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"cross_recording_{feature_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", path)
        plt.close(fig)

    return fig


def plot_combined_transitions(
    all_recording_features: List[Tuple[np.ndarray, LowGlucoseTimingInfo]],
    feature_name: str,
    save_dir: Optional[Path] = None,
    smooth_window: int = 9,
    trial_interval_s: float = 10.0,
    cell_type_label: str = "all",
) -> plt.Figure:
    """
    Pool units from all recordings and plot two aligned transitions:
      Left:  Normal -> High glucose  (trial 0 to low_glucose_min)
      Right: High -> Low glucose     (low_glucose_min - 5 min to end)
    Data are trimmed to the shortest common length across recordings.
    """
    left_segments: List[np.ndarray] = []
    right_segments: List[np.ndarray] = []
    high_glc_min = all_recording_features[0][1].high_glucose_min

    for features, info in all_recording_features:
        if features.size == 0:
            continue

        fs = features.copy()
        row_max = fs.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        fs = fs / row_max

        n_trials = fs.shape[1]

        end_trial = min(int(info.low_glucose_min * 60 / trial_interval_s), n_trials)
        left_segments.append(fs[:, :end_trial])

        start_trial = max(0, int((info.low_glucose_min - 5.0) * 60 / trial_interval_s))
        right_segments.append(fs[:, start_trial:])

    if not left_segments or not right_segments:
        return plt.figure()

    left_min = min(s.shape[1] for s in left_segments)
    left_all = np.vstack([s[:, :left_min] for s in left_segments])

    right_min = min(s.shape[1] for s in right_segments)
    right_all = np.vstack([s[:, :right_min] for s in right_segments])

    n_left = left_all.shape[0]
    n_right = right_all.shape[0]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Left: Normal -> High (then low) ----
    x_l = np.arange(left_min) * trial_interval_s / 60.0
    mean_l = left_all.mean(axis=0)
    sem_l = left_all.std(axis=0) / np.sqrt(n_left)

    ax_l.errorbar(x_l, mean_l, yerr=sem_l, fmt="o", color="black",
                  markersize=3, capsize=2, ecolor="gray", elinewidth=0.8,
                  label="Mean +/- SEM")
    smooth_l = median_mean_smooth(mean_l[np.newaxis, :], window=smooth_window)[0]
    ax_l.plot(x_l, smooth_l, color="blue", lw=2, alpha=0.7, label="Trend")
    ax_l.axvline(high_glc_min, color="red", ls="--", lw=1.5,
                 label=f"High glucose ({high_glc_min} min)")

    type_str = f" [{cell_type_label.upper()} cells]" if cell_type_label != "all" else ""
    ax_l.set_xlabel("Time (min)", fontsize=12)
    ax_l.set_ylabel("Normalized Response", fontsize=12)
    ax_l.set_title(f"Normal -> High glucose{type_str}\n"
                   f"{feature_name}  (n={n_left} units from "
                   f"{len(left_segments)} recordings)")
    ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.3)

    # ---- Right: High -> Low (x-axis relative to low-glucose onset) ----
    x_r = np.arange(right_min) * trial_interval_s / 60.0 - 5.0
    mean_r = right_all.mean(axis=0)
    sem_r = right_all.std(axis=0) / np.sqrt(n_right)

    ax_r.errorbar(x_r, mean_r, yerr=sem_r, fmt="o", color="black",
                  markersize=3, capsize=2, ecolor="gray", elinewidth=0.8,
                  label="Mean +/- SEM")
    smooth_r = median_mean_smooth(mean_r[np.newaxis, :], window=smooth_window)[0]
    ax_r.plot(x_r, smooth_r, color="blue", lw=2, alpha=0.7, label="Trend")
    ax_r.axvline(0, color="blue", ls="--", lw=1.5,
                 label="Low glucose (2mM)")
    ax_r.set_xlabel("Time relative to low glucose onset (min)", fontsize=12)
    ax_r.set_ylabel("Normalized Response", fontsize=12)
    ax_r.set_title(f"High -> Low glucose{type_str}\n"
                   f"{feature_name}  (n={n_right} units from "
                   f"{len(right_segments)} recordings)")
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ct_suffix = f"_{cell_type_label}" if cell_type_label != "all" else ""
        path = save_dir / f"combined_transitions_{feature_name}{ct_suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", path)
        plt.close(fig)

    return fig


def plot_mean_step_response(
    data: Dict[str, Any],
    info: LowGlucoseTimingInfo,
    save_dir: Optional[Path] = None,
    quality_threshold: float = 0.01,
) -> plt.Figure:
    """Plot mean step-response waveform averaged over all HQ units."""
    units = data.get("units", {})
    all_means: List[np.ndarray] = []

    for unit_data in units.values():
        qi = unit_data.get("quality_index", 0)
        if qi < quality_threshold:
            continue
        if "step_responses" not in unit_data:
            continue
        resp = np.array(unit_data["step_responses"])
        if resp.size > 0:
            all_means.append(resp.mean(axis=0))

    fig, ax = plt.subplots(figsize=(8, 4))
    if all_means:
        all_means_arr = np.array(all_means)
        grand_mean = all_means_arr.mean(axis=0)
        grand_std = all_means_arr.std(axis=0)
        x = np.arange(len(grand_mean)) / 10.0
        ax.plot(x, grand_mean, color="blue", lw=2)
        ax.fill_between(x, grand_mean - grand_std, grand_mean + grand_std,
                        color="blue", alpha=0.2)
        ax.set_title(f"{info.description}  -  Mean step response "
                     f"(n={len(all_means)} units)")
    else:
        ax.set_title(f"{info.description}  -  No high-quality units")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (Hz)")
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe = info.description.replace(" ", "_")
        path = save_dir / f"{safe}_mean_step_response.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", path)
        plt.close(fig)

    return fig


# =============================================================================
# Main Pipeline
# =============================================================================

def run_analysis(
    config: Optional[LowGlucosePipelineConfig] = None,
    overwrite: bool = False,
    feature_names: Optional[List[str]] = None,
    combined_only: bool = False,
):
    """Run the full low-glucose analysis pipeline."""
    if config is None:
        config = default_config

    if feature_names is None:
        feature_names = ["on_peak", "off_peak", "on_sustained", "max_min_range"]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load recordings ----
    logger.info("=" * 60)
    logger.info("Step 1: Loading recordings")
    logger.info("=" * 60)
    recording_results = load_all_recordings(config, overwrite=overwrite)
    logger.info("Loaded %d recordings", len(recording_results))

    if not combined_only:
        # ---- Step 2: per-recording mean step-response plot ----
        logger.info("=" * 60)
        logger.info("Step 2: Mean step-response waveforms")
        logger.info("=" * 60)
        for data, info in recording_results:
            plot_mean_step_response(
                data, info,
                save_dir=config.figures_dir,
                quality_threshold=config.quality.quality_threshold,
            )

    # ---- Step 3: per-recording feature time-courses ----
    logger.info("=" * 60)
    logger.info("Step 3: Per-recording feature time-courses")
    logger.info("=" * 60)

    quality_threshold = config.quality.quality_threshold
    trial_interval_s = config.response_analysis.trial_interval_s

    for feature_name in feature_names:
        logger.info("--- Feature: %s ---", feature_name)

        all_for_summary: List[Tuple[np.ndarray, LowGlucoseTimingInfo]] = []

        for data, info in recording_results:
            features = extract_recording_features(
                data, feature_name, quality_threshold,
            )
            if features.size == 0:
                logger.warning("  %s: no qualifying units", info.description)
                continue

            logger.info("  %s: %d units, %d trials",
                        info.description, features.shape[0], features.shape[1])

            if not combined_only:
                plot_recording_feature(
                    features, info, feature_name,
                    save_dir=config.figures_dir,
                    trial_interval_s=trial_interval_s,
                )
            all_for_summary.append((features, info))

        if not combined_only and len(all_for_summary) > 1:
            plot_cross_recording_summary(
                all_for_summary, feature_name,
                save_dir=config.figures_dir,
                trial_interval_s=trial_interval_s,
            )

        # ---- Combined transition plots (pool all recordings) ----
        if len(all_for_summary) > 1:
            plot_combined_transitions(
                all_for_summary, feature_name,
                save_dir=config.figures_dir,
                trial_interval_s=trial_interval_s,
            )

        # ---- ON / OFF cell combined plots ----
        for ct in ("on", "off"):
            ct_summary: List[Tuple[np.ndarray, LowGlucoseTimingInfo]] = []
            for data, info in recording_results:
                feats = extract_recording_features(
                    data, feature_name, quality_threshold, cell_type=ct,
                )
                if feats.size > 0:
                    ct_summary.append((feats, info))
            if len(ct_summary) > 1:
                n_units_total = sum(f.shape[0] for f, _ in ct_summary)
                logger.info("  %s cells (%s): %d units across %d recordings",
                            ct.upper(), feature_name, n_units_total, len(ct_summary))
                plot_combined_transitions(
                    ct_summary, feature_name,
                    save_dir=config.figures_dir,
                    trial_interval_s=trial_interval_s,
                    cell_type_label=ct,
                )

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("Figures saved in: %s", config.figures_dir)
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Low-glucose step-response analysis",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing HDF5 files",
    )
    parser.add_argument(
        "--features", nargs="+",
        default=None,
        help="Feature names to extract (default: on_peak off_peak on_sustained max_min_range)",
    )
    parser.add_argument(
        "--combined-only", action="store_true",
        help="Only generate the combined transition plots (skip per-recording figures)",
    )
    args = parser.parse_args()
    run_analysis(
        overwrite=args.overwrite,
        feature_names=args.features,
        combined_only=args.combined_only,
    )
