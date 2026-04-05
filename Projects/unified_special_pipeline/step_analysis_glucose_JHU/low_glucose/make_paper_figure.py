"""
Generate paper-quality Figure X: Hypoglycemia accelerates decay of
light-evoked responses in RGCs.

Panels:
  A - Representative OFF-RGC spike-rate traces evolving over time
      (subplots at regular trial intervals, gray=darkness, white=light)
  B - Population OFF-RGC time course (normalized response over recording,
      errorbar dots with Welch t-test significance markers)
  C - Representative ON-RGC spike-rate traces (same layout as A)
  D - Population ON-RGC time course (same layout as B)

Light onset at 1 s, offset at 4 s.  Gray shading = darkness, white = light.
Protocol: Normal glucose -> High glucose (2.5 min) -> Low glucose (15 min).

Usage:
  python make_paper_figure.py
  python make_paper_figure.py --separate    # individual panels only
  python make_paper_figure.py --combined    # combined 4-panel only
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

_THIS_DIR = Path(__file__).resolve().parent
_GLUCOSE_JHU_DIR = _THIS_DIR.parent
_USP_DIR = _GLUCOSE_JHU_DIR.parent

for _p in (str(_GLUCOSE_JHU_DIR), str(_USP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step_change_analysis.data_loader import load_recording_from_hdf5

_HG_ALONE_DIR = _GLUCOSE_JHU_DIR / "high-glucose-alone"
_parent_ra_path = _HG_ALONE_DIR / "run_analysis.py"
_spec_ra = importlib.util.spec_from_file_location("parent_run_analysis", _parent_ra_path)
_parent_ra = importlib.util.module_from_spec(_spec_ra)
_spec_ra.loader.exec_module(_parent_ra)
classify_unit_on_off = _parent_ra.classify_unit_on_off
extract_recording_features = _parent_ra.extract_recording_features
median_mean_smooth = _parent_ra.median_mean_smooth

_parent_cfg_path = _HG_ALONE_DIR / "specific_config.py"
_spec_pcfg = importlib.util.spec_from_file_location("parent_specific_config", _parent_cfg_path)
_parent_cfg = importlib.util.module_from_spec(_spec_pcfg)
_spec_pcfg.loader.exec_module(_parent_cfg)
HG_GlucosePipelineConfig = _parent_cfg.GlucosePipelineConfig
HG_default_config = _parent_cfg.default_config
hg_load_recording_info = _parent_cfg.load_recording_info
hg_get_output_hdf5_path = _parent_cfg.get_output_hdf5_path

_local_cfg_path = _THIS_DIR / "specific_config.py"
_spec_cfg = importlib.util.spec_from_file_location("low_glucose_config", _local_cfg_path)
_local_cfg = importlib.util.module_from_spec(_spec_cfg)
_spec_cfg.loader.exec_module(_local_cfg)
LowGlucosePipelineConfig = _local_cfg.LowGlucosePipelineConfig
default_config = _local_cfg.default_config
load_recording_info = _local_cfg.load_recording_info
get_output_hdf5_path = _local_cfg.get_output_hdf5_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Protocol timing ----
TRIAL_INTERVAL_S = 10.0
BINS_PER_TRIAL = 60
BIN_RATE_HZ = 10.0
LIGHT_ON_S = 1.0
LIGHT_OFF_S = 4.0
TRIAL_DURATION_S = BINS_PER_TRIAL / BIN_RATE_HZ

HIGH_GLUCOSE_MIN = 2.5
LOW_GLUCOSE_MIN = 15.0
HIGH_GLUCOSE_S = HIGH_GLUCOSE_MIN * 60
LOW_GLUCOSE_S = LOW_GLUCOSE_MIN * 60

CONTROL_TRIAL_END = int(HIGH_GLUCOSE_S / TRIAL_INTERVAL_S)
LOW_GLUCOSE_TRIAL_START = int(LOW_GLUCOSE_S / TRIAL_INTERVAL_S)

REPR_INTERVAL = 10  # average this many trials per subplot

# ---- Majority timing for high-glucose-alone alignment ----
MAJORITY_HG_OFF_MIN = 15.0
MAJORITY_HG_OFF_S = MAJORITY_HG_OFF_MIN * 60.0
MAJORITY_HG_OFF_TRIAL = int(MAJORITY_HG_OFF_S / TRIAL_INTERVAL_S)
HG_ON_TRIAL = CONTROL_TRIAL_END  # trial 15

# ---- External control data from low_glucose_alone ----
_ALONE_DIR = _THIS_DIR.parent / "low_glucose_alone"


def load_control_features() -> Tuple[np.ndarray, np.ndarray]:
    """Load control feature arrays from the low_glucose_alone pipeline.

    Returns (off_control, on_control) -- each (n_units, n_trials) already
    smoothed and per-unit-max normalized.  n_trials is 2 * repeat_num_clip
    (typically 166) because both recordings per aligned pair are concatenated.
    """
    npz_path = _ALONE_DIR / "data" / "extracted_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Control features not found: {npz_path}\n"
            "Run low_glucose_alone/make_paper_figure.py first."
        )
    data = np.load(npz_path)
    return data["off_control"], data["on_control"]


def load_low_glucose_alone_features() -> Tuple[np.ndarray, np.ndarray]:
    """Load low-glucose condition features from the low_glucose_alone pipeline.

    Returns (off_low, on_low) -- each (n_units, n_trials).
    """
    npz_path = _ALONE_DIR / "data" / "extracted_features.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Low-glucose-alone features not found: {npz_path}\n"
            "Run low_glucose_alone/make_paper_figure.py first."
        )
    data = np.load(npz_path)
    return data["off_low"], data["on_low"]


def load_high_glucose_only_features(
    cell_type: str,
    smooth_window: int = 9,
) -> Tuple[np.ndarray, List[float]]:
    """Load high-glucose-only recordings and extract max_min_range features.

    Uses the parent pipeline's HDF5 files (output/*.h5) and timing info.
    Each recording went through Normal -> High -> Normal glucose.

    Args:
        cell_type: "on" or "off"
        smooth_window: smoothing window for median_mean_smooth

    Returns:
        (features, align_points_s) where:
          features: (n_units, n_trials) smoothed, per-unit-max-normalized
          align_points_s: list of normal_glucose_min*60 (one per recording,
              repeated for each unit from that recording)
    """
    hg_config = HG_default_config
    hg_infos = hg_load_recording_info(hg_config.xlsx_path, hg_config.data_folder)

    all_features = []
    all_align_s = []

    for info in hg_infos:
        h5_path = hg_get_output_hdf5_path(info.cmcr, hg_config.output_dir)
        if not h5_path.exists():
            logger.warning("HG HDF5 not found: %s", h5_path)
            continue
        data = load_recording_from_hdf5(h5_path)
        feats = extract_recording_features(
            data, "max_min_range",
            quality_threshold=hg_config.quality.quality_threshold,
            cell_type=cell_type,
        )
        if feats.size == 0:
            continue

        if feats.shape[0] > 1 and smooth_window > 1:
            feats = median_mean_smooth(feats, window=smooth_window)

        row_max = feats.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        feats = feats / row_max

        n_units = feats.shape[0]
        align_s = info.normal_glucose_min * 60.0
        all_features.append(feats)
        all_align_s.extend([align_s] * n_units)

    if not all_features:
        return np.empty((0, 0)), []

    min_trials = min(f.shape[1] for f in all_features)
    all_features = [f[:, :min_trials] for f in all_features]
    combined = np.vstack(all_features)
    return combined, all_align_s


def _align_hg_features_to_majority(
    features: np.ndarray,
    hg_off_trial: int,
    majority_hg_off_trial: int = MAJORITY_HG_OFF_TRIAL,
    hg_on_trial: int = HG_ON_TRIAL,
) -> np.ndarray:
    """Align one recording's feature array so HG OFF matches majority timing.

    For recordings with shorter HG phase: break at midpoint, insert NaN.
    For recordings with longer HG phase: break at midpoint, remove excess.
    """
    n_hg_actual = hg_off_trial - hg_on_trial
    n_hg_majority = majority_hg_off_trial - hg_on_trial
    diff = n_hg_majority - n_hg_actual

    pre_hg = features[:, :hg_on_trial]
    hg_seg = features[:, hg_on_trial:hg_off_trial]
    post_hg = features[:, hg_off_trial:]

    if diff > 0:
        mid = n_hg_actual // 2
        hg_new = np.hstack([
            hg_seg[:, :mid],
            np.full((features.shape[0], diff), np.nan),
            hg_seg[:, mid:],
        ])
    elif diff < 0:
        excess = -diff
        mid = n_hg_actual // 2
        hg_new = np.hstack([
            hg_seg[:, :mid - excess // 2],
            hg_seg[:, mid + (excess - excess // 2):],
        ])
    else:
        hg_new = hg_seg

    return np.hstack([pre_hg, hg_new, post_hg])


def load_high_glucose_combined_features(
    cell_type: str,
    smooth_window: int = 9,
) -> np.ndarray:
    """Load all high-glucose-only recordings, align HG phases to majority
    timing, and return a single pooled feature array.

    Each recording's HG phase is adjusted so that HG OFF aligns at
    MAJORITY_HG_OFF_TRIAL. After alignment all traces are trimmed to a
    consistent length (the shortest aligned recording).

    Returns:
        combined: (n_units, n_trials) smoothed, per-unit-max-normalized,
            HG-phase-aligned feature array.
    """
    hg_config = HG_default_config
    hg_infos = hg_load_recording_info(hg_config.xlsx_path, hg_config.data_folder)

    aligned_features = []

    for info in hg_infos:
        hg_off_min = info.normal_glucose_min
        if hg_off_min > MAJORITY_HG_OFF_MIN:
            logger.info(
                "  Skipping %s (normal_glucose_min=%.1f > majority %.1f)",
                info.cmcr, hg_off_min, MAJORITY_HG_OFF_MIN,
            )
            continue

        h5_path = hg_get_output_hdf5_path(info.cmcr, hg_config.output_dir)
        if not h5_path.exists():
            logger.warning("HG HDF5 not found: %s", h5_path)
            continue
        data = load_recording_from_hdf5(h5_path)
        feats = extract_recording_features(
            data, "max_min_range",
            quality_threshold=hg_config.quality.quality_threshold,
            cell_type=cell_type,
        )
        if feats.size == 0:
            continue

        if feats.shape[0] > 1 and smooth_window > 1:
            feats = median_mean_smooth(feats, window=smooth_window)

        row_max = feats.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        feats = feats / row_max

        hg_off_trial = int(round(hg_off_min * 60.0 / TRIAL_INTERVAL_S))
        feats_aligned = _align_hg_features_to_majority(feats, hg_off_trial)
        aligned_features.append(feats_aligned)
        logger.info(
            "  %s: %d units, HG OFF trial %d -> aligned %s",
            h5_path.name, feats.shape[0], hg_off_trial, feats_aligned.shape,
        )

    if not aligned_features:
        return np.empty((0, 0))

    min_trials = min(f.shape[1] for f in aligned_features)
    aligned_features = [f[:, :min_trials] for f in aligned_features]
    combined = np.vstack(aligned_features)
    return combined


def load_high_glucose_majority_features(
    cell_type: str,
    smooth_window: int = 9,
) -> np.ndarray:
    """Load only the majority-timing (15.0 min) high-glucose recordings.

    No alignment or NaN filling is needed because all included recordings
    share the same protocol timing.

    Returns:
        combined: (n_units, n_trials) smoothed, per-unit-max-normalized.
    """
    hg_config = HG_default_config
    hg_infos = hg_load_recording_info(hg_config.xlsx_path, hg_config.data_folder)

    all_features = []

    for info in hg_infos:
        if info.normal_glucose_min != MAJORITY_HG_OFF_MIN:
            logger.info(
                "  Skipping %s (normal_glucose_min=%.1f != majority %.1f)",
                info.cmcr, info.normal_glucose_min, MAJORITY_HG_OFF_MIN,
            )
            continue

        h5_path = hg_get_output_hdf5_path(info.cmcr, hg_config.output_dir)
        if not h5_path.exists():
            logger.warning("HG HDF5 not found: %s", h5_path)
            continue
        data = load_recording_from_hdf5(h5_path)
        feats = extract_recording_features(
            data, "max_min_range",
            quality_threshold=hg_config.quality.quality_threshold,
            cell_type=cell_type,
        )
        if feats.size == 0:
            continue

        if feats.shape[0] > 1 and smooth_window > 1:
            feats = median_mean_smooth(feats, window=smooth_window)

        row_max = feats.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        feats = feats / row_max

        all_features.append(feats)
        logger.info(
            "  %s: %d units, %d trials",
            h5_path.name, feats.shape[0], feats.shape[1],
        )

    if not all_features:
        return np.empty((0, 0))

    min_trials = min(f.shape[1] for f in all_features)
    all_features = [f[:, :min_trials] for f in all_features]
    combined = np.vstack(all_features)
    return combined


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_all_h5(config: LowGlucosePipelineConfig):
    """Load all HDF5 files, return list of (data, info) tuples."""
    infos = load_recording_info(config.gsheet_csv_path, config.data_folders)
    results = []
    for info in infos:
        h5_path = get_output_hdf5_path(info.cmcr, config.output_dir)
        if not h5_path.exists():
            logger.warning("HDF5 not found: %s", h5_path)
            continue
        data = load_recording_from_hdf5(h5_path)
        results.append((data, info))
    return results


def collect_units(
    all_data: list,
    cell_type: str,
    quality_threshold: float = 0.01,
) -> List[np.ndarray]:
    """Collect step_responses arrays for all units of a given cell type."""
    collected = []
    for data, info in all_data:
        units = data.get("units", {})
        for uid, udata in units.items():
            qi = udata.get("quality_index", 0)
            if qi < quality_threshold:
                continue
            if "step_responses" not in udata:
                continue
            sr = np.array(udata["step_responses"])
            ct = classify_unit_on_off(sr)
            if ct == cell_type:
                collected.append(sr)
    return collected


def pick_representative_unit(units_sr: list) -> np.ndarray:
    """Pick a representative unit with clear, stable baseline response."""
    best_idx = 0
    best_score = -1.0
    for i, sr in enumerate(units_sr):
        ctrl_trials = sr[:CONTROL_TRIAL_END]
        if len(ctrl_trials) < 5:
            continue
        mean_resp = ctrl_trials.mean(axis=0)
        amplitude = mean_resp.max() - mean_resp.min()
        trial_amps = ctrl_trials.max(axis=1) - ctrl_trials.min(axis=1)
        consistency = 1.0 / (trial_amps.std() + 1e-6)
        score = amplitude * consistency
        if score > best_score:
            best_score = score
            best_idx = i
    return units_sr[best_idx]


# ---------------------------------------------------------------------------
# Panel A / C  --  representative trace subplots over time
# ---------------------------------------------------------------------------

def _add_light_shading(ax: plt.Axes, add_label: bool = False):
    """Gray = darkness (0-1 s, 4-6 s), white = light (1-4 s)."""
    kw = dict(color="gray", alpha=0.2, zorder=0)
    if add_label:
        kw["label"] = "Darkness"
    ax.axvspan(0, LIGHT_ON_S, **kw)
    if add_label:
        kw.pop("label", None)
    ax.axvspan(LIGHT_OFF_S, TRIAL_DURATION_S, **kw)


def plot_representative_traces(
    sr: np.ndarray,
    cell_label: str,
    interval: int = REPR_INTERVAL,
):
    """Panel A or C: row of subplots, each the average trace of *interval*
    consecutive trials.  Vertical dashed lines mark glucose transitions.

    Returns (fig, axes).
    """
    n_trials = sr.shape[0]
    n_subplots = n_trials // interval
    if n_subplots < 1:
        n_subplots = 1

    x_s = np.arange(BINS_PER_TRIAL) / BIN_RATE_HZ
    y_max = 0.0
    traces = []
    for k in range(n_subplots):
        avg = sr[k * interval:(k + 1) * interval].mean(axis=0)
        traces.append(avg)
        y_max = max(y_max, avg.max())
    y_lim_top = y_max * 1.15

    fig_w = max(n_subplots * 1.0, 6)
    fig, axs = plt.subplots(1, n_subplots, figsize=(fig_w, fig_w / 5 + 1.5),
                            sharey=True)
    if n_subplots == 1:
        axs = [axs]

    high_glucose_subplot = int(HIGH_GLUCOSE_S / TRIAL_INTERVAL_S) // interval
    low_glucose_subplot = int(LOW_GLUCOSE_S / TRIAL_INTERVAL_S) // interval

    for k, ax in enumerate(axs):
        _add_light_shading(ax, add_label=(k == 0))
        ax.plot(x_s, traces[k], color="black", lw=1.2)
        ax.set_ylim(0, y_lim_top)

        if k == 0:
            ax.set_ylabel("Firing rate (Hz)", fontsize=10)
        else:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        seg_start_s = k * interval * TRIAL_INTERVAL_S
        ax.text(TRIAL_DURATION_S / 2, y_lim_top * 0.95,
                f"{seg_start_s:.0f} s", ha="center", va="top",
                fontsize=8, color="black")

        if k == high_glucose_subplot:
            ax.axvline(0, color="red", ls="--", lw=1.2, zorder=5)
        if k == low_glucose_subplot:
            ax.axvline(0, color="blue", ls="--", lw=1.2, zorder=5)

    fig.suptitle(cell_label, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------------
# Panel B / D  --  population time course
# ---------------------------------------------------------------------------

BASELINE_WINDOW = 5  # last N control trials used for normalization


def compute_population_timecourse(
    units_sr: List[np.ndarray],
) -> dict:
    """Two-step normalization:
    1. Per-unit: normalize each unit's feature trace by its own max.
    2. Population: normalize the pooled array by the grand-mean of the
       last BASELINE_WINDOW control trials (just before high glucose).

    Returns dict with: x_s, mean, sem, n_units, per-trial stacked array.
    """
    min_trials = min(sr.shape[0] for sr in units_sr)
    baseline_start = max(0, CONTROL_TRIAL_END - BASELINE_WINDOW)

    # Step 1: per-unit normalization by own max
    feature_traces = []
    for sr in units_sr:
        feat = sr[:min_trials].max(axis=1) - sr[:min_trials].min(axis=1)
        feat = np.abs(feat)
        fmax = feat.max()
        if fmax > 0:
            feat = feat / fmax
        feature_traces.append(feat)

    arr = np.array(feature_traces)  # (n_units, min_trials)

    # Step 2: population normalization by grand-mean baseline
    pop_mean_curve = arr.mean(axis=0)
    baseline_val = pop_mean_curve[baseline_start:CONTROL_TRIAL_END].mean()
    if baseline_val > 0:
        arr = arr / baseline_val

    x_s = np.arange(min_trials) * TRIAL_INTERVAL_S
    mean_t = arr.mean(axis=0)
    sem_t = arr.std(axis=0) / np.sqrt(arr.shape[0])
    return {
        "x_s": x_s,
        "mean": mean_t,
        "sem": sem_t,
        "n_units": arr.shape[0],
        "arr": arr,
    }


def plot_population_timecourse(
    ax: plt.Axes,
    units_sr: List[np.ndarray],
    cell_label: str,
    step: int = 2,
    ctrl_features: Optional[np.ndarray] = None,
):
    """Panel B or D: population time course (normalized response) plotted
    over recording time in seconds.

    When *ctrl_features* is provided (from low_glucose_alone), two curves
    are overlaid:
      - Black: external control (grand-mean-max normalized)
      - Red: treatment (this pipeline's two-step normalization)
    Otherwise falls back to the single-curve layout.
    """
    tc = compute_population_timecourse(units_sr)
    x_s = tc["x_s"]
    mean_t = tc["mean"]
    sem_t = tc["sem"]
    arr = tc["arr"]
    n_units = tc["n_units"]

    # glucose phase shading
    ax.axvspan(HIGH_GLUCOSE_S, LOW_GLUCOSE_S,
               color="red", alpha=0.06, zorder=0)
    ax.axvspan(LOW_GLUCOSE_S, x_s[-1],
               color="blue", alpha=0.06, zorder=0)
    ax.axvline(HIGH_GLUCOSE_S, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(LOW_GLUCOSE_S, color="blue", ls="--", lw=1, alpha=0.7)

    idx = np.arange(0, len(mean_t), step)

    if ctrl_features is not None:
        # -- Two-curve mode --
        # Double-step normalization for control (same as treatment):
        #   Step 1 (per-unit max) already done in the npz data.
        #   Step 2: population baseline normalization.
        baseline_start = max(0, CONTROL_TRIAL_END - BASELINE_WINDOW)
        ctrl_pop_mean = np.nanmean(ctrl_features, axis=0)
        ctrl_baseline = np.nanmean(ctrl_pop_mean[baseline_start:CONTROL_TRIAL_END])
        if ctrl_baseline > 0:
            ctrl_norm = ctrl_features / ctrl_baseline
        else:
            ctrl_norm = ctrl_features.copy()

        n_valid_ctrl = np.sum(~np.isnan(ctrl_norm), axis=0)
        ctrl_mean = np.nanmean(ctrl_norm, axis=0)
        ctrl_sem = np.nanstd(ctrl_norm, axis=0) / np.sqrt(np.maximum(n_valid_ctrl, 1))
        n_ctrl = ctrl_norm.shape[0]

        idx_ctrl = idx[idx < len(ctrl_mean)]
        x_ctrl = idx_ctrl * TRIAL_INTERVAL_S

        # Control curve (black)
        ax.errorbar(
            x_ctrl, ctrl_mean[idx_ctrl], yerr=ctrl_sem[idx_ctrl],
            fmt="o-", capsize=3, color="black", markersize=3, lw=1.2,
            label=f"Control (n={n_ctrl})",
        )

        # Treatment curve -- full length (red)
        ax.errorbar(
            x_s[idx], mean_t[idx], yerr=sem_t[idx],
            fmt="o-", capsize=3, color="red", markersize=3, lw=1.2,
            label=f"Treatment (n={n_units})",
        )

        # Welch t-test at overlapping sampled points
        for i, j in zip(idx_ctrl, x_ctrl):
            if i >= arr.shape[1]:
                break
            c_vals = ctrl_norm[:, i][~np.isnan(ctrl_norm[:, i])]
            t_vals = arr[:, i]
            if len(c_vals) > 1 and len(t_vals) > 1:
                t_stat, p = ttest_ind(c_vals, t_vals, equal_var=False)
            else:
                p = 1.0
            if p < 0.05:
                y_star = max(
                    ctrl_mean[i] + ctrl_sem[i],
                    mean_t[i] + sem_t[i],
                ) + 0.02
                ax.text(j, y_star, "*", ha="center", va="bottom",
                        fontsize=10, color="k")

        title_text = (
            f"{cell_label} "
            f"(Control: n={n_ctrl}, Treatment: n={n_units})"
        )
    else:
        # -- Single-curve mode (original behavior) --
        ax.errorbar(x_s[idx], mean_t[idx], yerr=sem_t[idx],
                    fmt="o-", capsize=3, color="black", markersize=3, lw=1.2,
                    label=f"n = {n_units}")

        ctrl_idx = np.arange(0, CONTROL_TRIAL_END)
        ctrl_pool = arr[:, ctrl_idx].mean(axis=1)
        for i in idx:
            if i < CONTROL_TRIAL_END:
                continue
            trial_vals = arr[:, i]
            _, p = ttest_ind(ctrl_pool, trial_vals, equal_var=False)
            if p < 0.05:
                y_star = mean_t[i] + sem_t[i] + 0.02
                ax.text(x_s[i], y_star, "*", ha="center", va="bottom",
                        fontsize=10, color="k")

        title_text = f"{cell_label} (n = {n_units})"

    # phase labels
    ylim = ax.get_ylim()
    label_y = ylim[1] * 0.97 if ylim[1] > 0 else 1.0
    ax.text(HIGH_GLUCOSE_S / 2, label_y, "Normal",
            ha="center", fontsize=8, color="black", alpha=0.6)
    ax.text((HIGH_GLUCOSE_S + LOW_GLUCOSE_S) / 2, label_y, "High glucose",
            ha="center", fontsize=8, color="red", alpha=0.7)
    ax.text((LOW_GLUCOSE_S + x_s[-1]) / 2, label_y, "Low glucose",
            ha="center", fontsize=8, color="blue", alpha=0.7)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Normalized Response", fontsize=11)
    ax.set_title(title_text, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_single_panel(fig, save_dir: Path, panel_name: str):
    """Save a single-panel figure as PNG and SVG."""
    for fmt in ("png", "svg"):
        path = save_dir / f"Figure_Y_{panel_name}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.close(fig)


def plot_with_low_glucose_alone(
    ax: plt.Axes,
    units_sr: List[np.ndarray],
    cell_label: str,
    low_alone_features: np.ndarray,
    step: int = 2,
):
    """Plot treatment time course with low-glucose-alone data overlaid.

    The low-glucose-alone data is time-shifted to start at LOW_GLUCOSE_S,
    showing the separate low-glucose experiment alongside the continuous
    normal->high->low recording.
    """
    tc = compute_population_timecourse(units_sr)
    x_s = tc["x_s"]
    mean_t = tc["mean"]
    sem_t = tc["sem"]
    arr = tc["arr"]
    n_units = tc["n_units"]

    ax.axvspan(HIGH_GLUCOSE_S, LOW_GLUCOSE_S,
               color="red", alpha=0.06, zorder=0)
    ax.axvspan(LOW_GLUCOSE_S, x_s[-1],
               color="blue", alpha=0.06, zorder=0)
    ax.axvline(HIGH_GLUCOSE_S, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(LOW_GLUCOSE_S, color="blue", ls="--", lw=1, alpha=0.7)

    idx = np.arange(0, len(mean_t), step)

    ax.errorbar(
        x_s[idx], mean_t[idx], yerr=sem_t[idx],
        fmt="o-", capsize=3, color="red", markersize=3, lw=1.2,
        label=f"Treatment (n={n_units})",
    )

    baseline_start = max(0, CONTROL_TRIAL_END - BASELINE_WINDOW)
    la_pop_mean = np.nanmean(low_alone_features, axis=0)
    la_baseline = np.nanmean(la_pop_mean[:BASELINE_WINDOW])
    if la_baseline > 0:
        la_norm = low_alone_features / la_baseline
    else:
        la_norm = low_alone_features.copy()

    n_valid_la = np.sum(~np.isnan(la_norm), axis=0)
    la_mean = np.nanmean(la_norm, axis=0)
    la_sem = np.nanstd(la_norm, axis=0) / np.sqrt(np.maximum(n_valid_la, 1))
    n_la = la_norm.shape[0]

    la_trials = np.arange(len(la_mean))
    la_idx = la_trials[::step]
    x_la = LOW_GLUCOSE_S + la_idx * TRIAL_INTERVAL_S

    treat_end = x_s[-1]
    mask = x_la <= treat_end
    x_la = x_la[mask]
    la_idx = la_idx[mask]

    ax.errorbar(
        x_la, la_mean[la_idx], yerr=la_sem[la_idx],
        fmt="s-", capsize=3, color="blue", markersize=3, lw=1.2,
        label=f"Low glucose alone (n={n_la})",
    )

    for i in la_idx:
        trial_in_treatment = LOW_GLUCOSE_TRIAL_START + i
        if trial_in_treatment >= arr.shape[1]:
            break
        la_vals = la_norm[:, i][~np.isnan(la_norm[:, i])]
        t_vals = arr[:, trial_in_treatment]
        if len(la_vals) > 1 and len(t_vals) > 1:
            _, p = ttest_ind(la_vals, t_vals, equal_var=False)
        else:
            p = 1.0
        if p < 0.05:
            j = LOW_GLUCOSE_S + i * TRIAL_INTERVAL_S
            y_star = max(la_mean[i] + la_sem[i],
                         mean_t[trial_in_treatment] + sem_t[trial_in_treatment]) + 0.02
            ax.text(j, y_star, "*", ha="center", va="bottom",
                    fontsize=10, color="k")

    ylim = ax.get_ylim()
    label_y = ylim[1] * 0.97 if ylim[1] > 0 else 1.0
    ax.text(HIGH_GLUCOSE_S / 2, label_y, "Normal",
            ha="center", fontsize=8, color="black", alpha=0.6)
    ax.text((HIGH_GLUCOSE_S + LOW_GLUCOSE_S) / 2, label_y, "High glucose",
            ha="center", fontsize=8, color="red", alpha=0.7)
    ax.text((LOW_GLUCOSE_S + x_s[-1]) / 2, label_y, "Low glucose",
            ha="center", fontsize=8, color="blue", alpha=0.7)

    ax.set_xlim(x_s[0] - 10, treat_end + 10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Normalized Response", fontsize=11)
    ax.set_title(
        f"{cell_label} (Treatment: n={n_units}, Low glucose alone: n={n_la})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
# Panel E2 / F2 -- Treatment vs Low glucose alone, aligned at HG OFF
# ---------------------------------------------------------------------------

def plot_treatment_vs_low_alone_aligned(
    ax: plt.Axes,
    units_sr: List[np.ndarray],
    cell_label: str,
    low_alone_features: np.ndarray,
    step: int = 2,
    trim_red: int = 3,
):
    """Plot treatment vs low-glucose-alone, aligned at end of high glucose.

    The blue trace (low glucose alone) is scaled to match the red trace
    (treatment) at the high-glucose OFF time point (t=0).
    """
    tc = compute_population_timecourse(units_sr)
    x_abs = tc["x_s"]
    treat_mean = tc["mean"]
    treat_sem = tc["sem"]
    treat_arr = tc["arr"]
    n_treat = tc["n_units"]

    x_rel = x_abs - LOW_GLUCOSE_S
    treat_end_rel = x_rel[-1]

    la_pop_mean = np.nanmean(low_alone_features, axis=0)
    la_baseline = np.nanmean(la_pop_mean[:BASELINE_WINDOW])
    if la_baseline > 0:
        la_norm = low_alone_features / la_baseline
    else:
        la_norm = low_alone_features.copy()
    n_valid_la = np.sum(~np.isnan(la_norm), axis=0)
    la_mean = np.nanmean(la_norm, axis=0)
    la_sem = np.nanstd(la_norm, axis=0) / np.sqrt(np.maximum(n_valid_la, 1))
    n_la = la_norm.shape[0]
    la_x = np.arange(len(la_mean)) * TRIAL_INTERVAL_S
    la_mask = la_x <= treat_end_rel
    la_n_valid = int(la_mask.sum())

    treat_t0_idx = LOW_GLUCOSE_TRIAL_START
    treat_val_t0 = treat_mean[treat_t0_idx] if treat_t0_idx < len(treat_mean) else 1.0
    la_val_t0 = la_mean[0] if len(la_mean) > 0 else 1.0
    if la_val_t0 > 0:
        la_scale = treat_val_t0 / la_val_t0
        la_mean = la_mean * la_scale
        la_sem = la_sem * la_scale
        la_norm = la_norm * la_scale

    hg_on_rel = HIGH_GLUCOSE_S - LOW_GLUCOSE_S
    ax.axvspan(hg_on_rel, 0, color="red", alpha=0.06, zorder=0)
    ax.axvspan(0, treat_end_rel, color="blue", alpha=0.06, zorder=0)
    ax.axvline(hg_on_rel, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(0, color="blue", ls="--", lw=1, alpha=0.7)

    treat_idx = np.arange(trim_red * step, len(treat_mean), step)
    la_idx = np.arange(0, la_n_valid, step)

    ax.errorbar(
        x_rel[treat_idx], treat_mean[treat_idx], yerr=treat_sem[treat_idx],
        fmt="o-", capsize=3, color="red", markersize=3, lw=1.2,
        label=f"High to low glucose (n={n_treat})",
    )
    ax.errorbar(
        la_x[la_idx], la_mean[la_idx], yerr=la_sem[la_idx],
        fmt="s-", capsize=3, color="blue", markersize=3, lw=1.2,
        label=f"Normal to low glucose (n={n_la})",
    )

    n_treat_trials = treat_arr.shape[1]
    for idx_i in la_idx:
        t_rel = la_x[idx_i]
        treat_trial = LOW_GLUCOSE_TRIAL_START + int(round(t_rel / TRIAL_INTERVAL_S))
        if treat_trial >= n_treat_trials or idx_i >= la_norm.shape[1]:
            continue
        t_vals = treat_arr[:, treat_trial]
        la_col = la_norm[:, idx_i]
        la_valid = la_col[~np.isnan(la_col)]
        if len(la_valid) > 1 and len(t_vals) > 1:
            _, p = ttest_ind(la_valid, t_vals, equal_var=False)
            if p < 0.05:
                y_star = max(la_mean[idx_i] + la_sem[idx_i],
                             treat_mean[treat_trial] + treat_sem[treat_trial]) + 0.02
                ax.text(la_x[idx_i], y_star, "*", ha="center", va="bottom",
                        fontsize=10, color="k")

    ax.set_ylim(0.4, 1.1)
    ylim = ax.get_ylim()
    label_y = ylim[1] * 0.97 if ylim[1] > 0 else 1.0
    normal_center = (x_rel[0] + hg_on_rel) / 2
    ax.text(normal_center, label_y, "Normal",
            ha="center", fontsize=8, color="black", alpha=0.6)
    ax.text(hg_on_rel / 2, label_y, "High glucose",
            ha="center", fontsize=8, color="red", alpha=0.7)
    ax.text(treat_end_rel / 2, label_y, "Low glucose",
            ha="center", fontsize=8, color="blue", alpha=0.7)

    ax.set_xlim(x_rel[0] - 10, treat_end_rel + 10)
    ax.set_xlabel("Time relative to glucose OFF (s)", fontsize=11)
    ax.set_ylabel("Normalized Response", fontsize=11)
    ax.set_title(
        f"{cell_label} (n={n_treat}, n={n_la})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
# Panel G / H  --  high-glucose-only vs high/low glucose treatment
# ---------------------------------------------------------------------------

def _align_features_to_offset(
    features: np.ndarray,
    align_points_s: List[float],
    trial_interval_s: float = TRIAL_INTERVAL_S,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift feature arrays so that each unit's alignment point maps to t=0.

    Different recordings may have different alignment points (normal_glucose_min).
    We build a common time grid and place each unit's data onto it with NaN
    padding where data is unavailable.

    Returns:
        (aligned_arr, common_x_s) where aligned_arr is (n_units, n_common_trials)
        and common_x_s is the time axis in seconds relative to the alignment point.
    """
    n_units, n_trials = features.shape
    align_trials = [int(round(a / trial_interval_s)) for a in align_points_s]

    pre_max = max(align_trials)
    post_max = max(n_trials - at for at in align_trials)
    n_common = pre_max + post_max

    aligned = np.full((n_units, n_common), np.nan)
    for i in range(n_units):
        at = align_trials[i]
        offset = pre_max - at
        aligned[i, offset:offset + n_trials] = features[i, :]

    common_x = (np.arange(n_common) - pre_max) * trial_interval_s
    return aligned, common_x


def plot_high_vs_low_glucose(
    ax: plt.Axes,
    treatment_units_sr: List[np.ndarray],
    cell_label: str,
    hg_features: np.ndarray,
    hg_align_points_s: List[float],
    step: int = 2,
):
    """Overlay high-glucose-only and high/low glucose treatment data,
    aligned at the offset of high glucose (t=0).

    - Treatment (high/low glucose): red circles, aligned at LOW_GLUCOSE_S
    - High-glucose-only: green squares, aligned at each recording's
      normal_glucose_min (end of high glucose = return to normal)
    """
    # -- Treatment data (high/low glucose) --
    tc = compute_population_timecourse(treatment_units_sr)
    treat_x_abs = tc["x_s"]
    treat_arr = tc["arr"]
    n_treat = tc["n_units"]

    treat_x_rel = treat_x_abs - LOW_GLUCOSE_S
    treat_mean = tc["mean"]
    treat_sem = tc["sem"]

    # -- High-glucose-only data --
    hg_aligned, hg_x_rel = _align_features_to_offset(
        hg_features, hg_align_points_s,
    )

    baseline_start = max(0, CONTROL_TRIAL_END - BASELINE_WINDOW)
    hg_pop_mean = np.nanmean(hg_aligned, axis=0)
    baseline_trial_indices = np.where(
        (hg_x_rel >= -(CONTROL_TRIAL_END * TRIAL_INTERVAL_S))
        & (hg_x_rel < -(baseline_start * TRIAL_INTERVAL_S))
    )[0]
    if len(baseline_trial_indices) > 0:
        hg_baseline = np.nanmean(hg_pop_mean[baseline_trial_indices[-BASELINE_WINDOW:]])
    else:
        hg_baseline = np.nanmean(hg_pop_mean[:BASELINE_WINDOW])
    if hg_baseline > 0:
        hg_norm = hg_aligned / hg_baseline
    else:
        hg_norm = hg_aligned.copy()

    n_valid_hg = np.sum(~np.isnan(hg_norm), axis=0)
    hg_mean = np.nanmean(hg_norm, axis=0)
    hg_sem = np.nanstd(hg_norm, axis=0) / np.sqrt(np.maximum(n_valid_hg, 1))
    n_hg = hg_features.shape[0]

    valid_hg_mask = n_valid_hg >= max(3, n_hg * 0.3)

    # -- Phase shading (relative to alignment at t=0) --
    hg_start_rel = HIGH_GLUCOSE_S - LOW_GLUCOSE_S
    ax.axvspan(hg_start_rel, 0, color="red", alpha=0.06, zorder=0)
    ax.axvline(hg_start_rel, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(0, color="green", ls="--", lw=1.5, alpha=0.7,
               label="End of high glucose")

    # -- Treatment curve (red) --
    idx_t = np.arange(0, len(treat_mean), step)
    ax.errorbar(
        treat_x_rel[idx_t], treat_mean[idx_t], yerr=treat_sem[idx_t],
        fmt="o-", capsize=3, color="red", markersize=3, lw=1.2,
        label=f"High/Low glucose (n={n_treat})",
    )

    # -- High-glucose-only curve (green) --
    hg_idx = np.arange(0, len(hg_mean), step)
    hg_idx = hg_idx[valid_hg_mask[hg_idx]]
    ax.errorbar(
        hg_x_rel[hg_idx], hg_mean[hg_idx], yerr=hg_sem[hg_idx],
        fmt="s-", capsize=3, color="green", markersize=3, lw=1.2,
        label=f"High glucose only (n={n_hg})",
    )

    # -- Welch t-test at overlapping sampled time points --
    for i_hg in hg_idx:
        t_rel = hg_x_rel[i_hg]
        i_treat = np.argmin(np.abs(treat_x_rel - t_rel))
        if abs(treat_x_rel[i_treat] - t_rel) > TRIAL_INTERVAL_S * 0.6:
            continue
        hg_vals = hg_norm[:, i_hg][~np.isnan(hg_norm[:, i_hg])]
        t_vals = treat_arr[:, i_treat]
        if len(hg_vals) > 1 and len(t_vals) > 1:
            _, p = ttest_ind(hg_vals, t_vals, equal_var=False)
        else:
            p = 1.0
        if p < 0.05:
            y_star = max(
                hg_mean[i_hg] + hg_sem[i_hg],
                treat_mean[i_treat] + treat_sem[i_treat],
            ) + 0.02
            ax.text(t_rel, y_star, "*", ha="center", va="bottom",
                    fontsize=10, color="k")

    # -- Phase labels --
    ylim = ax.get_ylim()
    label_y = ylim[1] * 0.97 if ylim[1] > 0 else 1.0
    normal_mid = (treat_x_rel[0] + hg_start_rel) / 2
    ax.text(normal_mid, label_y, "Normal",
            ha="center", fontsize=8, color="black", alpha=0.6)
    ax.text(hg_start_rel / 2, label_y, "High glucose",
            ha="center", fontsize=8, color="red", alpha=0.7)

    treat_post_label_x = treat_x_rel[-1] / 2
    if treat_post_label_x > 20:
        ax.text(treat_post_label_x, label_y, "Low glucose (treatment)",
                ha="center", fontsize=8, color="blue", alpha=0.7)

    hg_post = hg_x_rel[valid_hg_mask]
    hg_post_positive = hg_post[hg_post > 0]
    if len(hg_post_positive) > 1:
        hg_post_label_x = hg_post_positive[-1] / 2
        ax.text(hg_post_label_x, label_y * 0.93, "Normal (HG only)",
                ha="center", fontsize=8, color="green", alpha=0.6)

    x_lo = min(treat_x_rel[0], hg_x_rel[valid_hg_mask][0] if valid_hg_mask.any() else 0) - 10
    x_hi = max(treat_x_rel[-1], hg_x_rel[valid_hg_mask][-1] if valid_hg_mask.any() else 0) + 10
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Time relative to end of high glucose (s)", fontsize=11)
    ax.set_ylabel("Normalized Response", fontsize=11)
    ax.set_title(
        f"{cell_label} (High/Low: n={n_treat}, High only: n={n_hg})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
# Panel I / J  --  high-glucose-alone combined population time course
# ---------------------------------------------------------------------------

def plot_high_glucose_combined(
    ax: plt.Axes,
    combined_features: np.ndarray,
    cell_label: str,
    step: int = 2,
    bin_size: int = 1,
):
    """Single-curve population time course for pooled high-glucose-alone data.

    The input *combined_features* is already per-unit-max normalized and
    HG-phase-aligned to the majority timing (MAJORITY_HG_OFF_TRIAL).

    Two-step normalization is completed here (step 2: population baseline).
    NaN-aware statistics are used throughout.

    When *bin_size* > 1, consecutive trials are averaged into non-overlapping
    bins before plotting (reduces noise, fewer data points).
    """
    n_units, n_trials = combined_features.shape
    baseline_start = max(0, HG_ON_TRIAL - BASELINE_WINDOW)

    pop_mean_raw = np.nanmean(combined_features, axis=0)
    baseline_val = np.nanmean(pop_mean_raw[baseline_start:HG_ON_TRIAL])
    if baseline_val > 0:
        arr = combined_features / baseline_val
    else:
        arr = combined_features.copy()

    if bin_size > 1:
        n_bins = n_trials // bin_size
        arr_binned = np.full((n_units, n_bins), np.nan)
        for b in range(n_bins):
            arr_binned[:, b] = np.nanmean(arr[:, b * bin_size:(b + 1) * bin_size], axis=1)
        arr = arr_binned
        x_s = (np.arange(n_bins) * bin_size + (bin_size - 1) / 2.0) * TRIAL_INTERVAL_S
        n_plot = n_bins
        hg_on_bin = HG_ON_TRIAL / bin_size
    else:
        x_s = np.arange(n_trials) * TRIAL_INTERVAL_S
        n_plot = n_trials
        hg_on_bin = HG_ON_TRIAL

    n_valid = np.sum(~np.isnan(arr), axis=0)
    mean_t = np.nanmean(arr, axis=0)
    sem_t = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n_valid, 1))

    min_valid = max(3, int(n_units * 0.3))
    valid_mask = n_valid >= min_valid

    ax.axvspan(HIGH_GLUCOSE_S, MAJORITY_HG_OFF_S,
               color="red", alpha=0.06, zorder=0)
    ax.axvline(HIGH_GLUCOSE_S, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(MAJORITY_HG_OFF_S, color="green", ls="--", lw=1, alpha=0.7)

    idx = np.arange(0, n_plot, step)
    idx = idx[valid_mask[idx]]

    ax.errorbar(
        x_s[idx], mean_t[idx], yerr=sem_t[idx],
        fmt="o-", capsize=3, color="black", markersize=3, lw=1.2,
        label=f"n = {n_units}",
    )

    baseline_bins = np.where(x_s < HIGH_GLUCOSE_S)[0]
    if len(baseline_bins) > BASELINE_WINDOW:
        baseline_bins = baseline_bins[-BASELINE_WINDOW:]
    baseline_pool = arr[:, baseline_bins]
    baseline_per_unit = np.nanmean(baseline_pool, axis=1)
    baseline_valid = baseline_per_unit[~np.isnan(baseline_per_unit)]
    for i in idx:
        if x_s[i] < HIGH_GLUCOSE_S:
            continue
        trial_vals = arr[:, i][~np.isnan(arr[:, i])]
        if len(trial_vals) > 1 and len(baseline_valid) > 1:
            _, p = ttest_ind(baseline_valid, trial_vals, equal_var=False)
        else:
            p = 1.0
        if p < 0.05:
            y_star = mean_t[i] + sem_t[i] + 0.02
            ax.text(x_s[i], y_star, "*", ha="center", va="bottom",
                    fontsize=10, color="k")

    ylim = ax.get_ylim()
    label_y = ylim[1] * 0.97 if ylim[1] > 0 else 1.0
    ax.text(HIGH_GLUCOSE_S / 2, label_y, "Normal",
            ha="center", fontsize=8, color="black", alpha=0.6)
    ax.text((HIGH_GLUCOSE_S + MAJORITY_HG_OFF_S) / 2, label_y, "High glucose",
            ha="center", fontsize=8, color="red", alpha=0.7)
    ax.text((MAJORITY_HG_OFF_S + x_s[idx[-1]]) / 2, label_y, "Normal",
            ha="center", fontsize=8, color="green", alpha=0.6)

    ax.set_xlim(x_s[0] - 10, x_s[idx[-1]] + 10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Normalized Response", fontsize=11)
    ax.set_title(
        f"{cell_label} (n = {n_units})",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
# Panel M / N  --  three-trace overlay aligned at glucose OFF
# ---------------------------------------------------------------------------

def plot_three_trace_overlay(
    ax: plt.Axes,
    units_sr: List[np.ndarray],
    cell_label: str,
    low_alone_features: np.ndarray,
    hg_combined_features: np.ndarray,
    step: int = 2,
    show_full_hg: bool = False,
    show_low_alone: bool = True,
    post_only: bool = False,
    custom_labels: Optional[Dict[str, str]] = None,
):
    """Overlay three traces aligned at high-glucose OFF (t=0).

    Style matches Figure E: discrete error-bar dots, phase shading,
    vertical dashed lines, and phase labels.

    Traces:
      - Treatment (red): full recording from the high/low glucose experiment
      - Low glucose alone (blue): from the separate low-glucose experiment
      - Normal recovery (green): recovery after HG from the
        high-glucose-only recordings
    """
    # -- 1. Treatment trace (two-step normalized) --
    tc = compute_population_timecourse(units_sr)
    treat_x_abs = tc["x_s"]
    treat_mean = tc["mean"]
    treat_sem = tc["sem"]
    n_treat = tc["n_units"]
    treat_x_rel = treat_x_abs - LOW_GLUCOSE_S
    treat_end_rel = treat_x_rel[-1]

    # -- 2. Low glucose alone trace --
    la_pop_mean = np.nanmean(low_alone_features, axis=0)
    la_baseline = np.nanmean(la_pop_mean[:BASELINE_WINDOW])
    if la_baseline > 0:
        la_norm = low_alone_features / la_baseline
    else:
        la_norm = low_alone_features.copy()
    n_valid_la = np.sum(~np.isnan(la_norm), axis=0)
    la_mean = np.nanmean(la_norm, axis=0)
    la_sem = np.nanstd(la_norm, axis=0) / np.sqrt(np.maximum(n_valid_la, 1))
    n_la = la_norm.shape[0]
    la_x_all = np.arange(len(la_mean)) * TRIAL_INTERVAL_S
    la_mask = la_x_all <= treat_end_rel
    la_n_valid = int(la_mask.sum())

    # -- 3. HG-alone trace (two-step normalized) --
    n_units_hg, n_trials_hg = hg_combined_features.shape
    baseline_start_hg = max(0, HG_ON_TRIAL - BASELINE_WINDOW)
    hg_pop_raw = np.nanmean(hg_combined_features, axis=0)
    hg_baseline = np.nanmean(hg_pop_raw[baseline_start_hg:HG_ON_TRIAL])
    if hg_baseline > 0:
        hg_arr = hg_combined_features / hg_baseline
    else:
        hg_arr = hg_combined_features.copy()
    n_valid_hg = np.sum(~np.isnan(hg_arr), axis=0)
    hg_mean_all = np.nanmean(hg_arr, axis=0)
    hg_sem_all = np.nanstd(hg_arr, axis=0) / np.sqrt(np.maximum(n_valid_hg, 1))

    if show_full_hg:
        hg_recovery_mean = hg_mean_all
        hg_recovery_sem = hg_sem_all
        hg_x_all = (np.arange(n_trials_hg) - MAJORITY_HG_OFF_TRIAL) * TRIAL_INTERVAL_S
        hg_n_valid = n_trials_hg
        hg_align_val = hg_mean_all[MAJORITY_HG_OFF_TRIAL]
    else:
        hg_recovery_mean = hg_mean_all[MAJORITY_HG_OFF_TRIAL:]
        hg_recovery_sem = hg_sem_all[MAJORITY_HG_OFF_TRIAL:]
        hg_x_all = np.arange(len(hg_recovery_mean)) * TRIAL_INTERVAL_S
        hg_mask = hg_x_all <= treat_end_rel
        hg_n_valid = int(hg_mask.sum())
        hg_align_val = hg_recovery_mean[0] if len(hg_recovery_mean) > 0 else 1.0

    # -- Align blue/green to red at t=0 (glucose OFF) --
    if not show_full_hg:
        treat_t0_idx = LOW_GLUCOSE_TRIAL_START
        treat_val_t0 = treat_mean[treat_t0_idx] if treat_t0_idx < len(treat_mean) else 1.0

        la_val_t0 = la_mean[0] if len(la_mean) > 0 else 1.0
        if la_val_t0 > 0:
            la_scale = treat_val_t0 / la_val_t0
            la_mean = la_mean * la_scale
            la_sem = la_sem * la_scale
            la_norm = la_norm * la_scale

        if hg_align_val > 0:
            hg_scale = treat_val_t0 / hg_align_val
            hg_recovery_mean = hg_recovery_mean * hg_scale
            hg_recovery_sem = hg_recovery_sem * hg_scale
            hg_arr = hg_arr * hg_scale

    # -- Phase shading (relative to glucose OFF = 0) --
    hg_on_rel = HIGH_GLUCOSE_S - LOW_GLUCOSE_S
    if not post_only:
        ax.axvspan(hg_on_rel, 0, color="red", alpha=0.06, zorder=0)
        ax.axvline(hg_on_rel, color="red", ls="--", lw=1, alpha=0.7)
        ax.axvspan(max(0, hg_on_rel), treat_end_rel,
                   color="blue", alpha=0.06, zorder=0)
    if not post_only:
        ax.axvline(0, color="blue", ls="--", lw=1, alpha=0.7)

    # -- Subsample indices --
    if post_only:
        t0_trial = LOW_GLUCOSE_TRIAL_START
        treat_idx = np.arange(t0_trial, len(treat_mean), step)
    else:
        treat_idx = np.arange(4 * step, len(treat_mean), step)
    la_idx = np.arange(0, la_n_valid, step)
    hg_idx = np.arange(0, hg_n_valid, step)

    # -- Resolve labels --
    _cl = custom_labels or {}
    lbl_red = _cl.get("red", f"High+Low glucose (n={n_treat})")
    lbl_blue = _cl.get("blue", f"Low glucose alone (n={n_la})")
    lbl_green = _cl.get("green", f"Normal recovery (n={n_units_hg})")
    if "{n}" in lbl_red:
        lbl_red = lbl_red.replace("{n}", str(n_treat))
    if "{n}" in lbl_blue:
        lbl_blue = lbl_blue.replace("{n}", str(n_la))
    if "{n}" in lbl_green:
        lbl_green = lbl_green.replace("{n}", str(n_units_hg))

    # -- Treatment (red) --
    ax.errorbar(
        treat_x_rel[treat_idx], treat_mean[treat_idx],
        yerr=treat_sem[treat_idx],
        fmt="o-", capsize=3, color="red", markersize=3, lw=1.2,
        label=lbl_red,
    )

    # -- Low glucose alone (blue) --
    if show_low_alone:
        ax.errorbar(
            la_x_all[la_idx], la_mean[la_idx], yerr=la_sem[la_idx],
            fmt="s-", capsize=3, color="blue", markersize=3, lw=1.2,
            label=lbl_blue,
        )

    # -- Normal recovery (green) --
    ax.errorbar(
        hg_x_all[hg_idx], hg_recovery_mean[hg_idx],
        yerr=hg_recovery_sem[hg_idx],
        fmt="^-", capsize=3, color="green", markersize=3, lw=1.2,
        label=lbl_green,
    )

    # -- Welch t-test: blue/green vs red at each plotted point (post_only) --
    if post_only:
        treat_arr = tc["arr"]
        n_treat_trials = treat_arr.shape[1]
        for idx_i in la_idx:
            t_rel = la_x_all[idx_i]
            treat_trial = LOW_GLUCOSE_TRIAL_START + int(round(t_rel / TRIAL_INTERVAL_S))
            if treat_trial >= n_treat_trials:
                continue
            t_vals = treat_arr[:, treat_trial]
            la_col = la_norm[:, idx_i] if idx_i < la_norm.shape[1] else None
            if la_col is not None:
                la_valid = la_col[~np.isnan(la_col)]
                if len(la_valid) > 1 and len(t_vals) > 1:
                    _, p = ttest_ind(la_valid, t_vals, equal_var=False)
                    if p < 0.05:
                        y_star = la_mean[idx_i] + la_sem[idx_i] + 0.01
                        ax.text(la_x_all[idx_i], y_star, "*",
                                ha="center", va="bottom", fontsize=9,
                                color="blue", fontweight="bold")

        hg_offset = MAJORITY_HG_OFF_TRIAL if not show_full_hg else 0
        for idx_i in hg_idx:
            t_rel = hg_x_all[idx_i]
            treat_trial = LOW_GLUCOSE_TRIAL_START + int(round(t_rel / TRIAL_INTERVAL_S))
            hg_trial = hg_offset + int(round(t_rel / TRIAL_INTERVAL_S)) if not show_full_hg else idx_i
            if treat_trial >= n_treat_trials or hg_trial >= hg_arr.shape[1]:
                continue
            t_vals = treat_arr[:, treat_trial]
            hg_col = hg_arr[:, hg_trial]
            hg_valid = hg_col[~np.isnan(hg_col)]
            if len(hg_valid) > 1 and len(t_vals) > 1:
                _, p = ttest_ind(hg_valid, t_vals, equal_var=False)
                if p < 0.05:
                    y_star = hg_recovery_mean[idx_i] + hg_recovery_sem[idx_i] + 0.01
                    ax.text(hg_x_all[idx_i], y_star, "*",
                            ha="center", va="bottom", fontsize=9,
                            color="green", fontweight="bold")

    # -- Phase labels --
    ax.set_ylim(0.4, 1.1)
    ylim = ax.get_ylim()
    label_y = ylim[1] * 0.97 if ylim[1] > 0 else 1.0
    if not post_only:
        normal_center = (treat_x_rel[0] + hg_on_rel) / 2
        ax.text(normal_center, label_y, "Normal",
                ha="center", fontsize=8, color="black", alpha=0.6)
        ax.text(hg_on_rel / 2, label_y, "High glucose",
                ha="center", fontsize=8, color="red", alpha=0.7)
        ax.text(treat_end_rel / 2, label_y, "Low glucose",
                ha="center", fontsize=8, color="blue", alpha=0.7)

    x_start = -10 if post_only else treat_x_rel[0] - 10
    ax.set_xlim(x_start, treat_end_rel + 10)
    ax.set_xlabel("Time relative to glucose OFF (s)", fontsize=11)
    ax.set_ylabel("Normalized Response", fontsize=11)
    ax.set_title(cell_label, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best", framealpha=0.8)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15)


# ---------------------------------------------------------------------------
# Separate panels
# ---------------------------------------------------------------------------

def make_separate_panels(
    config: Optional[LowGlucosePipelineConfig] = None,
):
    """Generate panels A, B, C, D as separate figures."""
    if config is None:
        config = default_config

    save_dir = _THIS_DIR / "paper_figure"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading recordings...")
    all_data = load_all_h5(config)
    logger.info("Loaded %d recordings", len(all_data))

    logger.info("Collecting OFF cells...")
    off_units = collect_units(all_data, "off")
    logger.info("  %d OFF cells", len(off_units))

    logger.info("Collecting ON cells...")
    on_units = collect_units(all_data, "on")
    logger.info("  %d ON cells", len(on_units))

    off_repr = pick_representative_unit(off_units)
    on_repr = pick_representative_unit(on_units)

    logger.info("Loading external control features from low_glucose_alone...")
    off_ctrl_alone, on_ctrl_alone = load_control_features()
    logger.info("  OFF control: %s, ON control: %s",
                off_ctrl_alone.shape, on_ctrl_alone.shape)

    # Panel A: Representative OFF-RGC traces (subplots over time)
    fig_a, _ = plot_representative_traces(off_repr, "OFF-RGC")
    _save_single_panel(fig_a, save_dir, "A_OFF_representative")

    # Panel B: Population OFF time course (with control overlay)
    fig_b, ax_b = plt.subplots(figsize=(10, 4))
    plot_population_timecourse(ax_b, off_units, "OFF-response",
                               ctrl_features=off_ctrl_alone)
    fig_b.tight_layout()
    _save_single_panel(fig_b, save_dir, "B_OFF_population")

    # Panel C: Representative ON-RGC traces (subplots over time)
    fig_c, _ = plot_representative_traces(on_repr, "ON-RGC")
    _save_single_panel(fig_c, save_dir, "C_ON_representative")

    # Panel D: Population ON time course (with control overlay)
    fig_d, ax_d = plt.subplots(figsize=(10, 4))
    plot_population_timecourse(ax_d, on_units, "ON-response",
                               ctrl_features=on_ctrl_alone)
    fig_d.tight_layout()
    _save_single_panel(fig_d, save_dir, "D_ON_population")

    # Panel E/F: Treatment with low-glucose-alone overlay
    logger.info("Loading low-glucose-alone features...")
    off_low_alone, on_low_alone = load_low_glucose_alone_features()
    logger.info("  OFF low alone: %s, ON low alone: %s",
                off_low_alone.shape, on_low_alone.shape)

    fig_e, ax_e = plt.subplots(figsize=(10, 4))
    plot_with_low_glucose_alone(ax_e, off_units, "OFF-response",
                                off_low_alone)
    fig_e.tight_layout()
    _save_single_panel(fig_e, save_dir, "E_OFF_with_low_alone")

    fig_f, ax_f = plt.subplots(figsize=(10, 4))
    plot_with_low_glucose_alone(ax_f, on_units, "ON-response",
                                on_low_alone)
    fig_f.tight_layout()
    _save_single_panel(fig_f, save_dir, "F_ON_with_low_alone")

    # Panel E2/F2: Treatment vs Low glucose alone, aligned at HG OFF
    fig_e2, ax_e2 = plt.subplots(figsize=(10, 4))
    plot_treatment_vs_low_alone_aligned(ax_e2, off_units, "OFF-Cell",
                                         off_low_alone)
    fig_e2.tight_layout()
    _save_single_panel(fig_e2, save_dir, "E2_OFF_treatment_vs_low_alone_aligned")

    fig_f2, ax_f2 = plt.subplots(figsize=(10, 4))
    plot_treatment_vs_low_alone_aligned(ax_f2, on_units, "ON-Cell",
                                         on_low_alone)
    fig_f2.tight_layout()
    _save_single_panel(fig_f2, save_dir, "F2_ON_treatment_vs_low_alone_aligned")

    # Panel G/H: High-glucose-only vs high/low glucose treatment
    logger.info("Loading high-glucose-only features...")
    hg_off_features, hg_off_align = load_high_glucose_only_features("off")
    hg_on_features, hg_on_align = load_high_glucose_only_features("on")
    logger.info("  HG OFF: %s, HG ON: %s",
                hg_off_features.shape, hg_on_features.shape)

    fig_g, ax_g = plt.subplots(figsize=(10, 4))
    plot_high_vs_low_glucose(ax_g, off_units, "OFF-response",
                              hg_off_features, hg_off_align)
    fig_g.tight_layout()
    _save_single_panel(fig_g, save_dir, "G_OFF_high_vs_low")

    fig_h, ax_h = plt.subplots(figsize=(10, 4))
    plot_high_vs_low_glucose(ax_h, on_units, "ON-response",
                              hg_on_features, hg_on_align)
    fig_h.tight_layout()
    _save_single_panel(fig_h, save_dir, "H_ON_high_vs_low")

    # Panel I/J: High-glucose-alone combined population time course
    logger.info("Loading high-glucose combined features (aligned to majority)...")
    hg_combined_off = load_high_glucose_combined_features("off")
    hg_combined_on = load_high_glucose_combined_features("on")
    logger.info("  HG combined OFF: %s, ON: %s",
                hg_combined_off.shape, hg_combined_on.shape)

    fig_i, ax_i = plt.subplots(figsize=(10, 4))
    plot_high_glucose_combined(ax_i, hg_combined_off, "OFF-response")
    fig_i.tight_layout()
    _save_single_panel(fig_i, save_dir, "I_OFF_high_glucose_combined")

    fig_j, ax_j = plt.subplots(figsize=(10, 4))
    plot_high_glucose_combined(ax_j, hg_combined_on, "ON-response")
    fig_j.tight_layout()
    _save_single_panel(fig_j, save_dir, "J_ON_high_glucose_combined")

    # Binned (4 trials per bin) versions of I/J
    fig_ib, ax_ib = plt.subplots(figsize=(10, 4))
    plot_high_glucose_combined(ax_ib, hg_combined_off, "OFF-response",
                                step=1, bin_size=4)
    fig_ib.tight_layout()
    _save_single_panel(fig_ib, save_dir, "I2_OFF_high_glucose_combined_binned")

    fig_jb, ax_jb = plt.subplots(figsize=(10, 4))
    plot_high_glucose_combined(ax_jb, hg_combined_on, "ON-response",
                                step=1, bin_size=4)
    fig_jb.tight_layout()
    _save_single_panel(fig_jb, save_dir, "J2_ON_high_glucose_combined_binned")

    # Panel K/L: Majority-only (15.0 min) high-glucose combined -- no NaN
    logger.info("Loading majority-only high-glucose features (15.0 min)...")
    hg_maj_off = load_high_glucose_majority_features("off")
    hg_maj_on = load_high_glucose_majority_features("on")
    logger.info("  HG majority OFF: %s, ON: %s",
                hg_maj_off.shape, hg_maj_on.shape)

    fig_k, ax_k = plt.subplots(figsize=(10, 4))
    plot_high_glucose_combined(ax_k, hg_maj_off, "OFF-response")
    fig_k.tight_layout()
    _save_single_panel(fig_k, save_dir, "K_OFF_high_glucose_majority")

    fig_l, ax_l = plt.subplots(figsize=(10, 4))
    plot_high_glucose_combined(ax_l, hg_maj_on, "ON-response")
    fig_l.tight_layout()
    _save_single_panel(fig_l, save_dir, "L_ON_high_glucose_majority")

    # Panel M/N: Three-trace overlay aligned at glucose OFF
    logger.info("Generating three-trace overlay plots (aligned at glucose OFF)...")
    fig_m, ax_m = plt.subplots(figsize=(10, 4))
    plot_three_trace_overlay(ax_m, off_units, "OFF-response",
                              off_low_alone, hg_combined_off)
    fig_m.tight_layout()
    _save_single_panel(fig_m, save_dir, "M_OFF_three_trace_overlay")

    fig_n, ax_n = plt.subplots(figsize=(10, 4))
    plot_three_trace_overlay(ax_n, on_units, "ON-response",
                              on_low_alone, hg_combined_on)
    fig_n.tight_layout()
    _save_single_panel(fig_n, save_dir, "N_ON_three_trace_overlay")

    # Panel O/P: Three-trace overlay with full HG-alone trace
    logger.info("Generating full-HG three-trace overlay plots...")
    fig_o, ax_o = plt.subplots(figsize=(10, 4))
    plot_three_trace_overlay(ax_o, off_units, "OFF-response",
                              off_low_alone, hg_combined_off,
                              show_full_hg=True)
    fig_o.tight_layout()
    _save_single_panel(fig_o, save_dir, "O_OFF_three_trace_full_hg")

    fig_p, ax_p = plt.subplots(figsize=(10, 4))
    plot_three_trace_overlay(ax_p, on_units, "ON-response",
                              on_low_alone, hg_combined_on,
                              show_full_hg=True)
    fig_p.tight_layout()
    _save_single_panel(fig_p, save_dir, "P_ON_three_trace_full_hg")

    # Panel Q/R: Two-trace overlay (treatment + normal recovery, no low alone)
    logger.info("Generating two-trace overlay plots (no low glucose alone)...")
    fig_q, ax_q = plt.subplots(figsize=(10, 4))
    plot_three_trace_overlay(ax_q, off_units, "OFF-response",
                              off_low_alone, hg_combined_off,
                              show_low_alone=False)
    fig_q.tight_layout()
    _save_single_panel(fig_q, save_dir, "Q_OFF_two_trace_overlay")

    fig_r, ax_r = plt.subplots(figsize=(10, 4))
    plot_three_trace_overlay(ax_r, on_units, "ON-response",
                              on_low_alone, hg_combined_on,
                              show_low_alone=False)
    fig_r.tight_layout()
    _save_single_panel(fig_r, save_dir, "R_ON_two_trace_overlay")

    # --- Panels S / T: post-glucose-OFF only (low glucose part) ---
    _post_labels = {
        "red": "High to low glucose (n={n})",
        "green": "High to normal glucose (n={n})",
        "blue": "Normal to low glucose (n={n})",
    }
    fig_s, ax_s = plt.subplots(figsize=(6, 4))
    plot_three_trace_overlay(ax_s, off_units, "OFF-Cell",
                              off_low_alone, hg_combined_off,
                              post_only=True, custom_labels=_post_labels)
    fig_s.tight_layout()
    _save_single_panel(fig_s, save_dir, "S_OFF_low_glucose_only")

    fig_t, ax_t = plt.subplots(figsize=(6, 4))
    plot_three_trace_overlay(ax_t, on_units, "ON-Cell",
                              on_low_alone, hg_combined_on,
                              post_only=True, custom_labels=_post_labels)
    fig_t.tight_layout()
    _save_single_panel(fig_t, save_dir, "T_ON_low_glucose_only")

    logger.info("All separate panels done.")


# ---------------------------------------------------------------------------
# Combined 4-panel figure (kept for convenience)
# ---------------------------------------------------------------------------

def make_figure(
    config: Optional[LowGlucosePipelineConfig] = None,
):
    """Generate the combined 4-panel paper figure."""
    if config is None:
        config = default_config

    save_dir = _THIS_DIR / "paper_figure"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading recordings...")
    all_data = load_all_h5(config)
    logger.info("Loaded %d recordings", len(all_data))

    logger.info("Collecting OFF cells...")
    off_units = collect_units(all_data, "off")
    logger.info("  %d OFF cells", len(off_units))

    logger.info("Collecting ON cells...")
    on_units = collect_units(all_data, "on")
    logger.info("  %d ON cells", len(on_units))

    off_repr = pick_representative_unit(off_units)
    on_repr = pick_representative_unit(on_units)

    logger.info("Loading external control features from low_glucose_alone...")
    off_ctrl_alone, on_ctrl_alone = load_control_features()

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.35,
                          wspace=0.3)

    # A: representative OFF traces (subplots)
    gs_a = gs[0, 0].subgridspec(1, 1)
    ax_a = fig.add_subplot(gs_a[0])
    ax_a.set_visible(False)
    fig_a_tmp, _ = plot_representative_traces(off_repr, "OFF-RGC")
    fig_a_tmp.savefig(save_dir / "_tmp_a.png", dpi=150, bbox_inches="tight")
    plt.close(fig_a_tmp)

    # B: population OFF time course (with control overlay)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_population_timecourse(ax_b, off_units, "OFF-response",
                               ctrl_features=off_ctrl_alone)
    ax_b.text(-0.12, 1.08, "B", transform=ax_b.transAxes,
              fontsize=20, fontweight="bold", va="top")

    # C: representative ON traces (subplots)
    gs_c = gs[1, 0].subgridspec(1, 1)
    ax_c = fig.add_subplot(gs_c[0])
    ax_c.set_visible(False)
    fig_c_tmp, _ = plot_representative_traces(on_repr, "ON-RGC")
    fig_c_tmp.savefig(save_dir / "_tmp_c.png", dpi=150, bbox_inches="tight")
    plt.close(fig_c_tmp)

    # D: population ON time course (with control overlay)
    ax_d = fig.add_subplot(gs[1, 1])
    plot_population_timecourse(ax_d, on_units, "ON-response",
                               ctrl_features=on_ctrl_alone)
    ax_d.text(-0.12, 1.08, "D", transform=ax_d.transAxes,
              fontsize=20, fontweight="bold", va="top")

    for fmt in ("png", "svg"):
        path = save_dir / f"Figure_Y_glucose.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", path)
    plt.close(fig)

    for tmp in (save_dir / "_tmp_a.png", save_dir / "_tmp_c.png"):
        tmp.unlink(missing_ok=True)

    logger.info("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--separate", action="store_true",
                        help="Generate each panel as a separate figure")
    parser.add_argument("--combined", action="store_true",
                        help="Generate the combined 4-panel figure")
    args = parser.parse_args()

    if not args.separate and not args.combined:
        args.separate = True
        args.combined = True

    if args.separate:
        make_separate_panels()
    if args.combined:
        make_figure()
