"""
Generate Figure X panels A-D for the low-glucose-alone study.

Reproduces the exact style from legacy visualize_chains_5.py:

  Panel A: Representative OFF-RGC spike-rate traces (subplots over time)
  Panel B: Population OFF time course  (control=black vs low glucose=red)
  Panel C: Representative ON-RGC spike-rate traces
  Panel D: Population ON time course

Usage:
  python make_paper_figure.py                    # generate all panels
  python make_paper_figure.py --skip-extract     # use cached .npz
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

_THIS_DIR = Path(__file__).resolve().parent
_GLUCOSE_JHU_DIR = _THIS_DIR.parent
_USP_DIR = _GLUCOSE_JHU_DIR.parent

for _p in (str(_GLUCOSE_JHU_DIR), str(_USP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_local_cfg_spec = importlib.util.spec_from_file_location(
    "low_glucose_alone_config", _THIS_DIR / "specific_config.py")
_local_cfg = importlib.util.module_from_spec(_local_cfg_spec)
_local_cfg_spec.loader.exec_module(_local_cfg)
default_config = _local_cfg.default_config
LowGlucoseAloneConfig = _local_cfg.LowGlucoseAloneConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- Legacy constants ----
LIGHT_ON_S = 1.0
LIGHT_OFF_S = 4.0
BINS_PER_TRIAL = 60
BIN_RATE_HZ = 10.0
TRIAL_DURATION_S = BINS_PER_TRIAL / BIN_RATE_HZ
REPR_INTERVAL = 10
REPR_Y_LIM = (0, 220)
POP_STEP = 2
POP_Y_LIM = (0.5, 1.1)


# ---- Helpers ----

def _add_light_shading(ax: plt.Axes):
    """Gray = darkness (before light-on, after light-off)."""
    ax.axvspan(0, LIGHT_ON_S, color="gray", alpha=0.2, zorder=0)
    ax.axvspan(LIGHT_OFF_S, TRIAL_DURATION_S, color="gray", alpha=0.2, zorder=0)


# ---- Panel A / C: Representative traces ----

def plot_representative_traces(
    raw_traces: np.ndarray,
    title: str = "",
    interval: int = REPR_INTERVAL,
    y_lim: Tuple[float, float] = REPR_Y_LIM,
    save_path: Optional[Path] = None,
):
    """Plot representative spike-rate trace evolution over time.

    raw_traces: (n_bins, n_trials) for one unit -- transposed from step_responses
    """
    n_bins, n_trials = raw_traces.shape
    total_sub = n_trials // interval

    fig, axs = plt.subplots(1, total_sub, figsize=(total_sub, total_sub / 5))
    if total_sub == 1:
        axs = [axs]
    x_sec = np.arange(n_bins) / BIN_RATE_HZ

    for i in range(total_sub):
        ax = axs[i]
        mean_trace = raw_traces[:, i * interval:(i + 1) * interval].mean(axis=1)
        ax.plot(x_sec, mean_trace, color="black", linewidth=0.7)
        ax.set_ylim(*y_lim)

        _add_light_shading(ax)

        if i == 0:
            ax.set_ylabel("Response (Hz)")
        else:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)
        ax.set_xlabel("Time (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        time_label = i * interval * default_config.stimulus_interval_s
        ax.text(1.2, y_lim[1] - 10, f"{int(time_label)} s",
                ha="center", va="top", fontsize=9)

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", save_path)
    plt.close(fig)
    return fig


# ---- Panel B / D: Population time course (two overlaid curves) ----

def plot_population_timecourse(
    ctrl_features: np.ndarray,
    low_features: np.ndarray,
    title: str = "",
    step: int = POP_STEP,
    stimulus_interval: int = 10,
    y_lim: Tuple[float, float] = POP_Y_LIM,
    normalize_mode: str = "max",
    save_path: Optional[Path] = None,
    fig_size: Tuple[float, float] = (12, 4),
):
    """Two-curve population time course (control vs low glucose).

    Exactly replicates ax_plot_compare_feature_list from legacy code.

    ctrl_features:  (n_ctrl_units, n_trials) -- already smoothed + per-unit normed
    low_features:   (n_low_units,  n_trials)
    """
    def _norm(arr, mode):
        if arr.shape[0] == 0:
            return arr
        m = np.nanmean(arr, axis=0)
        if mode == "max":
            denom = np.nanmax(m) if np.nanmax(m) > 0 else 1.0
        elif mode == "first":
            denom = np.nanmean(m[:5]) if len(m) >= 5 else (np.nanmean(m) if np.nanmean(m) > 0 else 1.0)
        else:
            raise ValueError(f"unsupported normalize_mode: {mode}")
        return arr / denom if denom > 0 else arr

    ctrl = _norm(ctrl_features, normalize_mode)
    low = _norm(low_features, normalize_mode)

    has_ctrl = ctrl.shape[0] > 0
    has_low = low.shape[0] > 0

    fig, ax = plt.subplots(figsize=fig_size)

    if has_ctrl:
        n_valid_ctrl = np.sum(~np.isnan(ctrl), axis=0)
        mean_ctrl = np.nanmean(ctrl, axis=0)
        sem_ctrl = np.nanstd(ctrl, axis=0) / np.sqrt(np.maximum(n_valid_ctrl, 1))
        idx_ctrl = np.arange(0, len(mean_ctrl), step)
        x_ctrl = idx_ctrl * stimulus_interval
        ax.errorbar(x_ctrl, mean_ctrl[idx_ctrl], yerr=sem_ctrl[idx_ctrl],
                    fmt="o-", capsize=3, color="black", label="Control")

    if has_low:
        n_valid_low = np.sum(~np.isnan(low), axis=0)
        mean_low = np.nanmean(low, axis=0)
        sem_low = np.nanstd(low, axis=0) / np.sqrt(np.maximum(n_valid_low, 1))
        idx_low = np.arange(0, len(mean_low), step)
        x_low = idx_low * stimulus_interval
        ax.errorbar(x_low, mean_low[idx_low], yerr=sem_low[idx_low],
                    fmt="o-", capsize=3, color="red", label="Low Glucose")

    if has_ctrl and has_low:
        common_len = min(len(mean_ctrl), len(mean_low))
        idx_common = np.arange(0, common_len, step)
        for i in idx_common:
            c_vals = ctrl[:, i][~np.isnan(ctrl[:, i])]
            l_vals = low[:, i][~np.isnan(low[:, i])]
            if len(c_vals) > 1 and len(l_vals) > 1:
                t, p = ttest_ind(c_vals, l_vals, equal_var=False)
            else:
                p = 1.0
            if p < 0.05:
                j = i * stimulus_interval
                y_star = max(mean_ctrl[i] + sem_ctrl[i],
                             mean_low[i] + sem_low[i]) + 0.02
                ax.text(j, y_star, "*", ha="center", va="bottom",
                        fontsize=12, color="k")

    n_ctrl = ctrl.shape[0] if has_ctrl else 0
    n_low = low.shape[0] if has_low else 0
    ax.set_xlabel("Time (sec)", fontsize=12)
    ax.set_ylabel("Normalized Response", fontsize=12)
    ax.set_ylim(*y_lim)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(
        f"{title} (Control: n = {n_ctrl}, "
        f"Low Glucose: n = {n_low})"
    )
    ax.legend(fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", save_path)
    plt.close(fig)
    return fig


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true",
                        help="Use cached extracted_features.npz")
    args = parser.parse_args()

    config = default_config
    config.paper_figure_dir.mkdir(parents=True, exist_ok=True)

    npz_path = config.output_dir / "extracted_features.npz"
    raw_npz_path = config.output_dir / "extracted_raw_traces.npz"

    if not npz_path.exists() or (not args.skip_extract):
        logger.info("Running feature extraction via run_analysis ...")
        _local_ra_spec = importlib.util.spec_from_file_location(
            "low_glucose_alone_run", _THIS_DIR / "run_analysis.py")
        _local_ra = importlib.util.module_from_spec(_local_ra_spec)
        _local_ra_spec.loader.exec_module(_local_ra)
        extract_all_features = _local_ra.extract_all_features
        convert_all_recordings = _local_ra.convert_all_recordings
        align_all_pairs = _local_ra.align_all_pairs
        convert_all_recordings(config)
        align_all_pairs(config)
        results = extract_all_features(config)

        save_dict = {}
        raw_save = {}
        for key, val in results.items():
            save_dict[key] = val["features"]
            for i, arr in enumerate(val["raw"]):
                raw_save[f"{key}_raw_{i}"] = np.array(arr)
        np.savez(npz_path, **save_dict)
        np.savez(raw_npz_path, **raw_save)
    else:
        logger.info("Loading cached features from %s", npz_path)
        loaded = np.load(npz_path)
        results = {}
        for key in ["off_control", "off_low", "on_control", "on_low"]:
            results[key] = {"features": loaded[key]}

        if raw_npz_path.exists():
            raw_loaded = np.load(raw_npz_path, allow_pickle=True)
            for key in ["off_control", "off_low", "on_control", "on_low"]:
                prefix = f"{key}_raw_"
                raw_list = []
                i = 0
                while f"{prefix}{i}" in raw_loaded:
                    raw_list.append(raw_loaded[f"{prefix}{i}"])
                    i += 1
                results[key]["raw"] = raw_list

    # --- Panel B: OFF population ---
    off_ctrl = results["off_control"]["features"]
    off_low = results["off_low"]["features"]
    logger.info("OFF control: %s, OFF low: %s", off_ctrl.shape, off_low.shape)

    plot_population_timecourse(
        off_ctrl, off_low,
        title="OFF Response",
        save_path=config.paper_figure_dir / "Figure_X_B_OFF_population.pdf",
    )

    # --- Panel D: ON population ---
    on_ctrl = results["on_control"]["features"]
    on_low = results["on_low"]["features"]
    logger.info("ON control: %s, ON low: %s", on_ctrl.shape, on_low.shape)

    plot_population_timecourse(
        on_ctrl, on_low,
        title="ON Response",
        save_path=config.paper_figure_dir / "Figure_X_D_ON_population.pdf",
    )

    # --- Panel A/C: Representative traces ---
    for cell_type, raw_key_ctrl, raw_key_low, panel_label in [
        ("OFF", "off_control", "off_low", "A"),
        ("ON", "on_control", "on_low", "C"),
    ]:
        ctrl_raw = results.get(raw_key_ctrl, {}).get("raw", [])
        low_raw = results.get(raw_key_low, {}).get("raw", [])

        if ctrl_raw:
            idx_ctrl = 0
            trace_ctrl = np.array(ctrl_raw[idx_ctrl]).T
            plot_representative_traces(
                trace_ctrl,
                title=f"{cell_type} RGC Control",
                save_path=config.paper_figure_dir
                / f"Figure_X_{panel_label}_{cell_type}_control_representative.pdf",
            )

        if low_raw:
            idx_low = min(31 if cell_type == "OFF" else 8, len(low_raw) - 1)
            trace_low = np.array(low_raw[idx_low]).T
            plot_representative_traces(
                trace_low,
                title=f"{cell_type} RGC Low Glucose",
                save_path=config.paper_figure_dir
                / f"Figure_X_{panel_label}_{cell_type}_low_representative.pdf",
            )

    logger.info("All panels saved to %s", config.paper_figure_dir)


if __name__ == "__main__":
    main()
