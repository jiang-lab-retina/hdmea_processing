#!/usr/bin/env python
"""
Plot bias-corrected STA for natural scene movies.

For each unit, loads the raw STA (60, 20, 20) from H5, subtracts the
uniform STA (one-spike-per-frame baseline that captures static luminance
bias), and saves a figure with 4 rows x 12 columns:
  - Rows 1-3: corrected STA for each movie (stu48, superfruit, dway)
  - Row 4: weighted average across movies (weight = spike fraction)

Frames shown at every 5th index from -60 to -5.

Usage:
    python plot_corrected_sta.py
    python plot_corrected_sta.py --h5 path/to/file.h5
    python plot_corrected_sta.py --start 0 --end 10
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_H5 = SCRIPT_DIR.parent / "export" / "2025.02.04-13.22.13-Rec.h5"
OUTPUT_DIR = SCRIPT_DIR / "sta_figures"
STIMULI_DIR = Path(r"M:\Python_Project\Design_Stimulation_Pattern\Data\Stimulations")

MOVIES = ["stu48_final", "superfruit_final", "dway_final"]
COVER_RANGE = (-60, 0)
FRAME_STEP = 5
DOWNSAMPLE_TARGET = 20


def downsample_movie(movie: np.ndarray, target: int = DOWNSAMPLE_TARGET) -> np.ndarray:
    n, h, w = movie.shape
    bh = h // target
    bw = w // target
    return (
        movie[:, : target * bh, : target * bw]
        .reshape(n, target, bh, target, bw)
        .mean(axis=(2, 4))
        .astype(np.float32)
    )


def compute_uniform_sta(movie: np.ndarray, cover_range=COVER_RANGE) -> np.ndarray:
    """
    Uniform STA: expected STA assuming one spike at every valid frame.

    uniform_sta[k] = mean of movie[i + (k + cover_range[0])]
                     for all valid spike frames i in [|cover_range[0]|, movie_length + cover_range[1])

    Returns shape (window_length, H, W).
    """
    window_length = cover_range[1] - cover_range[0]
    n_frames = movie.shape[0]
    n_valid = n_frames - window_length

    uniform = np.empty((window_length, movie.shape[1], movie.shape[2]), dtype=np.float32)
    for k in range(window_length):
        uniform[k] = movie[k : k + n_valid].mean(axis=0)
    return uniform


def load_uniform_stas() -> Dict[str, np.ndarray]:
    logger.info("Computing uniform STAs (movie bias) for %d movies...", len(MOVIES))
    result = {}
    for name in MOVIES:
        npy_path = STIMULI_DIR / f"{name}.npy"
        if not npy_path.exists():
            logger.warning("Stimulus not found: %s", npy_path)
            continue
        raw = np.load(npy_path)
        small = downsample_movie(raw)
        del raw
        uniform = compute_uniform_sta(small)
        result[name] = uniform
        logger.info("  %s: movie %s -> uniform STA %s", name, small.shape, uniform.shape)
    return result


def _plot_sta_row(
    fig, gs, row: int, n_cols: int, frame_indices: List[int],
    sta_2d: np.ndarray, label: str, is_first_row: bool,
):
    """Plot one row of STA subplots with a shared colorbar."""
    vmax = np.nanmax(np.abs(sta_2d))
    if vmax == 0:
        vmax = 1.0

    for col, fi in enumerate(frame_indices):
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(
            sta_2d[fi],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])

        if is_first_row:
            ax.set_title(f"{COVER_RANGE[0] + fi}", fontsize=8)
        if col == 0:
            ax.set_ylabel(label, fontsize=8)

    cax = fig.add_subplot(gs[row, n_cols])
    plt.colorbar(im, cax=cax)


def plot_unit(
    unit_id: str,
    sta_dict: Dict[str, np.ndarray],
    spike_counts: Dict[str, int],
    uniform_dict: Dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """Plot corrected STA for one unit: 3 movie rows + 1 weighted-average row."""
    frame_indices = list(range(0, -COVER_RANGE[0], FRAME_STEP))
    n_cols = len(frame_indices)
    n_rows = 4  # 3 movies + weighted average

    fig = plt.figure(figsize=(n_cols * 1.6, n_rows * 2.0 + 0.6))
    gs = GridSpec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05],
                  wspace=0.08, hspace=0.35)

    # Compute corrected STAs and collect for weighted average
    corrected_list: List[Tuple[np.ndarray, int]] = []

    for row, movie_name in enumerate(MOVIES):
        raw_sta = sta_dict.get(movie_name)
        uniform = uniform_dict.get(movie_name)
        if raw_sta is None or uniform is None:
            continue

        corrected = raw_sta - uniform
        n_spikes = spike_counts.get(movie_name, 0)
        corrected_list.append((corrected, n_spikes))

        _plot_sta_row(
            fig, gs, row, n_cols, frame_indices,
            corrected, movie_name.replace("_final", ""), is_first_row=(row == 0),
        )

    # Row 4: spike-weighted average across movies
    total_spikes = sum(n for _, n in corrected_list)
    if total_spikes > 0 and corrected_list:
        weighted = sum(c * (n / total_spikes) for c, n in corrected_list)
        _plot_sta_row(
            fig, gs, 3, n_cols, frame_indices,
            weighted, "weighted avg", is_first_row=False,
        )

    fig.suptitle(unit_id, fontsize=10, y=0.98)

    out_path = output_dir / f"{unit_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Plot bias-corrected natural scene STA")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5, help="Input H5 file")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start unit index")
    parser.add_argument("--end", type=int, default=None, help="End unit index (exclusive)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    uniform_dict = load_uniform_stas()
    if not uniform_dict:
        logger.error("No uniform STAs computed, exiting")
        sys.exit(1)

    logger.info("Opening %s", args.h5)
    with h5py.File(str(args.h5), "r") as f:
        unit_ids = sorted(f["units"].keys())
        subset = unit_ids[args.start : args.end]
        logger.info("Processing %d / %d units", len(subset), len(unit_ids))

        for i, uid in enumerate(subset):
            sta_dict: Dict[str, Optional[np.ndarray]] = {}
            spike_counts: Dict[str, int] = {}

            feat = f["units"][uid].get("features", {})
            for movie_name in MOVIES:
                key = f"sta_{movie_name}"
                if key in feat:
                    sta_dict[movie_name] = feat[key]["data"][...]
                    meta = feat[key].get("metadata", {})
                    if "n_spikes_used" in meta:
                        spike_counts[movie_name] = int(meta["n_spikes_used"][()])
                else:
                    logger.warning("  %s missing %s", uid, key)

            if not sta_dict:
                continue

            out = plot_unit(uid, sta_dict, spike_counts, uniform_dict, args.output)
            if (i + 1) % 50 == 0 or (i + 1) == len(subset):
                logger.info("  [%d/%d] saved %s", i + 1, len(subset), out.name)

    logger.info("Done. Figures saved to %s", args.output)


if __name__ == "__main__":
    main()
