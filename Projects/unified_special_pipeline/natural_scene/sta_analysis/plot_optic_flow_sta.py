#!/usr/bin/env python
"""
Plot optic-flow STA for natural scene movies.

Converts each movie to optic flow (Vx, Vy) via Farneback, computes STA on
both components using spike data from the H5, subtracts uniform bias, then
spike-weights across movies.  Each unit gets a 7-row figure:
  Row 1: normal pixel STA (RdBu_r, bias-corrected weighted avg)
  Row 2: flow amplitude  (viridis)
  Row 3: flow direction  (hsv, circular)
  Row 4: flow arrows     (quiver, colored by amplitude)
  Row 5: flow spatial gradient  (inferno)
  Row 6: HSV composite (hue=direction, value=amplitude)
  Row 7: temporal derivative of flow amplitude (RdBu_r)
All rows use a shared color range across columns.

Usage:
    python plot_optic_flow_sta.py
    python plot_optic_flow_sta.py --start 0 --end 3
"""

import argparse
import logging
import sys
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from hdmea.io.section_time import convert_sample_index_to_frame, PRE_MARGIN_FRAME_NUM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DEFAULT_H5 = SCRIPT_DIR.parent / "export" / "2025.02.04-13.22.13-Rec.h5"
OUTPUT_DIR = SCRIPT_DIR / "optic_flow_figures"
STIMULI_DIR = Path(r"M:\Python_Project\Design_Stimulation_Pattern\Data\Stimulations")

MOVIES = ["stu48_final", "superfruit_final", "dway_final"]
COVER_RANGE = (-120, 0)
FRAME_STEP = 5
DOWNSAMPLE_TARGET = 100


# ============================================================================
# Movie loading & optic flow
# ============================================================================


def _downsample_movie(movie: np.ndarray, target: int = DOWNSAMPLE_TARGET) -> np.ndarray:
    n, h, w = movie.shape
    bh, bw = h // target, w // target
    reshaped = movie[:, :target * bh, :target * bw].reshape(n, target, bh, target, bw)
    return np.median(reshaped, axis=(2, 4)).astype(np.float32)


def _compute_optical_flow(movie_f32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Farneback dense optical flow on a float32 movie (N, H, W).
    Returns (flow_vx, flow_vy), each shape (N-1, H, W).
    """
    n = movie_f32.shape[0]
    h, w = movie_f32.shape[1], movie_f32.shape[2]
    vx = np.empty((n - 1, h, w), dtype=np.float32)
    vy = np.empty((n - 1, h, w), dtype=np.float32)

    prev = movie_f32[0].astype(np.uint8)
    for i in range(1, n):
        curr = movie_f32[i].astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=5,
            iterations=3, poly_n=5, poly_sigma=1.1, flags=0,
        )
        vx[i - 1] = flow[..., 0]
        vy[i - 1] = flow[..., 1]
        prev = curr

    return vx, vy


def load_flow_movies() -> Tuple[
    Dict[str, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]
]:
    """Load, downsample, compute flow, move to GPU.

    Returns (flow_dict, luminance_dict) where:
      flow_dict = {name: (vx_gpu, vy_gpu)}
      luminance_dict = {name: movie_gpu}
    """
    flow_result = {}
    lum_result = {}
    for name in MOVIES:
        npy_path = STIMULI_DIR / f"{name}.npy"
        if not npy_path.exists():
            logger.warning("Stimulus not found: %s", npy_path)
            continue
        t0 = _time.time()
        raw = np.load(npy_path)
        small = _downsample_movie(raw)
        del raw
        vx, vy = _compute_optical_flow(small)
        flow_result[name] = (
            torch.from_numpy(vx).to(DEVICE),
            torch.from_numpy(vy).to(DEVICE),
        )
        lum_result[name] = torch.from_numpy(small).to(DEVICE)
        logger.info("  %s: flow shape %s, %.1fs", name, vx.shape, _time.time() - t0)
    return flow_result, lum_result


# ============================================================================
# Uniform flow STA (bias)
# ============================================================================


def _compute_uniform_flow_sta(
    flow: torch.Tensor, cover_range: Tuple[int, int] = COVER_RANGE,
) -> torch.Tensor:
    """Uniform STA on a single component tensor (N, H, W) -> (window, H, W) on GPU."""
    wlen = cover_range[1] - cover_range[0]
    n = flow.shape[0]
    n_valid = n - wlen
    cumsum = torch.cumsum(flow.double(), dim=0)
    parts = cumsum[wlen:] - cumsum[:n_valid]
    uniform = torch.empty((wlen, flow.shape[1], flow.shape[2]),
                          dtype=torch.float32, device=flow.device)
    uniform[0] = (cumsum[n_valid - 1] / n_valid).float()
    for k in range(1, wlen):
        uniform[k] = ((cumsum[k + n_valid - 1] - cumsum[k - 1]) / n_valid).float()
    return uniform


# ============================================================================
# STA computation from spike data
# ============================================================================


def _extract_valid_spikes(
    spike_times: np.ndarray,
    section_time: np.ndarray,
    frame_timestamps: np.ndarray,
    data_len: int,
    cover_range: Tuple[int, int],
    frame_offset: int = 0,
) -> Tuple[Optional[np.ndarray], int]:
    """Extract valid spike frame indices relative to each repeat's start.

    Args:
        frame_offset: extra offset to subtract (use -1 for flow indexing).
    Returns:
        (valid_spikes_array or None, n_used)
    """
    n_repeats = section_time.shape[0]
    all_rel: List[np.ndarray] = []
    for r in range(n_repeats):
        s0, s1 = int(section_time[r, 0]), int(section_time[r, 1])
        mask = (spike_times >= s0) & (spike_times <= s1)
        if mask.sum() == 0:
            continue
        spk = spike_times[mask]
        abs_frames = convert_sample_index_to_frame(spk, frame_timestamps)
        start_frame = int(convert_sample_index_to_frame(
            np.array([s0]), frame_timestamps
        )[0]) + PRE_MARGIN_FRAME_NUM
        all_rel.append(abs_frames - start_frame + frame_offset)
    if not all_rel:
        return None, 0
    pooled = np.concatenate(all_rel).astype(np.int64)
    valid = (pooled + cover_range[0] >= 0) & (pooled + cover_range[1] <= data_len)
    vs = pooled[valid]
    return (vs, len(vs)) if len(vs) > 0 else (None, 0)


def _gpu_sta(
    data: torch.Tensor,
    valid_spikes: np.ndarray,
    cover_range: Tuple[int, int],
) -> torch.Tensor:
    """Compute STA on GPU using chunked fancy indexing.

    data: (N, H, W) tensor on GPU.
    valid_spikes: 1-D int array of spike frame indices (CPU).
    Returns: (wlen, H, W) float32 tensor on GPU.
    """
    wlen = cover_range[1] - cover_range[0]
    offsets = torch.arange(cover_range[0], cover_range[1], device=data.device)
    spk_t = torch.from_numpy(valid_spikes).to(data.device)
    n_used = spk_t.shape[0]
    bytes_per_spike = wlen * data.shape[1] * data.shape[2] * 4
    chunk = max(1, int(200e6) // bytes_per_spike)
    sta_acc = torch.zeros((wlen, data.shape[1], data.shape[2]),
                          dtype=torch.float32, device=data.device)
    for i in range(0, n_used, chunk):
        batch = spk_t[i : i + chunk]
        indices = batch.unsqueeze(1) + offsets.unsqueeze(0)
        sta_acc += data[indices].sum(dim=0)
    return sta_acc / n_used


def _compute_flow_sta_for_unit(
    spike_times: np.ndarray,
    section_time: np.ndarray,
    frame_timestamps: np.ndarray,
    flow: torch.Tensor,
    cover_range: Tuple[int, int],
) -> Tuple[Optional[torch.Tensor], int]:
    """Compute STA on a flow component (GPU tensor) for one unit."""
    vs, n_used = _extract_valid_spikes(
        spike_times, section_time, frame_timestamps,
        flow.shape[0], cover_range, frame_offset=-1,
    )
    if vs is None:
        return None, 0
    return _gpu_sta(flow, vs, cover_range), n_used


def _compute_pixel_sta_for_unit(
    spike_times: np.ndarray,
    section_time: np.ndarray,
    frame_timestamps: np.ndarray,
    movie: torch.Tensor,
    cover_range: Tuple[int, int],
) -> Tuple[Optional[torch.Tensor], int]:
    """Compute STA on a luminance movie (GPU tensor) for one unit."""
    vs, n_used = _extract_valid_spikes(
        spike_times, section_time, frame_timestamps,
        movie.shape[0], cover_range, frame_offset=0,
    )
    if vs is None:
        return None, 0
    return _gpu_sta(movie, vs, cover_range), n_used


# ============================================================================
# HSV composite helper
# ============================================================================


def _flow_to_hsv_image(vx_frame: np.ndarray, vy_frame: np.ndarray,
                       global_mag_max: float = 0.0) -> np.ndarray:
    """Convert a single (H,W) Vx,Vy frame to an RGB image via HSV encoding."""
    angle = np.arctan2(vy_frame, vx_frame)
    mag = np.sqrt(vx_frame ** 2 + vy_frame ** 2)

    hue = (angle + np.pi) / (2 * np.pi)
    sat = np.ones_like(hue)
    norm = global_mag_max if global_mag_max > 0 else (mag.max() + 1e-8)
    val = np.clip(mag / norm, 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb


# ============================================================================
# Plotting
# ============================================================================


def _bin_mean(arr: np.ndarray, step: int) -> List[np.ndarray]:
    """Split arr (T, H, W) into consecutive bins of *step* frames and return their means."""
    n = arr.shape[0]
    return [arr[i : i + step].mean(axis=0) for i in range(0, n - step + 1, step)]


def _compute_flow_gradient(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Spatial gradient magnitude of a 2-D vector field (H, W).

    grad = sqrt( (dvx/dx)^2 + (dvx/dy)^2 + (dvy/dx)^2 + (dvy/dy)^2 )
    """
    dvx_dy, dvx_dx = np.gradient(vx)
    dvy_dy, dvy_dx = np.gradient(vy)
    return np.sqrt(dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2)


def _compute_divergence(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Divergence of a 2-D vector field: dvx/dx + dvy/dy.

    Smooths with uniform_filter(size=DOWNSAMPLE_TARGET//10) before
    computing gradients to capture structure at 1/10 of the window.
    """
    k = max(DOWNSAMPLE_TARGET // 10, 1)
    vx_s = uniform_filter(vx.astype(np.float64), size=k)
    vy_s = uniform_filter(vy.astype(np.float64), size=k)
    _, dvx_dx = np.gradient(vx_s)
    dvy_dy, _ = np.gradient(vy_s)
    return (dvx_dx + dvy_dy).astype(np.float32)


def _compute_curl(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Curl (z-component) of a 2-D vector field: dvy/dx - dvx/dy.

    Smooths with uniform_filter(size=DOWNSAMPLE_TARGET//10) before
    computing gradients to capture structure at 1/10 of the window.
    """
    k = max(DOWNSAMPLE_TARGET // 10, 1)
    vx_s = uniform_filter(vx.astype(np.float64), size=k)
    vy_s = uniform_filter(vy.astype(np.float64), size=k)
    dvx_dy, _ = np.gradient(vx_s)
    _, dvy_dx = np.gradient(vy_s)
    return (dvy_dx - dvx_dy).astype(np.float32)


def plot_unit(
    unit_id: str,
    pixel_sta: np.ndarray,
    vx_sta: np.ndarray,
    vy_sta: np.ndarray,
    output_dir: Path,
) -> Path:
    """
    9-row figure for one unit (weighted-average across movies).
    Each column is the mean of FRAME_STEP consecutive frames.
    All rows use a shared color range across the row.
    """
    wlen = -COVER_RANGE[0]
    n_cols = wlen // FRAME_STEP
    n_rows = 9

    pix_bins = _bin_mean(pixel_sta, FRAME_STEP)
    vx_bins = _bin_mean(vx_sta, FRAME_STEP)
    vy_bins = _bin_mean(vy_sta, FRAME_STEP)

    amp_full = np.sqrt(vx_sta ** 2 + vy_sta ** 2)
    amp_bins = _bin_mean(amp_full, FRAME_STEP)

    dir_full = np.arctan2(vy_sta, vx_sta)
    dir_bins = _bin_mean(dir_full, FRAME_STEP)

    grad_bins = [_compute_flow_gradient(vx_bins[c], vy_bins[c]) for c in range(n_cols)]
    div_bins = [_compute_divergence(vx_bins[c], vy_bins[c]) for c in range(n_cols)]
    curl_bins = [_compute_curl(vx_bins[c], vy_bins[c]) for c in range(n_cols)]

    d_amp_full = np.zeros_like(amp_full)
    d_amp_full[1:] = amp_full[1:] - amp_full[:-1]
    d_amp_bins = _bin_mean(d_amp_full, FRAME_STEP)

    # Shared color limits per row
    pix_vmax = max(np.nanmax(np.abs(b)) for b in pix_bins) if pix_bins else 1.0
    if pix_vmax == 0:
        pix_vmax = 1.0
    amp_max = max(np.nanmax(b) for b in amp_bins) if amp_bins else 1.0
    if amp_max == 0:
        amp_max = 1.0
    grad_max = max(np.nanmax(b) for b in grad_bins) if grad_bins else 1.0
    if grad_max == 0:
        grad_max = 1.0
    div_max = max(np.nanmax(np.abs(b)) for b in div_bins) if div_bins else 1.0
    if div_max == 0:
        div_max = 1.0
    curl_max = max(np.nanmax(np.abs(b)) for b in curl_bins) if curl_bins else 1.0
    if curl_max == 0:
        curl_max = 1.0
    d_max = max(np.nanmax(np.abs(b)) for b in d_amp_bins) if d_amp_bins else 1.0
    if d_max == 0:
        d_max = 1.0

    fig = plt.figure(figsize=(n_cols * 1.6, n_rows * 2.0 + 0.8))
    gs = GridSpec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05],
                  wspace=0.08, hspace=0.40)

    row_labels = ["pixel STA", "amplitude", "direction", "arrows",
                  "flow grad", "divergence", "curl", "HSV", "d(amp)/dt"]

    h, w = vx_sta.shape[1], vx_sta.shape[2]
    Y, X = np.mgrid[0:h, 0:w]

    for col in range(n_cols):
        bin_start = col * FRAME_STEP
        bin_end = bin_start + FRAME_STEP - 1
        time_label = f"{COVER_RANGE[0] + bin_start}:{COVER_RANGE[0] + bin_end}"
        vx_b = vx_bins[col]
        vy_b = vy_bins[col]

        # Row 0: pixel STA
        ax = fig.add_subplot(gs[0, col])
        im_pix = ax.imshow(pix_bins[col], cmap="RdBu_r",
                           vmin=-pix_vmax, vmax=pix_vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(time_label, fontsize=7)
        if col == 0:
            ax.set_ylabel(row_labels[0], fontsize=8)

        # Row 1: amplitude
        ax = fig.add_subplot(gs[1, col])
        im_amp = ax.imshow(amp_bins[col], cmap="viridis",
                           vmin=0, vmax=amp_max, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[1], fontsize=8)

        # Row 2: direction
        ax = fig.add_subplot(gs[2, col])
        im_dir = ax.imshow(dir_bins[col], cmap="hsv",
                           vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[2], fontsize=8)

        # Row 3: arrows
        ax = fig.add_subplot(gs[3, col])
        amp_norm = amp_bins[col] / amp_max
        ax.set_facecolor("#f0f0f0")
        ax.quiver(
            X, Y, vx_b, -vy_b,
            amp_norm, cmap="viridis", scale=None,
            clim=[0, 1], headwidth=4, headlength=4,
        )
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel(row_labels[3], fontsize=8)

        # Row 4: flow spatial gradient
        ax = fig.add_subplot(gs[4, col])
        im_grad = ax.imshow(grad_bins[col], cmap="inferno",
                            vmin=0, vmax=grad_max, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[4], fontsize=8)

        # Row 5: divergence
        ax = fig.add_subplot(gs[5, col])
        im_div = ax.imshow(div_bins[col], cmap="RdBu_r",
                           vmin=-div_max, vmax=div_max, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[5], fontsize=8)

        # Row 6: curl
        ax = fig.add_subplot(gs[6, col])
        im_curl = ax.imshow(curl_bins[col], cmap="PiYG",
                            vmin=-curl_max, vmax=curl_max, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[6], fontsize=8)

        # Row 7: HSV composite
        ax = fig.add_subplot(gs[7, col])
        rgb = _flow_to_hsv_image(vx_b, vy_b, global_mag_max=amp_max)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[7], fontsize=8)

        # Row 8: temporal derivative of amplitude
        ax = fig.add_subplot(gs[8, col])
        im_damp = ax.imshow(d_amp_bins[col], cmap="RdBu_r",
                            vmin=-d_max, vmax=d_max, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(row_labels[8], fontsize=8)

    # Colorbars
    cax = fig.add_subplot(gs[0, n_cols]); plt.colorbar(im_pix, cax=cax)
    cax = fig.add_subplot(gs[1, n_cols]); plt.colorbar(im_amp, cax=cax)
    cax = fig.add_subplot(gs[2, n_cols]); plt.colorbar(im_dir, cax=cax)
    fig.add_subplot(gs[3, n_cols]).axis("off")
    cax = fig.add_subplot(gs[4, n_cols]); plt.colorbar(im_grad, cax=cax)
    cax = fig.add_subplot(gs[5, n_cols]); plt.colorbar(im_div, cax=cax)
    cax = fig.add_subplot(gs[6, n_cols]); plt.colorbar(im_curl, cax=cax)
    fig.add_subplot(gs[7, n_cols]).axis("off")
    cax = fig.add_subplot(gs[8, n_cols]); plt.colorbar(im_damp, cax=cax)

    fig.suptitle(unit_id, fontsize=10, y=0.99)

    out_path = output_dir / f"{unit_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Plot optic-flow STA for natural scene movies")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5, help="Input H5 file")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Load movies, compute flow, keep luminance for pixel-STA bias
    logger.info("Computing optic flow for %d movies...", len(MOVIES))
    flow_movies, lum_movies = load_flow_movies()
    if not flow_movies:
        logger.error("No flow movies computed"); sys.exit(1)

    # 2. Compute uniform STAs (bias) on GPU
    logger.info("Computing uniform STAs (flow + luminance)...")
    uniform_vx: Dict[str, torch.Tensor] = {}
    uniform_vy: Dict[str, torch.Tensor] = {}
    uniform_lum: Dict[str, torch.Tensor] = {}
    for name, (vx, vy) in flow_movies.items():
        uniform_vx[name] = _compute_uniform_flow_sta(vx)
        uniform_vy[name] = _compute_uniform_flow_sta(vy)
        uniform_lum[name] = _compute_uniform_flow_sta(lum_movies[name])
        logger.info("  %s uniform done", name)

    # 3. Process units
    logger.info("Opening %s  (GPU: %s)", args.h5, DEVICE)
    wlen = COVER_RANGE[1] - COVER_RANGE[0]
    with h5py.File(str(args.h5), "r") as f:
        frame_timestamps = f["stimulus"]["frame_times"]["default"][...]

        section_times: Dict[str, np.ndarray] = {}
        for name in MOVIES:
            section_times[name] = f["stimulus"]["section_time"][name][...]

        unit_ids = sorted(f["units"].keys())
        subset = unit_ids[args.start : args.end]
        logger.info("Processing %d / %d units", len(subset), len(unit_ids))

        for i, uid in enumerate(subset):
            spike_times = f["units"][uid]["spike_times"][...]

            sta_vx_list: List[Tuple[torch.Tensor, int]] = []
            sta_vy_list: List[Tuple[torch.Tensor, int]] = []
            sta_pix_list: List[Tuple[torch.Tensor, int]] = []

            for name in MOVIES:
                if name not in flow_movies:
                    continue
                vx_flow, vy_flow = flow_movies[name]

                sta_vx, n_vx = _compute_flow_sta_for_unit(
                    spike_times, section_times[name], frame_timestamps,
                    vx_flow, COVER_RANGE,
                )
                sta_vy, n_vy = _compute_flow_sta_for_unit(
                    spike_times, section_times[name], frame_timestamps,
                    vy_flow, COVER_RANGE,
                )

                if sta_vx is None or sta_vy is None:
                    continue

                corr_vx = sta_vx - uniform_vx[name]
                corr_vy = sta_vy - uniform_vy[name]
                n_spikes = n_vx

                sta_vx_list.append((corr_vx, n_spikes))
                sta_vy_list.append((corr_vy, n_spikes))

                raw_pix, n_pix = _compute_pixel_sta_for_unit(
                    spike_times, section_times[name], frame_timestamps,
                    lum_movies[name], COVER_RANGE,
                )
                if raw_pix is not None:
                    corr_pix = raw_pix - uniform_lum[name]
                    sta_pix_list.append((corr_pix, n_pix))

            if not sta_vx_list:
                continue

            total_flow = sum(n for _, n in sta_vx_list)
            if total_flow == 0:
                continue
            w_vx = sum(c * (n / total_flow) for c, n in sta_vx_list)
            w_vy = sum(c * (n / total_flow) for c, n in sta_vy_list)

            if sta_pix_list:
                total_pix = sum(n for _, n in sta_pix_list)
                w_pix = sum(c * (n / total_pix) for c, n in sta_pix_list)
            else:
                w_pix = torch.zeros((wlen, DOWNSAMPLE_TARGET, DOWNSAMPLE_TARGET),
                                    dtype=torch.float32, device=DEVICE)

            out = plot_unit(
                uid,
                w_pix.cpu().numpy(),
                w_vx.cpu().numpy(),
                w_vy.cpu().numpy(),
                args.output,
            )
            if (i + 1) % 10 == 0 or (i + 1) == len(subset):
                logger.info("  [%d/%d] saved %s", i + 1, len(subset), out.name)

    logger.info("Done. Figures in %s", args.output)


if __name__ == "__main__":
    main()
