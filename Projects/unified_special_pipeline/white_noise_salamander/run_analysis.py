"""
White Noise Salamander STA Pipeline

Computes spike-triggered averages (STA) for salamander retinal ganglion cells
stimulated with dense white noise, and saves per-unit visualizations.

Two approaches are available:

  Default  -- Uses hdmea.features.compute_sta via a PipelineSession.
              Constructs a session with the loaded data and synthetic
              section_time, then delegates STA computation to the standard
              compute_sta function.  HDF5 is written by session.save().

  --legacy -- Direct per-unit STA using _compute_sta_for_unit with manual
              spike-to-frame conversion and custom HDF5 saving.

Usage (from the white_noise_salamander directory):
    python run_analysis.py                      # default (session)
    python run_analysis.py --legacy             # legacy direct approach
    python run_analysis.py --overwrite
    python run_analysis.py --all --overwrite
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# sys.path -- make sibling packages importable regardless of how we are run
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_USP_DIR = _THIS_DIR.parent  # unified_special_pipeline
_PROJECT_ROOT = _USP_DIR.parent.parent  # Data_Processing_2027
_SRC_DIR = _PROJECT_ROOT / "src"

_RF_MEASURE_DIR = _PROJECT_ROOT / "Projects" / "rf_sta_measure"

for _p in (str(_THIS_DIR), str(_USP_DIR), str(_SRC_DIR), str(_RF_MEASURE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step_change_analysis.data_loader import load_cmcr_cmtr_data

from hdmea.pipeline.runner import get_frame_timestamps
from hdmea.pipeline.session import PipelineSession
from hdmea.io.section_time import convert_sample_index_to_frame, PRE_MARGIN_FRAME_NUM
from hdmea.features.sta import compute_sta, _compute_sta_for_unit

from specific_config import (
    STAPipelineConfig,
    RecordingInfo,
    GSheetRecordingInfo,
    TEST_FILES,
    default_config,
    discover_recordings,
    discover_recordings_from_gsheet,
    get_output_figures_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STIMULUS_MOVIE_NAME = "perfect_dense_noise_15x15_5hz_r42_10min"


# =============================================================================
# Stimulus Loading
# =============================================================================

def load_stimulus_movie(stimulus_path: Path) -> np.ndarray:
    """Load the dense-noise stimulus movie from an .npy file."""
    if not stimulus_path.exists():
        raise FileNotFoundError(f"Stimulus file not found: {stimulus_path}")
    movie = np.load(stimulus_path)
    logger.info("Loaded stimulus movie: shape=%s, dtype=%s", movie.shape, movie.dtype)
    return movie


# =============================================================================
# Spike-to-Frame Conversion
# =============================================================================

def convert_spikes_to_frames(
    spike_times_us: np.ndarray,
    acquisition_rate: float,
    frame_timestamps: np.ndarray,
) -> np.ndarray:
    """
    Convert spike timestamps from microseconds to display-frame indices.

    Steps mirror the legacy sta_visualization.py:
      1. microseconds -> acquisition-rate sample indices
      2. sample indices -> display-frame numbers via frame_timestamps
    """
    if len(spike_times_us) == 0:
        return np.array([], dtype=np.int64)

    spike_samples = np.round(
        spike_times_us.astype(np.float64) * acquisition_rate / 1e6
    ).astype(np.int64)

    spike_frames = convert_sample_index_to_frame(spike_samples, frame_timestamps)
    return spike_frames


# =============================================================================
# Visualization
# =============================================================================

def plot_sta_spatial(
    sta: np.ndarray,
    unit_id: str,
    dataset_id: str,
    n_cols: int = 10,
    save_path: Optional[Path] = None,
) -> None:
    """Plot tiled spatial maps of the STA across time frames."""
    n_frames = sta.shape[0]
    n_rows = max(1, int(np.ceil(n_frames / n_cols)))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.4, n_rows * 1.4),
        squeeze=False,
    )
    vmin, vmax = sta.min(), sta.max()

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx < n_frames:
            ax.imshow(sta[idx], vmin=vmin, vmax=vmax, cmap="RdBu_r",
                      interpolation="nearest")
            ax.set_title(str(idx), fontsize=6)
        ax.axis("off")

    fig.suptitle(f"{dataset_id}  Unit {unit_id}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sta_extreme_trace(
    sta: np.ndarray,
    unit_id: str,
    dataset_id: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot the 1-D time course at the pixel with maximum deviation from mean.

    Matches the legacy logic:
        extreme_coordinate = argmax(|sta - sta.mean()|)
        plot sta[:, y, x]
    """
    extreme_idx = np.unravel_index(
        np.argmax(np.abs(sta - sta.mean())), sta.shape
    )

    trace = sta[:, extreme_idx[1], extreme_idx[2]]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(trace, color="r", linewidth=1.2)
    ax.set_xlabel("Frame offset")
    ax.set_ylabel("Mean intensity")
    ax.set_title(
        f"{dataset_id}  Unit {unit_id}  "
        f"pixel ({extreme_idx[1]}, {extreme_idx[2]})"
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _generate_figures_from_session(
    session: PipelineSession,
    fig_dir: Path,
    movie_name: str = STIMULUS_MOVIE_NAME,
) -> int:
    """Extract STA arrays from session and generate figures. Return count."""
    n_plotted = 0
    feature_key = f"sta_{movie_name}"

    for unit_id, unit_data in session.units.items():
        features = unit_data.get("features", {})
        if feature_key not in features:
            continue

        sta_entry = features[feature_key]
        sta = sta_entry.get("data") if isinstance(sta_entry, dict) else sta_entry
        if sta is None or not hasattr(sta, "shape"):
            continue

        plot_sta_spatial(
            sta, unit_id, session.dataset_id,
            save_path=fig_dir / f"sta_spatial_{unit_id}.png",
        )
        plot_sta_extreme_trace(
            sta, unit_id, session.dataset_id,
            save_path=fig_dir / f"sta_trace_{unit_id}.png",
        )
        n_plotted += 1

    return n_plotted


# =============================================================================
# RF geometry extraction
# =============================================================================

def _parse_frame_rate(movie_name: str) -> float:
    """Extract stimulus frame rate (Hz) from movie name, e.g. '..._5hz_...' -> 5.0."""
    import re
    m = re.search(r"_(\d+)hz_", movie_name, re.IGNORECASE)
    return float(m.group(1)) if m else 15.0


def _extract_rf_geometry_for_session(
    session: PipelineSession,
    movie_name: str = STIMULUS_MOVIE_NAME,
    frame_range: Tuple[int, int] = (40, 60),
    threshold_fraction: float = 0.5,
    cover_range: Tuple[int, int] = (-60, 0),
    stimulus_path: Optional[Path] = None,
) -> PipelineSession:
    """
    Run RF geometry extraction on every unit in *session* that has STA data.

    Stores results under  units/{uid}/features/sta_{movie_name}/sta_geometry
    in the session so they are saved to HDF5 by session.save().

    Also attempts LNL fitting when stimulus movie and spike data are available.
    """
    from rf_sta_measure import extract_rf_geometry, fit_lnl_model, RFGeometry
    from rf_session import _geometry_to_dict

    feature_key = f"sta_{movie_name}"
    frame_rate = _parse_frame_rate(movie_name)

    # --- prepare LNL data (stimulus movie + spike frames) ------------------
    movie_array: Optional[np.ndarray] = None
    spike_frames_dict: Optional[Dict[str, np.ndarray]] = None

    stimuli_dir = (stimulus_path or Path(".")).parent
    movie_npy = stimuli_dir / f"{movie_name}.npy"
    if movie_npy.exists():
        movie_array = np.load(str(movie_npy))
        logger.info("  Loaded stimulus movie for LNL fitting: %s  shape=%s",
                     movie_npy.name, movie_array.shape)

        frame_timestamps = None
        ft = session.stimulus.get("frame_times", {})
        if "frame_timestamps" in ft:
            frame_timestamps = np.array(ft["frame_timestamps"])
        elif "frame_timestamps" in session.metadata:
            frame_timestamps = np.array(session.metadata["frame_timestamps"])

        if frame_timestamps is not None:
            section_time_data = session.stimulus.get("section_time", {})
            section_time = section_time_data.get(movie_name)
            if section_time is not None and len(section_time) > 0:
                movie_start_sample = section_time[0, 0]
                movie_start_frame = int(convert_sample_index_to_frame(
                    np.array([movie_start_sample]), frame_timestamps
                )[0]) + PRE_MARGIN_FRAME_NUM

                spike_frames_dict = {}
                for uid, ud in session.units.items():
                    sec = ud.get("spike_times_sectioned", {}).get(movie_name, {})
                    trials = sec.get("trials_spike_times", {})
                    if 0 not in trials:
                        continue
                    sp_samples = np.array(trials[0])
                    sp_abs = convert_sample_index_to_frame(sp_samples, frame_timestamps)
                    spike_frames_dict[uid] = (sp_abs - movie_start_frame).astype(np.int32)
                logger.info("  Spike frames prepared for %d units", len(spike_frames_dict))

    do_lnl = movie_array is not None and spike_frames_dict is not None

    # --- geometry extraction per unit --------------------------------------
    processed = 0
    lnl_fitted = 0

    for unit_id, unit_data in session.units.items():
        features = unit_data.get("features", {})
        sta_entry = features.get(feature_key)
        if sta_entry is None:
            continue
        sta_data = sta_entry.get("data") if isinstance(sta_entry, dict) else sta_entry
        if sta_data is None or not hasattr(sta_data, "shape"):
            continue

        try:
            geometry = extract_rf_geometry(
                sta_data,
                frame_range=frame_range,
                threshold_fraction=threshold_fraction,
            )

            if do_lnl and unit_id in spike_frames_dict:
                try:
                    lnl_fit = fit_lnl_model(
                        sta=sta_data,
                        movie_array=movie_array,
                        spike_frames=spike_frames_dict[unit_id],
                        cover_range=cover_range,
                        frame_rate=frame_rate,
                    )
                    if lnl_fit is not None:
                        geometry.lnl_fit = lnl_fit
                        lnl_fitted += 1
                except Exception:
                    pass

            if feature_key not in features:
                features[feature_key] = {}
            features[feature_key]["sta_geometry"] = _geometry_to_dict(geometry)
            processed += 1
        except Exception as exc:
            logger.debug("RF geometry failed for %s: %s", unit_id, exc)

    logger.info("RF geometry extracted: %d units (LNL fitted: %d)", processed, lnl_fitted)
    session.completed_steps.add("extract_rf_geometry")
    return session


# =============================================================================
# Default approach: PipelineSession + compute_sta
# =============================================================================

def _build_session(
    rec: RecordingInfo,
    data: Dict,
    frame_timestamps: np.ndarray,
    config: STAPipelineConfig,
    movie_name: str = STIMULUS_MOVIE_NAME,
) -> PipelineSession:
    """
    Construct a PipelineSession from loaded CMCR/CMTR data.

    Populates the session with the fields that compute_sta expects:
      - session.units[uid]["spike_times_sectioned"][movie][trials_spike_times][0]
      - session.stimulus["section_time"][movie]
      - session.stimulus["frame_times"]["frame_timestamps"]
      - session.completed_steps contains "section_spike_times"
    """
    acq_rate = data["metadata"]["acquisition_rate"]
    frame_lo = config.section_time_frame_num[0]
    frame_hi = config.section_time_frame_num[1]

    session = PipelineSession(
        dataset_id=rec.dataset_id,
        output_dir=config.output_dir,
    )
    session.set_source_files(
        cmcr_path=rec.cmcr_path, cmtr_path=rec.cmtr_path,
    )
    session.add_metadata({"acquisition_rate": acq_rate})

    session.add_stimulus({
        "frame_times": {"frame_timestamps": frame_timestamps},
    })

    # compute_sta adds PRE_MARGIN_FRAME_NUM (60) to the frame derived from
    # section_time[0,0].  We offset the start sample backwards by that margin
    # so that movie_start_frame lands exactly on frame_lo.
    section_start_frame = max(0, frame_lo - PRE_MARGIN_FRAME_NUM)
    last_valid_frame = len(frame_timestamps) - 1
    if not np.isinf(frame_hi):
        last_valid_frame = min(int(frame_hi), last_valid_frame)

    start_sample = int(frame_timestamps[section_start_frame])
    end_sample = int(frame_timestamps[min(last_valid_frame, len(frame_timestamps) - 1)])
    section_time_array = np.array([[start_sample, end_sample]], dtype=np.int64)

    session.add_stimulus({
        "section_time": {movie_name: section_time_array},
    })

    for unit_id, unit_data in data["units"].items():
        spike_times_us = unit_data.get("spike_times")
        if spike_times_us is None or len(spike_times_us) == 0:
            continue

        spike_samples = np.round(
            spike_times_us.astype(np.float64) * acq_rate / 1e6
        ).astype(np.int64)

        mask = (spike_samples >= start_sample) & (spike_samples <= end_sample)
        sectioned_spikes = spike_samples[mask]

        session.units[unit_id] = {
            "spike_times": spike_samples,
            "row": unit_data.get("row", 0),
            "col": unit_data.get("col", 0),
            "spike_times_sectioned": {
                movie_name: {
                    "trials_spike_times": {0: sectioned_spikes},
                },
            },
        }

    session.mark_step_complete("load_recording")
    session.mark_step_complete("add_section_time")
    session.mark_step_complete("section_spike_times")

    return session


def process_single_recording_session(
    rec: RecordingInfo,
    config: STAPipelineConfig,
    overwrite: bool = False,
    movie_name: str = STIMULUS_MOVIE_NAME,
    stimulus_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Default approach: build a PipelineSession, call compute_sta, session.save().

    For gsheet-discovered recordings, movie_name and stimulus_path are set
    per-recording via GSheetRecordingInfo.
    """
    fig_dir = get_output_figures_dir(rec.dataset_id, config.figures_dir)

    marker = fig_dir / "_done.txt"
    if marker.exists() and not overwrite:
        return True, "skipped (already done)"

    t0 = time.time()

    try:
        data = load_cmcr_cmtr_data(rec.cmcr_path, rec.cmtr_path)
    except Exception as exc:
        return False, f"load failed: {exc}"

    light_ref = data["light_reference"]
    lr_key = config.frame_channel_key
    if lr_key not in light_ref:
        return False, f"light_reference key '{lr_key}' not found (have {list(light_ref.keys())})"

    frame_timestamps = get_frame_timestamps(light_ref[lr_key])
    logger.info("Frame timestamps: %d frames detected", len(frame_timestamps))

    session = _build_session(rec, data, frame_timestamps, config, movie_name=movie_name)
    del data

    stimuli_dir = (stimulus_path or config.stimulus_path).parent

    session = compute_sta(
        cover_range=config.cover_range,
        use_multiprocessing=False,
        stimuli_dir=stimuli_dir,
        session=session,
    )

    logger.info("Extracting RF geometry ...")
    session = _extract_rf_geometry_for_session(
        session,
        movie_name=movie_name,
        cover_range=config.cover_range,
        stimulus_path=stimulus_path or config.stimulus_path,
    )

    h5_path = config.output_dir / f"{rec.dataset_id}.h5"
    session.save(output_path=h5_path, overwrite=overwrite)

    n_plotted = _generate_figures_from_session(session, fig_dir, movie_name=movie_name)

    marker.write_text(f"processed {session.unit_count} units, plotted {n_plotted} in {time.time() - t0:.1f}s\n")

    elapsed = time.time() - t0
    return True, f"{n_plotted} units plotted ({session.unit_count} total) in {elapsed:.1f}s"


# =============================================================================
# Legacy approach: direct _compute_sta_for_unit + custom HDF5
# =============================================================================

def save_sta_to_hdf5(
    hdf5_path: Path,
    dataset_id: str,
    units_data: Dict[str, dict],
    sta_results: Dict[str, Dict],
    frame_timestamps: np.ndarray,
    config: STAPipelineConfig,
    overwrite: bool = False,
) -> Path:
    """Save STA results and unit metadata to HDF5 (legacy format)."""
    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    if hdf5_path.exists() and not overwrite:
        logger.info("HDF5 exists, skipping: %s", hdf5_path)
        return hdf5_path

    with h5py.File(hdf5_path, "w") as f:
        f.attrs["pipeline"] = "white_noise_salamander_sta"
        f.attrs["dataset_id"] = dataset_id
        f.attrs["cover_range"] = list(config.cover_range)
        f.attrs["section_time_frame_num_lo"] = config.section_time_frame_num[0]
        f.attrs["stimulus"] = config.stimulus_path.name

        stim_grp = f.create_group("stimulus")
        stim_grp.create_dataset("frame_timestamps", data=frame_timestamps)

        meta_grp = f.create_group("metadata")
        meta_grp.attrs["section_time_frame_num_lo"] = config.section_time_frame_num[0]
        meta_grp.attrs["cover_range"] = list(config.cover_range)

        units_grp = f.create_group("units")

        for unit_id, sta_info in sta_results.items():
            unit_grp = units_grp.create_group(unit_id)

            ud = units_data.get(unit_id, {})
            if "spike_times" in ud:
                unit_grp.create_dataset(
                    "spike_times",
                    data=np.array(ud["spike_times"], dtype=np.uint64),
                )
            if "row" in ud:
                unit_grp.attrs["row"] = ud["row"]
            if "col" in ud:
                unit_grp.attrs["col"] = ud["col"]

            feat_grp = unit_grp.create_group("features")
            sta_ds = feat_grp.create_dataset(
                "sta", data=sta_info["sta"], dtype=np.float32,
            )
            sta_ds.attrs["n_spikes_used"] = sta_info["n_used"]
            sta_ds.attrs["n_spikes_excluded"] = sta_info["n_excluded"]
            sta_ds.attrs["cover_range"] = list(config.cover_range)

    logger.info("Saved HDF5: %s (%d units)", hdf5_path, len(sta_results))
    return hdf5_path


def process_single_recording_legacy(
    rec: RecordingInfo,
    movie_array: np.ndarray,
    config: STAPipelineConfig,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    Legacy approach: manual spike-to-frame, _compute_sta_for_unit, custom HDF5.
    """
    fig_dir = get_output_figures_dir(rec.dataset_id, config.figures_dir)

    marker = fig_dir / "_done.txt"
    if marker.exists() and not overwrite:
        return True, "skipped (already done)"

    t0 = time.time()

    try:
        data = load_cmcr_cmtr_data(rec.cmcr_path, rec.cmtr_path)
    except Exception as exc:
        return False, f"load failed: {exc}"

    acq_rate = data["metadata"]["acquisition_rate"]
    light_ref = data["light_reference"]

    lr_key = config.frame_channel_key
    if lr_key not in light_ref:
        return False, f"light_reference key '{lr_key}' not found (have {list(light_ref.keys())})"

    frame_timestamps = get_frame_timestamps(light_ref[lr_key])
    logger.info("Frame timestamps: %d frames detected", len(frame_timestamps))

    frame_lo = config.section_time_frame_num[0]
    frame_hi = config.section_time_frame_num[1]

    units = data["units"]
    sta_results: Dict[str, Dict] = {}

    for unit_id, unit_data in units.items():
        spike_times_us = unit_data.get("spike_times")
        if spike_times_us is None or len(spike_times_us) == 0:
            continue

        spike_frames = convert_spikes_to_frames(
            spike_times_us, acq_rate, frame_timestamps,
        )

        mask = spike_frames >= frame_lo
        if not np.isinf(frame_hi):
            mask &= spike_frames < int(frame_hi)
        spike_frames = spike_frames[mask] - frame_lo

        if len(spike_frames) == 0:
            continue

        sta, n_used, n_excluded = _compute_sta_for_unit(
            spike_frames, movie_array, config.cover_range,
        )

        if n_used == 0:
            continue

        row = unit_data.get("row", "?")
        col = unit_data.get("col", "?")
        logger.info(
            "Unit %s (row=%s, col=%s): %d spikes used, %d excluded",
            unit_id, row, col, n_used, n_excluded,
        )

        sta_results[unit_id] = {
            "sta": sta, "n_used": n_used, "n_excluded": n_excluded,
        }

        plot_sta_spatial(
            sta, unit_id, rec.dataset_id,
            save_path=fig_dir / f"sta_spatial_{unit_id}.png",
        )
        plot_sta_extreme_trace(
            sta, unit_id, rec.dataset_id,
            save_path=fig_dir / f"sta_trace_{unit_id}.png",
        )

    h5_path = config.output_dir / f"{rec.dataset_id}.h5"
    save_sta_to_hdf5(
        h5_path, rec.dataset_id, units, sta_results,
        frame_timestamps, config, overwrite=overwrite,
    )

    n_processed = len(sta_results)
    marker.write_text(f"processed {n_processed} units in {time.time() - t0:.1f}s\n")

    elapsed = time.time() - t0
    return True, f"{n_processed} units in {elapsed:.1f}s"


# =============================================================================
# Batch Runner
# =============================================================================

def _build_recording_list(
    use_all: bool,
    config: STAPipelineConfig,
) -> List[RecordingInfo]:
    """Return either TEST_FILES or full discovery depending on use_all."""
    if use_all:
        return discover_recordings(config.data_folders)

    recordings = []
    for cmcr_path, cmtr_path in TEST_FILES:
        if not cmcr_path.exists():
            logger.warning("CMCR not found: %s", cmcr_path)
            continue
        if not cmtr_path.exists():
            logger.warning("CMTR not found: %s", cmtr_path)
            continue
        recordings.append(RecordingInfo(
            cmcr_path=cmcr_path,
            cmtr_path=cmtr_path,
            dataset_id=cmcr_path.stem,
        ))
    return recordings


def run_batch(
    config: STAPipelineConfig,
    start_index: int = 0,
    end_index: Optional[int] = None,
    overwrite: bool = False,
    use_all: bool = False,
    legacy: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Process recordings sequentially.

    By default uses PipelineSession + compute_sta.
    Pass legacy=True for the direct _compute_sta_for_unit approach.
    """
    recordings = _build_recording_list(use_all, config)
    if not recordings:
        logger.warning("No recordings found")
        return [], [], []

    if end_index is not None:
        recordings = recordings[start_index:end_index]
    else:
        recordings = recordings[start_index:]

    movie_array = None
    if legacy:
        logger.info("Loading stimulus movie (legacy mode) ...")
        movie_array = load_stimulus_movie(config.stimulus_path)

    successful: List[str] = []
    skipped: List[str] = []
    failed: List[str] = []

    for i, rec in enumerate(recordings):
        logger.info(
            "=== [%d/%d] %s ===", i + 1, len(recordings), rec.dataset_id
        )

        if legacy:
            ok, msg = process_single_recording_legacy(
                rec, movie_array, config, overwrite,
            )
        else:
            ok, msg = process_single_recording_session(
                rec, config, overwrite,
            )

        if ok and "skipped" in msg:
            skipped.append(rec.dataset_id)
            logger.info("  -> %s", msg)
        elif ok:
            successful.append(rec.dataset_id)
            logger.info("  -> %s", msg)
        else:
            failed.append(rec.dataset_id)
            logger.error("  -> FAILED: %s", msg)

    return successful, skipped, failed


def run_batch_gsheet(
    config: STAPipelineConfig,
    date_filter: str,
    overwrite: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Process play_movie recordings discovered from Google Sheet for a given date.

    Each recording may use a different stimulus movie; the movie name and
    stimulus path come from the Condition column via GSheetRecordingInfo.
    """
    recordings = discover_recordings_from_gsheet(date_filter)
    if not recordings:
        logger.warning("No gsheet recordings to process for '%s'", date_filter)
        return [], [], []

    successful: List[str] = []
    skipped: List[str] = []
    failed: List[str] = []

    for i, rec in enumerate(recordings):
        logger.info(
            "=== [%d/%d] %s  movie=%s ===",
            i + 1, len(recordings), rec.dataset_id, rec.movie_name,
        )
        ok, msg = process_single_recording_session(
            rec, config, overwrite,
            movie_name=rec.movie_name,
            stimulus_path=rec.stimulus_path,
        )

        if ok and "skipped" in msg:
            skipped.append(rec.dataset_id)
            logger.info("  -> %s", msg)
        elif ok:
            successful.append(rec.dataset_id)
            logger.info("  -> %s", msg)
        else:
            failed.append(rec.dataset_id)
            logger.error("  -> FAILED: %s", msg)

    return successful, skipped, failed


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="White-noise STA pipeline for salamander recordings",
    )
    parser.add_argument("--all", action="store_true",
                        help="Process all discovered recordings (default: TEST_FILES only)")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy direct STA approach instead of PipelineSession")
    parser.add_argument("--gsheet-date", type=str, default=None,
                        help="Date filter for Google Sheet discovery (e.g. 2026.03.03)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index in the recording list (default: 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="End index (exclusive) in the recording list")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if output already exists")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG-level logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = default_config
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if args.gsheet_date:
        mode_label = f"Google Sheet date={args.gsheet_date}"
        approach_label = "session (compute_sta) per-recording stimulus"
    else:
        mode_label = "ALL recordings" if args.all else "TEST_FILES only"
        approach_label = "legacy (direct)" if args.legacy else "session (compute_sta)"

    print("=" * 60)
    print("White Noise Salamander STA Pipeline")
    print("=" * 60)
    print(f"  Mode     : {mode_label}")
    print(f"  Approach : {approach_label}")
    if not args.gsheet_date:
        print(f"  Stimulus : {config.stimulus_path.name}")
    print(f"  Frames   : {config.section_time_frame_num}")
    print(f"  STA range: {config.cover_range}")
    print(f"  HDF5 out : {config.output_dir}")
    print(f"  Figures  : {config.figures_dir}")
    print("=" * 60)

    t0 = time.time()

    if args.gsheet_date:
        successful, skipped, failed = run_batch_gsheet(
            config,
            date_filter=args.gsheet_date,
            overwrite=args.overwrite,
        )
    else:
        successful, skipped, failed = run_batch(
            config,
            start_index=args.start,
            end_index=args.end,
            overwrite=args.overwrite,
            use_all=args.all,
            legacy=args.legacy,
        )

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  Successful: {len(successful)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("  Failed IDs:")
        for fid in failed:
            print(f"    - {fid}")
    print("=" * 60)


if __name__ == "__main__":
    main()
