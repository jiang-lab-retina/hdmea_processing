#!/usr/bin/env python
"""
Batch Processing: Natural Scene Pipeline

Processes recordings from file_name_natural_scene.xlsx through either:
  - Full unified pipeline (Steps 1-11) for set6 protocol
  - Basic pipeline (Steps 1-5+) for natural scene protocol

Input: file_name_natural_scene.xlsx
Output: export/{dataset_id}.h5

Usage:
    python batch_pipeline.py
    python batch_pipeline.py --start 0 --end 10
    python batch_pipeline.py --overwrite
"""

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import PipelineSession, create_session
from hdmea.io.section_time import convert_sample_index_to_frame, PRE_MARGIN_FRAME_NUM

from Projects.unified_pipeline.steps import (
    load_recording_step,
    add_section_time_step,
    add_section_time_analog_step,
    section_spike_times_step,
    section_spike_times_analog_step,
    compute_sta_step,
    add_metadata_step,
    extract_soma_geometry_step,
    extract_rf_geometry_step,
    add_gsheet_step,
    add_cell_type_step,
    compute_ap_tracking_step,
    section_by_direction_step,
)

from Projects.unified_pipeline.config import (
    setup_logging,
    LoadRecordingConfig,
    SectionTimeConfig,
    SectionTimeAnalogConfig,
    GeometryConfig,
    APTrackingConfig,
    DSGCConfig,
    green_success,
    red_warning,
)

from Projects.unified_special_pipeline.natural_scene.specific_config import (
    EXCEL_PATH,
    OUTPUT_DIR,
    PROTOCOL_SET6,
    PROTOCOL_NATURAL_SCENE,
    NaturalSceneSectionTimeConfig,
    Set6SectionTimeConfig,
    load_excel_data,
    build_recording_list,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Natural Scene STA
# =============================================================================

NATURAL_SCENE_STIMULI_DIR = Path(
    r"M:\Python_Project\Design_Stimulation_Pattern\Data\Stimulations"
)

NATURAL_SCENE_MOVIES = ["stu48_final", "superfruit_final", "dway_final"]

DOWNSAMPLE_TARGET = 20  # 500x500 -> 20x20


def _downsample_movie(movie: np.ndarray, target_size: int = DOWNSAMPLE_TARGET) -> np.ndarray:
    """Downsample movie frames from (N, H, W) to (N, target_size, target_size) by block-mean."""
    n_frames, h, w = movie.shape
    block_h = h // target_size
    block_w = w // target_size
    return (
        movie[:, :target_size * block_h, :target_size * block_w]
        .reshape(n_frames, target_size, block_h, target_size, block_w)
        .mean(axis=(2, 4))
        .astype(np.float32)
    )


def compute_sta_natural_scene(
    session: PipelineSession,
    cover_range: Tuple[int, int] = (-60, 0),
    stimuli_dir: Path = NATURAL_SCENE_STIMULI_DIR,
) -> PipelineSession:
    """
    Compute STA for natural scene movies with 500x500 -> 20x20 downsampling.

    Iterates over all movies in section_time, pools spikes from all repeats
    (interleaved in the playlist), and stores sta_{movie_name} per unit.
    """
    start_time = time.time()

    section_time_data: Dict[str, np.ndarray] = session.stimulus.get("section_time", {})

    # Get frame_timestamps
    frame_timestamps = None
    if "frame_times" in session.stimulus and "frame_timestamps" in session.stimulus["frame_times"]:
        frame_timestamps = np.array(session.stimulus["frame_times"]["frame_timestamps"])
    elif "frame_timestamps" in session.metadata:
        frame_timestamps = np.array(session.metadata["frame_timestamps"])

    if frame_timestamps is None:
        session.warnings.append("No frame_timestamps found in session")
        return session

    unit_ids = list(session.units.keys())
    if not unit_ids:
        session.warnings.append("No units in session for STA")
        session.mark_step_complete("compute_sta")
        return session

    window_length = cover_range[1] - cover_range[0]
    total_units_processed = 0

    for movie_name in NATURAL_SCENE_MOVIES:
        section_time = section_time_data.get(movie_name)
        if section_time is None:
            logger.warning(f"No section_time for movie '{movie_name}', skipping STA")
            continue

        section_time = np.asarray(section_time)
        n_repeats = section_time.shape[0]
        logger.info(f"STA for '{movie_name}': {n_repeats} repeats, cover_range={cover_range}")

        # Load and downsample movie
        npy_path = stimuli_dir / f"{movie_name}.npy"
        if not npy_path.exists():
            session.warnings.append(f"Stimulus not found: {npy_path}")
            continue
        raw_movie = np.load(npy_path)
        movie_array = _downsample_movie(raw_movie)
        del raw_movie
        logger.info(f"  Movie shape after downsample: {movie_array.shape}")

        # Precompute movie_start_frame for each repeat
        repeat_start_frames = []
        for r in range(n_repeats):
            start_sample = section_time[r, 0]
            start_frame = int(convert_sample_index_to_frame(
                np.array([start_sample]), frame_timestamps
            )[0]) + PRE_MARGIN_FRAME_NUM
            repeat_start_frames.append(start_frame)

        # Process each unit
        for unit_id in unit_ids:
            unit_data = session.units[unit_id]

            if "spike_times_sectioned" not in unit_data:
                continue
            if movie_name not in unit_data["spike_times_sectioned"]:
                continue

            sectioned = unit_data["spike_times_sectioned"][movie_name]
            trials_data = sectioned.get("trials_spike_times", {})

            # Pool spike frames across all repeats
            all_spike_frames: List[np.ndarray] = []
            for r in range(n_repeats):
                trial_spikes = trials_data.get(r)
                if trial_spikes is None or len(trial_spikes) == 0:
                    continue
                spike_samples = np.array(trial_spikes)
                spike_frames_abs = convert_sample_index_to_frame(spike_samples, frame_timestamps)
                spike_frames_rel = spike_frames_abs - repeat_start_frames[r]
                all_spike_frames.append(spike_frames_rel)

            if not all_spike_frames:
                continue

            pooled_spikes = np.concatenate(all_spike_frames).astype(int)

            # Edge filtering
            valid_mask = (
                (pooled_spikes + cover_range[0] >= 0) &
                (pooled_spikes + cover_range[1] <= movie_array.shape[0])
            )
            valid_spikes = pooled_spikes[valid_mask]
            n_used = len(valid_spikes)
            n_excluded = len(pooled_spikes) - n_used

            if n_used == 0:
                sta = np.full(
                    (window_length, movie_array.shape[1], movie_array.shape[2]),
                    np.nan, dtype=np.float32,
                )
            else:
                window_offsets = np.arange(cover_range[0], cover_range[1])
                all_indices = valid_spikes[:, np.newaxis] + window_offsets
                windows = movie_array[all_indices]
                sta = windows.mean(axis=0).astype(np.float32)

            session.add_feature(
                unit_id,
                f"sta_{movie_name}",
                sta,
                {
                    "n_spikes_used": n_used,
                    "n_spikes_excluded": n_excluded,
                    "n_repeats_pooled": n_repeats,
                    "cover_range": list(cover_range),
                    "movie_name": movie_name,
                    "downsampled_from": "500x500",
                    "downsampled_to": f"{DOWNSAMPLE_TARGET}x{DOWNSAMPLE_TARGET}",
                },
            )
            total_units_processed += 1

    session.mark_step_complete("compute_sta")
    elapsed = time.time() - start_time
    logger.info(f"Natural scene STA complete: {total_units_processed} unit-movie pairs, {elapsed:.1f}s")
    return session


# =============================================================================
# Single Recording Processing
# =============================================================================


def _process_set6(
    session: PipelineSession,
    section_config: Set6SectionTimeConfig,
    section_analog_config: SectionTimeAnalogConfig,
    geometry_config: GeometryConfig,
    ap_config: APTrackingConfig,
    dsgc_config: DSGCConfig,
) -> PipelineSession:
    """Full set6 pipeline: Steps 2-11."""
    # Step 2: Section time from playlist
    session = add_section_time_step(
        playlist_name=section_config.playlist_name,
        session=session,
    )

    # Step 3: Section spike times
    session = section_spike_times_step(
        pad_margin=section_config.pad_margin,
        session=session,
    )

    # Step 3b: Analog section time (ipRGC test)
    session = add_section_time_analog_step(
        config=section_analog_config,
        session=session,
    )

    # Step 3c: Section spike times for analog stimuli
    session = section_spike_times_analog_step(
        movie_name=section_analog_config.movie_name,
        pad_margin=section_analog_config.pad_margin,
        session=session,
    )

    # Step 4: Compute STA
    session = compute_sta_step(
        cover_range=section_config.cover_range,
        session=session,
    )

    # Step 5: CMTR/CMCR metadata
    session = add_metadata_step(session=session)

    # Step 6: Soma geometry
    session = extract_soma_geometry_step(
        frame_range=geometry_config.frame_range,
        threshold_fraction=geometry_config.threshold_fraction,
        session=session,
    )

    # Step 7: RF geometry
    session = extract_rf_geometry_step(
        frame_range=geometry_config.frame_range,
        threshold_fraction=geometry_config.threshold_fraction,
        session=session,
    )

    # Step 8: Google Sheet metadata
    session = add_gsheet_step(session=session)

    # Step 9: Cell type labels
    session = add_cell_type_step(session=session)

    # Step 10: AP tracking
    session = compute_ap_tracking_step(
        config=ap_config,
        session=session,
    )

    # Step 11: DSGC direction sectioning
    session = section_by_direction_step(
        config=dsgc_config,
        session=session,
    )

    return session


def _process_natural_scene(
    session: PipelineSession,
    section_config: NaturalSceneSectionTimeConfig,
) -> PipelineSession:
    """Natural scene pipeline: Steps 2-5+."""
    # Step 2: Section time from playlist
    session = add_section_time_step(
        playlist_name=section_config.playlist_name,
        session=session,
    )

    # Step 3: Section spike times
    session = section_spike_times_step(
        pad_margin=section_config.pad_margin,
        session=session,
    )

    # Step 4: Compute STA (natural scene: downsample 500x500 -> 20x20, pool all repeats)
    session = compute_sta_natural_scene(
        session=session,
        cover_range=section_config.cover_range,
    )

    # Step 5: CMTR/CMCR metadata
    session = add_metadata_step(session=session)

    # Step 8: Google Sheet metadata
    session = add_gsheet_step(session=session)

    return session


def process_single_recording(
    cmcr_path: Path,
    cmtr_path: Path,
    dataset_id: str,
    protocol: str,
    output_dir: Path,
    load_config: Optional[LoadRecordingConfig] = None,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Process a single recording through the appropriate pipeline path.

    Returns:
        (success, error_message) -- error_message is "skipped" when file exists.
    """
    if load_config is None:
        load_config = LoadRecordingConfig()

    output_path = output_dir / f"{dataset_id}.h5"

    if output_path.exists() and not overwrite:
        return True, "skipped"

    if not cmcr_path.exists():
        return False, f"CMCR not found: {cmcr_path}"
    if not cmtr_path.exists():
        return False, f"CMTR not found: {cmtr_path}"

    logger.info(f"Processing {dataset_id} [{protocol}]...")
    logger.debug(f"  CMCR: {cmcr_path}")
    logger.debug(f"  CMTR: {cmtr_path}")

    # Step 1: Load recording
    session = create_session(dataset_id=dataset_id)
    session = load_recording_step(
        cmcr_path=cmcr_path,
        cmtr_path=cmtr_path,
        duration_s=load_config.duration_s,
        spike_limit=load_config.spike_limit,
        window_range=load_config.window_range,
        session=session,
    )

    # Steps 2+ depend on protocol
    if protocol == PROTOCOL_SET6:
        session = _process_set6(
            session=session,
            section_config=Set6SectionTimeConfig(),
            section_analog_config=SectionTimeAnalogConfig(),
            geometry_config=GeometryConfig(),
            ap_config=APTrackingConfig(),
            dsgc_config=DSGCConfig(),
        )
    elif protocol == PROTOCOL_NATURAL_SCENE:
        session = _process_natural_scene(
            session=session,
            section_config=NaturalSceneSectionTimeConfig(),
        )
    else:
        return False, f"Unknown protocol: {protocol}"

    # Save
    session.save(output_path=output_path, overwrite=overwrite)
    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Units: {session.unit_count}, Steps: {len(session.completed_steps)}")
    if session.warnings:
        logger.debug(f"  Warnings: {len(session.warnings)}")

    return True, None


# =============================================================================
# Batch Processing
# =============================================================================


def run_batch(
    excel_path: Path = EXCEL_PATH,
    output_dir: Path = OUTPUT_DIR,
    start_index: int = 0,
    end_index: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Run batch processing on all recordings from the Excel file.

    Returns:
        (successful, skipped, failed) lists
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading Excel: {excel_path}")
    df = load_excel_data(excel_path)
    recordings = build_recording_list(df)
    total = len(recordings)

    # Report discovery results
    found = sum(1 for r in recordings if r["cmcr_path"] and r["cmtr_path"])
    missing = total - found
    logger.info(f"Found {found}/{total} recordings on disk ({missing} missing)")

    if end_index is not None:
        recordings = recordings[start_index:end_index]
    else:
        recordings = recordings[start_index:]

    logger.info(f"Processing {len(recordings)} of {total} recordings")

    successful: List[str] = []
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []

    start_time = time.time()

    for i, rec in enumerate(recordings, 1):
        dataset_id = rec["dataset_id"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"[{i}/{len(recordings)}] {dataset_id}  ({rec['protocol']})")
        logger.info(f"{'=' * 60}")

        if rec["cmcr_path"] is None or rec["cmtr_path"] is None:
            msg = "CMCR/CMTR not found on any drive"
            logger.error(f"  {msg}")
            failed.append((dataset_id, msg))
            continue

        try:
            success, error = process_single_recording(
                cmcr_path=rec["cmcr_path"],
                cmtr_path=rec["cmtr_path"],
                dataset_id=dataset_id,
                protocol=rec["protocol"],
                output_dir=output_dir,
                overwrite=overwrite,
            )

            if success:
                if error == "skipped":
                    logger.info("  Skipped - output already exists")
                    skipped.append(dataset_id)
                else:
                    successful.append(dataset_id)
            else:
                logger.error(f"  Failed: {error}")
                failed.append((dataset_id, error or "Unknown error"))

        except Exception as e:
            logger.error(f"  Exception: {e}")
            logger.debug(traceback.format_exc())
            failed.append((dataset_id, str(e)))

    elapsed = time.time() - start_time

    logger.info(f"\n{'=' * 60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total: {len(recordings)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    if failed:
        logger.info("\nFailed recordings:")
        for did, error in failed[:10]:
            logger.info(f"  - {did}: {error}")
        if len(failed) > 10:
            logger.info(f"  ... and {len(failed) - 10} more")

    return successful, skipped, failed


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch process natural scene recordings through the unified pipeline"
    )
    parser.add_argument(
        "--excel", type=Path, default=EXCEL_PATH,
        help=f"Excel mapping file (default: {EXCEL_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Starting index (0-based, default: 0)",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Ending index (exclusive, default: all)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    print("=" * 70)
    print("Batch Processing: Natural Scene Pipeline")
    print("=" * 70)
    print(f"Excel:     {args.excel}")
    print(f"Output:    {args.output}")
    print(f"Range:     {args.start} to {args.end or 'end'}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        successful, skipped, failed = run_batch(
            excel_path=args.excel,
            output_dir=args.output,
            start_index=args.start,
            end_index=args.end,
            overwrite=args.overwrite,
        )

        print("\n" + "=" * 70)
        if len(failed) == 0:
            print(green_success("BATCH COMPLETE - ALL SUCCESSFUL"))
        else:
            print(red_warning(f"BATCH COMPLETE - {len(failed)} FAILED"))
        print("=" * 70)
        print(f"Successful: {len(successful)}")
        print(f"Skipped:    {len(skipped)}")
        print(f"Failed:     {len(failed)}")

    except Exception as e:
        print(red_warning(f"\nBatch processing failed: {e}"))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
