#!/usr/bin/env python
"""
Unified Pipeline Part 1: Steps 1-7

Processes a recording from CMCR/CMTR through the first 7 steps:
  1. Load recording with eimage_sta
  2. Add section time
  3. Section spike times
  4. Compute STA
  5. Add CMTR/CMCR metadata
  6. Extract soma geometry
  7. Extract RF geometry

Saves the intermediate result to HDF5 for further processing.

Usage:
    python run_steps_1_to_7.py --dataset 2024.08.08-10.40.20-Rec
    python run_steps_1_to_7.py --cmcr path/to/file.cmcr --cmtr path/to/file.cmtr
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import create_session

# Import step wrappers (steps 1-7 only)
from Projects.unified_pipeline.steps import (
    load_recording_step,
    add_section_time_step,
    section_spike_times_step,
    compute_sta_step,
    add_metadata_step,
    extract_soma_geometry_step,
    extract_rf_geometry_step,
)

from Projects.unified_pipeline.config import (
    setup_logging,
    LoadRecordingConfig,
    SectionTimeConfig,
    GeometryConfig,
    TEST_DATASET_ID,
    DEFAULT_OUTPUT_DIR,
    green_success,
    red_warning,
)


# =============================================================================
# Configuration
# =============================================================================

DATASET_ID = TEST_DATASET_ID
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


# =============================================================================
# Path Lookup
# =============================================================================

def find_cmcr_cmtr_paths(dataset_id: str) -> tuple:
    """Find CMCR/CMTR paths from CSV mapping."""
    import csv
    from Projects.unified_pipeline.config import CSV_MAPPING_PATH
    
    if not CSV_MAPPING_PATH.exists():
        raise FileNotFoundError(f"CSV mapping file not found: {CSV_MAPPING_PATH}")
    
    with open(CSV_MAPPING_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cmtr_path = row.get('cmtr_path', '')
            if dataset_id in cmtr_path:
                return row['cmcr_path'], row['cmtr_path']
    
    raise ValueError(f"Dataset not found in CSV mapping: {dataset_id}")


# =============================================================================
# Pipeline Execution
# =============================================================================

def run_pipeline_steps_1_to_7(
    cmcr_path: Path,
    cmtr_path: Path,
    dataset_id: str,
    output_dir: Path,
) -> Path:
    """
    Run pipeline steps 1-7 and save result.
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
        dataset_id: Dataset identifier
        output_dir: Output directory
    
    Returns:
        Path to output HDF5 file
    """
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    output_path = output_dir / f"{dataset_id}_steps1-7.h5"
    
    print("=" * 70)
    print("Unified Pipeline: Steps 1-7 (Load → Geometry)")
    print("=" * 70)
    print(f"Dataset:  {dataset_id}")
    print(f"CMCR:     {cmcr_path}")
    print(f"CMTR:     {cmtr_path}")
    print(f"Output:   {output_path}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    load_config = LoadRecordingConfig()
    section_config = SectionTimeConfig()
    geometry_config = GeometryConfig()
    
    # =========================================================================
    # Step 1: Load recording with eimage_sta
    # =========================================================================
    session = create_session(dataset_id=dataset_id)
    session = load_recording_step(
        cmcr_path=cmcr_path,
        cmtr_path=cmtr_path,
        duration_s=load_config.duration_s,
        spike_limit=load_config.spike_limit,
        window_range=load_config.window_range,
        session=session,
    )
    
    # =========================================================================
    # Step 2: Add section time
    # =========================================================================
    session = add_section_time_step(
        playlist_name=section_config.playlist_name,
        session=session,
    )
    
    # =========================================================================
    # Step 3: Section spike times
    # =========================================================================
    session = section_spike_times_step(
        pad_margin=section_config.pad_margin,
        session=session,
    )
    
    # =========================================================================
    # Step 4: Compute STA
    # =========================================================================
    session = compute_sta_step(
        cover_range=section_config.cover_range,
        session=session,
    )
    
    # =========================================================================
    # Step 5: Add CMTR/CMCR metadata
    # =========================================================================
    session = add_metadata_step(session=session)
    
    # =========================================================================
    # Step 6: Extract soma geometry
    # =========================================================================
    session = extract_soma_geometry_step(
        frame_range=geometry_config.frame_range,
        threshold_fraction=geometry_config.threshold_fraction,
        session=session,
    )
    
    # =========================================================================
    # Step 7: Extract RF geometry
    # =========================================================================
    session = extract_rf_geometry_step(
        frame_range=geometry_config.frame_range,
        threshold_fraction=geometry_config.threshold_fraction,
        session=session,
    )
    
    # =========================================================================
    # Save Result
    # =========================================================================
    logger.info("Saving steps 1-7 result to HDF5...")
    session.save(output_path=output_path, overwrite=True)
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(green_success("STEPS 1-7 COMPLETE"))
    print("=" * 70)
    print(f"Output:   {output_path}")
    print(f"Units:    {session.unit_count}")
    print(f"Steps:    {len(session.completed_steps)}")
    print(f"Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    if session.warnings:
        print(f"\nWarnings ({len(session.warnings)}):")
        for warning in session.warnings[:5]:
            print(f"  - {warning}")
    
    print("\nCompleted steps:")
    for step in sorted(session.completed_steps):
        status = "⚠️" if ":skipped" in step or ":failed" in step else "✓"
        print(f"  {status} {step}")
    
    print(f"\nTo continue processing, run:")
    print(f"  python run_steps_8_to_end.py --input {output_path}")
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process recording through pipeline steps 1-7"
    )
    parser.add_argument(
        "--cmcr", type=Path, default=None,
        help="Path to CMCR file"
    )
    parser.add_argument(
        "--cmtr", type=Path, default=None,
        help="Path to CMTR file"
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_ID,
        help=f"Dataset ID (default: {DATASET_ID})"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    cmcr_path = args.cmcr
    cmtr_path = args.cmtr
    
    if cmcr_path is None or cmtr_path is None:
        try:
            cmcr_path, cmtr_path = find_cmcr_cmtr_paths(args.dataset)
            cmcr_path = Path(cmcr_path)
            cmtr_path = Path(cmtr_path)
        except (FileNotFoundError, ValueError) as e:
            print(red_warning(f"Error: {e}"))
            print("Please provide --cmcr and --cmtr paths directly")
            sys.exit(1)
    
    if not cmcr_path.exists():
        print(red_warning(f"CMCR file not found: {cmcr_path}"))
        sys.exit(1)
    
    if not cmtr_path.exists():
        print(red_warning(f"CMTR file not found: {cmtr_path}"))
        sys.exit(1)
    
    try:
        output_path = run_pipeline_steps_1_to_7(
            cmcr_path=cmcr_path,
            cmtr_path=cmtr_path,
            dataset_id=args.dataset,
            output_dir=args.output,
        )
        print(f"\nSuccess! Output saved to: {output_path}")
    except Exception as e:
        print(red_warning(f"\nPipeline failed: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

