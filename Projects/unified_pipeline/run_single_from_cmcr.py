#!/usr/bin/env python
"""
Unified Pipeline: Process Recording from CMCR/CMTR

This script processes a single recording from raw CMCR/CMTR files
through all 11 pipeline steps and saves the result to an HDF5 file.

Usage:
    python run_single_from_cmcr.py
    
    # Or with custom paths:
    python run_single_from_cmcr.py --cmcr path/to/file.cmcr --cmtr path/to/file.cmtr
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

# Import step wrappers
from Projects.unified_pipeline.steps import (
    load_recording_step,
    add_section_time_step,
    section_spike_times_step,
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
    GeometryConfig,
    APTrackingConfig,
    DSGCConfig,
    TEST_DATASET_ID,
    DEFAULT_OUTPUT_DIR,
    green_success,
    red_warning,
)


# =============================================================================
# Configuration
# =============================================================================

# Test file paths - update these for your test file
# The paths will be looked up from the CSV mapping file
CMCR_PATH = None  # Set to specific path or leave None to use CSV lookup
CMTR_PATH = None  # Set to specific path or leave None to use CSV lookup

# Dataset to process
DATASET_ID = TEST_DATASET_ID  # "2024.08.08-10.40.20-Rec"

# Output directory
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


# =============================================================================
# Path Lookup
# =============================================================================

def find_cmcr_cmtr_paths(dataset_id: str) -> tuple:
    """
    Find CMCR/CMTR paths for a dataset from the CSV mapping file.
    
    Args:
        dataset_id: Dataset identifier (e.g., "2024.08.08-10.40.20-Rec")
    
    Returns:
        Tuple of (cmcr_path, cmtr_path)
    """
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

def run_pipeline(
    cmcr_path: Path,
    cmtr_path: Path,
    dataset_id: str,
    output_dir: Path,
) -> Path:
    """
    Run the complete 11-step pipeline.
    
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
    output_path = output_dir / f"{dataset_id}.h5"
    
    print("=" * 70)
    print("Unified Pipeline: Process from CMCR/CMTR")
    print("=" * 70)
    print(f"Dataset:  {dataset_id}")
    print(f"CMCR:     {cmcr_path}")
    print(f"CMTR:     {cmtr_path}")
    print(f"Output:   {output_path}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    load_config = LoadRecordingConfig()
    section_config = SectionTimeConfig()
    geometry_config = GeometryConfig()
    ap_config = APTrackingConfig()
    dsgc_config = DSGCConfig()
    
    # =========================================================================
    # Pipeline Steps 1-4: In-memory processing
    # =========================================================================
    
    # Step 1: Create session and load recording
    session = create_session(dataset_id=dataset_id)
    session = load_recording_step(
        cmcr_path=cmcr_path,
        cmtr_path=cmtr_path,
        duration_s=load_config.duration_s,
        spike_limit=load_config.spike_limit,
        window_range=load_config.window_range,
        session=session,
    )
    
    # Step 2: Add section time
    session = add_section_time_step(
        playlist_name=section_config.playlist_name,
        session=session,
    )
    
    # Step 3: Section spike times
    session = section_spike_times_step(
        pad_margin=section_config.pad_margin,
        session=session,
    )
    
    # Step 4: Compute STA
    session = compute_sta_step(
        cover_range=section_config.cover_range,
        session=session,
    )
    
    # =========================================================================
    # Pipeline Steps 5-11: All in deferred mode
    # =========================================================================
    
    # Step 5: Add CMTR/CMCR metadata (deferred)
    session = add_metadata_step(session=session)
    
    # Step 6: Extract soma geometry
    session = extract_soma_geometry_step(
        frame_range=geometry_config.frame_range,
        threshold_fraction=geometry_config.threshold_fraction,
        session=session,
    )
    
    # Step 7: Extract RF geometry
    session = extract_rf_geometry_step(
        frame_range=geometry_config.frame_range,
        threshold_fraction=geometry_config.threshold_fraction,
        session=session,
    )
    
    # Step 8: Add gsheet metadata
    session = add_gsheet_step(session=session)
    
    # Step 9: Add cell type labels
    session = add_cell_type_step(session=session)
    
    # Step 10: Compute AP tracking (deferred - works on session data)
    session = compute_ap_tracking_step(
        config=ap_config,
        session=session,
    )
    
    # Step 11: Section by direction (DSGC) (deferred)
    session = section_by_direction_step(
        config=dsgc_config,
        session=session,
    )
    
    # =========================================================================
    # Single Save at End
    # =========================================================================
    # All steps worked in deferred mode. Now save everything to HDF5.
    logger.info("Saving all data to HDF5...")
    session.save(output_path=output_path, overwrite=True)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(green_success("PIPELINE COMPLETE"))
    print("=" * 70)
    print(f"Output:   {output_path}")
    print(f"Units:    {session.unit_count}")
    print(f"Steps:    {len(session.completed_steps)}")
    print(f"Time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    if session.warnings:
        print(f"\nWarnings ({len(session.warnings)}):")
        for warning in session.warnings[:5]:
            print(f"  - {warning}")
        if len(session.warnings) > 5:
            print(f"  ... and {len(session.warnings) - 5} more")
    
    print("\nCompleted steps:")
    for step in sorted(session.completed_steps):
        status = "⚠️" if ":skipped" in step or ":failed" in step else "✓"
        print(f"  {status} {step}")
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process recording from CMCR/CMTR through unified pipeline"
    )
    parser.add_argument(
        "--cmcr", type=Path, default=None,
        help="Path to CMCR file (optional, uses CSV lookup if not provided)"
    )
    parser.add_argument(
        "--cmtr", type=Path, default=None,
        help="Path to CMTR file (optional, uses CSV lookup if not provided)"
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
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Get paths
    cmcr_path = args.cmcr or CMCR_PATH
    cmtr_path = args.cmtr or CMTR_PATH
    
    if cmcr_path is None or cmtr_path is None:
        try:
            cmcr_path, cmtr_path = find_cmcr_cmtr_paths(args.dataset)
            cmcr_path = Path(cmcr_path)
            cmtr_path = Path(cmtr_path)
        except (FileNotFoundError, ValueError) as e:
            print(red_warning(f"Error: {e}"))
            print("Please provide --cmcr and --cmtr paths directly")
            sys.exit(1)
    
    # Validate paths
    if not cmcr_path.exists():
        print(red_warning(f"CMCR file not found: {cmcr_path}"))
        sys.exit(1)
    
    if not cmtr_path.exists():
        print(red_warning(f"CMTR file not found: {cmtr_path}"))
        sys.exit(1)
    
    # Run pipeline
    try:
        output_path = run_pipeline(
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

