#!/usr/bin/env python
"""
Load CMCR/CMTR: Steps 1-5 of the Unified Pipeline

This script processes the steps that require CMCR/CMTR file access:
- Step 1: Load recording with eimage_sta
- Step 2: Add section time from playlist
- Step 3: Section spike times
- Step 4: Compute STA (dense noise)
- Step 5: Add CMTR/CMCR metadata

After running this, all data is in memory and can be:
1. Saved to HDF5 for later continuation
2. Passed directly to remaining steps (6-11)

Usage:
    python load_cmcr_cmtr.py
    
    # With custom paths:
    python load_cmcr_cmtr.py --cmcr path/to/file.cmcr --cmtr path/to/file.cmtr
    
    # Save output for later continuation:
    python load_cmcr_cmtr.py --output path/to/output.h5
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import PipelineSession, create_session

# Import step wrappers
from Projects.unified_pipeline.steps import (
    load_recording_step,
    add_section_time_step,
    section_spike_times_step,
    compute_sta_step,
    add_metadata_step,
)

from Projects.unified_pipeline.config import (
    setup_logging,
    LoadRecordingConfig,
    SectionTimeConfig,
    TEST_DATASET_ID,
    DEFAULT_OUTPUT_DIR,
    green_success,
    red_warning,
)


# =============================================================================
# Configuration
# =============================================================================

# Default dataset to process
DATASET_ID = TEST_DATASET_ID  # "2024.08.08-10.40.20-Rec"

# Output directory
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


# =============================================================================
# Path Lookup
# =============================================================================

def find_cmcr_cmtr_paths(dataset_id: str) -> Tuple[Path, Path]:
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
                return Path(row['cmcr_path']), Path(row['cmtr_path'])
    
    raise ValueError(f"Dataset not found in CSV mapping: {dataset_id}")


# =============================================================================
# Main Processing Function
# =============================================================================

def load_from_cmcr_cmtr(
    cmcr_path: Path,
    cmtr_path: Path,
    dataset_id: str,
    load_config: Optional[LoadRecordingConfig] = None,
    section_config: Optional[SectionTimeConfig] = None,
) -> PipelineSession:
    """
    Load recording from CMCR/CMTR and run steps 1-5.
    
    These are all the steps that require access to the raw CMCR/CMTR files.
    After this function returns, all data is in memory and no further
    access to CMCR/CMTR is needed.
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
        dataset_id: Dataset identifier
        load_config: Configuration for loading (optional)
        section_config: Configuration for sectioning (optional)
    
    Returns:
        PipelineSession with steps 1-5 complete
    """
    logger = logging.getLogger(__name__)
    
    if load_config is None:
        load_config = LoadRecordingConfig()
    if section_config is None:
        section_config = SectionTimeConfig()
    
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("Loading from CMCR/CMTR (Steps 1-5)")
    logger.info("=" * 60)
    logger.info(f"Dataset:  {dataset_id}")
    logger.info(f"CMCR:     {cmcr_path}")
    logger.info(f"CMTR:     {cmtr_path}")
    
    # =========================================================================
    # Step 1: Create session and load recording
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
    # Step 2: Add section time from playlist
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
    # Step 4: Compute STA (dense noise)
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
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    logger.info("-" * 60)
    logger.info(f"Steps 1-5 complete in {elapsed:.1f}s")
    logger.info(f"Units loaded: {session.unit_count}")
    logger.info(f"Completed steps: {len(session.completed_steps)}")
    
    return session


def save_session(
    session: PipelineSession,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """
    Save session to HDF5 file.
    
    Args:
        session: PipelineSession to save
        output_path: Path to output HDF5 file
        overwrite: Whether to overwrite existing file
    
    Returns:
        Path to saved file
    """
    logger = logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving session to: {output_path}")
    session.save(output_path=output_path, overwrite=overwrite)
    logger.info(f"  Saved {session.unit_count} units, {len(session.completed_steps)} steps")
    
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load recording from CMCR/CMTR (Steps 1-5)"
    )
    parser.add_argument(
        "--cmcr", type=Path, default=None,
        help="Path to CMCR file (uses CSV lookup if not provided)"
    )
    parser.add_argument(
        "--cmtr", type=Path, default=None,
        help="Path to CMTR file (uses CSV lookup if not provided)"
    )
    parser.add_argument(
        "--dataset", type=str, default=DATASET_ID,
        help=f"Dataset ID (default: {DATASET_ID})"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output HDF5 file path (optional, saves if provided)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output file"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get paths
    cmcr_path = args.cmcr
    cmtr_path = args.cmtr
    
    if cmcr_path is None or cmtr_path is None:
        try:
            cmcr_path, cmtr_path = find_cmcr_cmtr_paths(args.dataset)
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
    
    print("=" * 70)
    print("Load CMCR/CMTR: Steps 1-5")
    print("=" * 70)
    print(f"Dataset:  {args.dataset}")
    print(f"CMCR:     {cmcr_path}")
    print(f"CMTR:     {cmtr_path}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run steps 1-5
    try:
        session = load_from_cmcr_cmtr(
            cmcr_path=cmcr_path,
            cmtr_path=cmtr_path,
            dataset_id=args.dataset,
        )
        
        # Save if output specified
        if args.output:
            output_path = args.output
        else:
            # Default: save to output_dir with _steps1-5 suffix
            output_path = args.output_dir / f"{args.dataset}_steps1-5.h5"
        
        save_session(session, output_path, overwrite=args.overwrite)
        
        print("\n" + "=" * 70)
        print(green_success("STEPS 1-5 COMPLETE"))
        print("=" * 70)
        print(f"Output:   {output_path}")
        print(f"Units:    {session.unit_count}")
        print(f"Steps:    {len(session.completed_steps)}")
        
        print("\nCompleted steps:")
        for step in sorted(session.completed_steps):
            print(f"  [+] {step}")
        
        print(f"\nTo continue processing, load this file and run steps 6-11:")
        print(f"  python run_steps_6_to_end.py --input {output_path}")
        
    except Exception as e:
        print(red_warning(f"\nFailed: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

