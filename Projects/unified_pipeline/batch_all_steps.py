#!/usr/bin/env python
"""
Batch Processing: All Pipeline Steps (1-11)

Processes all matched CMTR/CMCR pairs from the CSV mapping file
through the complete unified pipeline (all 11 steps).

Steps processed:
- Step 1: Load recording with eimage_sta
- Step 2: Add section time from playlist
- Step 3: Section spike times
- Step 3b: Add section time from analog signal (ipRGC test)
- Step 3c: Section spike times for analog stimuli
- Step 4: Compute STA (dense noise)
- Step 5: Add CMTR/CMCR metadata
- Step 6: Extract soma geometry
- Step 7: Extract RF geometry
- Step 8: Add Google Sheet metadata
- Step 9: Add manual cell type labels
- Step 10: Compute AP tracking (CNN)
- Step 11: Section by direction (DSGC)

Usage:
    python batch_all_steps.py
    
    # Process specific range:
    python batch_all_steps.py --start 0 --end 10
    
    # Force overwrite existing files:
    python batch_all_steps.py --overwrite
"""

import argparse
import csv
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import PipelineSession, create_session

# Import step wrappers
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
    CSV_MAPPING_PATH,
    green_success,
    red_warning,
)


# =============================================================================
# Configuration
# =============================================================================

# Output directory for batch processing
OUTPUT_DIR = Path(__file__).parent / "export_all_steps"

# CSV mapping file
CSV_PATH = CSV_MAPPING_PATH


# =============================================================================
# Helper Functions
# =============================================================================

def get_dataset_id_from_cmtr(cmtr_path: str) -> str:
    """
    Extract dataset_id from cmtr filename.
    Example: "2024.01.17-11.15.41-Rec-.cmtr" -> "2024.01.17-11.15.41-Rec"
    """
    cmtr_filename = Path(cmtr_path).stem  # "2024.01.17-11.15.41-Rec-"
    return cmtr_filename.rstrip("-")  # "2024.01.17-11.15.41-Rec"


def load_matched_pairs(csv_path: Path) -> List[dict]:
    """
    Load matched CMCR/CMTR pairs from CSV file.
    
    Args:
        csv_path: Path to CSV mapping file
    
    Returns:
        List of matched row dictionaries
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV mapping file not found: {csv_path}")
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        matched_pairs = [row for row in reader if row.get("matched") == "True"]
    
    return matched_pairs


# =============================================================================
# Single Recording Processing (All Steps)
# =============================================================================

def process_single_recording(
    cmcr_path: str,
    cmtr_path: str,
    output_dir: Path,
    load_config: Optional[LoadRecordingConfig] = None,
    section_config: Optional[SectionTimeConfig] = None,
    section_analog_config: Optional[SectionTimeAnalogConfig] = None,
    geometry_config: Optional[GeometryConfig] = None,
    ap_config: Optional[APTrackingConfig] = None,
    dsgc_config: Optional[DSGCConfig] = None,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Process a single recording through all 11 steps.
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
        output_dir: Output directory
        load_config: Configuration for loading
        section_config: Configuration for sectioning
        geometry_config: Configuration for geometry extraction
        ap_config: Configuration for AP tracking
        dsgc_config: Configuration for DSGC
        overwrite: Whether to overwrite existing files
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    logger = logging.getLogger(__name__)
    
    # Use default configs if not provided
    if load_config is None:
        load_config = LoadRecordingConfig()
    if section_config is None:
        section_config = SectionTimeConfig()
    if section_analog_config is None:
        section_analog_config = SectionTimeAnalogConfig()
    if geometry_config is None:
        geometry_config = GeometryConfig()
    if ap_config is None:
        ap_config = APTrackingConfig()
    if dsgc_config is None:
        dsgc_config = DSGCConfig()
    
    dataset_id = get_dataset_id_from_cmtr(cmtr_path)
    output_path = output_dir / f"{dataset_id}.h5"
    
    # Check if output already exists
    if output_path.exists() and not overwrite:
        return True, "skipped"
    
    # Validate input files exist
    cmcr_path_obj = Path(cmcr_path)
    cmtr_path_obj = Path(cmtr_path)
    
    if not cmcr_path_obj.exists():
        return False, f"CMCR file not found: {cmcr_path}"
    
    if not cmtr_path_obj.exists():
        return False, f"CMTR file not found: {cmtr_path}"
    
    logger.info(f"Processing {dataset_id}...")
    logger.debug(f"  CMCR: {cmcr_path}")
    logger.debug(f"  CMTR: {cmtr_path}")
    
    # =========================================================================
    # Steps 1-4: Load and prepare data
    # =========================================================================
    
    # Step 1: Create session and load recording
    session = create_session(dataset_id=dataset_id)
    session = load_recording_step(
        cmcr_path=cmcr_path_obj,
        cmtr_path=cmtr_path_obj,
        duration_s=load_config.duration_s,
        spike_limit=load_config.spike_limit,
        window_range=load_config.window_range,
        session=session,
    )
    
    # Step 2: Add section time from playlist
    session = add_section_time_step(
        playlist_name=section_config.playlist_name,
        session=session,
    )
    
    # Step 3: Section spike times
    session = section_spike_times_step(
        pad_margin=section_config.pad_margin,
        session=session,
    )
    
    # Step 3b: Add section time from analog signal (ipRGC test)
    session = add_section_time_analog_step(
        config=section_analog_config,
        session=session,
    )
    
    # Step 3c: Section spike times for analog-detected stimuli
    session = section_spike_times_analog_step(
        movie_name=section_analog_config.movie_name,
        pad_margin=section_analog_config.pad_margin,
        session=session,
    )
    
    # Step 4: Compute STA (dense noise)
    session = compute_sta_step(
        cover_range=section_config.cover_range,
        session=session,
    )
    
    # =========================================================================
    # Steps 5-11: Feature extraction and analysis
    # =========================================================================
    
    # Step 5: Add CMTR/CMCR metadata
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
    
    # Step 8: Add Google Sheet metadata
    session = add_gsheet_step(session=session)
    
    # Step 9: Add cell type labels
    session = add_cell_type_step(session=session)
    
    # Step 10: Compute AP tracking
    session = compute_ap_tracking_step(
        config=ap_config,
        session=session,
    )
    
    # Step 11: Section by direction (DSGC)
    session = section_by_direction_step(
        config=dsgc_config,
        session=session,
    )
    
    # =========================================================================
    # Save final result
    # =========================================================================
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
    csv_path: Path = CSV_PATH,
    output_dir: Path = OUTPUT_DIR,
    start_index: int = 0,
    end_index: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Run batch processing on all matched pairs.
    
    Args:
        csv_path: Path to CSV mapping file
        output_dir: Output directory
        start_index: Starting index (0-based)
        end_index: Ending index (exclusive, None for all)
        overwrite: Whether to overwrite existing files
    
    Returns:
        Tuple of (successful, skipped, failed) lists
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load matched pairs
    logger.info(f"Reading CSV: {csv_path}")
    matched_pairs = load_matched_pairs(csv_path)
    total_pairs = len(matched_pairs)
    
    # Apply index range
    if end_index is not None:
        matched_pairs = matched_pairs[start_index:end_index]
    else:
        matched_pairs = matched_pairs[start_index:]
    
    logger.info(f"Processing {len(matched_pairs)} of {total_pairs} matched pairs")
    
    # Track results
    successful: List[str] = []
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []
    
    # Process each matched pair
    start_time = time.time()
    
    for i, row in enumerate(matched_pairs, 1):
        cmcr_path = row["cmcr_path"]
        cmtr_path = row["cmtr_path"]
        dataset_id = get_dataset_id_from_cmtr(cmtr_path)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(matched_pairs)}] {dataset_id}")
        logger.info(f"{'='*60}")
        
        try:
            success, error = process_single_recording(
                cmcr_path=cmcr_path,
                cmtr_path=cmtr_path,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            
            if success:
                if error == "skipped":
                    logger.info(f"  Skipped - output file already exists")
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
    
    # Summary
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total pairs processed: {len(matched_pairs)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Skipped (already exists): {len(skipped)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    if failed:
        logger.info("\nFailed recordings:")
        for dataset_id, error in failed[:10]:
            logger.info(f"  - {dataset_id}: {error}")
        if len(failed) > 10:
            logger.info(f"  ... and {len(failed) - 10} more")
    
    return successful, skipped, failed


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process CMCR/CMTR files through all 11 pipeline steps"
    )
    parser.add_argument(
        "--csv", type=Path, default=CSV_PATH,
        help=f"CSV mapping file (default: {CSV_PATH})"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Starting index (0-based, default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Ending index (exclusive, default: all)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    print("=" * 70)
    print("Batch Processing: All Pipeline Steps (1-11)")
    print("=" * 70)
    print(f"CSV:       {args.csv}")
    print(f"Output:    {args.output}")
    print(f"Range:     {args.start} to {args.end or 'end'}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        successful, skipped, failed = run_batch(
            csv_path=args.csv,
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

