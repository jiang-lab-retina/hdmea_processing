#!/usr/bin/env python
"""
Batch Processing: Discover CMCR files from folders and run unified pipeline.

This script:
1. Discovers all *.cmcr files recursively from specified search folders
2. Finds matching *.cmtr files (same stem with optional '-' suffix)
3. Processes each pair through the full unified pipeline (Steps 1-11)
4. Optionally runs RF geometry extraction with LNL fitting

Usage:
    # Use folders defined in config.py
    python batch_from_folders.py
    
    # Override with CLI arguments
    python batch_from_folders.py --search-folders "M:\\data1" "M:\\data2" --output "M:\\output"
    
    # Process specific range
    python batch_from_folders.py --start 0 --end 10
    
    # Skip RF update for faster processing
    python batch_from_folders.py --skip-rf-update
"""

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Projects" / "rf_sta_measure"))

from hdmea.pipeline import create_session

# Import step wrappers from unified_pipeline
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

# Import RF geometry session functions for LNL fitting
from rf_session import extract_rf_geometry_session, prepare_lnl_data_from_session

# Import local config
from yan_pipeline.config import (
    SEARCH_FOLDERS,
    OUTPUT_DIR,
    SKIP_RF_UPDATE,
    OVERWRITE_EXISTING,
    DEBUG,
    CREATE_MANUAL_LABEL,
    MANUAL_LABEL_DIR,
    MANUAL_LABEL_PLOTS,
)

# Import manual label plot generation
from yan_pipeline.manual_label_plots import generate_manual_label_images

logger = logging.getLogger(__name__)


# =============================================================================
# CMCR/CMTR Discovery
# =============================================================================

def discover_cmcr_files(folders: List[Path]) -> List[Path]:
    """
    Recursively discover all *.cmcr files in the given folders.
    
    Args:
        folders: List of directories to search
    
    Returns:
        Sorted list of CMCR file paths
    """
    cmcr_files = []
    
    for folder in folders:
        if not folder.exists():
            logger.warning(f"Search folder does not exist: {folder}")
            continue
        
        # Recursively find all .cmcr files
        found = list(folder.rglob("*.cmcr"))
        logger.info(f"Found {len(found)} CMCR files in {folder}")
        cmcr_files.extend(found)
    
    # Sort by name for consistent ordering
    cmcr_files.sort(key=lambda p: p.name)
    
    return cmcr_files


def find_matching_cmtr(cmcr_path: Path) -> Optional[Path]:
    """
    Find the matching CMTR file for a given CMCR file.
    
    Looks for:
    1. {stem}-.cmtr (with trailing dash) - primary
    2. {stem}.cmtr (exact stem match) - fallback
    
    Args:
        cmcr_path: Path to CMCR file
    
    Returns:
        Path to matching CMTR file, or None if not found
    """
    cmcr_dir = cmcr_path.parent
    stem = cmcr_path.stem  # e.g., "2024.01.17-11.15.41-Rec"
    
    # Try with trailing dash first (common pattern)
    cmtr_with_dash = cmcr_dir / f"{stem}-.cmtr"
    if cmtr_with_dash.exists():
        return cmtr_with_dash
    
    # Try exact stem match
    cmtr_exact = cmcr_dir / f"{stem}.cmtr"
    if cmtr_exact.exists():
        return cmtr_exact
    
    return None


def get_dataset_id_from_cmcr(cmcr_path: Path) -> str:
    """
    Extract dataset_id from CMCR filename.
    
    Example: "2024.01.17-11.15.41-Rec.cmcr" -> "2024.01.17-11.15.41-Rec"
    """
    return cmcr_path.stem


# =============================================================================
# Single Recording Processing
# =============================================================================

def process_single_recording(
    cmcr_path: Path,
    cmtr_path: Path,
    output_dir: Path,
    load_config: Optional[LoadRecordingConfig] = None,
    section_config: Optional[SectionTimeConfig] = None,
    section_analog_config: Optional[SectionTimeAnalogConfig] = None,
    geometry_config: Optional[GeometryConfig] = None,
    ap_config: Optional[APTrackingConfig] = None,
    dsgc_config: Optional[DSGCConfig] = None,
    skip_rf_update: bool = False,
    overwrite: bool = False,
    create_manual_label: bool = True,
    manual_label_dir: Optional[Path] = None,
    manual_label_plots: Optional[dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Process a single recording through all pipeline steps.
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
        output_dir: Output directory
        load_config: Configuration for loading
        section_config: Configuration for sectioning
        section_analog_config: Configuration for analog sectioning
        geometry_config: Configuration for geometry extraction
        ap_config: Configuration for AP tracking
        dsgc_config: Configuration for DSGC
        skip_rf_update: Skip RF geometry + LNL fitting
        overwrite: Whether to overwrite existing files
        create_manual_label: Generate manual label images
        manual_label_dir: Directory for manual label images
        manual_label_plots: Dict of plot types to include
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
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
    
    dataset_id = get_dataset_id_from_cmcr(cmcr_path)
    output_path = output_dir / f"{dataset_id}.h5"
    
    # Check if output already exists
    if output_path.exists() and not overwrite:
        return True, "skipped"
    
    # Validate input files exist
    if not cmcr_path.exists():
        return False, f"CMCR file not found: {cmcr_path}"
    
    if not cmtr_path.exists():
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
        cmcr_path=cmcr_path,
        cmtr_path=cmtr_path,
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
    # Optional: RF Geometry with LNL Fitting
    # =========================================================================
    
    if not skip_rf_update:
        logger.info(f"  Running RF geometry + LNL fitting...")
        cover_range = (-60, 0)
        frame_rate = 15.0
        
        # Prepare LNL data (stimulus movie and spike frames)
        try:
            movie_array, spike_frames_dict = prepare_lnl_data_from_session(
                session=session,
                cover_range=cover_range,
            )
            
            if movie_array is not None and spike_frames_dict is not None:
                logger.debug(f"  Stimulus movie loaded: {movie_array.shape}")
                logger.debug(f"  Spike frames available for {len(spike_frames_dict)} units")
                
                # Clear RF geometry step from completed_steps so it runs again
                session.completed_steps.discard("extract_rf_geometry")
                
                # Run RF geometry extraction with LNL
                session = extract_rf_geometry_session(
                    session=session,
                    movie_array=movie_array,
                    spike_frames_dict=spike_frames_dict,
                    cover_range=cover_range,
                    frame_rate=frame_rate,
                )
            else:
                logger.info(f"  LNL fitting skipped (missing stimulus or spike data)")
        except Exception as e:
            logger.warning(f"  LNL fitting failed: {e}")
    
    # =========================================================================
    # Save final result
    # =========================================================================
    session.save(output_path=output_path, overwrite=overwrite)
    
    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Units: {session.unit_count}, Steps: {len(session.completed_steps)}")
    
    if session.warnings:
        logger.debug(f"  Warnings: {len(session.warnings)}")
    
    # =========================================================================
    # Generate Manual Label Images
    # =========================================================================
    if create_manual_label:
        logger.info(f"  Generating manual label images...")
        try:
            label_dir = manual_label_dir or MANUAL_LABEL_DIR
            label_output = label_dir / dataset_id
            plot_types = manual_label_plots or MANUAL_LABEL_PLOTS
            
            generated = generate_manual_label_images(
                hdf5_path=output_path,
                output_dir=label_output,
                plot_types=plot_types,
            )
            logger.info(f"  Generated {len(generated)} manual label images")
        except Exception as e:
            logger.warning(f"  Manual label generation failed: {e}")
    
    return True, None


# =============================================================================
# Batch Processing
# =============================================================================

def run_batch(
    search_folders: List[Path],
    output_dir: Path,
    start_index: int = 0,
    end_index: Optional[int] = None,
    skip_rf_update: bool = False,
    overwrite: bool = False,
    create_manual_label: bool = True,
    manual_label_dir: Optional[Path] = None,
    manual_label_plots: Optional[dict] = None,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Run batch processing on all discovered CMCR/CMTR pairs.
    
    Args:
        search_folders: Directories to search for CMCR files
        output_dir: Output directory
        start_index: Starting index (0-based)
        end_index: Ending index (exclusive, None for all)
        skip_rf_update: Skip RF geometry + LNL fitting
        overwrite: Whether to overwrite existing files
        create_manual_label: Generate manual label images
        manual_label_dir: Directory for manual label images
        manual_label_plots: Dict of plot types to include
    
    Returns:
        Tuple of (successful, skipped, failed) lists
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover CMCR files
    logger.info(f"Searching for CMCR files in {len(search_folders)} folder(s)...")
    cmcr_files = discover_cmcr_files(search_folders)
    
    if not cmcr_files:
        logger.error("No CMCR files found in search folders")
        return [], [], []
    
    total_files = len(cmcr_files)
    logger.info(f"Found {total_files} CMCR files total")
    
    # Find matching CMTR files
    matched_pairs = []
    unmatched = []
    
    for cmcr_path in cmcr_files:
        cmtr_path = find_matching_cmtr(cmcr_path)
        if cmtr_path:
            matched_pairs.append((cmcr_path, cmtr_path))
        else:
            unmatched.append(cmcr_path)
    
    if unmatched:
        logger.warning(f"{len(unmatched)} CMCR files have no matching CMTR:")
        for p in unmatched[:5]:
            logger.warning(f"  - {p.name}")
        if len(unmatched) > 5:
            logger.warning(f"  ... and {len(unmatched) - 5} more")
    
    logger.info(f"Matched {len(matched_pairs)} CMCR/CMTR pairs")
    
    # Apply index range
    if end_index is not None:
        matched_pairs = matched_pairs[start_index:end_index]
    else:
        matched_pairs = matched_pairs[start_index:]
    
    logger.info(f"Processing {len(matched_pairs)} pairs (index {start_index} to {end_index or total_files})")
    
    # Track results
    successful: List[str] = []
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []
    
    # Process each matched pair
    start_time = time.time()
    
    for i, (cmcr_path, cmtr_path) in enumerate(matched_pairs, 1):
        dataset_id = get_dataset_id_from_cmcr(cmcr_path)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(matched_pairs)}] {dataset_id}")
        logger.info(f"{'='*60}")
        
        try:
            success, error = process_single_recording(
                cmcr_path=cmcr_path,
                cmtr_path=cmtr_path,
                output_dir=output_dir,
                skip_rf_update=skip_rf_update,
                overwrite=overwrite,
                create_manual_label=create_manual_label,
                manual_label_dir=manual_label_dir,
                manual_label_plots=manual_label_plots,
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
        description="Batch process CMCR/CMTR files through unified pipeline (11 steps + RF/LNL)"
    )
    parser.add_argument(
        "--search-folders", "-s",
        type=Path,
        nargs="+",
        default=None,
        help="Directories to search for CMCR files (default: from config.py)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index (0-based, default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending index (exclusive, default: all)"
    )
    parser.add_argument(
        "--skip-rf-update",
        action="store_true",
        default=None,
        help="Skip RF geometry + LNL fitting step"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=None,
        help="Enable debug logging"
    )
    parser.add_argument(
        "--skip-manual-label",
        action="store_true",
        default=False,
        help="Skip manual label image generation"
    )
    
    args = parser.parse_args()
    
    # Merge CLI args with config defaults
    search_folders = args.search_folders if args.search_folders else SEARCH_FOLDERS
    output_dir = args.output if args.output else OUTPUT_DIR
    skip_rf_update = args.skip_rf_update if args.skip_rf_update is not None else SKIP_RF_UPDATE
    overwrite = args.overwrite if args.overwrite is not None else OVERWRITE_EXISTING
    debug = args.debug if args.debug is not None else DEBUG
    create_manual_label = not args.skip_manual_label and CREATE_MANUAL_LABEL
    
    # Validate search folders
    if not search_folders:
        print(red_warning("No search folders specified!"))
        print("Either:")
        print("  1. Add folders to yan_pipeline/config.py SEARCH_FOLDERS list")
        print("  2. Use --search-folders CLI argument")
        sys.exit(1)
    
    # Setup logging
    setup_logging(level=logging.DEBUG if debug else logging.INFO)
    
    print("=" * 70)
    print("Batch Processing: CMCR Discovery + Unified Pipeline")
    print("=" * 70)
    print(f"Search folders: {len(search_folders)}")
    for folder in search_folders:
        print(f"  - {folder}")
    print(f"Output:         {output_dir}")
    print(f"Range:          {args.start} to {args.end or 'end'}")
    print(f"Skip RF update: {skip_rf_update}")
    print(f"Overwrite:      {overwrite}")
    print(f"Manual label:   {create_manual_label}")
    print(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        successful, skipped, failed = run_batch(
            search_folders=search_folders,
            output_dir=output_dir,
            start_index=args.start,
            end_index=args.end,
            skip_rf_update=skip_rf_update,
            overwrite=overwrite,
            create_manual_label=create_manual_label,
            manual_label_dir=MANUAL_LABEL_DIR,
            manual_label_plots=MANUAL_LABEL_PLOTS,
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
