#!/usr/bin/env python
"""
Batch Processing: Set6a Image Alignment Pipeline

Processes recordings from the Excel mapping file through the unified pipeline.
Uses the playlist "play_optimization_set6_a_ipRGC_manual" for section time.

Input: Yan Dulce Corresponding Data.xlsx
Output: export/{dataset_id}.h5

Usage:
    python batch_pipeline.py
    
    # Process specific range:
    python batch_pipeline.py --start 0 --end 10
    
    # Force overwrite existing files:
    python batch_pipeline.py --overwrite
"""

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import PipelineSession, create_session

# Import step wrappers from unified pipeline
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
    set_6_compatibility_step,
)

# Import base configs from unified pipeline
from Projects.unified_pipeline.config import (
    setup_logging,
    LoadRecordingConfig,
    GeometryConfig,
    APTrackingConfig,
    DSGCConfig,
    green_success,
    red_warning,
)

# Import specific configs for this pipeline
from Projects.unified_special_pipeline.set6a_image_alignment.specific_config import (
    EXCEL_PATH,
    OUTPUT_DIR,
    COL_DATA_LOCATION,
    COL_DATA_FOLDER,
    COL_DATA_FILENAME,
    SectionTimeConfigOverride,
    SectionTimeAnalogConfigOverride,
    Set6CompatibilityConfig,
    resolve_drive_path,
    get_dataset_id,
)


# =============================================================================
# Helper Functions
# =============================================================================

def load_excel_data(excel_path: Path) -> pd.DataFrame:
    """
    Load recording data from Excel file.
    
    Args:
        excel_path: Path to Excel file
    
    Returns:
        DataFrame with MEA data columns
    
    Raises:
        FileNotFoundError: If Excel file doesn't exist
        ValueError: If required columns are missing
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    df = pd.read_excel(excel_path)
    
    # Validate required columns
    required_cols = [COL_DATA_LOCATION, COL_DATA_FOLDER, COL_DATA_FILENAME]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns in Excel file: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Filter out rows with missing data
    df_valid = df.dropna(subset=required_cols)
    
    return df_valid


def build_recording_list(df: pd.DataFrame) -> List[dict]:
    """
    Build list of recording dictionaries from DataFrame.
    
    Handles cells with multiple filenames separated by newlines.
    
    Args:
        df: DataFrame with MEA data columns
    
    Returns:
        List of dicts with cmcr_path, cmtr_path, dataset_id
    """
    recordings = []
    
    for _, row in df.iterrows():
        location = str(row[COL_DATA_LOCATION]).strip()
        folder = str(row[COL_DATA_FOLDER]).strip()
        filename_cell = str(row[COL_DATA_FILENAME]).strip()
        
        # Split by newlines in case multiple filenames are in one cell
        filenames = [f.strip() for f in filename_cell.split('\n') if f.strip()]
        
        for filename in filenames:
            try:
                cmcr_path, cmtr_path = resolve_drive_path(location, folder, filename)
                dataset_id = get_dataset_id(filename)
                
                recordings.append({
                    "cmcr_path": cmcr_path,
                    "cmtr_path": cmtr_path,
                    "dataset_id": dataset_id,
                    "location": location,
                    "folder": folder,
                    "filename": filename,
                })
            except ValueError as e:
                logging.warning(f"Skipping '{filename}' due to error: {e}")
                continue
    
    return recordings


# =============================================================================
# Single Recording Processing (All Steps)
# =============================================================================

def process_single_recording(
    cmcr_path: Path,
    cmtr_path: Path,
    dataset_id: str,
    output_dir: Path,
    load_config: Optional[LoadRecordingConfig] = None,
    section_config: Optional[SectionTimeConfigOverride] = None,
    section_analog_config: Optional[SectionTimeAnalogConfigOverride] = None,
    geometry_config: Optional[GeometryConfig] = None,
    ap_config: Optional[APTrackingConfig] = None,
    dsgc_config: Optional[DSGCConfig] = None,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Process a single recording through all pipeline steps.
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
        dataset_id: Unique identifier for the dataset
        output_dir: Output directory
        load_config: Configuration for loading
        section_config: Configuration for sectioning (with playlist override)
        section_analog_config: Configuration for analog section time
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
        section_config = SectionTimeConfigOverride()
    if section_analog_config is None:
        section_analog_config = SectionTimeAnalogConfigOverride()
    if geometry_config is None:
        geometry_config = GeometryConfig()
    if ap_config is None:
        ap_config = APTrackingConfig()
    if dsgc_config is None:
        dsgc_config = DSGCConfig()
    
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
    
    # Step 2: Add section time from playlist (using override config)
    session = add_section_time_step(
        playlist_name=section_config.playlist_name,
        session=session,
    )
    
    # Step 3: Section spike times
    session = section_spike_times_step(
        pad_margin=section_config.pad_margin,
        session=session,
    )
    
    # Step 3a: Create set6-compatible green_blue data from last 3 repeats
    compat_config = Set6CompatibilityConfig()
    session = set_6_compatibility_step(
        source_movie=compat_config.source_movie,
        target_movie=compat_config.target_movie,
        repeat_slice=compat_config.repeat_slice,
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
    excel_path: Path = EXCEL_PATH,
    output_dir: Path = OUTPUT_DIR,
    start_index: int = 0,
    end_index: Optional[int] = None,
    overwrite: bool = False,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Run batch processing on all recordings from Excel file.
    
    Args:
        excel_path: Path to Excel mapping file
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
    
    # Load Excel data
    logger.info(f"Reading Excel: {excel_path}")
    df = load_excel_data(excel_path)
    recordings = build_recording_list(df)
    total_recordings = len(recordings)
    
    logger.info(f"Found {total_recordings} valid recordings in Excel file")
    
    # Apply index range
    if end_index is not None:
        recordings = recordings[start_index:end_index]
    else:
        recordings = recordings[start_index:]
    
    logger.info(f"Processing {len(recordings)} of {total_recordings} recordings")
    
    # Track results
    successful: List[str] = []
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []
    
    # Process each recording
    start_time = time.time()
    
    for i, rec in enumerate(recordings, 1):
        dataset_id = rec["dataset_id"]
        cmcr_path = rec["cmcr_path"]
        cmtr_path = rec["cmtr_path"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(recordings)}] {dataset_id}")
        logger.info(f"{'='*60}")
        
        try:
            success, error = process_single_recording(
                cmcr_path=cmcr_path,
                cmtr_path=cmtr_path,
                dataset_id=dataset_id,
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
    logger.info(f"Total recordings processed: {len(recordings)}")
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
        description="Batch process set6a image alignment recordings through the unified pipeline"
    )
    parser.add_argument(
        "--excel", type=Path, default=EXCEL_PATH,
        help=f"Excel mapping file (default: {EXCEL_PATH})"
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
    print("Batch Processing: Set6a Image Alignment Pipeline")
    print("=" * 70)
    print(f"Excel:     {args.excel}")
    print(f"Output:    {args.output}")
    print(f"Playlist:  {SectionTimeConfigOverride.playlist_name}")
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
