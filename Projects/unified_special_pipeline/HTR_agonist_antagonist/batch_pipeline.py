#!/usr/bin/env python
"""
Batch Processing: HTR Agonist/Antagonist Pipeline

Processes CMCR/CMTR recordings with dynamic playlist selection based on
the gsheet "Condition" column. If the condition matches an available
playlist (after stripping "()"), runs the full 11-step pipeline. Otherwise,
runs basic loading only.

Usage:
    python batch_pipeline.py
    
    # Process specific range:
    python batch_pipeline.py --start 0 --end 3
    
    # Force overwrite existing files:
    python batch_pipeline.py --overwrite

Steps executed (when playlist available):
- Step 1: Load recording with eimage_sta
- Step 2: Add section time from playlist (dynamic)
- Step 3: Section spike times
- Step 3b: Add section time from analog signal
- Step 3c: Section spike times for analog stimuli
- Step 4: Compute STA
- Step 5: Add CMTR/CMCR metadata
- Step 6: Extract soma geometry
- Step 7: Extract RF geometry
- Step 8: Add Google Sheet metadata
- Step 9: Add manual cell type labels
- Step 10: Compute AP tracking
- Step 11: Section by direction (DSGC)

Steps executed (basic load only):
- Step 1: Load recording with eimage_sta
- Step 5: Add CMTR/CMCR metadata
- Step 8: Add Google Sheet metadata
"""

import argparse
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from hdmea.pipeline import create_session

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
)

# Import config classes from unified pipeline
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

# Import local config (handle both module and direct execution)
try:
    from .specific_config import (
        DATA_FOLDER,
        OUTPUT_DIR,
        GSHEET_CSV_PATH,
        PLAYLIST_CSV_PATH,
        TEST_FILES,
        get_dataset_id_from_cmcr,
        get_cmcr_cmtr_paths,
    )
except ImportError:
    from specific_config import (
        DATA_FOLDER,
        OUTPUT_DIR,
        GSHEET_CSV_PATH,
        PLAYLIST_CSV_PATH,
        TEST_FILES,
        get_dataset_id_from_cmcr,
        get_cmcr_cmtr_paths,
    )


# =============================================================================
# Playlist Resolution
# =============================================================================

def load_available_playlists(playlist_csv_path: Path = PLAYLIST_CSV_PATH) -> Set[str]:
    """
    Load available playlist names from CSV.
    
    Args:
        playlist_csv_path: Path to playlist.csv
    
    Returns:
        Set of available playlist names
    """
    logger = logging.getLogger(__name__)
    
    if not playlist_csv_path.exists():
        logger.warning(f"Playlist CSV not found: {playlist_csv_path}")
        return set()
    
    try:
        df = pd.read_csv(playlist_csv_path)
        playlists = set(df["playlist_name"].dropna().tolist())
        logger.debug(f"Loaded {len(playlists)} playlists from {playlist_csv_path}")
        return playlists
    except Exception as e:
        logger.error(f"Failed to load playlist CSV: {e}")
        return set()


def load_gsheet_dataframe(gsheet_csv_path: Path = GSHEET_CSV_PATH) -> Optional[pd.DataFrame]:
    """
    Load gsheet data from CSV cache.
    
    Args:
        gsheet_csv_path: Path to gsheet_table.csv
    
    Returns:
        DataFrame with gsheet data, or None if load fails
    """
    logger = logging.getLogger(__name__)
    
    if not gsheet_csv_path.exists():
        logger.warning(f"Gsheet CSV not found: {gsheet_csv_path}")
        return None
    
    try:
        df = pd.read_csv(gsheet_csv_path)
        logger.debug(f"Loaded {len(df)} rows from gsheet CSV")
        return df
    except Exception as e:
        logger.error(f"Failed to load gsheet CSV: {e}")
        return None


def get_condition_from_gsheet(
    cmcr_filename: str,
    gsheet_df: pd.DataFrame,
) -> Optional[str]:
    """
    Look up the "Condition" column for a given CMCR file.
    
    Handles filename format differences (dashes vs dots):
    - Actual files: "2025.10.01-09.33.32-Rec.cmcr"
    - Gsheet format: "2025.10.01.09.33.32.Rec.cmcr"
    
    Args:
        cmcr_filename: CMCR filename (e.g., "2025.10.01-09.33.32-Rec.cmcr")
        gsheet_df: DataFrame with gsheet data
    
    Returns:
        Condition value, or None if not found
    """
    logger = logging.getLogger(__name__)
    
    # Normalize filename for matching: convert dashes to dots for comparison
    # "2025.10.01-09.33.32-Rec.cmcr" -> "2025.10.01.09.33.32.Rec.cmcr"
    normalized_filename = cmcr_filename.replace("-", ".")
    
    # Also extract the base pattern (date.time) for fuzzy matching
    # "2025.10.01-09.33.32-Rec" -> "2025.10.01" and "09.33.32"
    stem = Path(cmcr_filename).stem  # Remove .cmcr
    parts = stem.replace("-", ".").split(".")
    if len(parts) >= 6:
        date_pattern = ".".join(parts[0:3])  # 2025.10.01
        time_pattern = ".".join(parts[3:6])  # 09.33.32
    else:
        date_pattern = ""
        time_pattern = ""
    
    # Search for the file in the File_name column
    for _, row in gsheet_df.iterrows():
        file_name = str(row.get("File_name", ""))
        
        # Try multiple matching strategies
        if (cmcr_filename in file_name or 
            file_name in cmcr_filename or
            normalized_filename in file_name or
            file_name in normalized_filename or
            (date_pattern and time_pattern and date_pattern in file_name and time_pattern in file_name)):
            condition = row.get("Condition", "")
            if pd.isna(condition):
                return None
            return str(condition).strip()
    
    logger.warning(f"No gsheet row found for {cmcr_filename}")
    return None


def resolve_playlist(
    condition: Optional[str],
    available_playlists: Set[str],
) -> Optional[str]:
    """
    Resolve condition to playlist name.
    
    Strips trailing "()" from condition and checks if it exists
    in the available playlists.
    
    Args:
        condition: Condition value from gsheet
        available_playlists: Set of available playlist names
    
    Returns:
        Playlist name if found, None otherwise
    """
    if not condition:
        return None
    
    # Strip trailing "()" if present
    playlist_name = condition.rstrip("()")
    
    # Check if it's in the available playlists
    if playlist_name in available_playlists:
        return playlist_name
    
    return None


# =============================================================================
# Single Recording Processing
# =============================================================================

def process_single_recording(
    cmcr_path: Path,
    cmtr_path: Path,
    output_dir: Path,
    playlist_name: Optional[str] = None,
    gsheet_df: Optional[pd.DataFrame] = None,
    load_config: Optional[LoadRecordingConfig] = None,
    section_config: Optional[SectionTimeConfig] = None,
    section_analog_config: Optional[SectionTimeAnalogConfig] = None,
    geometry_config: Optional[GeometryConfig] = None,
    ap_config: Optional[APTrackingConfig] = None,
    dsgc_config: Optional[DSGCConfig] = None,
    overwrite: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Process a single recording.
    
    If playlist_name is provided, runs full 11-step pipeline.
    Otherwise, runs basic loading only (steps 1, 5, 8).
    
    Args:
        cmcr_path: Path to CMCR file
        cmtr_path: Path to CMTR file
        output_dir: Output directory
        playlist_name: Playlist name for section time (None for basic load)
        gsheet_df: Pre-loaded gsheet DataFrame for add_gsheet_step
        load_config: Configuration for loading
        section_config: Configuration for sectioning
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
        section_config = SectionTimeConfig()
    if section_analog_config is None:
        section_analog_config = SectionTimeAnalogConfig()
    if geometry_config is None:
        geometry_config = GeometryConfig()
    if ap_config is None:
        ap_config = APTrackingConfig()
    if dsgc_config is None:
        dsgc_config = DSGCConfig()
    
    dataset_id = get_dataset_id_from_cmcr(cmcr_path.name)
    output_path = output_dir / f"{dataset_id}.h5"
    
    # Check if output already exists
    if output_path.exists() and not overwrite:
        return True, "skipped"
    
    # Validate input files exist
    if not cmcr_path.exists():
        return False, f"CMCR file not found: {cmcr_path}"
    
    if not cmtr_path.exists():
        return False, f"CMTR file not found: {cmtr_path}"
    
    mode = "full pipeline" if playlist_name else "basic load"
    logger.info(f"Processing {dataset_id} ({mode})...")
    if playlist_name:
        logger.info(f"  Playlist: {playlist_name}")
    logger.debug(f"  CMCR: {cmcr_path}")
    logger.debug(f"  CMTR: {cmtr_path}")
    
    # =========================================================================
    # Step 1: Load recording with eimage_sta (always run)
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
    
    if playlist_name:
        # =====================================================================
        # Full Pipeline: Steps 2-11
        # =====================================================================
        
        # Step 2: Add section time from playlist (dynamic)
        session = add_section_time_step(
            playlist_name=playlist_name,
            session=session,
        )
        
        # Step 3: Section spike times
        session = section_spike_times_step(
            pad_margin=section_config.pad_margin,
            session=session,
        )
        
        # Step 3b: Add section time from analog signal
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
        
        # Step 5: Add metadata
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
        session = add_gsheet_step(
            gsheet_df=gsheet_df,
            session=session,
        )
        
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
    else:
        # =====================================================================
        # Basic Load: Steps 1, 5, 8 only
        # =====================================================================
        
        # Step 5: Add metadata
        session = add_metadata_step(session=session)
        
        # Step 8: Add gsheet metadata
        session = add_gsheet_step(
            gsheet_df=gsheet_df,
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

def discover_cmcr_files(data_folder: Path) -> List[str]:
    """
    Discover all CMCR files in the data folder.
    
    Args:
        data_folder: Folder to search for CMCR files
    
    Returns:
        List of CMCR filenames (sorted)
    """
    logger = logging.getLogger(__name__)
    
    if not data_folder.exists():
        logger.warning(f"Data folder not found: {data_folder}")
        return []
    
    cmcr_files = sorted([f.name for f in data_folder.glob("*.cmcr")])
    logger.info(f"Discovered {len(cmcr_files)} CMCR files in {data_folder}")
    
    return cmcr_files


def run_batch(
    data_folder: Path = DATA_FOLDER,
    output_dir: Path = OUTPUT_DIR,
    test_files: Optional[List[str]] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    overwrite: bool = False,
    discover_all: bool = False,
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Run batch processing on files.
    
    Args:
        data_folder: Folder containing CMCR/CMTR files
        output_dir: Output directory
        test_files: List of CMCR filenames to process (ignored if discover_all=True)
        start_index: Starting index (0-based)
        end_index: Ending index (exclusive, None for all)
        overwrite: Whether to overwrite existing files
        discover_all: If True, discover all CMCR files in data_folder
    
    Returns:
        Tuple of (successful, skipped, failed) lists
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine files to process
    if discover_all:
        all_files = discover_cmcr_files(data_folder)
    else:
        all_files = test_files if test_files is not None else TEST_FILES
    
    # Load playlist and gsheet data
    logger.info("Loading configuration data...")
    available_playlists = load_available_playlists()
    gsheet_df = load_gsheet_dataframe()
    
    if not available_playlists:
        logger.warning("No playlists loaded - all files will use basic load")
    
    if gsheet_df is None:
        logger.warning("Gsheet not loaded - condition lookup will fail")
    
    # Apply index range
    if end_index is not None:
        files_to_process = all_files[start_index:end_index]
    else:
        files_to_process = all_files[start_index:]
    
    logger.info(f"Processing {len(files_to_process)} of {len(all_files)} files")
    
    # Track results
    successful: List[str] = []
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []
    
    # Process each file
    start_time = time.time()
    
    for i, cmcr_filename in enumerate(files_to_process, 1):
        dataset_id = get_dataset_id_from_cmcr(cmcr_filename)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(files_to_process)}] {dataset_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Get file paths
            cmcr_path, cmtr_path = get_cmcr_cmtr_paths(cmcr_filename, data_folder)
            
            # Look up condition and resolve playlist
            condition = None
            playlist_name = None
            
            if gsheet_df is not None:
                condition = get_condition_from_gsheet(cmcr_filename, gsheet_df)
                if condition:
                    playlist_name = resolve_playlist(condition, available_playlists)
                    if playlist_name:
                        logger.info(f"  Condition: {condition}")
                        logger.info(f"  Resolved playlist: {playlist_name}")
                    else:
                        logger.info(f"  Condition: {condition} (no matching playlist)")
            
            # Process the recording
            success, error = process_single_recording(
                cmcr_path=cmcr_path,
                cmtr_path=cmtr_path,
                output_dir=output_dir,
                playlist_name=playlist_name,
                gsheet_df=gsheet_df,
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
    logger.info(f"Total files processed: {len(files_to_process)}")
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
        description="Batch process HTR agonist/antagonist recordings with dynamic playlist selection"
    )
    parser.add_argument(
        "--data-folder", type=Path, default=DATA_FOLDER,
        help=f"Folder containing CMCR/CMTR files (default: {DATA_FOLDER})"
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
        "--all", action="store_true", dest="discover_all",
        help="Process all CMCR files in data folder (instead of TEST_FILES list)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    # Determine file count for display
    if args.discover_all:
        file_count = len(discover_cmcr_files(args.data_folder))
        file_source = "all CMCR files in folder"
    else:
        file_count = len(TEST_FILES)
        file_source = "TEST_FILES list"
    
    print("=" * 70)
    print("Batch Processing: HTR Agonist/Antagonist Pipeline")
    print("=" * 70)
    print(f"Data:      {args.data_folder}")
    print(f"Output:    {args.output}")
    print(f"Files:     {file_count} ({file_source})")
    print(f"Range:     {args.start} to {args.end or 'end'}")
    print(f"Overwrite: {args.overwrite}")
    print(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        successful, skipped, failed = run_batch(
            data_folder=args.data_folder,
            output_dir=args.output,
            start_index=args.start,
            end_index=args.end,
            overwrite=args.overwrite,
            discover_all=args.discover_all,
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
