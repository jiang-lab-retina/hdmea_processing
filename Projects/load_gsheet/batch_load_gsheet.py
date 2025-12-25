"""
Batch Google Sheet to HDF5 Metadata Loader - Session Workflow

Processes multiple HDF5 files from rf_sta_output_20251225 folder,
adds Google Sheet metadata using the session workflow pattern,
and exports to export folder.

Workflow:
1. Load existing HDF5 data into a session
2. Add gsheet metadata row to session
3. Save results to export HDF5 file (preserves all existing data)

Features:
- Uses PipelineSession for consistent workflow pattern
- Loads gsheet only once for efficiency
- Preserves all existing HDF5 data (only adds/updates gsheet_row)
- Progress tracking and error handling

Usage:
    python batch_load_gsheet.py

Author: Generated for experimental analysis
Date: 2024-12
"""

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import pandas as pd

# Import session utilities
from hdmea.pipeline import create_session, PipelineSession

# Import functions from load_gsheet module
from load_gsheet import (
    import_gsheet_v2,
    find_gsheet_row,
    hdf5_to_gsheet_filename,
    _write_value_to_hdf5,
    CREDENTIALS_PATH,
    SHEET_NAME,
    CSV_CACHE_PATH,
)

# Enable logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Input directory containing HDF5 files
INPUT_DIR = Path(__file__).parent.parent / "rf_sta_measure" / "rf_sta_output_20251225"

# Output directory for processed files  
EXPORT_DIR = Path(__file__).parent / "export_gsheet_20251225"

# Index range: (start, end) for slicing files, or None for all files
# Examples:
#   None          -> process all files
#   (0, 10)       -> process first 10 files (index 0-9)
#   (5, 15)       -> process files at index 5-14
#   (2, 3)        -> process only file at index 2 (third file)
INDEX_RANGE = None  # Default: process all files


# =============================================================================
# Session Loading
# =============================================================================

def load_hdf5_to_session(hdf5_path: Path, dataset_id: str = None) -> PipelineSession:
    """
    Load an existing HDF5 file into a PipelineSession.
    
    Loads minimal data needed for gsheet integration (metadata only).
    
    Args:
        hdf5_path: Path to existing HDF5 file
        dataset_id: Optional dataset ID (uses filename stem if not provided)
        
    Returns:
        PipelineSession with source file tracked
    """
    if dataset_id is None:
        dataset_id = hdf5_path.stem
    
    session = create_session(dataset_id=dataset_id)
    session.hdf5_path = hdf5_path  # Track source file
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get dataset_id from attributes if available
        if 'dataset_id' in f.attrs:
            session.metadata['original_dataset_id'] = f.attrs['dataset_id']
        
        # Load existing metadata (to check for existing gsheet_row)
        if 'metadata' in f:
            for key in f['metadata'].keys():
                try:
                    item = f[f'metadata/{key}']
                    if isinstance(item, h5py.Dataset):
                        session.metadata[key] = item[()]
                    elif isinstance(item, h5py.Group):
                        session.metadata[key] = _read_group_to_dict(item)
                except Exception as e:
                    logger.debug(f"Could not read metadata/{key}: {e}")
    
    session.completed_steps.add('load_hdf5')
    logger.info(f"Loaded session: {dataset_id}")
    return session


def _read_group_to_dict(group: h5py.Group) -> Dict[str, Any]:
    """Recursively read HDF5 group to dict."""
    result = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = item[()]
        elif isinstance(item, h5py.Group):
            result[key] = _read_group_to_dict(item)
    return result


# =============================================================================
# Gsheet Session Integration
# =============================================================================

def add_gsheet_to_session(
    session: PipelineSession,
    gsheet_df: pd.DataFrame,
) -> PipelineSession:
    """
    Add gsheet metadata row to the session.
    
    Finds the matching gsheet row for the session's source file
    and stores it in session.metadata['gsheet_row'].
    
    Args:
        session: PipelineSession with source file loaded
        gsheet_df: Pre-loaded gsheet DataFrame
        
    Returns:
        Updated PipelineSession with gsheet_row in metadata
    """
    # Find matching gsheet row
    gsheet_row = find_gsheet_row(gsheet_df, str(session.hdf5_path))
    
    if gsheet_row is None:
        logger.warning(f"No gsheet row found for {session.dataset_id}")
        return session
    
    logger.info(f"Found gsheet match: {hdf5_to_gsheet_filename(str(session.hdf5_path))}")
    
    # Convert Series to dict and store in session metadata
    gsheet_dict = {}
    for col_name, value in gsheet_row.items():
        # Sanitize key for HDF5 compatibility
        safe_key = str(col_name).replace('/', '_').replace('\\', '_')
        gsheet_dict[safe_key] = value
    
    session.metadata['gsheet_row'] = gsheet_dict
    session.completed_steps.add('add_gsheet')
    
    logger.info(f"Added {len(gsheet_dict)} gsheet fields to session")
    return session


# =============================================================================
# HDF5 Saving
# =============================================================================

def save_gsheet_to_hdf5(session: PipelineSession, output_path: Path = None) -> Path:
    """
    Save gsheet metadata to HDF5 file.
    
    Copies the source HDF5 file to the export directory and adds gsheet_row
    to metadata, preserving all existing data.
    
    Args:
        session: PipelineSession with gsheet_row in metadata
        output_path: Path to save to (uses EXPORT_DIR/{dataset_id}.h5 if not provided)
        
    Returns:
        Path to saved file
    """
    if output_path is None:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORT_DIR / f"{session.dataset_id}.h5"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy source file to preserve all existing data
    source_path = session.hdf5_path
    if source_path and source_path.exists():
        logger.info(f"Copying source HDF5 to: {output_path}")
        shutil.copy2(source_path, output_path)
    else:
        logger.warning(f"Source file not found: {source_path}")
        return None
    
    # Get gsheet data from session
    gsheet_data = session.metadata.get('gsheet_row')
    
    if gsheet_data is None:
        logger.warning(f"No gsheet_row in session metadata, skipping save")
        return output_path
    
    logger.info(f"Adding gsheet metadata to: {output_path}")
    
    # Open in append mode to preserve existing data
    with h5py.File(output_path, 'a') as f:
        # Ensure metadata group exists
        if 'metadata' not in f:
            f.create_group('metadata')
        
        # Remove existing gsheet_row if present (to update)
        gsheet_path = 'metadata/gsheet_row'
        if gsheet_path in f:
            del f[gsheet_path]
        
        # Create gsheet_row group
        gsheet_group = f.create_group(gsheet_path)
        
        # Write each column as a dataset
        for col_name, value in gsheet_data.items():
            try:
                _write_value_to_hdf5(gsheet_group, col_name, value)
            except Exception as e:
                logger.warning(f"Could not write column '{col_name}': {e}")
    
    session.completed_steps.add('save_gsheet')
    logger.info(f"Saved gsheet metadata ({len(gsheet_data)} fields) to: {output_path}")
    return output_path


# =============================================================================
# Batch Processing
# =============================================================================

def process_single_file(
    input_path: Path,
    output_path: Path,
    gsheet_df: pd.DataFrame,
) -> bool:
    """
    Process a single HDF5 file using session workflow.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        gsheet_df: Pre-loaded gsheet DataFrame
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Step 1: Load HDF5 into session
        session = load_hdf5_to_session(input_path)
        
        # Step 2: Add gsheet to session
        session = add_gsheet_to_session(session, gsheet_df)
        
        # Check if gsheet was added
        if 'gsheet_row' not in session.metadata:
            logger.warning(f"No gsheet data added for {input_path.name}")
            return False
        
        # Step 3: Save to HDF5
        result_path = save_gsheet_to_hdf5(session, output_path)
        
        if result_path is None:
            return False
        
        logger.info(f"Session completed: {session.completed_steps}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_gsheet_in_hdf5(hdf5_path: Path) -> bool:
    """
    Verify that gsheet_row exists in the HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        True if gsheet_row exists and has data
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'metadata/gsheet_row' in f:
                gsheet_group = f['metadata/gsheet_row']
                return len(gsheet_group.keys()) > 0
        return False
    except Exception:
        return False


def batch_process(index_range: tuple = None):
    """
    Process HDF5 files in the input directory using session workflow.
    
    Args:
        index_range: Tuple (start, end) for slicing files, or None for all files.
                     Uses Python slice semantics: [start:end] (end is exclusive).
                     Examples:
                       None      -> all files
                       (0, 10)   -> first 10 files (index 0-9)
                       (5, 15)   -> files at index 5-14
                       (2, 3)    -> only file at index 2
    """
    print("=" * 70)
    print("Batch Google Sheet to HDF5 - Session Workflow")
    print("=" * 70)
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {EXPORT_DIR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check input directory
    if not INPUT_DIR.exists():
        print(f"\nError: Input directory not found: {INPUT_DIR}")
        return
    
    # Get list of HDF5 files (sorted by name)
    all_hdf5_files = sorted(INPUT_DIR.glob("*.h5"))
    
    if not all_hdf5_files:
        print(f"\nError: No HDF5 files found in {INPUT_DIR}")
        return
    
    total_files = len(all_hdf5_files)
    print(f"\nFound {total_files} HDF5 files total")
    
    # Apply index range filter
    if index_range is not None:
        start_idx, end_idx = index_range
        # Validate range
        if start_idx < 0:
            start_idx = 0
        if end_idx > total_files:
            end_idx = total_files
        if start_idx >= end_idx:
            print(f"\nError: Invalid range ({start_idx}, {end_idx})")
            return
        
        hdf5_files = all_hdf5_files[start_idx:end_idx]
        print(f"Processing range [{start_idx}:{end_idx}] -> {len(hdf5_files)} files")
        if len(hdf5_files) <= 5:
            for idx, f in enumerate(hdf5_files):
                print(f"  [{start_idx + idx}] {f.name}")
    else:
        hdf5_files = all_hdf5_files
        print("Processing all files")
    
    # Create output directory
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Google Sheet (ONCE for all files)
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 1: Loading Google Sheet (once for all files)")
    print("-" * 70)
    
    gsheet_df = import_gsheet_v2(
        sheet_name=SHEET_NAME,
        cred=str(CREDENTIALS_PATH),
        csv_path=str(CSV_CACHE_PATH),
    )
    
    print(f"Loaded {len(gsheet_df)} rows from Google Sheet")
    
    # =========================================================================
    # Step 2: Process files using session workflow
    # =========================================================================
    print("\n" + "-" * 70)
    print("Step 2: Processing files (session workflow)")
    print("-" * 70)
    
    successful = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    for i, input_path in enumerate(hdf5_files, 1):
        print(f"\n[{i}/{len(hdf5_files)}] {input_path.name}")
        
        output_path = EXPORT_DIR / input_path.name
        
        # Skip if output already exists and has gsheet data
        if output_path.exists() and verify_gsheet_in_hdf5(output_path):
            logger.info(f"  Output already exists with gsheet data, skipping")
            skipped += 1
            continue
        
        file_start = time.time()
        
        if process_single_file(input_path, output_path, gsheet_df):
            successful += 1
            elapsed = time.time() - file_start
            logger.info(f"  Completed in {elapsed:.1f}s")
            
            # Verify the result
            if verify_gsheet_in_hdf5(output_path):
                logger.info(f"  Verified: gsheet_row added successfully")
            else:
                logger.warning(f"  Warning: gsheet_row verification failed")
        else:
            failed += 1
    
    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total files:   {len(hdf5_files)}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")
    print(f"Skipped:       {skipped}")
    print(f"Total time:    {total_time:.1f}s")
    print(f"Output dir:    {EXPORT_DIR}")
    
    if successful > 0:
        print(f"\nProcessed files saved to: {EXPORT_DIR}")


def main():
    """
    Main entry point for batch processing.
    """
    # Run with configured index range (None = all files)
    batch_process(index_range=INDEX_RANGE)


if __name__ == "__main__":
    main()
