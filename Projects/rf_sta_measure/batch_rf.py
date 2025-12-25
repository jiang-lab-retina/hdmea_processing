"""
Batch RF-STA Receptive Field Measurement

Processes all HDF5 files from eimage_sta_output_20251225 folder,
performs RF geometry extraction (Gaussian, DoG, ON/OFF fits),
and exports to rf_sta_output_20251225 folder.

Usage:
    python batch_rf.py

Author: Generated for experimental analysis
Date: 2024-12
"""

from pathlib import Path
import time
from datetime import datetime

# Import RF session functions
from rf_session import (
    load_hdf5_to_session,
    extract_rf_geometry_session,
    save_rf_geometry_to_hdf5,
    STA_FEATURE_NAME,
    FRAME_RANGE,
    THRESHOLD_FRACTION,
)

# Enable logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Input directory containing HDF5 files
INPUT_DIR = Path(__file__).parent.parent / "sta_quantification" / "eimage_sta_output_20251225"

# Output directory for processed files
OUTPUT_DIR = Path(__file__).parent / "rf_sta_output_20251225"


# =============================================================================
# Batch Processing
# =============================================================================

def process_single_file(input_path: Path, output_path: Path) -> bool:
    """
    Process a single HDF5 file for RF geometry extraction.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Loading: {input_path.name}")
        session = load_hdf5_to_session(input_path)
        
        if len(session.units) == 0:
            logger.warning(f"No units found in {input_path.name}, skipping")
            return False
        
        # Count units with STA data
        units_with_sta = sum(
            1 for u in session.units.values()
            if 'features' in u and STA_FEATURE_NAME in u.get('features', {}) 
            and 'data' in u['features'][STA_FEATURE_NAME]
        )
        
        if units_with_sta == 0:
            logger.warning(f"No units with STA data in {input_path.name}, skipping")
            return False
        
        logger.info(f"  Found {units_with_sta} units with STA data")
        
        # Extract RF geometry
        logger.info(f"  Extracting RF geometry...")
        session = extract_rf_geometry_session(
            session,
            frame_range=FRAME_RANGE,
            threshold_fraction=THRESHOLD_FRACTION,
        )
        
        # Save to output
        logger.info(f"  Saving to: {output_path.name}")
        save_rf_geometry_to_hdf5(session, output_path)
        
        # Count successful geometries
        geom_count = sum(
            1 for u in session.units.values()
            if u.get('features', {}).get(STA_FEATURE_NAME, {}).get('sta_geometry') is not None
        )
        logger.info(f"  Completed: {geom_count}/{units_with_sta} units with RF geometry")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_process():
    """
    Process all HDF5 files in the input directory.
    """
    print("=" * 70)
    print("Batch RF-STA Receptive Field Measurement")
    print("=" * 70)
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Feature:    {STA_FEATURE_NAME}")
    print(f"Frame range: {FRAME_RANGE}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check input directory
    if not INPUT_DIR.exists():
        print(f"\nError: Input directory not found: {INPUT_DIR}")
        return
    
    # Get list of HDF5 files
    hdf5_files = sorted(INPUT_DIR.glob("*.h5"))
    
    if not hdf5_files:
        print(f"\nError: No HDF5 files found in {INPUT_DIR}")
        return
    
    print(f"\nFound {len(hdf5_files)} HDF5 files to process")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    successful = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    print("\n" + "-" * 70)
    print("Processing files...")
    print("-" * 70)
    
    for i, input_path in enumerate(hdf5_files, 1):
        print(f"\n[{i}/{len(hdf5_files)}] {input_path.name}")
        
        output_path = OUTPUT_DIR / input_path.name
        
        # Skip if output already exists
        if output_path.exists():
            logger.info(f"  Output already exists, skipping")
            skipped += 1
            continue
        
        file_start = time.time()
        
        if process_single_file(input_path, output_path):
            successful += 1
            elapsed = time.time() - file_start
            logger.info(f"  Time: {elapsed:.1f}s")
        else:
            failed += 1
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total files:   {len(hdf5_files)}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")
    print(f"Skipped:       {skipped}")
    print(f"Total time:    {total_time:.1f}s ({total_time/60:.1f} min)")
    if successful > 0:
        print(f"Avg time/file: {total_time/successful:.1f}s")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Write processing log
    log_path = OUTPUT_DIR / "processing_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Batch RF-STA Processing Log\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Input dir:  {INPUT_DIR}\n")
        f.write(f"Output dir: {OUTPUT_DIR}\n")
        f.write(f"Feature:    {STA_FEATURE_NAME}\n")
        f.write(f"Frame range: {FRAME_RANGE}\n")
        f.write(f"\nTotal files:   {len(hdf5_files)}\n")
        f.write(f"Successful:    {successful}\n")
        f.write(f"Failed:        {failed}\n")
        f.write(f"Skipped:       {skipped}\n")
        f.write(f"Total time:    {total_time:.1f}s\n")
        f.write(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nProcessed files:\n")
        for input_path in hdf5_files:
            output_path = OUTPUT_DIR / input_path.name
            status = "OK" if output_path.exists() else "FAILED"
            f.write(f"  [{status}] {input_path.name}\n")
    
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    batch_process()

