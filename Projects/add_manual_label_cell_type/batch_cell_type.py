"""
Batch Processing: Add Manual Label Cell Type

Processes all HDF5 files in the input directory and adds manual cell type labels.
Follows the pipeline principles with progress tracking, skip existing, and error handling.
"""

from pathlib import Path
from datetime import datetime
import time
import logging

from add_manual_label_cell_type import process_single_file, MANUAL_LABEL_FOLDER

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path(r"M:\Python_Project\Data_Processing_2027\Projects\load_gsheet\export_gsheet_20251225")
OUTPUT_DIR = Path(__file__).parent / "manual_label_20251225"


# =============================================================================
# Batch Processing
# =============================================================================

def batch_process():
    """Process all HDF5 files in input directory."""
    print("=" * 70)
    print("Batch Processing: Add Manual Label Cell Type")
    print("=" * 70)
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Label dir:  {MANUAL_LABEL_FOLDER}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate input
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return
    
    hdf5_files = sorted(INPUT_DIR.glob("*.h5"))
    if not hdf5_files:
        print(f"Error: No HDF5 files found in {INPUT_DIR}")
        return
    
    print(f"\nFound {len(hdf5_files)} files to process")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process with tracking
    successful, failed, skipped = 0, 0, 0
    start_time = time.time()
    
    for i, input_path in enumerate(hdf5_files, 1):
        print(f"\n[{i}/{len(hdf5_files)}] {input_path.name}")
        
        output_path = OUTPUT_DIR / input_path.name
        
        # Skip if already processed (allows resume)
        if output_path.exists():
            print("  Already exists, skipping")
            skipped += 1
            continue
        
        file_start = time.time()
        
        if process_single_file(input_path, output_path, MANUAL_LABEL_FOLDER):
            successful += 1
            print(f"  Time: {time.time() - file_start:.1f}s")
        else:
            failed += 1
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Skipped:    {skipped}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Write processing log
    write_processing_log(hdf5_files, successful, failed, skipped, total_time)


def write_processing_log(files, successful, failed, skipped, total_time):
    """Write a log file summarizing the batch processing."""
    log_path = OUTPUT_DIR / "processing_log.txt"
    with open(log_path, 'w') as f:
        f.write("Batch Processing Log: Add Manual Label Cell Type\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Input:  {INPUT_DIR}\n")
        f.write(f"Output: {OUTPUT_DIR}\n")
        f.write(f"Labels: {MANUAL_LABEL_FOLDER}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Successful: {successful}\n")
        f.write(f"  Failed:     {failed}\n")
        f.write(f"  Skipped:    {skipped}\n")
        f.write(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nFiles ({len(files)} total):\n")
        for path in files:
            output_path = OUTPUT_DIR / path.name
            status = "OK" if output_path.exists() else "FAILED"
            f.write(f"  [{status}] {path.name}\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    batch_process()

