"""
Batch Processing Script - Adding CMTR/CMCR Metadata to Existing HDF5 Files

This script batch processes all HDF5 files in a folder and adds:
  - Extended CMTR unit metadata (row, column, SNR, separability, etc.) -> units/*/unit_meta
  - System metadata from CMCR -> metadata/cmcr_meta
  - System metadata from CMTR -> metadata/cmtr_meta

NOTE: As of the latest update, load_recording_with_eimage_sta() now includes
both unit_meta (load_unit_meta=True) and sys_meta (load_sys_meta=True) loading
by default. This script is for:
  1. Adding metadata to HDF5 files created BEFORE this integration
  2. Updating metadata in existing files if source data has changed
"""
from hdmea.io.cmtr import add_cmtr_unit_info
from hdmea.io.cmcr import add_sys_meta_info
from pathlib import Path
import time
import shutil

# Enable logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================
DATA_FOLDER = Path("Projects/pipeline_test/data")

# Set to False to process all files, True for dry run (just list files)
DRY_RUN = False

# Set to True to overwrite existing unit_meta groups
FORCE_OVERWRITE = True

# NEW: Set to True to only process the first file
FIRST_FILE_ONLY = False

# NEW: Output folder for processed files (None = update in place)
OUTPUT_FOLDER: Path | None = Path("Projects/pipeline_test/output")


# =============================================================================
# Batch Processing
# =============================================================================
def batch_add_metadata(
    data_folder: Path,
    dry_run: bool = False,
    force: bool = True,
    first_only: bool = False,
    output_folder: Path | None = None,
):
    """
    Batch process all HDF5 files in a folder to add CMTR/CMCR metadata.
    
    Adds:
      - Extended CMTR unit metadata (row, column, SNR, etc.) -> units/*/unit_meta
      - CMCR file metadata -> metadata/cmcr_meta
      - CMTR file metadata -> metadata/cmtr_meta
    
    Args:
        data_folder: Path to folder containing HDF5 files
        dry_run: If True, only list files without processing
        force: If True, overwrite existing metadata groups
        first_only: If True, only process the first file
        output_folder: If provided, copy files here before processing (instead of in-place)
    """
    # Find all HDF5 files
    hdf5_files = sorted(data_folder.glob("*.h5"))
    
    # Limit to first file if requested
    if first_only and hdf5_files:
        hdf5_files = hdf5_files[:1]
    
    print("=" * 70)
    print(f"BATCH PROCESSING: Adding CMTR/CMCR Metadata")
    print("=" * 70)
    print(f"  Data folder: {data_folder}")
    print(f"  HDF5 files found: {len(hdf5_files)}")
    print(f"  Dry run: {dry_run}")
    print(f"  Force overwrite: {force}")
    print(f"  First file only: {first_only}")
    print(f"  Output folder: {output_folder or '(in-place)'}")
    print("=" * 70)
    print()
    
    if not hdf5_files:
        print("No HDF5 files found in folder!")
        return
    
    # Create output folder if specified
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    start_time = time.time()
    
    for i, src_path in enumerate(hdf5_files, 1):
        print(f"\n[{i}/{len(hdf5_files)}] Processing: {src_path.name}")
        
        if dry_run:
            print(f"  [DRY RUN] Would process: {src_path}")
            skipped.append(src_path)
            continue
        
        # Determine working path (copy to output folder if specified)
        if output_folder:
            hdf5_path = output_folder / src_path.name
            print(f"  Copying to: {hdf5_path}")
            shutil.copy2(src_path, hdf5_path)
        else:
            hdf5_path = src_path
        
        try:
            # Add unit_meta from CMTR (extended per-unit metadata)
            # Paths are auto-detected from HDF5 source_files
            result_unit = add_cmtr_unit_info(
                hdf5_path=hdf5_path,
                cmtr_path=None,
                force=force,
            )
            units_updated = result_unit.get('units_updated', '?')
            
            # Add cmcr_meta and cmtr_meta (file-level metadata)
            # Paths are auto-detected from HDF5 source_files
            result_sys = add_sys_meta_info(
                hdf5_path=hdf5_path,
                cmcr_path=None,
                cmtr_path=None,
                force=force,
            )
            cmcr_fields = result_sys.get('cmcr_fields', '?')
            cmtr_fields = result_sys.get('cmtr_fields', '?')
            
            print(f"  ✓ Success: {units_updated} units, {cmcr_fields} cmcr, {cmtr_fields} cmtr fields")
            successful.append(hdf5_path)
            
        except FileNotFoundError as e:
            print(f"  ✗ Source file not found: {e}")
            failed.append((hdf5_path, str(e)))
            
        except ValueError as e:
            print(f"  ✗ Error: {e}")
            failed.append((hdf5_path, str(e)))
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            failed.append((hdf5_path, str(e)))
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total files: {len(hdf5_files)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    if failed:
        print("\nFailed files:")
        for path, error in failed:
            print(f"  - {path.name}: {error}")
    
    print("=" * 70)
    
    return {
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    results = batch_add_metadata(
        data_folder=DATA_FOLDER,
        dry_run=DRY_RUN,
        force=FORCE_OVERWRITE,
        first_only=FIRST_FILE_ONLY,
        output_folder=OUTPUT_FOLDER,
    )