"""
Batch DSGC Direction Sectioning for all HDF5 files.

This script:
1. Finds all HDF5 files in the source directory
2. Runs section_by_direction on each file
3. Saves outputs to a separate export directory
4. Logs progress and results

Usage:
    python batch_dsgc_section.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path for development
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import h5py

from hdmea.features import section_by_direction


# =============================================================================
# Configuration
# =============================================================================

SOURCE_DIR = Path(
    r"M:\Python_Project\Data_Processing_2027\Projects\ap_trace_hdf5"
    r"\export_ap_tracking_20251226"
)

EXPORT_DIR = Path(
    r"M:\Python_Project\Data_Processing_2027\Projects\dsgc_section"
    r"\export_dsgc_section_20251226"
)

MOVIE_NAME = "moving_h_bar_s5_d8_3x"

# Use _hd version (single pixel) instead of _area_hd (25x25 area average)
ON_OFF_DICT_PATH = Path(
    r"M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern"
    r"\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_hd.pkl"
)

PADDING_FRAMES = 10


def get_hdf5_files(source_dir: Path) -> list[Path]:
    """Get all HDF5 files in source directory, sorted by name."""
    files = list(source_dir.glob("*.h5"))
    return sorted(files, key=lambda p: p.name)


def process_file(
    source_path: Path,
    output_path: Path,
    on_off_dict_path: Path,
    movie_name: str,
    padding_frames: int,
) -> dict:
    """
    Process a single HDF5 file.
    
    Returns a dict with processing results.
    """
    result_info = {
        "source": source_path.name,
        "output": output_path.name,
        "success": False,
        "units_processed": 0,
        "units_skipped": 0,
        "elapsed_seconds": 0,
        "error": None,
    }
    
    try:
        # Check if movie exists in this file
        with h5py.File(source_path, "r") as f:
            section_time_path = f"stimulus/section_time/{movie_name}"
            if section_time_path not in f:
                result_info["error"] = f"Movie '{movie_name}' not found in file"
                result_info["units_skipped"] = -1  # Mark as N/A
                return result_info
        
        # Run direction sectioning
        result = section_by_direction(
            source_path,
            movie_name=movie_name,
            on_off_dict_path=on_off_dict_path,
            padding_frames=padding_frames,
            force=True,
            output_path=output_path,
        )
        
        result_info["success"] = True
        result_info["units_processed"] = result.units_processed
        result_info["units_skipped"] = result.units_skipped
        result_info["elapsed_seconds"] = result.elapsed_seconds
        result_info["skipped_units"] = result.skipped_units
        
        if result.warnings:
            result_info["warnings"] = result.warnings
        
    except Exception as e:
        result_info["error"] = str(e)
    
    return result_info


def main():
    """Batch process all HDF5 files."""
    print("=" * 80)
    print("DSGC Direction Sectioning - Batch Processing")
    print("=" * 80)
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check paths
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return 1
    
    if not ON_OFF_DICT_PATH.exists():
        print(f"ERROR: On/off dictionary not found: {ON_OFF_DICT_PATH}")
        return 1
    
    # Get list of HDF5 files
    hdf5_files = get_hdf5_files(SOURCE_DIR)
    print(f"\nSource directory: {SOURCE_DIR}")
    print(f"Export directory: {EXPORT_DIR}")
    print(f"On/off dictionary: {ON_OFF_DICT_PATH.name}")
    print(f"Movie: {MOVIE_NAME}")
    print(f"Padding frames: {PADDING_FRAMES}")
    print(f"\nFound {len(hdf5_files)} HDF5 files to process")
    
    if not hdf5_files:
        print("No HDF5 files found!")
        return 1
    
    # Create export directory
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    results = []
    successful = 0
    skipped = 0
    failed = 0
    
    print("\n" + "-" * 80)
    print("Processing files...")
    print("-" * 80)
    
    for i, source_path in enumerate(hdf5_files, 1):
        output_path = EXPORT_DIR / source_path.name
        
        print(f"\n[{i}/{len(hdf5_files)}] {source_path.name}")
        print(f"  Size: {source_path.stat().st_size / (1024**3):.2f} GB")
        
        result = process_file(
            source_path=source_path,
            output_path=output_path,
            on_off_dict_path=ON_OFF_DICT_PATH,
            movie_name=MOVIE_NAME,
            padding_frames=PADDING_FRAMES,
        )
        
        results.append(result)
        
        if result["success"]:
            successful += 1
            print(f"  ✓ Processed: {result['units_processed']} units "
                  f"({result['elapsed_seconds']:.1f}s)")
            if result["units_skipped"] > 0:
                skipped_list = result.get("skipped_units", [])
                if skipped_list:
                    print(f"    Skipped {result['units_skipped']} units (invalid STA): "
                          f"{', '.join(skipped_list[:5])}"
                          f"{'...' if len(skipped_list) > 5 else ''}")
        elif result["units_skipped"] == -1:
            skipped += 1
            print(f"  ⊘ Skipped: {result['error']}")
        else:
            failed += 1
            print(f"  ✗ FAILED: {result['error']}")
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nTimestamp: {timestamp}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Skipped (no movie): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total files: {len(hdf5_files)}")
    
    # Total units processed
    total_units = sum(r["units_processed"] for r in results if r["success"])
    total_skipped_units = sum(
        r["units_skipped"] for r in results 
        if r["success"] and r["units_skipped"] >= 0
    )
    print(f"\nTotal units processed: {total_units}")
    print(f"Total units skipped: {total_skipped_units}")
    
    # Save log
    log_path = EXPORT_DIR / "batch_processing_log.txt"
    with open(log_path, "w") as f:
        f.write("DSGC Direction Sectioning - Batch Processing Log\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Source: {SOURCE_DIR}\n")
        f.write(f"Export: {EXPORT_DIR}\n")
        f.write(f"Movie: {MOVIE_NAME}\n")
        f.write(f"On/off dict: {ON_OFF_DICT_PATH}\n")
        f.write(f"Padding frames: {PADDING_FRAMES}\n\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Skipped: {skipped}\n")
        f.write(f"Failed: {failed}\n\n")
        f.write("-" * 60 + "\n")
        f.write("File Details:\n")
        f.write("-" * 60 + "\n\n")
        
        for r in results:
            f.write(f"{r['source']}\n")
            if r["success"]:
                f.write(f"  Status: SUCCESS\n")
                f.write(f"  Units processed: {r['units_processed']}\n")
                f.write(f"  Units skipped: {r['units_skipped']}\n")
                if r.get("skipped_units"):
                    f.write(f"  Skipped units: {', '.join(r['skipped_units'])}\n")
                f.write(f"  Time: {r['elapsed_seconds']:.1f}s\n")
            elif r["units_skipped"] == -1:
                f.write(f"  Status: SKIPPED\n")
                f.write(f"  Reason: {r['error']}\n")
            else:
                f.write(f"  Status: FAILED\n")
                f.write(f"  Error: {r['error']}\n")
            f.write("\n")
    
    print(f"\nLog saved: {log_path}")
    
    # Print failed files if any
    if failed > 0:
        print("\nFailed files:")
        for r in results:
            if not r["success"] and r["units_skipped"] != -1:
                print(f"  - {r['source']}: {r['error']}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

