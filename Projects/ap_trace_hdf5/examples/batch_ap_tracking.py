#!/usr/bin/env python
"""
Batch Processing: AP Tracking Analysis

Processes all HDF5 files in the input directory and adds AP tracking features.
Follows the pipeline principles with progress tracking, skip existing, and error handling.
"""

import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import h5py
import numpy as np


# =============================================================================
# Configuration - Edit these parameters
# =============================================================================

# Input/Output paths
INPUT_DIR = Path(r"M:\Python_Project\Data_Processing_2027\Projects\add_manual_label_cell_type\manual_label_20251225")
OUTPUT_DIR = Path(__file__).parent.parent / "export_ap_tracking_20251226"
MODEL_PATH = Path(__file__).parent.parent / "model" / "CNN_3d_with_velocity_model_from_all_process.pth"

# GPU/CPU processing
FORCE_CPU = False  # Set to True to force CPU processing (ignore GPU)

# Cell type filtering
FILTER_BY_CELL_TYPE = True  # Set to False to process all units
CELL_TYPE_FILTER = "rgc"    # Only process units with this cell type

# AP tracking parameters
MIN_POINTS_FOR_FIT = 10    # Minimum tracked axon points for line fitting
R2_THRESHOLD = 0.8         # Minimum R² for valid line fit
MAX_DISPLACEMENT = 100     # Maximum centroid displacement between frames (default 5, set high to keep all)
CENTROID_START_FRAME = 10  # Exclude centroids before this frame (default 0, set to 10 to skip early frames)
MAX_DISPLACEMENT_POST = 5.0  # Post-processing: remove centroids with >5px jump to neighbors

# Bad lanes preprocessing
FIX_BAD_LANES = True       # Set to False to skip bad lanes preprocessing


# =============================================================================
# GPU Detection
# =============================================================================

def check_gpu_availability() -> dict:
    """Check GPU availability and return device information."""
    info = {
        "cuda_available": False,
        "cuda_version": None,
        "device_name": None,
        "device_memory_gb": None,
    }
    
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass
    
    return info


def print_gpu_status(force_cpu: bool = False) -> dict:
    """Print GPU status information and return info dict."""
    info = check_gpu_availability()
    
    print("\nGPU Status:")
    if info["cuda_available"]:
        print(f"   CUDA available: Yes (v{info['cuda_version']})")
        print(f"   Device: {info['device_name']}")
        print(f"   Memory: {info['device_memory_gb']:.1f} GB")
        if force_cpu:
            print("   Mode: CPU (forced by FORCE_CPU=True)")
        else:
            print("   Mode: GPU (using CUDA)")
    else:
        print("   CUDA available: No")
        print("   Mode: CPU")
    
    return info


# =============================================================================
# Bad Lanes Processing
# =============================================================================

def parse_bad_lanes(bad_lanes_str: str) -> list[int]:
    """Parse bad lanes string to list of 0-indexed column indices."""
    if not bad_lanes_str or bad_lanes_str.strip() == "":
        return []
    
    lanes = []
    for part in bad_lanes_str.split(","):
        part = part.strip()
        if part:
            try:
                lane_1indexed = int(part)
                lane_0indexed = lane_1indexed - 1
                lanes.append(lane_0indexed)
            except ValueError:
                pass
    
    return sorted(lanes)


def read_bad_lanes_from_hdf5(hdf5_path: Path) -> list[int]:
    """Read Bad_lanes from HDF5 metadata/gsheet_row/Bad_lanes."""
    with h5py.File(hdf5_path, "r") as f:
        if "metadata/gsheet_row/Bad_lanes" not in f:
            return []
        
        bad_lanes_data = f["metadata/gsheet_row/Bad_lanes"][()]
        
        if isinstance(bad_lanes_data, bytes):
            bad_lanes_str = bad_lanes_data.decode("utf-8")
        elif isinstance(bad_lanes_data, np.ndarray):
            if bad_lanes_data.size > 0:
                val = bad_lanes_data.flat[0]
                if isinstance(val, bytes):
                    bad_lanes_str = val.decode("utf-8")
                else:
                    bad_lanes_str = str(val)
            else:
                bad_lanes_str = ""
        else:
            bad_lanes_str = str(bad_lanes_data)
        
        return parse_bad_lanes(bad_lanes_str)


def fix_bad_lanes_in_sta(sta_data: np.ndarray, bad_lanes: list[int]) -> np.ndarray:
    """Replace bad lane columns with the mean of the entire STA data."""
    if len(bad_lanes) == 0:
        return sta_data
    
    result = sta_data.copy()
    sta_mean = np.mean(sta_data)
    n_cols = sta_data.shape[-1]
    
    for lane_idx in bad_lanes:
        if 0 <= lane_idx < n_cols:
            result[:, :, lane_idx] = sta_mean
    
    return result


def preprocess_bad_lanes(hdf5_path: Path) -> int:
    """
    Preprocess all STA data in the HDF5 file to fix bad lanes.
    
    Returns:
        Number of units processed
    """
    bad_lanes = read_bad_lanes_from_hdf5(hdf5_path)
    
    if not bad_lanes:
        return 0
    
    processed = 0
    with h5py.File(hdf5_path, "r+") as f:
        if "units" not in f:
            return 0
        
        for unit_id in f["units"].keys():
            sta_path = f"units/{unit_id}/features/eimage_sta/data"
            if sta_path not in f:
                continue
            
            sta_data = f[sta_path][:]
            fixed_data = fix_bad_lanes_in_sta(sta_data, bad_lanes)
            
            del f[sta_path]
            f.create_dataset(sta_path, data=fixed_data, dtype=fixed_data.dtype)
            
            processed += 1
    
    return processed


# =============================================================================
# Single File Processing
# =============================================================================

def process_single_file(input_path: Path, output_path: Path) -> bool:
    """
    Process a single HDF5 file with AP tracking.
    
    Args:
        input_path: Path to source HDF5 file
        output_path: Path to save processed file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from hdmea.features.ap_tracking import compute_ap_tracking
        
        # Copy source to output location
        shutil.copy(input_path, output_path)
        
        # Preprocess: fix bad lanes in STA data
        if FIX_BAD_LANES:
            bad_lanes = read_bad_lanes_from_hdf5(output_path)
            if bad_lanes:
                n_fixed = preprocess_bad_lanes(output_path)
                print(f"  Fixed bad lanes {[l+1 for l in bad_lanes]} in {n_fixed} units")
        
        # Count units by cell type for logging
        with h5py.File(output_path, "r") as f:
            if "units" not in f:
                print(f"  ERROR: No 'units' group in file")
                return False
            
            unit_ids = list(f["units"].keys())
            filter_count = 0
            for uid in unit_ids:
                label_path = f"units/{uid}/auto_label/axon_type"
                if label_path in f:
                    cell_type = f[label_path][()]
                    if isinstance(cell_type, bytes):
                        cell_type = cell_type.decode('utf-8')
                    if cell_type == CELL_TYPE_FILTER:
                        filter_count += 1
            
            print(f"  Units: {len(unit_ids)} total, {filter_count} {CELL_TYPE_FILTER.upper()}")
        
        # Run AP tracking
        print(f"  Running AP tracking...")
        compute_ap_tracking(
            output_path,
            MODEL_PATH,
            max_units=None,
            force_cpu=FORCE_CPU,
            filter_by_cell_type=FILTER_BY_CELL_TYPE,
            cell_type_filter=CELL_TYPE_FILTER,
            min_points_for_fit=MIN_POINTS_FOR_FIT,
            r2_threshold=R2_THRESHOLD,
            max_displacement=MAX_DISPLACEMENT,
            centroid_start_frame=CENTROID_START_FRAME,
            max_displacement_post=MAX_DISPLACEMENT_POST,
        )
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up failed output file
        if output_path.exists():
            output_path.unlink()
        
        return False


# =============================================================================
# Batch Processing
# =============================================================================

def batch_process():
    """Process all HDF5 files in input directory."""
    print("=" * 70)
    print("Batch Processing: AP Tracking Analysis")
    print("=" * 70)
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Model:      {MODEL_PATH}")
    print(f"Cell type filter: {CELL_TYPE_FILTER if FILTER_BY_CELL_TYPE else 'None'}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show GPU status
    gpu_info = print_gpu_status(FORCE_CPU)
    
    # Validate paths
    if not INPUT_DIR.exists():
        print(f"\nError: Input directory not found: {INPUT_DIR}")
        return
    
    if not MODEL_PATH.exists():
        print(f"\nError: Model file not found: {MODEL_PATH}")
        return
    
    hdf5_files = sorted(INPUT_DIR.glob("*.h5"))
    if not hdf5_files:
        print(f"\nError: No HDF5 files found in {INPUT_DIR}")
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
            # Verify the file has ap_tracking data
            try:
                with h5py.File(output_path, "r") as f:
                    units = list(f["units"].keys())
                    has_ap = any(f"units/{u}/features/ap_tracking" in f for u in units[:5])
                    if has_ap:
                        print("  Already processed, skipping")
                        skipped += 1
                        continue
                    else:
                        print("  Output exists but incomplete, reprocessing...")
                        output_path.unlink()
            except Exception:
                print("  Output exists but corrupted, reprocessing...")
                output_path.unlink()
        
        file_start = time.time()
        
        if process_single_file(input_path, output_path):
            successful += 1
            elapsed = time.time() - file_start
            print(f"  Completed in {elapsed:.1f}s")
        else:
            failed += 1
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
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
    gpu_info = check_gpu_availability()
    
    with open(log_path, 'w') as f:
        f.write("Batch Processing Log: AP Tracking Analysis\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Input:  {INPUT_DIR}\n")
        f.write(f"Output: {OUTPUT_DIR}\n")
        f.write(f"Model:  {MODEL_PATH}\n")
        f.write(f"\nGPU Configuration:\n")
        f.write(f"  CUDA available:  {gpu_info['cuda_available']}\n")
        if gpu_info['cuda_available']:
            f.write(f"  Device:          {gpu_info['device_name']}\n")
            f.write(f"  Memory:          {gpu_info['device_memory_gb']:.1f} GB\n")
        f.write(f"  Mode used:       {'CPU (forced)' if FORCE_CPU else ('GPU' if gpu_info['cuda_available'] else 'CPU')}\n")
        f.write(f"\nParameters:\n")
        f.write(f"  filter_by_cell_type:   {FILTER_BY_CELL_TYPE}\n")
        f.write(f"  cell_type_filter:      {CELL_TYPE_FILTER}\n")
        f.write(f"  min_points_for_fit:    {MIN_POINTS_FOR_FIT}\n")
        f.write(f"  r2_threshold:          {R2_THRESHOLD}\n")
        f.write(f"  max_displacement:      {MAX_DISPLACEMENT}\n")
        f.write(f"  centroid_start_frame:  {CENTROID_START_FRAME}\n")
        f.write(f"  max_displacement_post: {MAX_DISPLACEMENT_POST}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Successful: {successful}\n")
        f.write(f"  Failed:     {failed}\n")
        f.write(f"  Skipped:    {skipped}\n")
        f.write(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)\n")
        f.write(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nFiles ({len(files)} total):\n")
        for path in files:
            output_path = OUTPUT_DIR / path.name
            if output_path.exists():
                status = "OK"
            else:
                status = "FAILED"
            f.write(f"  [{status}] {path.name}\n")
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    batch_process()
