#!/usr/bin/env python
"""
Example script for running AP tracking on HDF5 files.

This script demonstrates how to use the AP tracking feature module
to process HDF5 recordings and extract axon tracking features.

The script:
1. Copies the source file to the export folder
2. Runs AP tracking analysis on the copy
3. Validates the output structure
"""

import shutil
import sys
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
# Use a file with bad lanes for testing (lanes 39, 47)
SOURCE_FILE = project_root / "Projects/add_manual_label_cell_type/manual_label_20251225/2024.03.25-15.38.58-Rec.h5"
EXPORT_FOLDER = project_root / "Projects/ap_trace_hdf5/export"
MODEL_PATH = project_root / "Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth"

# GPU/CPU processing
FORCE_CPU = False  # Set to True to force CPU processing (ignore GPU)

# Cell type filtering
FILTER_BY_CELL_TYPE = True  # Set to False to process all units
CELL_TYPE_FILTER = "rgc"    # Only process units with this cell type

# AP tracking parameters
MAX_UNITS = None           # Limit number of units to process (None = all)
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


def print_gpu_status(force_cpu: bool = False):
    """Print GPU status information."""
    info = check_gpu_availability()
    
    print("\n   GPU Status:")
    if info["cuda_available"]:
        print(f"      CUDA available: Yes (v{info['cuda_version']})")
        print(f"      Device: {info['device_name']}")
        print(f"      Memory: {info['device_memory_gb']:.1f} GB")
        if force_cpu:
            print("      Mode: CPU (forced by FORCE_CPU=True)")
        else:
            print("      Mode: GPU (using CUDA)")
    else:
        print("      CUDA available: No")
        print("      Mode: CPU")
    
    return info


# =============================================================================
# Bad Lanes Processing
# =============================================================================

def parse_bad_lanes(bad_lanes_str: str) -> list[int]:
    """
    Parse bad lanes string to list of 0-indexed column indices.
    
    Args:
        bad_lanes_str: Comma-separated string of lane numbers (1-indexed)
                      e.g., "33, 34, 37, 39, 45, 47"
    
    Returns:
        List of 0-indexed column indices
    """
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
    """
    Read Bad_lanes from HDF5 metadata/gsheet_row/Bad_lanes.
    
    Args:
        hdf5_path: Path to HDF5 file
        
    Returns:
        List of 0-indexed bad lane column indices
    """
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
    """
    Replace bad lane columns with the mean of the entire STA data.
    
    Args:
        sta_data: STA array with shape (time, y, x)
        bad_lanes: List of 0-indexed column (x) indices to replace
        
    Returns:
        Modified STA data with bad lanes replaced by mean
    """
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
    
    Reads Bad_lanes from metadata and replaces those columns with the mean
    value in all units' eimage_sta/data.
    
    Args:
        hdf5_path: Path to HDF5 file (modified in place)
        
    Returns:
        Number of units processed
    """
    bad_lanes = read_bad_lanes_from_hdf5(hdf5_path)
    
    if not bad_lanes:
        print("   No bad lanes found in metadata")
        return 0
    
    print(f"   Bad lanes (1-indexed): {[l+1 for l in bad_lanes]}")
    
    processed = 0
    with h5py.File(hdf5_path, "r+") as f:
        if "units" not in f:
            return 0
        
        for unit_id in f["units"].keys():
            sta_path = f"units/{unit_id}/features/eimage_sta/data"
            if sta_path not in f:
                continue
            
            # Read, fix, and write back
            sta_data = f[sta_path][:]
            fixed_data = fix_bad_lanes_in_sta(sta_data, bad_lanes)
            
            # Overwrite the dataset
            del f[sta_path]
            f.create_dataset(sta_path, data=fixed_data, dtype=fixed_data.dtype)
            
            processed += 1
    
    print(f"   Fixed bad lanes in {processed} units")
    return processed


# =============================================================================
# Main
# =============================================================================

def main():
    """Run AP tracking example."""
    print("=" * 60)
    print("AP Tracking Example - HDF5 Pipeline")
    print("=" * 60)
    
    # Show GPU status
    print_gpu_status(FORCE_CPU)

    # Define target file path
    target_file = EXPORT_FOLDER / SOURCE_FILE.name

    # Validate paths
    print("\n1. Validating paths...")
    if not SOURCE_FILE.exists():
        print(f"   ERROR: Source file not found: {SOURCE_FILE}")
        return 1

    if not MODEL_PATH.exists():
        print(f"   ERROR: Model file not found: {MODEL_PATH}")
        return 1

    print(f"   Source: {SOURCE_FILE}")
    print(f"   Output: {target_file}")
    print(f"   Model:  {MODEL_PATH}")

    # Create export folder and copy source to target
    print("\n2. Preparing output file...")
    EXPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE_FILE, target_file)
    print(f"   Created: {target_file.name}")

    # Preprocess: fix bad lanes in STA data
    if FIX_BAD_LANES:
        print("\n3. Preprocessing: fixing bad lanes in STA data...")
        preprocess_bad_lanes(target_file)
    
    # Check file structure before processing
    print("\n4. Inspecting source file structure...")
    with h5py.File(target_file, "r") as f:
        if "units" not in f:
            print("   ERROR: No 'units' group in file")
            return 1

        unit_ids = list(f["units"].keys())
        print(f"   Found {len(unit_ids)} units total")
        
        # Count units by cell type (from auto_label/axon_type)
        cell_type_counts = {}
        for uid in unit_ids:
            label_path = f"units/{uid}/auto_label/axon_type"
            if label_path in f:
                cell_type = f[label_path][()]
                if isinstance(cell_type, bytes):
                    cell_type = cell_type.decode('utf-8')
                cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
            else:
                cell_type_counts["no_label"] = cell_type_counts.get("no_label", 0) + 1
        
        print(f"   Cell type distribution: {cell_type_counts}")
        
        if FILTER_BY_CELL_TYPE:
            filter_count = cell_type_counts.get(CELL_TYPE_FILTER, 0)
            print(f"   {CELL_TYPE_FILTER.upper()} units (will be processed): {filter_count}")

        # Check for eimage_sta in first unit
        first_unit = unit_ids[0]
        has_sta = "eimage_sta" in f[f"units/{first_unit}/features"]
        print(f"   First unit ({first_unit}) has eimage_sta: {has_sta}")

        if has_sta and "data" in f[f"units/{first_unit}/features/eimage_sta"]:
            sta_path = f"units/{first_unit}/features/eimage_sta/data"
            sta_shape = f[sta_path].shape
            print(f"   STA data shape: {sta_shape}")

    # Run AP tracking
    print("\n5. Running AP tracking analysis...")
    try:
        from hdmea.features.ap_tracking import compute_ap_tracking

        compute_ap_tracking(
            target_file,
            MODEL_PATH,
            max_units=MAX_UNITS,
            force_cpu=FORCE_CPU,
            filter_by_cell_type=FILTER_BY_CELL_TYPE,
            cell_type_filter=CELL_TYPE_FILTER,
            min_points_for_fit=MIN_POINTS_FOR_FIT,
            r2_threshold=R2_THRESHOLD,
            max_displacement=MAX_DISPLACEMENT,
            centroid_start_frame=CENTROID_START_FRAME,
            max_displacement_post=MAX_DISPLACEMENT_POST,
        )
        print("   AP tracking completed successfully!")

    except Exception as e:
        print(f"   ERROR: AP tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Validate output structure
    print("\n6. Validating output structure...")
    with h5py.File(target_file, "r") as f:
        processed_count = 0
        for unit_id in list(f["units"].keys())[:]:
            ap_path = f"units/{unit_id}/features/ap_tracking"

            if ap_path not in f:
                continue

            processed_count += 1
            ap_group = f[ap_path]

            # Check required fields
            required_fields = [
                "DV_position", "NT_position", "LR_position",
                "refined_soma", "axon_initial_segment",
                "prediction_sta_data", "post_processed_data",
                "ap_pathway", "all_ap_intersection", "soma_polar_coordinates",
            ]

            missing = [field for field in required_fields if field not in ap_group]
            if missing:
                print(f"   Unit {unit_id}: Missing fields: {missing}")
            else:
                # Read some values for verification
                dv = ap_group["DV_position"][()]
                nt = ap_group["NT_position"][()]
                lr = ap_group["LR_position"][()]
                if isinstance(lr, bytes):
                    lr = lr.decode("utf-8")

                soma_t = ap_group["refined_soma/t"][()]
                soma_x = ap_group["refined_soma/x"][()]
                soma_y = ap_group["refined_soma/y"][()]

                pred_shape = ap_group["prediction_sta_data"].shape

                print(f"   Unit {unit_id}: OK")
                print(f"      DVNT: DV={dv}, NT={nt}, LR={lr}")
                print(f"      Soma: t={soma_t}, x={soma_x}, y={soma_y}")
                print(f"      Prediction shape: {pred_shape}")
        
        if processed_count == 0:
            print("   WARNING: No units with ap_tracking found in first 5 units")

    print("\n" + "=" * 60)
    print("AP Tracking Example Complete!")
    print(f"Output file: {target_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
