#!/usr/bin/env python
"""
Example script for running AP tracking on HDF5 files.

This script demonstrates how to use the AP tracking feature module
to process HDF5 recordings and extract axon tracking features.

Usage:
    python run_ap_tracking.py

The script:
1. Copies the test file to the export folder (preserves original)
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


def main():
    """Run AP tracking example."""
    print("=" * 60)
    print("AP Tracking Example - HDF5 Pipeline")
    print("=" * 60)

    # Define paths
    source_file = project_root / "Projects/load_gsheet/export_gsheet_20251225/2024.05.23-12.05.03-Rec.h5"
    export_folder = project_root / "Projects/ap_trace_hdf5/export"
    target_file = export_folder / "2024.05.23-12.05.03-Rec.h5"
    model_path = project_root / "Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth"

    # Validate paths
    print("\n1. Validating paths...")
    if not source_file.exists():
        print(f"   ERROR: Source file not found: {source_file}")
        return 1

    if not model_path.exists():
        print(f"   ERROR: Model file not found: {model_path}")
        return 1

    print(f"   Source file: {source_file}")
    print(f"   Model file: {model_path}")
    print(f"   Export folder: {export_folder}")

    # Create export folder if needed
    export_folder.mkdir(parents=True, exist_ok=True)

    # Copy file to export folder (never modify originals!)
    print("\n2. Copying source file to export folder...")
    if target_file.exists():
        print(f"   Target already exists, removing old copy...")
        target_file.unlink()
    shutil.copy(source_file, target_file)
    print(f"   Copied to: {target_file}")

    # Check file structure before processing
    print("\n3. Inspecting source file structure...")
    with h5py.File(target_file, "r") as f:
        if "units" not in f:
            print("   ERROR: No 'units' group in file")
            return 1

        unit_ids = list(f["units"].keys())
        print(f"   Found {len(unit_ids)} units")

        # Check for eimage_sta in first unit
        first_unit = unit_ids[0]
        sta_path = f"units/{first_unit}/features/eimage_sta/data"
        has_sta = "eimage_sta" in f[f"units/{first_unit}/features"]
        print(f"   First unit ({first_unit}) has eimage_sta: {has_sta}")

        if has_sta and "data" in f[f"units/{first_unit}/features/eimage_sta"]:
            sta_shape = f[sta_path].shape
            print(f"   STA data shape: {sta_shape}")

    # Run AP tracking
    print("\n4. Running AP tracking analysis...")
    try:
        from hdmea.features.ap_tracking import compute_ap_tracking

        compute_ap_tracking(
            target_file,
            model_path,
            max_units=None,  # Process all units
            force_cpu=False,  # Use GPU if available
        )
        print("   AP tracking completed successfully!")

    except Exception as e:
        print(f"   ERROR: AP tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Validate output structure
    print("\n5. Validating output structure...")
    with h5py.File(target_file, "r") as f:
        for unit_id in list(f["units"].keys())[:5]:
            ap_path = f"units/{unit_id}/features/ap_tracking"

            if ap_path not in f:
                print(f"   WARNING: No ap_tracking for unit {unit_id}")
                continue

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

    print("\n" + "=" * 60)
    print("AP Tracking Example Complete!")
    print(f"Output file: {target_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

