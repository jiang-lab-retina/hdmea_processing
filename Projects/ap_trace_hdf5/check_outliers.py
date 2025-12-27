"""Check outlier information in HDF5 file."""
import h5py
from pathlib import Path

hdf5_path = Path("Projects/ap_trace_hdf5/export/2024.03.07-10.12.28-Rec.h5")

with h5py.File(hdf5_path, "r") as f:
    int_grp = f["metadata/ap_tracking/all_ap_intersection"]
    print("ONH Result:")
    print(f"  x: {int_grp['x'][()]:.2f}")
    print(f"  y: {int_grp['y'][()]:.2f}")
    print(f"  method: {int_grp['method'][()].decode()}")
    print(f"  n_cells_used: {int_grp['n_cells_used'][()]}")
    print(f"  rmse: {int_grp['rmse'][()]:.4f}")
    
    if "outlier_unit_ids" in int_grp:
        outliers = [x.decode() for x in int_grp["outlier_unit_ids"][:]]
        print(f"  outliers ({len(outliers)}): {outliers}")
    else:
        print("  outliers: None")
    
    if "kept_unit_ids" in int_grp:
        kept = [x.decode() for x in int_grp["kept_unit_ids"][:]]
        print(f"  kept ({len(kept)}): {kept}")
    else:
        print("  kept: None")
    
    if "max_outlier_fraction" in int_grp:
        print(f"  max_outlier_fraction: {int_grp['max_outlier_fraction'][()]}")

