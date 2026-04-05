"""Check how many H5 files have DVNT positions."""
import h5py
import os
import numpy as np

h5_dir = r"M:\Python_Project\Data_Processing_2027\Projects\unified_pipeline\export_dsgc_sta_updated"
files = sorted([f for f in os.listdir(h5_dir) if f.endswith(".h5")])

has_dv = 0
has_pathway = 0
total = len(files)

for fname in files:
    h5_path = os.path.join(h5_dir, fname)
    with h5py.File(h5_path, "r") as f:
        meta = "metadata/ap_tracking"
        if f"{meta}/DV_position" in f:
            has_dv += 1
        if "units" in f:
            for uid in list(f["units"].keys())[:30]:
                pw = f"units/{uid}/features/ap_tracking/ap_pathway/slope"
                if pw in f and not np.isnan(f[pw][()]):
                    has_pathway += 1
                    break

print(f"Total H5 files: {total}")
print(f"Files with DV_position: {has_dv}")
print(f"Files with valid pathways: {has_pathway}")
