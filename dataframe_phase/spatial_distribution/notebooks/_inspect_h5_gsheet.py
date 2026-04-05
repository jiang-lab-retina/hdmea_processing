"""Check for gsheet_row/Center_xy in H5 files."""
import h5py
import os
import numpy as np

h5_dir = r"M:\Python_Project\Data_Processing_2027\Projects\unified_pipeline\export_dsgc_sta_updated"
files = sorted([f for f in os.listdir(h5_dir) if f.endswith(".h5")])

has_center_xy = 0
has_gsheet = 0
printed = 0

for fname in files[:50]:  # Only check first 50 for speed
    h5_path = os.path.join(h5_dir, fname)
    with h5py.File(h5_path, "r") as f:
        if "metadata/gsheet_row" in f:
            has_gsheet += 1
            gsheet = f["metadata/gsheet_row"]
            if "Center_xy" in gsheet:
                has_center_xy += 1
                if printed < 3:
                    val = gsheet["Center_xy"][()]
                    if isinstance(val, bytes):
                        val = val.decode()
                    print(f"{fname}: Center_xy = {val}")
                    # Also list all gsheet_row keys
                    if printed == 0:
                        print(f"  gsheet_row keys: {list(gsheet.keys())[:15]}")
                    printed += 1
            elif printed == 0:
                print(f"{fname}: gsheet_row exists but no Center_xy")
                print(f"  gsheet_row keys: {list(gsheet.keys())[:15]}")
                printed += 1

print(f"\nOf first 50 files:")
print(f"  Has metadata/gsheet_row: {has_gsheet}")
print(f"  Has Center_xy: {has_center_xy}")
