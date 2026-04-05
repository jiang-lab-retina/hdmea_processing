"""Quick inspection of H5 files to verify pathway/soma/DVNT structure."""
import h5py
import os
import numpy as np

h5_dir = r"M:\Python_Project\Data_Processing_2027\Projects\unified_pipeline\export_dsgc_sta_updated"
files = sorted([f for f in os.listdir(h5_dir) if f.endswith(".h5")])

found = 0
for fname in files:
    h5_path = os.path.join(h5_dir, fname)
    with h5py.File(h5_path, "r") as f:
        meta = "metadata/ap_tracking"
        has_dv = f"{meta}/DV_position" in f
        has_nt = f"{meta}/NT_position" in f

        # Check for valid pathway
        has_valid_pw = False
        if "units" in f:
            for uid in list(f["units"].keys())[:30]:
                pw = f"units/{uid}/features/ap_tracking/ap_pathway/slope"
                if pw in f:
                    val = f[pw][()]
                    if not np.isnan(val):
                        has_valid_pw = True
                        break

        if has_valid_pw:
            found += 1
            if found <= 2:
                print(f"=== {fname} ===")
                dv_val = f[f"{meta}/DV_position"][()] if has_dv else "MISSING"
                nt_val = f[f"{meta}/NT_position"][()] if has_nt else "MISSING"
                print(f"  DV_position: {dv_val}")
                print(f"  NT_position: {nt_val}")

                # Show all metadata keys
                if meta in f:
                    print(f"  Metadata keys: {list(f[meta].keys())}")

                # Show a few valid pathways
                count = 0
                for uid in f["units"].keys():
                    pw = f"units/{uid}/features/ap_tracking/ap_pathway"
                    sm = f"units/{uid}/features/ap_tracking/refined_soma"
                    if pw in f:
                        slope = f[f"{pw}/slope"][()]
                        if not np.isnan(slope):
                            r_val = f[f"{pw}/r_value"][()]
                            intercept = f[f"{pw}/intercept"][()]
                            soma_x = f[f"{sm}/x"][()] if sm in f else "N/A"
                            soma_y = f[f"{sm}/y"][()] if sm in f else "N/A"
                            print(
                                f"  {uid}: slope={slope:.4f} intercept={intercept:.4f}"
                                f" r={r_val:.4f} soma=({soma_x},{soma_y})"
                            )
                            count += 1
                            if count >= 4:
                                break
                print()
            if found >= 3:
                break

print(f"Files with valid pathways found so far: {found}")
