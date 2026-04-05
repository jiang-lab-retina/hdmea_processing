"""Check for non-empty Center_xy values."""
import h5py
import os

h5_dir = r"M:\Python_Project\Data_Processing_2027\Projects\unified_pipeline\export_dsgc_sta_updated"
files = sorted([f for f in os.listdir(h5_dir) if f.endswith(".h5")])

non_empty = 0
total = 0
samples = []

for fname in files:
    h5_path = os.path.join(h5_dir, fname)
    with h5py.File(h5_path, "r") as f:
        total += 1
        if "metadata/gsheet_row/Center_xy" in f:
            val = f["metadata/gsheet_row/Center_xy"][()]
            if isinstance(val, bytes):
                val = val.decode()
            if val.strip():
                non_empty += 1
                if len(samples) < 5:
                    samples.append((fname, val))

print(f"Total: {total}, Non-empty Center_xy: {non_empty}")
for fname, val in samples:
    print(f"  {fname}: '{val}'")
