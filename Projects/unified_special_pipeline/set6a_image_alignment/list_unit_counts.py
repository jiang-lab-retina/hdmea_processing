"""Quick script to list unit counts in H5 files."""
import h5py
from pathlib import Path

export_dir = Path(__file__).parent / "export"

print(f"{'File':<40} {'Units':>8}")
print("-" * 50)

total_units = 0
files = sorted(export_dir.glob("*.h5"))

for h5_file in files:
    try:
        with h5py.File(h5_file, "r") as f:
            if "units" in f:
                unit_count = len(f["units"].keys())
            else:
                unit_count = 0
        print(f"{h5_file.name:<40} {unit_count:>8}")
        total_units += unit_count
    except Exception as e:
        print(f"{h5_file.name:<40} ERROR: {e}")

print("-" * 50)
print(f"{'Total':<40} {total_units:>8}")
print(f"Files: {len(files)}")
