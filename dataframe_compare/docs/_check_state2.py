"""Check if alignment & export need re-running for new data."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from pathlib import Path

BASE = Path(r"m:\Python_Project\Data_Processing_2027\Projects\unified_special_pipeline\blocker_alignment_analysis")
EXPS = ["_ptx_str", "_ptx", "_str"]

for exp in EXPS:
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp}")
    print(f"{'='*60}")

    out_dir = BASE / f"output{exp}"
    aligned_dir = out_dir / "aligned"
    export_dir = BASE / f"output_export{exp}"
    fi_path = out_dir / "file_index.csv"

    # file_index
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        print(f"  file_index rows:    {len(fi)}")
        if "condition" in fi.columns:
            print(f"  conditions: {fi['condition'].value_counts().to_dict()}")
    else:
        print(f"  file_index:         NOT FOUND")

    # batch_pipeline H5
    h5_files = sorted(out_dir.glob("*.h5"))
    h5_names = set(f.stem for f in h5_files)
    print(f"  batch H5 count:     {len(h5_files)}")

    # aligned H5
    aligned_files = sorted(aligned_dir.glob("*.h5")) if aligned_dir.exists() else []
    print(f"  aligned H5 count:   {len(aligned_files)}")

    # export H5
    export_files = sorted(export_dir.glob("*.h5")) if export_dir.exists() else []
    export_names = set(f.stem for f in export_files)
    print(f"  export H5 count:    {len(export_files)}")

    # Check which batch H5 files are NOT in export
    if fi_path.exists():
        fi_names = set(fi["dataset_id"]) if "dataset_id" in fi.columns else set()
        batch_not_in_fi = h5_names - fi_names
        fi_not_in_batch = fi_names - h5_names
        print(f"  batch H5 not in file_index: {len(batch_not_in_fi)}")
        print(f"  file_index not in batch H5: {len(fi_not_in_batch)}")
