"""Check if alignment/export need re-running by comparing timestamps."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from datetime import datetime

BASE = Path(r"m:\Python_Project\Data_Processing_2027\Projects\unified_special_pipeline\blocker_alignment_analysis")
EXPS = ["_ptx_str", "_ptx", "_str"]

for exp in EXPS:
    print(f"\n{'='*60}")
    print(f"  Experiment: {exp}")
    print(f"{'='*60}")

    out_dir = BASE / f"output{exp}"
    aligned_dir = out_dir / "aligned"
    export_dir = BASE / f"output_export{exp}"

    # newest batch H5
    batch_h5 = sorted(out_dir.glob("*.h5"), key=lambda f: f.stat().st_mtime)
    if batch_h5:
        newest_batch = batch_h5[-1]
        oldest_batch = batch_h5[0]
        print(f"  Batch H5 range: {datetime.fromtimestamp(oldest_batch.stat().st_mtime):%Y-%m-%d %H:%M} -> {datetime.fromtimestamp(newest_batch.stat().st_mtime):%Y-%m-%d %H:%M}")
        print(f"    newest: {newest_batch.name}")

    # newest aligned H5
    aligned_h5 = sorted(aligned_dir.glob("*.h5"), key=lambda f: f.stat().st_mtime) if aligned_dir.exists() else []
    if aligned_h5:
        newest_aligned = aligned_h5[-1]
        print(f"  Aligned H5 newest: {datetime.fromtimestamp(newest_aligned.stat().st_mtime):%Y-%m-%d %H:%M} ({newest_aligned.name})")
    else:
        print(f"  Aligned H5:        NONE")

    # newest export H5
    export_h5 = sorted(export_dir.glob("*.h5"), key=lambda f: f.stat().st_mtime) if export_dir.exists() else []
    if export_h5:
        newest_export = export_h5[-1]
        print(f"  Export H5 newest:  {datetime.fromtimestamp(newest_export.stat().st_mtime):%Y-%m-%d %H:%M} ({newest_export.name})")
    else:
        print(f"  Export H5:         NONE")

    # Count batch H5 that are NEWER than the newest aligned
    if aligned_h5 and batch_h5:
        aligned_time = newest_aligned.stat().st_mtime
        new_batch = [f for f in batch_h5 if f.stat().st_mtime > aligned_time]
        print(f"  Batch H5 newer than alignment: {len(new_batch)}")
        if new_batch:
            for f in new_batch[:5]:
                print(f"    {f.name}")
            if len(new_batch) > 5:
                print(f"    ... and {len(new_batch) - 5} more")

    # Check file_index.csv columns
    fi_path = out_dir / "file_index.csv"
    if fi_path.exists():
        import pandas as pd
        fi = pd.read_csv(fi_path)
        print(f"  file_index columns: {list(fi.columns)}")
        print(f"  file_index sample rows:")
        print(fi.head(2).to_string(index=False))
