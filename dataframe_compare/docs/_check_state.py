"""Check upstream pipeline state for all 3 experiments."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
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

    h5_count = len(list(out_dir.glob("*.h5"))) if out_dir.exists() else 0
    aligned_count = len(list(aligned_dir.glob("*.h5"))) if aligned_dir.exists() else 0
    export_count = len(list(export_dir.glob("*.h5"))) if export_dir.exists() else 0

    print(f"  batch_pipeline H5:  {h5_count}")
    print(f"  aligned H5:         {aligned_count}")
    print(f"  export H5:          {export_count}")

    # Check dataframe_compare outputs
    compare_dir = Path(r"m:\Python_Project\Data_Processing_2027\dataframe_compare") / f"output{exp}"
    if compare_dir.exists():
        parquets = sorted(f.name for f in compare_dir.glob("*.parquet"))
        print(f"  compare outputs:    {len(parquets)} parquets")
        for p in parquets[:5]:
            print(f"    {p}")
        if len(parquets) > 5:
            print(f"    ... and {len(parquets) - 5} more")
    else:
        print(f"  compare outputs:    (dir not found)")
