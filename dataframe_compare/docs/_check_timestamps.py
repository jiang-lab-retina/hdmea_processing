"""Check all parquet timestamps to see if pipeline was interrupted."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from datetime import datetime

COMPARE_DIR = Path(r"m:\Python_Project\Data_Processing_2027\dataframe_compare")
EXPS = ["_ptx_str", "_ptx", "_str"]

STEP_MAP = {
    "pair_index.parquet": 1,
    "before_movies.parquet": 1,
    "after_movies.parquet": 1,
    "before_features.parquet": 1,
    "after_features.parquet": 1,
    "compared_dataframe.parquet": 1,
    "compared_dataframe_v2.parquet": 1,
    "compared_dataframe_v2_labeled.parquet": 2,
    "compared_dataframe_v2_labeled_spatial.parquet": 3,
    "hexbin_before_all.parquet": 4,
    "hexbin_after_all.parquet": 4,
    "hexbin_delta_all.parquet": 4,
    "hexbin_before_pergroup.parquet": 4,
    "hexbin_after_pergroup.parquet": 4,
    "hexbin_delta_pergroup.parquet": 4,
    "spatial_metrics_compare.parquet": 4,
    "spatial_quant_before.parquet": 5,
    "spatial_quant_after.parquet": 5,
    "spatial_quant_delta.parquet": 5,
    "spatial_quant_combined.parquet": 5,
    "radial_center_before.parquet": 6,
    "radial_center_after.parquet": 6,
    "radial_center_delta.parquet": 6,
    "radial_center_combined.parquet": 6,
}

for exp in EXPS:
    print(f"\n{'='*70}")
    print(f"  Experiment: {exp}")
    print(f"{'='*70}")
    compare_out = COMPARE_DIR / f"output{exp}"
    if not compare_out.exists():
        print("  NOT FOUND")
        continue

    entries = []
    for name, step in sorted(STEP_MAP.items(), key=lambda x: (x[1], x[0])):
        p = compare_out / name
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            entries.append((step, name, mtime))
        else:
            entries.append((step, name, None))

    current_step = None
    for step, name, mtime in entries:
        if step != current_step:
            current_step = step
            print(f"\n  --- Step {step} ---")
        if mtime:
            print(f"    {mtime:%Y-%m-%d %H:%M:%S}  {name}")
        else:
            print(f"    MISSING             {name}")

    # Also check figure timestamps
    fig_dir = COMPARE_DIR / f"figure{exp}"
    if fig_dir.exists():
        print(f"\n  --- Figures ---")
        for subdir in ["spatial/all_cells", "spatial/per_group", "spatial/figures_quant",
                        "spatial/figures_radial", "validation"]:
            sd = fig_dir / subdir
            if sd.exists():
                pngs = sorted(sd.glob("*.png"), key=lambda f: f.stat().st_mtime)
                if pngs:
                    newest = pngs[-1]
                    oldest = pngs[0]
                    print(f"    {subdir}: {len(pngs)} PNGs, "
                          f"oldest={datetime.fromtimestamp(oldest.stat().st_mtime):%Y-%m-%d %H:%M}, "
                          f"newest={datetime.fromtimestamp(newest.stat().st_mtime):%Y-%m-%d %H:%M}")
