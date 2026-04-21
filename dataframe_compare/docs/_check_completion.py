"""Check completion status of the dataframe_compare pipeline for all 3 experiments."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from datetime import datetime

COMPARE_DIR = Path(r"m:\Python_Project\Data_Processing_2027\dataframe_compare")
UPSTREAM_DIR = Path(r"m:\Python_Project\Data_Processing_2027\Projects\unified_special_pipeline\blocker_alignment_analysis")
EXPS = ["_ptx_str", "_ptx", "_str"]

EXPECTED_PARQUETS = [
    "pair_index.parquet",
    "before_movies.parquet",
    "after_movies.parquet",
    "before_features.parquet",
    "after_features.parquet",
    "compared_dataframe.parquet",
    "compared_dataframe_v2.parquet",
    "compared_dataframe_v2_labeled.parquet",
    "compared_dataframe_v2_labeled_spatial.parquet",
    "hexbin_before_all.parquet",
    "hexbin_after_all.parquet",
    "hexbin_delta_all.parquet",
    "hexbin_before_pergroup.parquet",
    "hexbin_after_pergroup.parquet",
    "hexbin_delta_pergroup.parquet",
    "spatial_metrics_compare.parquet",
    "spatial_quant_before.parquet",
    "spatial_quant_after.parquet",
    "spatial_quant_delta.parquet",
    "spatial_quant_combined.parquet",
    "radial_center_before.parquet",
    "radial_center_after.parquet",
    "radial_center_delta.parquet",
    "radial_center_combined.parquet",
]

STEP_MAP = {
    "pair_index.parquet": "Step 1 (compare)",
    "before_movies.parquet": "Step 1 (compare)",
    "after_movies.parquet": "Step 1 (compare)",
    "before_features.parquet": "Step 1 (compare)",
    "after_features.parquet": "Step 1 (compare)",
    "compared_dataframe.parquet": "Step 1 (compare)",
    "compared_dataframe_v2.parquet": "Step 1 (compare)",
    "compared_dataframe_v2_labeled.parquet": "Step 2 (classify)",
    "compared_dataframe_v2_labeled_spatial.parquet": "Step 3 (improve_onh)",
    "hexbin_before_all.parquet": "Step 4 (spatial_plots)",
    "hexbin_after_all.parquet": "Step 4 (spatial_plots)",
    "hexbin_delta_all.parquet": "Step 4 (spatial_plots)",
    "hexbin_before_pergroup.parquet": "Step 4 (spatial_plots)",
    "hexbin_after_pergroup.parquet": "Step 4 (spatial_plots)",
    "hexbin_delta_pergroup.parquet": "Step 4 (spatial_plots)",
    "spatial_metrics_compare.parquet": "Step 4 (spatial_plots)",
    "spatial_quant_before.parquet": "Step 5 (spatial_quant)",
    "spatial_quant_after.parquet": "Step 5 (spatial_quant)",
    "spatial_quant_delta.parquet": "Step 5 (spatial_quant)",
    "spatial_quant_combined.parquet": "Step 5 (spatial_quant)",
    "radial_center_before.parquet": "Step 6 (radial_center)",
    "radial_center_after.parquet": "Step 6 (radial_center)",
    "radial_center_delta.parquet": "Step 6 (radial_center)",
    "radial_center_combined.parquet": "Step 6 (radial_center)",
}

for exp in EXPS:
    print(f"\n{'='*70}")
    print(f"  Experiment: {exp}")
    print(f"{'='*70}")

    # Upstream status
    out_dir = UPSTREAM_DIR / f"output{exp}"
    aligned_dir = out_dir / "aligned"
    export_dir = UPSTREAM_DIR / f"output_export{exp}"
    h5_count = len(list(out_dir.glob("*.h5"))) if out_dir.exists() else 0
    aligned_count = len(list(aligned_dir.glob("*.h5"))) if aligned_dir.exists() else 0
    export_count = len(list(export_dir.glob("*.h5"))) if export_dir.exists() else 0
    print(f"  Upstream: {h5_count} batch H5, {aligned_count} aligned, {export_count} export")

    # Compare output status
    compare_out = COMPARE_DIR / f"output{exp}"
    if not compare_out.exists():
        print(f"  Compare output dir: NOT FOUND")
        continue

    present = []
    missing = []
    last_step_completed = "None"
    for pq in EXPECTED_PARQUETS:
        p = compare_out / pq
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            present.append((pq, mtime))
            last_step_completed = STEP_MAP.get(pq, "?")
        else:
            missing.append(pq)

    print(f"  Present: {len(present)}/{len(EXPECTED_PARQUETS)} expected parquets")
    print(f"  Last step completed: {last_step_completed}")

    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for m in missing:
            print(f"    - {m}  ({STEP_MAP.get(m, '?')})")

    # Show most recent files
    if present:
        present.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Most recent outputs:")
        for name, mtime in present[:5]:
            print(f"    {mtime:%Y-%m-%d %H:%M}  {name}")

    # Check figure directories
    fig_dir = COMPARE_DIR / f"figure{exp}"
    if fig_dir.exists():
        spatial_dir = fig_dir / "spatial"
        validation_dir = fig_dir / "validation"
        all_cells = spatial_dir / "all_cells" if spatial_dir.exists() else None
        per_group = spatial_dir / "per_group" if spatial_dir.exists() else None
        quant_figs = spatial_dir / "figures_quant" if spatial_dir.exists() else None
        radial_figs = spatial_dir / "figures_radial" if spatial_dir.exists() else None

        print(f"\n  Figures:")
        if all_cells and all_cells.exists():
            print(f"    spatial/all_cells:    {len(list(all_cells.glob('*.png')))} PNGs")
        else:
            print(f"    spatial/all_cells:    NOT FOUND")
        if per_group and per_group.exists():
            print(f"    spatial/per_group:    {len(list(per_group.glob('*.png')))} PNGs")
        else:
            print(f"    spatial/per_group:    NOT FOUND")
        if quant_figs and quant_figs.exists():
            print(f"    spatial/figures_quant: {len(list(quant_figs.glob('*.png')))} PNGs")
        else:
            print(f"    spatial/figures_quant: NOT FOUND")
        if radial_figs and radial_figs.exists():
            print(f"    spatial/figures_radial: {len(list(radial_figs.glob('*.png')))} PNGs")
        else:
            print(f"    spatial/figures_radial: NOT FOUND")
        if validation_dir and validation_dir.exists():
            print(f"    validation:           {len(list(validation_dir.glob('*.png')))} PNGs")
        else:
            print(f"    validation:           NOT FOUND")
    else:
        print(f"\n  Figure dir: NOT FOUND")
