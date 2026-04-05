# Blocker Comparison Pipeline

End-to-end pipeline for comparing retinal ganglion cell responses
before and after pharmacological blocker application (PTX+STR, PTX-only, STR-only).

## Prerequisites

1. **Upstream blocker alignment pipeline** must have completed for the target experiment:
   - `discover_files.py` -- file index
   - `batch_pipeline.py` -- per-recording H5 processing
   - `alignment.py` -- before/after unit pairing
   - `batch_update.py` -- exported H5 files with features

   These live in `Projects/unified_special_pipeline/blocker_alignment_analysis/`.

2. **Trained classification models** from `dataframe_phase/classification_v2/divide_conquer_method/`
   (autoencoder + DEC weights, cluster assignments). Required by Step 2.

3. **Python environment** with: numpy, pandas, h5py, scipy, matplotlib, pygam, torch, tqdm.

## Experiment Configuration

The experiment is selected via `specific_config.py` in the upstream blocker
alignment pipeline. Three selection methods, in priority order:

| Priority | Method | Example |
|----------|--------|---------|
| 1 | CLI flag | `--experiment _ptx` |
| 2 | Environment variable | `set BLOCKER_EXPERIMENT=_ptx` |
| 3 | Default in `specific_config.py` | `ACTIVE_EXPERIMENT = "_ptx_str"` |

Available experiments: `_ptx_str`, `_ptx`, `_str`.

The experiment controls:
- `FOLDER_POSTFIX` -- appended to `output/` and `figure/` directory names
- `ALIGNED_DIR`, `EXPORT_DIR` -- upstream H5 file locations
- Date ranges, subfolder names, and other blocker-specific parameters

`compare_config.py` imports `FOLDER_POSTFIX` from `specific_config.py`
and derives all local paths (`OUTPUT_DIR`, `FIG_DIR_BASE`) from it.

## Pipeline Steps

### Step 1: Compare (`pipeline_compare.py`)

Build the core comparison DataFrame from aligned H5 recording pairs.

| | |
|---|---|
| **Input** | `{ALIGNED_DIR}/*_aligned.h5`, `{EXPORT_DIR}/*.h5` |
| **Output** | `{OUTPUT_DIR}/compared_dataframe.parquet`, `compared_dataframe_v2.parquet` |
| **Sub-steps** | 0: pair index, 1: firing rates, 2: HDF5 features, 3: derived features, 4: merge, 5: intensity GB |

Supports `--start`, `--end`, `--start-step`, `--no-features`, `--output-suffix`
via `--s1`.

### Step 2: Classify (`classify_blocker.py`)

Assign each paired unit to a known RGC subtype using trained AE+DEC models.

| | |
|---|---|
| **Input** | `compared_dataframe_v2.parquet` |
| **Output** | `compared_dataframe_v2_labeled.parquet` |

Supports `--input`, `--output`, `--models-dir`, `--results-dir`, `--device`, `--debug`
via `--s2`.

### Step 3: Improve ONH (`spatial_improve_onh.py`)

Re-estimate the optic nerve head position per recording using robust
median+MAD intersection, then recompute `improved_tx`/`improved_ty` coordinates.

| | |
|---|---|
| **Input** | `compared_dataframe_v2_labeled.parquet`, `{EXPORT_DIR}/*.h5` |
| **Output** | `compared_dataframe_v2_labeled_spatial.parquet` |

### Step 4: Spatial Plots (`spatial_plots_compare.py`)

Generate hexbin + GAM triptych plots (Before | After | Delta) for curated
features, at both all-cells and per-group levels.

| | |
|---|---|
| **Input** | `compared_dataframe_v2_labeled_spatial.parquet` |
| **Output** | `hexbin_{before,after,delta}_{all,pergroup}.parquet`, `spatial_metrics_compare.parquet` |
| **Figures** | `{FIG_DIR_BASE}/spatial/all_cells/Triptych_*.png`, `.../per_group/Triptych_*.png` |

### Step 5: Spatial Quantification (`spatial_quantification_compare.py`)

Compute spatial statistics on the hexbin data: gradient magnitude/direction,
$R^2$, Moran's I, radial correlation, permutation p-values, FDR correction.

| | |
|---|---|
| **Input** | `hexbin_{before,after,delta}_{all,pergroup}.parquet` |
| **Output** | `spatial_quant_{before,after,delta}.parquet`, `spatial_quant_combined.parquet` |
| **Summary** | `{FIG_DIR_BASE}/spatial/spatial_quantification_summary.md` |

### Step 6: Radial Center (`spatial_radial_center.py`)

Search for the optimal retinal center that maximizes radial correlation,
on all-cells hexbin data only.

| | |
|---|---|
| **Input** | `hexbin_{before,after,delta}_all.parquet` |
| **Output** | `radial_center_{before,after,delta}.parquet`, `radial_center_combined.parquet` |
| **Summary** | `{FIG_DIR_BASE}/spatial/radial_center_summary.md` |

### Step 7: Visualize Quantification (`spatial_visualize_quant.py`)

Create comparison bar charts, scatter plots, and dashboards from the
quantification results.

| | |
|---|---|
| **Input** | `spatial_quant_combined.parquet` |
| **Figures** | `{FIG_DIR_BASE}/spatial/figures_quant/*.png` |

### Step 8: Visualize Radial (`spatial_visualize_radial.py`)

Create radial center comparison visualizations and dashboards.

| | |
|---|---|
| **Input** | `radial_center_combined.parquet` |
| **Figures** | `{FIG_DIR_BASE}/spatial/figures_radial/*.png` |

### Step 9: Validation Plots (`plot_step_up_validation.py`)

Compare step-up response traces across Reference (normal dataset),
Before Blocker, and After Blocker for each RGC group.

| | |
|---|---|
| **Input** | `compared_dataframe_v2_labeled.parquet`, reference `labeled_dataframe.parquet` |
| **Figures** | `{FIG_DIR_BASE}/validation/step_up_validation_*.png` |

## Dependency Graph

```
Step 1 (compare)
  |
  v
Step 2 (classify)
  |
  +---> Step 9 (validation)
  |
  v
Step 3 (improve_onh)
  |
  v
Step 4 (spatial_plots)
  |
  +---> Step 5 (spatial_quant) ---> Step 7 (viz_quant)
  |
  +---> Step 6 (radial_center) ---> Step 8 (viz_radial)
```

Steps 5 and 6 are independent of each other (both depend on Step 4).
Steps 7 and 8 are independent of each other.
Step 9 depends only on Step 2.

## Usage

All commands are run from the `dataframe_compare/` directory.

### Run the full pipeline

```powershell
python main_pipeline.py
```

### Run with a specific experiment

```powershell
python main_pipeline.py --experiment _ptx
python main_pipeline.py --experiment _str
python main_pipeline.py --experiment _ptx_str
```

### Run specific steps

```powershell
# Single step
python main_pipeline.py --steps 4

# Multiple steps
python main_pipeline.py --steps 5 6 7 8

# Range
python main_pipeline.py --steps 4-9

# Mixed
python main_pipeline.py --steps 1-3 9
```

### Pass arguments to individual steps

Steps 1 and 2 accept their own CLI arguments via `--s1` and `--s2`:

```powershell
# Run step 1 with --end 2 (process only first 2 pairs, for testing)
python main_pipeline.py --steps 1 --s1 "--end 2"

# Run step 1 resuming from its internal sub-step 5
python main_pipeline.py --steps 1 --s1 "--start-step 5"

# Run step 2 on CPU
python main_pipeline.py --steps 2 --s2 "--device cpu"
```

### Dry run (preview only)

```powershell
python main_pipeline.py --dry-run
python main_pipeline.py --experiment _ptx --steps 4-9 --dry-run
```

### Run from the project root

```powershell
python dataframe_compare/main_pipeline.py --experiment _ptx
```

## Output Directory Structure

After a full pipeline run, the output and figure directories look like:

```
dataframe_compare/
  output_ptx_str/
    pair_index.parquet
    before_movies.parquet
    after_movies.parquet
    before_features.parquet
    after_features.parquet
    compared_dataframe.parquet
    compared_dataframe_v2.parquet
    compared_dataframe_v2_labeled.parquet
    compared_dataframe_v2_labeled_spatial.parquet
    hexbin_before_all.parquet
    hexbin_after_all.parquet
    hexbin_delta_all.parquet
    hexbin_before_pergroup.parquet
    hexbin_after_pergroup.parquet
    hexbin_delta_pergroup.parquet
    spatial_metrics_compare.parquet
    spatial_quant_before.parquet
    spatial_quant_after.parquet
    spatial_quant_delta.parquet
    spatial_quant_combined.parquet
    radial_center_before.parquet
    radial_center_after.parquet
    radial_center_delta.parquet
    radial_center_combined.parquet
  figure_ptx_str/
    spatial/
      all_cells/
        Triptych_*.png
      per_group/
        Triptych_*.png
      figures_quant/
        *.png
      figures_radial/
        *.png
      spatial_quantification_summary.md
      radial_center_summary.md
    validation/
      step_up_validation_*.png
```

## Troubleshooting

**Step 2 fails with CUDA errors**: Pass `--s2 "--device cpu"` to run on CPU.

**Step 1 runs out of memory**: Use `--s1 "--start 0 --end 5"` to process
pairs in batches, then combine.

**Missing hexbin parquets for steps 5-8**: Re-run step 4 first. Steps 5-8
depend on the hexbin parquets produced by step 4.

**Wrong experiment selected**: Use `--dry-run` to verify the experiment
before running. The banner shows the resolved experiment name.
