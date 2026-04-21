# GB Control Spatial Analysis Pipeline

## Overview

This pipeline performs spatial visualization and topology analysis of
**green-blue chromatic response features** across the retina, using the
*before-blocker* recordings from all three blocker experiments as a combined
control dataset.

The key scientific question is: **do green-blue chromatic response properties
vary systematically with retinal position (relative to the ONH)?**

Because all three experiments share the same recording protocol and the
before-blocker condition is a pharmacological control with no manipulation, the
recordings can be pooled into a single high-powered control dataset.

```
Inputs (3 experiments)
  _ptx_str  -- 14,748 cells (14,254 with valid coords)
  _ptx      -- 7,970  cells (7,804  with valid coords)
  _str      -- 8,359  cells (7,791  with valid coords)
  ---
  Combined  -- 29,849 cells with valid spatial coordinates

 --> Prepare --> Spatial Plots --> Quantification --> Radial Center
                    |                  |                   |
                 Hexbin PNGs      Quant parquets    Center parquets
                 Hexbin parquets   Markdown summary  Markdown summary
                                       |                   |
                                   Viz Quant          Viz Radial
                                   (6 PNGs)           (5 PNGs)
```

---

## Prerequisites

- `dataframe_compare` pipeline (Steps 1-3) must have completed for all three
  experiments, producing:
  - `output_ptx_str/compared_dataframe_v2_labeled_spatial.parquet`
  - `output_ptx/compared_dataframe_v2_labeled_spatial.parquet`
  - `output_str/compared_dataframe_v2_labeled_spatial.parquet`

- These parquets contain `before_*` green-blue feature columns (added by
  `pipeline_compare.py` Step 1 sub-step 5) and spatial coordinates
  `improved_tx`, `improved_ty` (added by `spatial_improve_onh.py`).

---

## Feature Set (36 columns)

After loading, the `before_` prefix is stripped so all column names are
unprefixed feature names.

### Ratios (8)

| Column | Description |
|--------|-------------|
| `green_blue_on_ratio` | ON-phase green/blue amplitude ratio (all 9 trials averaged) |
| `green_blue_on_ratio_low` | ON ratio, low-intensity subset (trials 0-2, stimulus level 64) |
| `green_blue_on_ratio_mid` | ON ratio, mid-intensity subset (trials 3-5, stimulus level 128) |
| `green_blue_on_ratio_high` | ON ratio, high-intensity subset (trials 6-8, stimulus level 255) |
| `green_blue_off_ratio` | OFF-phase green/blue amplitude ratio (all trials) |
| `green_blue_off_ratio_low` | OFF ratio, low intensity |
| `green_blue_off_ratio_mid` | OFF ratio, mid intensity |
| `green_blue_off_ratio_high` | OFF ratio, high intensity |

### Baseline (8)

| Column | Description |
|--------|-------------|
| `gb_base_mean` | Mean baseline firing rate during green-blue stimulus (all trials) |
| `gb_base_mean_low/mid/high` | Baseline mean for each intensity level |
| `gb_base_std` | Baseline firing rate SD (all trials) |
| `gb_base_std_low/mid/high` | Baseline SD for each intensity level |

### Peak Extremes (16)

For each combination of `{color}` in `[green, blue]`, `{phase}` in `[on, off]`,
and `{level}` in `["", _low, _mid, _high]`:

- `{color}_{phase}_peak_extreme{level}` -- signed peak amplitude relative to
  baseline (positive = excitatory, negative = inhibitory)

### Timing (4)

| Column | Validity |
|--------|----------|
| `time_to_green_on_peak` | ~92% valid |
| `time_to_blue_on_peak` | ~91% valid |
| `time_to_green_off_peak` | ~87% valid |
| `time_to_blue_off_peak` | ~86% valid |

---

## Directory Layout

```
dataframe_compare/
  gb_spatial_control/
    config.py                  -- all shared constants (paths, feature lists, colors)
    prepare_data.py            -- Step 1
    spatial_plots.py           -- Step 2
    spatial_quantification.py  -- Step 3
    spatial_radial_center.py   -- Step 4
    visualize_quant.py         -- Step 5
    visualize_radial.py        -- Step 6
    main_pipeline.py           -- orchestrator
    PIPELINE.md                -- this file

  output_gb_control/
    combined_gb_control.parquet            -- 29,849 x 42 cols
    hexbin_all_cells.parquet               -- 33,407 bins x 7 cols
    hexbin_per_group.parquet               -- 17,903 bins x 7 cols
    spatial_metrics.parquet                -- 180 rows x 13 cols
    spatial_quant_all.parquet              -- 36 features x 30 cols
    spatial_quant_per_group.parquet        -- 144 rows x 30 cols
    spatial_quant_combined.parquet         -- 180 rows x 30 cols
    radial_center_all.parquet              -- 72 rows x 13 cols
    radial_center_per_group.parquet        -- 144 rows x 13 cols
    radial_center_combined.parquet         -- 216 rows x 13 cols

  figure_gb_control/
    spatial/
      all_cells/
        Hexbin_{feature}.png               -- 36 PNGs (raw + GAM)
      per_group/
        Hexbin_{group}_{feature}.png       -- 144 PNGs (4 groups x 36 features)
      figures_quant/
        gradient_polar.png
        plane_vs_gam_r2.png
        moran_bar.png
        radial_bar.png
        gradient_magnitude_ranked.png
        summary_dashboard.png
      figures_radial/
        radial_center_map.png
        origin_vs_optimal.png
        radial_profiles_top.png
        category_clustering.png
        radial_dashboard.png
      spatial_quantification_summary.md
      radial_center_summary.md
```

---

## Script Details

### `config.py` -- Shared Configuration

Central location for all paths, feature lists, and plotting constants.
Nothing in the pipeline is hardcoded; all scripts import from here.

Key exports:

| Symbol | Value | Description |
|--------|-------|-------------|
| `SOURCE_PARQUETS` | dict | Paths to source parquets keyed by experiment |
| `OUTPUT_DIR` | Path | `output_gb_control/` |
| `FIG_DIR_BASE` | Path | `figure_gb_control/` |
| `ALL_GB_FEATURES` | list[str] | All 36 unprefixed feature names |
| `GB_RATIO/BASELINE/PEAK/TIMING_FEATURES` | list[str] | Feature sub-groups |
| `CAT_COLORS` | dict | Plot colors: Ratio=blue, Baseline=green, Peak=orange, Timing=purple |
| `COORD_SCALE` | 16 | Converts ONH-normalized units to micrometers |
| `COORD_LIMIT` | 100 | Max retinal extent in ONH units (~1600 um) |

---

### Step 1: `prepare_data.py` -- Combine and Clean

**Runtime**: < 1 min

Loads the three source parquets, extracts the `before_*` GB columns, strips
the prefix, applies coordinate filtering, and saves a single combined parquet.

**What it does**:
1. For each experiment, reads `compared_dataframe_v2_labeled_spatial.parquet`.
2. Selects the 36 `before_{feat}` columns, `improved_tx`, `improved_ty`,
   `group`, `subtype`, `before_dataset_id`.
3. Renames `before_green_blue_on_ratio_low` -> `green_blue_on_ratio_low`, etc.
4. Adds `source_experiment` column (`_ptx_str`, `_ptx`, or `_str`).
5. Drops rows where `improved_tx` or `improved_ty` is NaN, or where the
   coordinate magnitude exceeds `COORD_LIMIT` (100 ONH units = 1600 um).
6. Concatenates all three experiments and saves.

**Output**: `combined_gb_control.parquet` (29,849 rows x 42 columns)

Columns:
- 36 GB feature columns (float)
- `improved_tx`, `improved_ty` -- ONH-centered retinal coordinates (ONH units)
- `group` -- RGC class: `DSGC`, `OSGC`, `Other`, `ipRGC`
- `subtype` -- finer classification (e.g. `DSGC_3`)
- `before_dataset_id` -- recording ID for traceability
- `source_experiment` -- `_ptx_str`, `_ptx`, or `_str`

Group breakdown: DSGC 2,595 | OSGC 1,520 | Other 4,966 | ipRGC 795
(remaining ~20,000 cells have `group == ""`, i.e. unclassified)

---

### Step 2: `spatial_plots.py` -- Hexbin Heatmaps

**Runtime**: ~50 min (GAM fitting on 36 features x 29,849 cells)

Creates spatial heatmaps showing how each GB feature varies across the retina.
Each figure has two panels: **raw hexbin** (left) and **GAM-smoothed** (right).

**Coordinate system**: coordinates are scaled by `COORD_SCALE=16`, so the
x/y axes are in micrometers relative to the ONH. The plotted range is
$\pm 1600$ um in both axes.

**All-cells pass** (36 figures, `GRIDSIZE=40`, `MINCNT=2`):
- For each feature, bins all cells across the retina into 40x40 hexagonal grid.
- Fits a tensor-product GAM (`te(x, y, n_splines=[30,30])`) using `pygam`.
  GAM family is selected automatically: Logistic for binary, Poisson for count,
  Linear otherwise.
- Color scale: centered at the feature mean $\pm 50\%$ of |mean|.
- Saves figure to `figure_gb_control/spatial/all_cells/Hexbin_{feature}.png`.
- Records bin centers, raw means, counts, and GAM predictions in
  `hexbin_all_cells.parquet`.

**Per-group pass** (144 figures, 4 groups x 36 features, `GRIDSIZE=15`, `MINCNT=1`):
- Same procedure restricted to cells from each RGC group.
- No GAM predictions stored for per-group (only raw hexbin saved to parquet).
- Saves to `figure_gb_control/spatial/per_group/Hexbin_{group}_{feature}.png`.
- Records in `hexbin_per_group.parquet`.

**`hexbin_all_cells.parquet` schema** (33,407 rows):

| Column | Type | Description |
|--------|------|-------------|
| `scope` | str | `"all_cells"` |
| `feature` | str | Unprefixed feature name |
| `bin_x` | float | Bin center x in micrometers |
| `bin_y` | float | Bin center y in micrometers |
| `count` | int | Number of cells in bin |
| `raw_mean` | float | Mean feature value in bin |
| `gam_pred` | float | GAM-predicted value at bin center |

**`spatial_metrics.parquet` schema** (180 rows = 36 features x 5 scopes):

| Column | Description |
|--------|-------------|
| `n_valid` | Cells with valid feature values |
| `n_bins` | Number of populated hexbins |
| `overall_mean/std` | Global mean and SD of the feature |
| `hexbin_cv` | Coefficient of variation of bin means |
| `gradient_mag/dir_deg/r2` | Planar gradient magnitude, direction, explained variance |
| `radial_r/p` | Pearson r of feature vs radial distance from ONH |
| `moran_i` | Global Moran's I (kNN, k=6) |

---

### Step 3: `spatial_quantification.py` -- Quantification

**Runtime**: ~1 min

Runs eight metric categories on each feature and scope combination using the
stored hexbin data as input.

**Analysis per feature per scope**:

| # | Metric | Method |
|---|--------|--------|
| 1 | **Weighted plane fit** | WLS: $z \sim b_0 + b_x x + b_y y$, weights = bin counts / total |
| 2 | **GAM structure** | GAM R2, nonlinearity $\Delta R^2$ over plane, dynamic range, hotspot area |
| 3 | **Global Moran's I** | kNN weight matrix ($k=6$), raw bin means |
| 4 | **Radial correlation** | Pearson $r$ between radial distance from ONH and bin mean; 499-bootstrap 95% CI |
| 5 | **Quadrant ANOVA** | One-way ANOVA on bins split into four 90-deg sectors from ONH |
| 6 | **Permutation p-values** | 499 permutations of bin values; null for plane R2 and Moran's I |
| 7 | **FDR correction** | Benjamini-Hochberg across all features within each scope |

**`spatial_quant_all.parquet` schema** (36 rows x 30 columns):

| Column group | Columns |
|--------------|---------|
| Identity | `feature`, `scope`, `n_bins` |
| Summary | `overall_mean`, `overall_std` |
| Plane | `bx`, `by`, `b0`, `grad_mag`, `grad_dir_deg`, `plane_r2` |
| GAM | `gam_r2`, `gam_delta_r2`, `gam_dynamic_range`, `gam_max/min_x/y`, `hotspot_area_frac` |
| Moran | `moran_i` |
| Radial | `radial_r`, `radial_p`, `radial_r_lo`, `radial_r_hi` |
| Quadrant | `quadrant_F`, `quadrant_p` |
| Significance | `perm_p_plane_r2`, `perm_p_moran`, `fdr_q_plane_r2`, `fdr_q_moran` |

**Top results** (all-cells, sorted by plane $R^2$):

| Feature | Plane R2 | Moran's I | Radial r | Gradient direction |
|---------|----------|-----------|----------|--------------------|
| `blue_off_peak_extreme_high` | 0.217 | 0.307 | -0.099 | -22 deg |
| `green_off_peak_extreme_mid` | 0.214 | 0.308 | -0.114 | -14 deg |
| `green_off_peak_extreme_high` | 0.194 | 0.298 | -0.110 | -24 deg |
| `blue_off_peak_extreme_mid` | 0.192 | 0.303 | -0.091 | -11 deg |
| `blue_off_peak_extreme` | 0.191 | 0.305 | -0.096 | -14 deg |
| `green_off_peak_extreme` | 0.189 | 0.301 | -0.110 | -17 deg |

The **OFF peak extremes** show the strongest and most consistent spatial
gradients (~$R^2 \approx 0.19$-0.22), with gradient directions clustered
near $-15$ deg (approximately superior-nasal). Positive Moran's I (~0.30)
confirms spatial clustering rather than random scatter.

---

### Step 4: `spatial_radial_center.py` -- Radial Center Search

**Runtime**: < 1 min

For each feature and scope, searches for the retinal position that maximizes
the absolute Pearson $|r|$ between radial distance and feature value. The
optimal center may differ from the ONH (anatomical optic nerve head) if a
feature has a center-surround organization shifted away from the ONH.

**Search algorithm**:
1. **Baseline**: compute $r$ from the ONH origin $(0, 0)$.
2. **Coarse grid**: $\pm 1200$ um in 200 um steps; track best $(c_x, c_y)$.
3. **Fine grid**: $\pm 300$ um around coarse best, 50 um steps.
4. **Nelder-Mead refinement**: `scipy.optimize.minimize` on $-|r|$, bounded
   at $\pm 1800$ um.

Both `raw_mean` and `gam_pred` hexbin values are analyzed (tagged as
`data_type` in the output).

**`radial_center_all.parquet` schema** (72 rows = 36 features x 2 data types):

| Column | Description |
|--------|-------------|
| `origin_r` | Pearson r at ONH origin (0, 0) |
| `origin_p` | p-value at origin |
| `origin_slope` | Slope of feature ~ radius at origin |
| `best_center_x/y` | Optimal center coordinates (micrometers) |
| `best_r` | Pearson r at optimal center |
| `best_p` | p-value at optimal center |
| `best_slope` | Slope of feature ~ radius at optimal center |
| `abs_r_improvement` | `|best_r| - |origin_r|` |
| `data_type` | `"raw_mean"` or `"gam_pred"` |
| `scope` | `"all_cells"` or group name |
| `n_bins` | Number of bins used |

**Top features by radial correlation improvement** (raw mean, all cells):

| Feature | origin_r | best_r | Best Cx (um) | Best Cy (um) | Improvement |
|---------|----------|--------|--------------|--------------|-------------|
| `blue_off_peak_extreme_high` | ~0.05 | ~0.43 | 942 | ... | 0.380 |
| `green_off_peak_extreme_mid` | ~0.06 | ~0.41 | 1237 | ... | 0.353 |
| `green_off_peak_extreme_high` | ~0.06 | ~0.41 | 845 | ... | 0.354 |
| `green_off_peak_extreme` | ~0.06 | ~0.40 | 908 | ... | 0.346 |
| `blue_off_peak_extreme` | ~0.06 | ~0.40 | 993 | ... | 0.356 |

The large improvement (0.35-0.38) over origin indicates that the true
functional center of radial organization for OFF peak responses is shifted
substantially (~900-1200 um) from the anatomical ONH.

---

### Step 5: `visualize_quant.py` -- Quantification Figures

**Runtime**: < 1 min

Reads `spatial_quant_combined.parquet`, filters to `scope == "all_cells"`.
Features are color-coded by category: Ratio=blue, Baseline=green,
Peak=orange, Timing=purple.

| Figure | Description |
|--------|-------------|
| `gradient_polar.png` | Polar scatter: angle = gradient direction, radius = plane R2. Shows dominant gradient directions across all features. |
| `plane_vs_gam_r2.png` | Scatter of plane R2 (x) vs GAM R2 (y); points above the diagonal indicate nonlinear spatial structure beyond a linear gradient. |
| `moran_bar.png` | Horizontal bar chart of Moran's I per feature. Positive values indicate spatial clustering; negative values indicate spatial dispersion. |
| `radial_bar.png` | Horizontal bar chart of radial r per feature with 95% bootstrap CI error bars. Negative values indicate center-high (higher values near ONH); positive indicates periphery-high. |
| `gradient_magnitude_ranked.png` | Features ranked by gradient magnitude (color = category). Shows which features have the steepest spatial variation across the retina. |
| `summary_dashboard.png` | 2x3 panel overview: plane R2, Moran's I, radial r (top row) and gradient magnitude, GAM R2, GAM dynamic range (bottom row). |

---

### Step 6: `visualize_radial.py` -- Radial Center Figures

**Runtime**: < 1 min

Reads `radial_center_combined.parquet`, filters to
`data_type == "raw_mean"` and `scope == "all_cells"`.

| Figure | Description |
|--------|-------------|
| `radial_center_map.png` | Scatter of optimal centers in retinal space. Marker size is proportional to `|best_r|`. Red cross = anatomical ONH at origin. Reveals whether chromatic features share a common functional center. |
| `origin_vs_optimal.png` | Horizontal bar chart comparing `|origin_r|` (ONH) vs `|best_r|` (optimal). Sorted by improvement. Shows how much radial structure is missed when using the ONH as reference. |
| `radial_profiles_top.png` | 6-panel scatter of feature value vs distance for the six features with strongest `|best_r|`. Includes a running-average smoothing line. |
| `category_clustering.png` | Same as `radial_center_map` but grouped by feature category with 1-SD ellipses. Shows whether Ratio, Baseline, Peak, and Timing features share a common functional center. |
| `radial_dashboard.png` | 4-panel summary: |best_r| ranked bar, improvement bar, center distance from ONH, radial direction count (center-high vs periphery-high). |

---

### Orchestrator: `main_pipeline.py`

**Usage**:

```powershell
cd m:\Python_Project\Data_Processing_2027\dataframe_compare\gb_spatial_control

# Run all 6 steps
python main_pipeline.py

# Run specific steps
python main_pipeline.py --steps 2
python main_pipeline.py --steps 3 4
python main_pipeline.py --steps 5-6

# Preview without running
python main_pipeline.py --dry-run
python main_pipeline.py --steps 2-4 --dry-run
```

Step dependency graph:

```
Step 1 (prepare)
  |
  v
Step 2 (spatial_plots)  <-- produces hexbin parquets
  |
  +---> Step 3 (spatial_quant)  ---> Step 5 (viz_quant)
  |
  +---> Step 4 (radial_center)  ---> Step 6 (viz_radial)
```

Steps 3 and 4 are independent of each other (both depend on Step 2).
Steps 5 and 6 are independent of each other.

---

## Coordinate System

All spatial coordinates use the **improved ONH-centered system** computed by
`spatial_improve_onh.py` in the upstream `dataframe_compare` pipeline:

- **Origin** $(0, 0)$ = Optic Nerve Head (ONH), estimated robustly from AP
  pathway intersections with MAD outlier rejection.
- **Axes** are DVNT-angle-corrected so the D (dorsal) direction is up.
- **Units**: ONH-normalized units; multiply by `COORD_SCALE = 16` to get
  micrometers (so COORD_LIMIT=100 ONH units = 1600 um from ONH).
- In hexbin parquets and radial center parquets, all `bin_x/y` and
  `best_center_x/y` coordinates are already in **micrometers**.

---

## Key Scientific Findings

Based on the quantification results (`spatial_quant_all.parquet`):

1. **OFF peak extremes show the strongest retinal gradients** across all
   features, with plane $R^2 \approx 0.19$-0.22 and gradient directions
   near $-15$ to $-24$ deg (approximately superior-nasal).

2. **Spatial clustering is consistent across all features** (Moran's I
   $\approx 0.20$-0.31 for all features), indicating that GB response
   properties are not randomly distributed but are topographically organized.

3. **The optimal radial center is shifted from the ONH** for OFF peak features
   (by ~900-1200 um), with large improvements in radial correlation
   (0.35-0.38 above origin). This suggests that the functional center of
   chromatic OFF organization does not coincide with the anatomical ONH.

4. **Intensity-dependent differences**: high-intensity features tend to have
   stronger spatial gradients than low-intensity equivalents, particularly
   for the OFF peak extremes.

---

## Extending the Pipeline

**Add new features**: Edit `ALL_GB_FEATURES` in `config.py`, then re-run
Steps 1-6 (`python main_pipeline.py`). Step 1 will attempt to load the new
`before_{feat}` columns from the source parquets.

**Change plot appearance**: Edit plot constants in `config.py`
(`CMAP`, `CAT_COLORS`, `GRIDSIZE_ALL`, `N_SPLINES_ALL`, etc.), then
re-run Steps 2 and/or 5-6.

**Regenerate only figures**: Re-run Steps 5-6 without re-running the slow
GAM fitting (Step 2) or permutation tests (Step 3):
```powershell
python main_pipeline.py --steps 5 6
```

**Change hexbin resolution**: Edit `GRIDSIZE_ALL` / `GRIDSIZE_GRP` in
`config.py`, then re-run Steps 2-6.
