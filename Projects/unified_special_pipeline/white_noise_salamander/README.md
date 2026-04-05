# White Noise Salamander STA Pipeline

Spike-triggered average (STA) analysis and receptive-field geometry extraction
for salamander retinal ganglion cells stimulated with dense white noise.

## Directory structure

```
white_noise_salamander/
  run_analysis.py          # Main STA + geometry pipeline
  analyze_rf_geometry.py   # Post-hoc RF geometry comparison (young vs old)
  specific_config.py       # Paths, parameters, recording discovery
  output/                  # HDF5 files (one per recording)
  figures/                 # Per-recording STA plots + rf_geometry_analysis/
```

## Prerequisites

- Python environment with `hdmea`, `numpy`, `h5py`, `matplotlib`, `scipy`,
  `pandas`, `tqdm` installed.
- Access to network drives containing raw CMCR/CMTR recordings and stimulus
  `.npy` files (configured in `specific_config.py`).

---

## 1. `run_analysis.py` — STA + RF geometry pipeline

### Quick start

```bash
# From the white_noise_salamander/ directory:

# Process the default test recording (session approach)
python run_analysis.py

# Process all recordings discovered in DATA_FOLDERS
python run_analysis.py --all

# Process recordings from Google Sheet by date
python run_analysis.py --gsheet-date 2024.03.04
python run_analysis.py --gsheet-date 2026.03.03

# Overwrite existing results
python run_analysis.py --gsheet-date 2024.03.04 --overwrite
```

### CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--all` | off | Process all recordings found in `DATA_FOLDERS` (instead of `TEST_FILES` only) |
| `--legacy` | off | Use the legacy direct STA approach (`_compute_sta_for_unit`) instead of the default session-based `compute_sta` |
| `--gsheet-date DATE` | None | Discover recordings from the Google Sheet cache filtered by date (e.g. `2024.03.04`). Only processes recordings whose `Condition` contains `play_movie(...)` |
| `--start N` | 0 | Start index in the recording list (for partial runs) |
| `--end N` | None | End index (exclusive) in the recording list |
| `--overwrite` | off | Recompute even if output already exists |
| `--debug` | off | Enable DEBUG-level logging |

### Configuration (`specific_config.py`)

All pipeline parameters are defined in `specific_config.py` and bundled into
`STAPipelineConfig`:

| Parameter | Default | Description |
|---|---|---|
| `DATA_FOLDERS` | `S:\20240227_salamander`, ... | Directories searched for CMCR/CMTR files |
| `STIMULUS_PATH` | `...perfect_dense_noise_15x15_5hz_r42_10min.npy` | Default stimulus movie file |
| `SECTION_TIME_FRAME_NUM` | `(184, inf)` | Frame range for spike sectioning. Frame 184 is the first valid stimulus frame after the pre-stimulus period |
| `STA_COVER_RANGE` | `(-60, 0)` | Number of stimulus frames before each spike to include in STA |
| `FRAME_CHANNEL_KEY` | `"raw_ch2"` | Light-reference channel for frame timestamp detection |
| `OUTPUT_DIR` | `./output` | Directory for HDF5 output files |
| `FIGURES_DIR` | `./figures` | Directory for per-unit STA figure output |

### Google Sheet mode

When `--gsheet-date` is provided, the pipeline reads a cached CSV
(`Projects/load_gsheet/gsheet_table.csv`) from the MEA dashboard Google Sheet.
It filters rows where:

1. `File_name` starts with the given date string, and
2. `Condition` matches `play_movie("filename.npy")`.

The movie filename is extracted and the corresponding `.npy` file is loaded
from the same directory as the default stimulus. This enables batch processing
of recordings with different stimulus movies.

### Output

- **HDF5** (`output/<dataset_id>.h5`): Contains per-unit STA arrays,
  spike data, and `sta_geometry` (Gaussian, DoG, ON/OFF, LNL fits).
- **Figures** (`figures/<dataset_id>/`): Per-unit spatial and temporal
  STA plots.

---

## 2. `analyze_rf_geometry.py` — RF geometry comparison

Compares receptive-field properties between age groups:

- **Young larval** — 2024.03.04 recordings
- **Old larval** — 2026.03.03 recordings

### Quick start

```bash
# Run with default settings (thresholds 0.5, 0.7, 0.8, 0.9)
python analyze_rf_geometry.py

# Override HDF5 input directory
python analyze_rf_geometry.py --output-dir /path/to/h5/files
```

### CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--r2-threshold` | 0.5 | (Currently unused by the multi-threshold loop; the script iterates over 0.5, 0.7, 0.8, 0.9 automatically) |
| `--output-dir` | `./output` | Directory containing HDF5 files to analyze |

### Recommended threshold

**$R^2 \geq 0.8$ is the recommended threshold.** It provides the best
balance between data quality and sample size:

- Strict enough to remove poorly-fit units (427 of 679 removed), keeping
  only cells with reliable Gaussian RF fits.
- Retains 252 units (50 old, 202 young) -- enough statistical power to
  detect significant differences on all key RF size metrics.
- At $R^2 \geq 0.9$ the old-larval group drops to only 18 units, making
  some comparisons underpowered; at $R^2 \geq 0.5$ nearly all units pass,
  admitting many poor fits that add noise.

Use the `R2_0p8/` output folder for publication-quality figures and tables.

### What it does

1. Loads `sta_geometry` from all HDF5 files in `output/`.
2. Filters to `15x15_5hz` stimulus only.
3. Plots unfiltered Gaussian $R^2$ distributions.
4. For each threshold (0.5, 0.7, 0.8, 0.9):
   - Filters units by Gaussian $R^2 \geq$ threshold.
   - Generates violin plots, bar charts (mean +/- SEM with significance
     stars), RF center scatter, peak frame histograms, and surround
     index scatter.
   - Copies 10 sample STA spatial plots per age group into
     `sample_sta_plots/` for visual quality inspection.
   - Runs Mann-Whitney U tests for all metrics.
   - Saves CSVs: `statistical_tests.csv`, `summary_table.csv`,
     `filtered_units.csv`.

### Output structure

```
figures/rf_geometry_analysis/
  r2_distribution.png            # R2 histogram before filtering
  R2_0p5/                        # Threshold >= 0.5
    bar_comparison_15x15_5hz.png
    comparison_15x15_5hz.png     # Violin plots
    rf_centers_15x15_5hz.png
    peak_frame_15x15_5hz.png
    surround_index_15x15_5hz.png
    sample_sta_plots/            # 10 old + 10 young STA examples
    statistical_tests.csv
    summary_table.csv
    filtered_units.csv
  R2_0p7/                        # Threshold >= 0.7
    ...
  R2_0p8/                        # Threshold >= 0.8  ** recommended **
    ...
  R2_0p9/                        # Threshold >= 0.9
    ...
```

### Metrics compared

| Metric | Column | Description |
|---|---|---|
| RF area | `area` | Thresholded pixel count |
| Equiv. diameter | `equivalent_diameter` | Diameter of circle with same area |
| Gaussian sigma (geo) | `gauss_sigma_mean` | Geometric mean of sigma_x and sigma_y |
| Gaussian sigma_x, sigma_y | `gauss_sigma_x`, `gauss_sigma_y` | Individual axis widths |
| Gaussian R2 | `gauss_r2` | Goodness of 2D Gaussian fit |
| DoG R2 | `dog_r2` | Goodness of center-surround fit |
| DoG sigma_exc | `dog_sigma_exc` | Excitatory (center) width |
| DoG sigma_inh | `dog_sigma_inh` | Inhibitory (surround) width |
| Surround strength | `surround_strength` | Ratio of inhibitory to excitatory amplitude |
| ON/OFF ratio | `on_off_ratio` | Balance of ON vs OFF subfields |
| Peak frame | `peak_frame` | STA frame with strongest response |
| LNL bits/spike | `lnl_bits_per_spike` | Information transmission rate |
| Rectification index | `lnl_rectification_index` | Degree of output rectification |
| Nonlinearity index | `lnl_nonlinearity_index` | Deviation from linear transfer |
