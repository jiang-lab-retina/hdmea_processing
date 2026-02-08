# HTR Agonist/Antagonist Analysis Pipeline

> **See [PIPELINE_CLI.md](PIPELINE_CLI.md) for command line usage with multiple datasets.**

Analysis pipeline for HTR (5-HT receptor) agonist experiments on mouse retinal ganglion cells (RGCs). This pipeline processes MEA recordings, aligns units across multiple recording sessions, classifies cell types, and generates statistical visualizations comparing WT vs KO genotypes.

## Overview

This pipeline processes data from experiments where:
1. **Before recording**: Baseline with step light stimulus (for cell type classification)
2. **Middle recordings**: Time series during agonist application (multiple 10-min recordings)
3. **After recording**: Final recording with step light stimulus

The analysis tracks individual RGC units across all recordings to measure the effect of agonist on firing rate, comparing Httr WT vs HttrB KO genotypes.

## Pipeline Architecture

```
Raw CMCR/CMTR files
        │
        ▼
┌─────────────────────┐
│  batch_pipeline.py  │  → Process raw files to HDF5
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    alignment.py     │  → Align BEFORE→AFTER units, classify cell types
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│ middle_recordings_alignment.py  │  → Align middle recordings to chain
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  middle_visualization.py    │  → Generate figures and statistics
└─────────────────────────────┘
```

## Files Description

### Core Pipeline

| File | Description |
|------|-------------|
| `specific_config.py` | Configuration: paths, parameters, data sources |
| `batch_pipeline.py` | Process raw CMCR/CMTR files to HDF5 format |
| `alignment.py` | Align units between BEFORE and AFTER recordings with cell type classification |
| `middle_recordings_alignment.py` | **Default method**: Align middle recordings using BEFORE→AFTER chain with zero-fill for missing data |
| `middle_recordings_alignment_strict.py` | **Alternative method**: Strict chain matching (requires match in all recordings) |
| `middle_visualization.py` | Generate time course plots, statistics, and summary markdown |

### Additional Tools

| File | Description |
|------|-------------|
| `visualize_alignment.py` | Visualize BEFORE→AFTER alignment results |
| `baseline_change_alignment.py` | Align recordings for baseline change analysis |
| `baseline_change_visualization.py` | Visualize baseline changes over time |
| `debug_cell_type.py` | Debug cell type classification |
| `debug_workflow.py` | Debug overall workflow |

## Cell Type Classification

Units are classified based on their step light response (from the BEFORE recording):

| Cell Type | Criteria |
|-----------|----------|
| **ON** | Increased firing 1-3s after light ON |
| **OFF** | Increased firing 6-8s (1-3s after light OFF) |
| **ON_OFF** | Both ON and OFF responses |
| **unknown** | Does not meet classification thresholds |

**Classification parameters:**
- Minimum response rate: 5.0 Hz
- Minimum difference from baseline: 3.0 Hz
- Quality index threshold: 0.1

## Alignment Methods

### Default Method (`middle_recordings_alignment.py`)

1. Uses BEFORE→AFTER alignment as the primary unit chain
2. **First middle recording**: Must have a real match (no zero-fill)
3. **Subsequent middle recordings**: Zero-filled if match not found
4. Ensures all tracked units have reliable starting data

### Strict Method (`middle_recordings_alignment_strict.py`)

- Requires successful match in ALL recordings
- No zero-filling allowed
- Results in fewer units but complete data chains

## Usage

### Full Pipeline

```bash
# 1. Process raw files to HDF5
python batch_pipeline.py

# 2. Align BEFORE→AFTER recordings
python alignment.py

# 3. Align middle recordings (default method)
python middle_recordings_alignment.py

# 4. Generate visualizations
python middle_visualization.py
```

### Process Additional Data Folder

```bash
# Process all files in a different folder
python batch_pipeline.py --data_folder "P:/20251002_HttrB_antagonist" --skip_existing
```

## Output Structure

```
output/
├── *.h5                          # Processed HDF5 files
├── aligned/                      # BEFORE→AFTER aligned pairs
│   └── {before}_to_{after}_aligned.h5
├── aligned_middle/               # Middle recordings aligned
│   └── {first_middle}_to_{last_middle}_middle_aligned.h5
├── figures/                      # BEFORE→AFTER visualization
│   ├── *.png
│   └── SUMMARY.md
└── figures_middle/               # Middle recordings visualization
    ├── timecourse_by_genotype.png
    ├── timecourse_by_cell_type.png
    ├── genotype_change_comparison.png
    ├── wt_ko_timecourse_comparison.png
    ├── baseline_vs_agonist_scatter.png
    ├── effect_size_comparison.png
    ├── cell_type_*.png
    └── STATISTICAL_SUMMARY.md

**Example filenames:**
- `2025.10.01-09.33.32-Rec_to_2025.10.01-10.15.53-Rec_aligned.h5` (BEFORE→AFTER)
- `2025.10.01.09.45.32.Rec_to_2025.10.01.10.05.45.Rec_middle_aligned.h5` (middle)
```

## Key Visualizations

| Figure | Description |
|--------|-------------|
| `genotype_change_comparison.png` | WT vs KO percent change by cell type with significance |
| `wt_ko_timecourse_comparison.png` | WT and KO firing rate time courses overlaid |
| `timecourse_by_cell_type.png` | Time courses separated by genotype and cell type |
| `effect_size_comparison.png` | Cohen's d effect size comparison |
| `baseline_vs_agonist_scatter.png` | Individual unit baseline vs agonist response |
| `STATISTICAL_SUMMARY.md` | Complete statistical analysis report |

## Configuration

Edit `specific_config.py` to change:

```python
# Data source
DATA_FOLDER = Path("O:/20251001_HttrB_agonist")

# Google Sheet metadata
GSHEET_CSV_PATH = PROJECT_ROOT / "Projects/load_gsheet/gsheet_table.csv"

# Playlist for stimulus conditions
PLAYLIST_CSV_PATH = Path("//Jiangfs1/.../playlist.csv")
```

## Dependencies

- numpy
- pandas
- h5py
- matplotlib
- scipy

Uses internal `hdmea` library for MEA data processing.

## Data Format

### Input
- **CMCR files**: MaxWell Biosystems raw recording data
- **CMTR files**: Spike sorting results
- **Google Sheet CSV**: Metadata including chip ID, genotype, condition

### Output (HDF5)
- `/units/unit_XXX/`: Unit data (waveform, spike_times, coordinates)
- `/metadata/`: Recording metadata
- `/aligned_units/`: Aligned unit chains with cell type labels

## Genotype Order

Visualizations display:
1. **Httr WT** (first/top/left) - shown in green
2. **HttrB KO** (second/bottom/right) - shown in red

## Statistical Methods

- **Paired t-test**: Within-genotype baseline vs agonist comparison
- **Independent t-test**: WT vs KO comparison of agonist effect magnitude
- **Mann-Whitney U test**: Non-parametric alternative for WT vs KO
- **Cohen's d**: Effect size (small: 0.2, medium: 0.5, large: 0.8)

## Time Parameters

- **Sample rate**: 20 kHz
- **Bin size**: 30 seconds
- **Agonist start**: 5 minutes (first recording only)
- **Recording duration**: ~10 minutes each
