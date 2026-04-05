# Blocker Alignment Analysis Pipeline

Analysis pipeline for GABA/glycine blocker experiments on mouse retinal ganglion cells (RGCs). This pipeline discovers recordings from the Google Sheet, locates raw files on O: drive, processes MEA recordings, aligns units across before/after recording sessions, classifies cell types, and generates a structured file index.

## Overview

This pipeline processes data from experiments where:
1. **Before recording**: Baseline with step light stimulus (for cell type classification) - condition: `play_optimization_set6_a_ipRGC_without_step()`
2. **After recording**: Recording with GABA and glycine blockers applied - condition: `play_optimization_set6_a_ipRGC_manual(), gaba, glycine`

The analysis tracks individual RGC units across both recordings to measure the effect of GABA/glycine blockers on firing patterns, comparing BLK vs OpnCre/Rtdt genotypes.

## Data Summary

- **Date range**: 2025.09.04 to 2025.09.19 (8 experiment dates)
- **Total recordings**: 96 (48 before + 48 gaba_gly)
- **Recording pairs**: 48 (one before/after pair per chip per date)
- **Genotypes**: BLK, OpnCre/Rtdt
- **Chips**: ~20 unique chips across all dates

## Pipeline Architecture

```
Google Sheet CSV + O: Drive
        |
        v
+---------------------+
|  discover_files.py   |  -> CSV index of all recordings
+---------------------+
        |
        v
+---------------------+
|  batch_pipeline.py   |  -> Process raw files to HDF5
+---------------------+
        |
        v
+---------------------+
|    alignment.py      |  -> Align BEFORE->AFTER units, classify cell types
+---------------------+
```

## Files Description

### Core Pipeline

| File | Description |
|------|-------------|
| `specific_config.py` | Configuration: paths, date range, folder layout, helper functions |
| `discover_files.py` | Discover files from gsheet + O: drive, generate CSV index |
| `batch_pipeline.py` | Process raw CMCR/CMTR files to HDF5 format using CSV index |
| `alignment.py` | Align units between before and after recordings with cell type classification |

## Data Sources

### Google Sheet CSV
- **Path**: `Projects/load_gsheet/gsheet_table.csv`
- **Key columns**: `File_name`, `Condition`, `Chip`, `Genotype`, `Note`
- **Filter**: Rows where `File_name` date is between 2025.09.04 and 2025.09.19

### O: Drive Data Folders
Raw recordings are organized by date on O: drive:

```
O:/
+-- 20250904_gaba_gly/
|   +-- before_gaba_gly/    <- step stimulus recordings
|   |   +-- 2025.09.04-10.11.09-Rec.cmcr
|   |   +-- 2025.09.04-10.11.09-Rec-.cmtr
|   +-- gaba_gly/            <- blocker recordings
|       +-- 2025.09.04-10.38.19-Rec.cmcr
|       +-- 2025.09.04-10.38.19-Rec-.cmtr
+-- 20250905_gaba_gly/
+-- 20250909_gaba_gly/
+-- 20250911_gaba_gly/
+-- 20250912_gaba_gly/
+-- 20250916_gaba_gly/
+-- 20250917_gaba_gly/
+-- 20250919_gaba_gly/
```

### Filename Convention Differences

| Format | Example |
|--------|---------|
| **Google Sheet** (dots) | `2025.09.04.10.11.09.Rec.cmcr` |
| **Disk CMCR** (hyphens) | `2025.09.04-10.11.09-Rec.cmcr` |
| **Disk CMTR** (trailing hyphen) | `2025.09.04-10.11.09-Rec-.cmtr` |

The pipeline handles these conversions automatically.

### Condition Column Parsing

The Condition column for blocker recordings contains comma-separated values:
```
play_optimization_set6_a_ipRGC_manual(), gaba, glycine
```

The pipeline extracts the part **before the first comma** for playlist resolution:
```
play_optimization_set6_a_ipRGC_manual()  ->  strip ()  ->  play_optimization_set6_a_ipRGC_manual
```

## Usage

### Step 1: Discover files and generate CSV index

```bash
python discover_files.py

# With debug logging:
python discover_files.py --debug

# Refresh gsheet from Google Sheets API first:
python discover_files.py --refresh-gsheet
```

This creates `output/file_index.csv` with all recording metadata and file paths.

### Step 2: Process raw files to HDF5

```bash
# Process all recordings:
python batch_pipeline.py

# Process a specific range:
python batch_pipeline.py --start 0 --end 10

# Force overwrite existing files:
python batch_pipeline.py --overwrite
```

### Step 3: Align before/after recordings

```bash
python alignment.py

# With custom parameters:
python alignment.py --quality-threshold 0.05 --waveform-weight 10.0
```

## Output Structure

```
output/
+-- file_index.csv              <- CSV index of all discovered files
+-- *.h5                        <- Processed HDF5 files (96 files)
+-- aligned/                    <- Before/after aligned pairs
    +-- {before}_to_{after}_aligned.h5
```

### CSV Index Columns

| Column | Description |
|--------|-------------|
| `pair_id` | Links before/after recordings (e.g., `pair_000`) |
| `recording_type` | `before` or `gaba_gly` |
| `date` | Recording date (e.g., `2025.09.04`) |
| `time` | Recording time (e.g., `10.11.09`) |
| `chip` | Chip ID |
| `genotype` | `BLK` or `OpnCre/Rtdt` |
| `gsheet_filename` | Filename as it appears in Google Sheet |
| `disk_cmcr` | CMCR filename on disk (with hyphens) |
| `disk_cmtr` | CMTR filename on disk (with trailing hyphen) |
| `cmcr_path` | Full path to CMCR file on O: drive |
| `cmtr_path` | Full path to CMTR file on O: drive |
| `cmcr_exists` | Whether CMCR was found on disk |
| `cmtr_exists` | Whether CMTR was found on disk |
| `condition` | Full Condition value from gsheet |
| `playlist_condition` | Part before first comma (for playlist resolution) |
| `data_folder` | O: drive folder path |
| `note` | Notes from gsheet |

## Pairing Logic

Before/after recordings are paired by:
1. **Same Chip ID** on the **same date**
2. **Temporal validation**: The gaba_gly recording must **immediately follow** the before recording in time (sorted by filename timestamp)
3. **No intervening recordings** between the pair

## Cell Type Classification

Units are classified based on their step light response (from the before recording):

| Cell Type | Criteria |
|-----------|----------|
| **ON** | Increased firing 1-3s after light ON |
| **OFF** | Increased firing 6-8s (1-3s after light OFF) |
| **ON_OFF** | Both ON and OFF responses |
| **unknown** | Does not meet classification thresholds |

## Dependencies

- numpy
- pandas
- h5py
- matplotlib
- scipy

Uses internal `hdmea` library for MEA data processing.

## Configuration

Edit `specific_config.py` to change:

```python
# Date range
DATE_START = "2025.09.04"
DATE_END = "2025.09.19"

# Google Sheet CSV cache
GSHEET_CSV_PATH = PROJECT_ROOT / "Projects/load_gsheet/gsheet_table.csv"

# Playlist CSV (network path)
PLAYLIST_CSV_PATH = Path("//Jiangfs1/.../playlist.csv")
```
