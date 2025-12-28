# Unified Pipeline: Batch Processing All Steps

This guide explains how to batch process CMCR/CMTR recordings through all 11 pipeline steps in a single run.

## Overview

The `batch_all_steps.py` script processes recordings through the complete unified pipeline:

| Step | Name | Description |
|------|------|-------------|
| 1 | Load Recording | Load CMCR/CMTR with eimage_sta |
| 2 | Section Time | Add timing from playlist |
| 3 | Section Spikes | Section spike times by stimulus |
| 4 | Compute STA | Compute dense noise STA |
| 5 | Add Metadata | Add CMCR/CMTR file metadata |
| 6 | Soma Geometry | Extract soma geometry from eimage_sta |
| 7 | RF Geometry | Extract RF geometry from dense noise |
| 8 | Google Sheet | Add metadata from Google Sheet |
| 9 | Cell Type | Add manual cell type labels |
| 10 | AP Tracking | Compute AP pathway tracking (CNN) |
| 11 | DSGC | Section by direction analysis |

---

## Quick Start

### Process all recordings
```bash
python Projects/unified_pipeline/batch_all_steps.py
```

### Process first 10 recordings
```bash
python Projects/unified_pipeline/batch_all_steps.py --start 0 --end 10
```

---

## Prerequisites

### 1. CSV Mapping File
The script reads CMCR/CMTR paths from:
```
tool_box/generate_data_path_list/pkl_to_cmtr_mapping.csv
```

Required CSV columns:
| Column | Description |
|--------|-------------|
| `cmcr_path` | Full path to CMCR file |
| `cmtr_path` | Full path to CMTR file |
| `matched` | "True" for valid pairs |

### 2. Raw Data Access
CMCR and CMTR files must be accessible at the paths specified in the CSV.

### 3. External Resources
The following files are needed for steps 8-11:
- **Google Sheet credentials** or local CSV cache (`Projects/load_gsheet/gsheet_table.csv`)
- **Manual label files** (`.pkl` in configured label folder)
- **CNN model** (`Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth`)
- **ON/OFF dictionary** for DSGC analysis

---

## Usage

### Basic Usage
```bash
# Process all matched recordings
python Projects/unified_pipeline/batch_all_steps.py
```

### Process Specific Range
```bash
# Process recordings 0-9 (first 10)
python Projects/unified_pipeline/batch_all_steps.py --start 0 --end 10

# Process recordings 50-99
python Projects/unified_pipeline/batch_all_steps.py --start 50 --end 100

# Process from index 100 to the end
python Projects/unified_pipeline/batch_all_steps.py --start 100
```

### Force Overwrite
```bash
# Overwrite existing output files
python Projects/unified_pipeline/batch_all_steps.py --overwrite
```

### Custom Paths
```bash
# Use custom CSV mapping file
python Projects/unified_pipeline/batch_all_steps.py --csv path/to/custom_mapping.csv

# Use custom output directory
python Projects/unified_pipeline/batch_all_steps.py --output path/to/custom_output
```

### Debug Mode
```bash
# Enable verbose debug logging
python Projects/unified_pipeline/batch_all_steps.py --debug
```

### Combine Options
```bash
python Projects/unified_pipeline/batch_all_steps.py \
    --start 0 \
    --end 50 \
    --output /data/pipeline_output \
    --debug
```

---

## Output

### Directory Structure
```
Projects/unified_pipeline/export_all_steps/
├── 2024.01.17-11.15.41-Rec.h5
├── 2024.03.01-14.40.14-Rec.h5
├── 2024.08.08-10.40.20-Rec.h5
└── ...
```

### HDF5 File Contents
Each output file contains:
- **units/** - Per-unit data and features
  - `eimage_sta` - EImage STA data
  - `sta_perfect_dense_noise_*` - Dense noise STA
  - `auto_label` - Cell type labels
  - `ap_tracking` - AP pathway analysis
  - `spike_times_sectioned` - Sectioned spike times
- **metadata/** - Recording metadata
  - `cmcr_meta`, `cmtr_meta` - Source file info
  - `gsheet_row` - Google Sheet data
  - `ap_tracking` - ONH intersection results
- **pipeline/** - Processing info
  - `completed_steps` - List of completed steps

---

## Behavior

### Skip Existing Files
By default, the script **skips** recordings that already have output files:
```
[1/100] 2024.01.17-11.15.41-Rec
  Skipped - output file already exists
```

Use `--overwrite` to force reprocessing.

### Error Handling
- Failed recordings are logged and tracked
- Processing continues with remaining recordings
- Summary shows success/skip/fail counts

### Progress Tracking
```
============================================================
[5/100] 2024.03.01-14.40.14-Rec
============================================================
Processing 2024.03.01-14.40.14-Rec...
  Saved: export_all_steps/2024.03.01-14.40.14-Rec.h5
  Units: 85, Steps: 15
```

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `tool_box/.../pkl_to_cmtr_mapping.csv` | CSV mapping file path |
| `--output` | `export_all_steps/` | Output directory |
| `--start` | `0` | Starting index (0-based) |
| `--end` | `None` (all) | Ending index (exclusive) |
| `--overwrite` | `False` | Overwrite existing files |
| `--debug` | `False` | Enable debug logging |

---

## Troubleshooting

### "CSV mapping file not found"
Generate the mapping file:
```bash
python tool_box/generate_data_path_list/generate_pkl_to_cmtr_mapping.py
```

### "CMCR/CMTR file not found"
- Verify file paths in the CSV are correct
- Check if network drives are mounted
- Update CSV if files have moved

### "Google Sheet unavailable"
The script automatically falls back to local CSV:
```
Projects/load_gsheet/gsheet_table.csv
```

### Memory issues
Process in smaller batches:
```bash
python batch_all_steps.py --start 0 --end 25
python batch_all_steps.py --start 25 --end 50
python batch_all_steps.py --start 50 --end 75
# etc.
```

### Step-specific failures
Check the log for which step failed:
```bash
python batch_all_steps.py --debug 2>&1 | tee batch_log.txt
```

---

## Comparison with Two-Stage Processing

| Approach | When to Use |
|----------|-------------|
| **batch_all_steps.py** | Processing everything in one go |
| **Two-stage** (1-5 then 6-11) | When raw files may become unavailable |

The two-stage approach saves intermediate results after step 5, allowing step 6-11 to be re-run without needing CMCR/CMTR files. See `README_batch_processing_two_stages.md` for details.

---

## Related Scripts

| Script | Description |
|--------|-------------|
| `batch_all_steps.py` | Batch: All 11 steps |
| `batch_cmcr_cmtr_1_to_5.py` | Batch: Steps 1-5 only |
| `run_steps_6_to_end.py` | Single: Steps 6-11 from HDF5 |
| `run_single_from_cmcr.py` | Single: All steps |

---

## Example Workflow

```bash
# 1. Generate CSV mapping (if needed)
python tool_box/generate_data_path_list/generate_pkl_to_cmtr_mapping.py

# 2. Test with a few recordings
python Projects/unified_pipeline/batch_all_steps.py --start 0 --end 3 --debug

# 3. Run full batch
python Projects/unified_pipeline/batch_all_steps.py

# 4. Check results
ls Projects/unified_pipeline/export_all_steps/
```

