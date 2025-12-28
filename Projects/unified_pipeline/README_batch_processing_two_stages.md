# Unified Pipeline: Batch Processing Guide

This guide explains how to batch process CMCR/CMTR recordings through the unified pipeline.

## Overview

The pipeline is split into two stages to optimize processing:

| Stage | Steps | Requires | Output |
|-------|-------|----------|--------|
| **Stage 1** | Steps 1-5 | CMCR/CMTR files | Intermediate HDF5 |
| **Stage 2** | Steps 6-11 | HDF5 from Stage 1 | Final HDF5 |

This separation allows you to:
- Process raw files once and store intermediate results
- Re-run analysis steps (6-11) without re-reading large raw files
- Distribute processing across machines (Stage 1 needs raw data access, Stage 2 doesn't)

---

## Stage 1: Batch Process CMCR/CMTR (Steps 1-5)

### Script
```
Projects/unified_pipeline/batch_cmcr_cmtr_1_to_5.py
```

### What it does
- **Step 1**: Load recording with eimage_sta
- **Step 2**: Add section time from playlist
- **Step 3**: Section spike times
- **Step 4**: Compute STA (dense noise)
- **Step 5**: Add CMTR/CMCR metadata

### Prerequisites

1. **CSV Mapping File** - Contains paths to matched CMCR/CMTR pairs:
   ```
   tool_box/generate_data_path_list/pkl_to_cmtr_mapping.csv
   ```
   
   Required columns:
   - `cmcr_path`: Full path to CMCR file
   - `cmtr_path`: Full path to CMTR file
   - `matched`: "True" for valid pairs

2. **Access to raw data** - The CMCR/CMTR files must be accessible

### Usage

#### Process all matched recordings
```bash
python Projects/unified_pipeline/batch_cmcr_cmtr_1_to_5.py
```

#### Process a specific range (e.g., first 10)
```bash
python Projects/unified_pipeline/batch_cmcr_cmtr_1_to_5.py --start 0 --end 10
```

#### Force overwrite existing files
```bash
python Projects/unified_pipeline/batch_cmcr_cmtr_1_to_5.py --overwrite
```

#### Use custom paths
```bash
python Projects/unified_pipeline/batch_cmcr_cmtr_1_to_5.py \
    --csv path/to/custom_mapping.csv \
    --output path/to/custom_output
```

#### Enable debug logging
```bash
python Projects/unified_pipeline/batch_cmcr_cmtr_1_to_5.py --debug
```

### Output

Files are saved to:
```
Projects/unified_pipeline/export_cmcr_cmtr_1_to_5/
├── 2024.01.17-11.15.41-Rec.h5
├── 2024.03.01-14.40.14-Rec.h5
├── 2024.08.08-10.40.20-Rec.h5
└── ...
```

### Skipping Logic

- By default, existing output files are **skipped**
- Use `--overwrite` to force reprocessing
- Skipped files are counted separately in the summary

---

## Stage 2: Complete Processing (Steps 6-11)

### Script
```
Projects/unified_pipeline/run_steps_6_to_end.py
```

### What it does
- **Step 6**: Extract soma geometry
- **Step 7**: Extract RF geometry
- **Step 8**: Add Google Sheet metadata
- **Step 9**: Add manual cell type labels
- **Step 10**: Compute AP tracking (CNN model)
- **Step 11**: Section by direction (DSGC)

### Prerequisites

1. **HDF5 files from Stage 1** - Must have steps 1-5 complete
2. **External resources**:
   - Google Sheet credentials (or local CSV cache)
   - Manual label files (`.pkl`)
   - CNN model file (`.pth`)
   - ON/OFF timing dictionary (`.pkl`)

### Usage

#### Process single file
```bash
python Projects/unified_pipeline/run_steps_6_to_end.py \
    --input export_cmcr_cmtr_1_to_5/2024.08.08-10.40.20-Rec.h5
```

#### Specify output path
```bash
python Projects/unified_pipeline/run_steps_6_to_end.py \
    --input input.h5 \
    --output final_output.h5
```

### Batch Processing Stage 2

For batch processing of Stage 2, you can use a simple shell loop:

**PowerShell:**
```powershell
Get-ChildItem "Projects/unified_pipeline/export_cmcr_cmtr_1_to_5/*.h5" | ForEach-Object {
    python Projects/unified_pipeline/run_steps_6_to_end.py --input $_.FullName
}
```

**Bash:**
```bash
for f in Projects/unified_pipeline/export_cmcr_cmtr_1_to_5/*.h5; do
    python Projects/unified_pipeline/run_steps_6_to_end.py --input "$f"
done
```

---

## Single File Processing

For processing individual recordings:

### From CMCR/CMTR (all steps)
```bash
python Projects/unified_pipeline/run_single_from_cmcr.py \
    --cmcr path/to/file.cmcr \
    --cmtr path/to/file.cmtr \
    --dataset 2024.08.08-10.40.20-Rec
```

### From existing HDF5 (steps 8-11)
```bash
python Projects/unified_pipeline/run_steps_8_to_end.py \
    --input path/to/steps1-7.h5
```

---

## Pipeline Steps Reference

| Step | Name | Description | Requires |
|------|------|-------------|----------|
| 1 | `load_recording_step` | Load CMCR/CMTR with eimage_sta | CMCR, CMTR files |
| 2 | `add_section_time_step` | Add timing from playlist | Playlist JSON |
| 3 | `section_spike_times_step` | Section spikes by stimulus | Session data |
| 4 | `compute_sta_step` | Compute dense noise STA | Session data |
| 5 | `add_metadata_step` | Add file metadata | CMCR, CMTR files |
| 6 | `extract_soma_geometry_step` | Extract soma geometry | Session data |
| 7 | `extract_rf_geometry_step` | Extract RF geometry | Session data |
| 8 | `add_gsheet_step` | Add Google Sheet data | API or CSV |
| 9 | `add_cell_type_step` | Add manual labels | Label PKL files |
| 10 | `compute_ap_tracking_step` | AP pathway tracking | CNN model |
| 11 | `section_by_direction_step` | DSGC analysis | ON/OFF dict |

---

## Troubleshooting

### "CSV mapping file not found"
Ensure the CSV file exists at the expected path. Create it using:
```bash
python tool_box/generate_data_path_list/generate_pkl_to_cmtr_mapping.py
```

### "CMCR/CMTR file not found"
The paths in the CSV may be outdated. Verify the files exist and update the CSV if needed.

### "Google Sheet unavailable"
The script will automatically fall back to the local CSV cache:
```
Projects/load_gsheet/gsheet_table.csv
```

### Memory issues
For large batches, process in chunks using `--start` and `--end`:
```bash
python batch_cmcr_cmtr_1_to_5.py --start 0 --end 50
python batch_cmcr_cmtr_1_to_5.py --start 50 --end 100
# etc.
```

---

## File Structure

```
Projects/unified_pipeline/
├── batch_cmcr_cmtr_1_to_5.py      # Batch: Steps 1-5
├── run_steps_6_to_end.py          # Single: Steps 6-11
├── run_steps_8_to_end.py          # Single: Steps 8-11
├── run_single_from_cmcr.py        # Single: All steps
├── run_single_from_hdf5.py        # Single: From HDF5
├── config.py                       # Configuration
├── steps/                          # Step implementations
│   ├── load_recording.py
│   ├── section_time.py
│   ├── metadata.py
│   ├── geometry.py
│   ├── gsheet.py
│   ├── cell_type.py
│   ├── ap_tracking.py
│   └── dsgc.py
├── export_cmcr_cmtr_1_to_5/       # Stage 1 output
├── test_output/                    # Test outputs
└── validation_plots/               # Validation plots
```

