---
name: Manual Label Image Generation
overview: Add a manual label image generation option to yan_pipeline that reuses the existing validation plots (eimage_sta, geometry, ap_tracking, dsgc) to create per-unit images for manual classification, similar to the legacy code workflow.
todos:
  - id: update-config
    content: Add CREATE_MANUAL_LABEL and MANUAL_LABEL_DIR to config.py
    status: pending
  - id: update-batch
    content: Add manual label generation to batch_from_folders.py with CLI options
    status: pending
    dependencies:
      - update-config
  - id: test-run
    content: Verify the script runs with --help and shows new options
    status: pending
    dependencies:
      - update-batch
---

# Manual Label Image Generation for yan_pipeline

## Overview

Add functionality to `batch_from_folders.py` that generates per-unit visualization images for manual labeling, reusing the existing validation plot modules from `Projects/unified_pipeline/validation_plots/`.

## Changes

### 1. Update `yan_pipeline/config.py`

Add new configuration option:

```python
# Generate manual label images for each processed recording
CREATE_MANUAL_LABEL = True

# Output directory for manual label images
MANUAL_LABEL_DIR = Path(__file__).parent / "manual_label_images"
```

### 2. Update `yan_pipeline/batch_from_folders.py`

Add:

- Import validation plot runners from `Projects/unified_pipeline/validation_plots/run_all.py`
- New CLI arguments: `--create-manual-label` (default from config) and `--skip-manual-label`
- After saving each HDF5 file, call validation plot runner to generate images
- Save images to `yan_pipeline/manual_label_images/{dataset_id}/`

**Key code addition in `process_single_recording()`:**

```python
# After session.save() and before return:
if create_manual_label:
    from Projects.unified_pipeline.validation_plots.run_all import run_all_validations
    label_output_dir = manual_label_dir / dataset_id
    run_all_validations(output_path, label_output_dir)
```

### 3. Generated Output Structure

```
yan_pipeline/
├── export/                        # HDF5 output files
│   └── 2024.01.17-11.15.41-Rec.h5
└── manual_label_images/           # Manual label images
    └── 2024.01.17-11.15.41-Rec/
        ├── 01_eimage_sta/
        │   ├── unit_001_eimage_sta.png
        │   └── unit_002_eimage_sta.png
        ├── 02_geometry/
        │   ├── unit_001_geometry.png
        │   └── unit_002_geometry.png
        ├── 03_ap_tracking/
        │   ├── unit_001_ap_tracking.png
        │   └── summary_onh.png
        └── 04_dsgc/
            ├── unit_001_dsgc.png
            └── unit_002_dsgc.png
```

## CLI Usage

```bash
# Default: generate manual label images
python yan_pipeline/batch_from_folders.py

# Explicitly enable
python yan_pipeline/batch_from_folders.py --create-manual-label

# Skip image generation (faster processing)
python yan_pipeline/batch_from_folders.py --skip-manual-label
```

## Implementation Todos