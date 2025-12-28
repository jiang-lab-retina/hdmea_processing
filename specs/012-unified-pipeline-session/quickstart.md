# Quickstart: Unified Pipeline Session

**Branch**: `012-unified-pipeline-session` | **Date**: 2025-12-28

## Overview

This guide shows how to use the unified pipeline to process HD-MEA recordings from raw files or existing HDF5 files.

---

## Installation

```bash
# Install the hdmea package in development mode
pip install -e ".[dev]"

# Verify installation
python -c "from hdmea.pipeline import create_session; print('OK')"
```

---

## Example 1: Process from CMCR/CMTR Files

```python
"""
run_single_from_cmcr.py

Process a recording from raw CMCR/CMTR files through all 11 pipeline steps.
"""
from pathlib import Path

from hdmea.pipeline import create_session
from hdmea.pipeline import load_recording_with_eimage_sta
from hdmea.io import add_section_time, section_spike_times
from hdmea.features import compute_sta

# Import step wrappers from unified pipeline
from Projects.unified_pipeline.steps import (
    add_cmtr_cmcr_metadata,
    extract_soma_geometry,
    extract_rf_geometry,
    add_gsheet_metadata,
    add_cell_type_labels,
    compute_ap_tracking_step,
    section_by_direction_step,
)

# === Configuration ===
CMCR_PATH = Path("path/to/2024.08.08-10.40.20-Rec.cmcr")
CMTR_PATH = Path("path/to/2024.08.08-10.40.20-Rec-.cmtr")
OUTPUT_PATH = Path("Projects/unified_pipeline/output/2024.08.08-10.40.20-Rec.h5")

# === Pipeline Execution ===

# 1. Create session
session = create_session(dataset_id="2024.08.08-10.40.20-Rec")

# 2. Load recording with eimage_sta
session = load_recording_with_eimage_sta(
    cmcr_path=str(CMCR_PATH),
    cmtr_path=str(CMTR_PATH),
    duration_s=120.0,
    spike_limit=10000,
    window_range=(-10, 40),
    session=session,
)

# 3. Add section time using playlist
session = add_section_time(
    playlist_name="play_optimization_set6_ipRGC_manual",
    session=session,
)

# 4. Section spike times
session = section_spike_times(
    pad_margin=(0.0, 0.0),
    session=session,
)

# 5. Compute STA
session = compute_sta(
    cover_range=(-60, 0),
    session=session,
)

# 6. Add CMTR/CMCR metadata
session = add_cmtr_cmcr_metadata(session=session)

# 7. Extract soma geometry
session = extract_soma_geometry(
    frame_range=(10, 14),
    threshold_fraction=0.5,
    session=session,
)

# 8. Extract RF-STA geometry
session = extract_rf_geometry(
    frame_range=(10, 14),
    threshold_fraction=0.5,
    session=session,
)

# 9. Add Google Sheet metadata
session = add_gsheet_metadata(
    sheet_name="Sheet_Name",
    session=session,
)

# 10. Add cell type labels
session = add_cell_type_labels(
    label_folder=Path("path/to/manual_label_data"),
    session=session,
)

# 11. Compute AP tracking
session = compute_ap_tracking_step(
    model_path=Path("path/to/model.pth"),
    cell_type_filter="rgc",
    session=session,
)

# 12. Section by direction (DSGC)
session = section_by_direction_step(
    movie_name="moving_h_bar_s5_d8_3x",
    on_off_dict_path=Path("path/to/on_off_dict.pkl"),
    padding_frames=10,
    session=session,
)

# === Save Results ===
session.save(output_path=OUTPUT_PATH)  # overwrite=False by default

print(f"Pipeline complete!")
print(f"  Units: {session.unit_count}")
print(f"  Steps: {session.completed_steps}")
print(f"  Output: {OUTPUT_PATH}")
```

---

## Example 2: Resume from Existing HDF5

```python
"""
run_single_from_hdf5.py

Load an existing HDF5 file and run additional pipeline steps.
"""
from pathlib import Path

from hdmea.pipeline.loader import load_session_from_hdf5

from Projects.unified_pipeline.steps import (
    compute_ap_tracking_step,
    section_by_direction_step,
)

# === Configuration ===
INPUT_HDF5 = Path("Projects/pipeline_test/data/2024.08.08-10.40.20-Rec.h5")
OUTPUT_PATH = Path("Projects/unified_pipeline/output/2024.08.08-10.40.20-Rec.h5")

# === Load Existing Data ===
session = load_session_from_hdf5(
    hdf5_path=INPUT_HDF5,
    # Optional: load only specific features
    # load_features=["eimage_sta", "sta"],
)

print(f"Loaded session: {session.dataset_id}")
print(f"  Units: {session.unit_count}")
print(f"  Completed steps: {session.completed_steps}")

# === Run Additional Steps ===

# Only run steps not yet completed
if "compute_ap_tracking" not in session.completed_steps:
    session = compute_ap_tracking_step(
        model_path=Path("path/to/model.pth"),
        cell_type_filter="rgc",
        session=session,
    )

if "section_by_direction" not in session.completed_steps:
    session = section_by_direction_step(
        movie_name="moving_h_bar_s5_d8_3x",
        on_off_dict_path=Path("path/to/on_off_dict.pkl"),
        padding_frames=10,
        session=session,
    )

# === Save Results ===
session.save(output_path=OUTPUT_PATH)

print(f"Pipeline complete!")
print(f"  Output: {OUTPUT_PATH}")
```

---

## Example 3: Using Checkpoints

```python
"""
Checkpoint example for long-running pipelines.
"""
from pathlib import Path
from hdmea.pipeline import create_session

# ... run first few steps ...

# Save checkpoint after expensive steps
checkpoint_path = Path("checkpoints/step5_checkpoint.h5")
session.checkpoint(checkpoint_path)
print(f"Checkpoint saved to: {checkpoint_path}")

# Later, resume from checkpoint
from hdmea.pipeline.loader import load_session_from_hdf5

session = load_session_from_hdf5(checkpoint_path)
print(f"Resumed from checkpoint. Completed steps: {session.completed_steps}")

# Continue with remaining steps...
```

---

## Example 4: Handling External Service Failures

```python
"""
Graceful handling when Google Sheet is unavailable.
"""
# The add_gsheet_metadata step will:
# 1. Log a RED warning if gsheet is unavailable
# 2. Skip the step and continue
# 3. Mark step as "add_gsheet_metadata:skipped" in completed_steps

session = add_gsheet_metadata(
    sheet_name="Sheet_Name",
    session=session,
)

# Check if step was skipped
if "add_gsheet_metadata:skipped" in session.completed_steps:
    print("Warning: Google Sheet metadata was not loaded")
else:
    print("Google Sheet metadata loaded successfully")
```

---

## Common Patterns

### Check if Step Already Completed

```python
if "compute_sta" not in session.completed_steps:
    session = compute_sta(..., session=session)
```

### Selective Feature Loading

```python
# Load only eimage_sta and sta features (reduces memory)
session = load_session_from_hdf5(
    hdf5_path=input_path,
    load_features=["eimage_sta", "sta"],
)
```

### Save to New File (Safe Mode)

```python
# This is the default - will error if file exists
session.save(output_path=new_path)

# Explicit overwrite
session.save(output_path=existing_path, overwrite=True)
```

---

## Troubleshooting

### Memory Issues

For recordings with many units (> 200), consider:
1. Processing in batches
2. Using selective feature loading
3. Saving checkpoints after memory-intensive steps

### Step Debugging

Each step wrapper is a thin function in `Projects/unified_pipeline/steps/`. To debug a specific step:

```python
# Import the step directly
from Projects.unified_pipeline.steps.geometry import extract_soma_geometry

# Run with a minimal session
session = load_session_from_hdf5(input_path, load_features=["eimage_sta"])
session = extract_soma_geometry(session=session)
```

### Viewing Completed Steps

```python
print(f"Completed: {session.completed_steps}")
print(f"Warnings: {session.warnings}")
```

