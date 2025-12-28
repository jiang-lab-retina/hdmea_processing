# Unified Pipeline - Implementation Complete

## Original Requirements ✅

The goal was to achieve a unified pipeline processing from CMCR/CMTR through every step using `PipelineSession` class to fulfill the following requirements:

| # | Requirement | Status |
|---|------------|--------|
| 1 | Pipeline can start from CMCR/CMTR or HDF5 | ✅ Implemented |
| 2 | Universal HDF5 loader loads everything to session | ✅ `load_session_from_hdf5()` |
| 3 | Concise, chainable pipeline code | ✅ See examples below |
| 4 | Two example files (CMCR/CMTR and HDF5) | ✅ Created |
| 5 | Deferred saving for speed, with optional early saves | ✅ Implemented |
| 6 | Easy to add new feature extraction steps | ✅ Template provided |
| 7 | Call functions from Projects subfolders | ✅ Step wrappers |
| 8 | Export as new file or overwrite | ✅ `overwrite` parameter |
| 9 | Testing doesn't overwrite files | ✅ `overwrite=False` default |
| 10 | Follow batch processing sequence | ✅ 11 steps defined |
| 11 | Test with `2024.08.08-10.40.20-Rec` | ✅ Config set up |
| 12 | Create git branch | ✅ `012-unified-pipeline-session` |

## Pipeline Flow (11 Steps)

```
CMCR/CMTR Files                           Existing HDF5
      │                                         │
      ▼                                         ▼
┌─────────────────┐                   ┌─────────────────┐
│  Step 1: Load   │                   │ Universal Load  │
│  Recording      │                   │ from HDF5       │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         ▼                                     │
┌─────────────────┐                            │
│  Step 2: Add    │                            │
│  Section Time   │                            │
└────────┬────────┘                            │
         │                                     │
         ▼                                     │
┌─────────────────┐                            │
│  Step 3: Section│                            │
│  Spike Times    │                            │
└────────┬────────┘                            │
         │                                     │
         ▼                                     │
┌─────────────────┐                            │
│  Step 4: Compute│                            │
│  STA            │◄───────────────────────────┘
└────────┬────────┘           (skip completed steps)
         │
         ▼ (Save intermediate - needed for steps 5-11)
         │
┌─────────────────┐
│  Step 5: Add    │
│  CMTR/CMCR Meta │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 6: Soma   │
│  Geometry       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 7: RF-STA │
│  Geometry       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 8: Google │
│  Sheet Metadata │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 9: Cell   │
│  Type Labels    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 10: AP    │
│  Tracking       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Step 11: DSGC  │
│  Section        │
└────────┬────────┘
         │
         ▼
    Final HDF5
```

## Concise Pipeline Code

```python
from hdmea.pipeline import create_session
from Projects.unified_pipeline.steps import *

# Create session
session = create_session(dataset_id="2024.08.08-10.40.20-Rec")

# Steps 1-4: In-memory processing
session = load_recording_step(cmcr_path, cmtr_path, session=session)
session = add_section_time_step(playlist_name="...", session=session)
session = section_spike_times_step(session=session)
session = compute_sta_step(session=session)

# Save intermediate (needed for HDF5-based steps)
session.save(output_path, overwrite=True)

# Steps 5-11: HDF5-based processing
session = add_metadata_step(session=session)
session = extract_soma_geometry_step(session=session)
session = extract_rf_geometry_step(session=session)
session = add_gsheet_step(session=session)
session = add_cell_type_step(session=session)
session = compute_ap_tracking_step(session=session)
session = section_by_direction_step(session=session)

# Done!
```

## Resume from HDF5

```python
from hdmea.pipeline import load_session_from_hdf5

# Load existing HDF5
session = load_session_from_hdf5("intermediate.h5")

# Completed steps are automatically skipped
session = add_gsheet_step(session=session)  # Runs if not done
session = compute_ap_tracking_step(session=session)  # Runs if not done
```

## Key Files

| File | Purpose |
|------|---------|
| `run_single_from_cmcr.py` | Example: full pipeline from raw files |
| `run_single_from_hdf5.py` | Example: resume from HDF5 |
| `steps/*.py` | Step wrapper functions |
| `steps/template.py` | Template for adding new steps |
| `config.py` | Configuration constants |
| `README.md` | Full documentation |

## Adding New Steps

1. Copy `steps/template.py` to `steps/my_step.py`
2. Update `STEP_NAME` and function
3. Add to `steps/__init__.py`
4. Use in example scripts

See `steps/template.py` for detailed instructions.

## Testing

```bash
# Run integration tests
pytest tests/integration/test_unified_pipeline.py -v

# Test full pipeline
python run_single_from_cmcr.py --dataset 2024.08.08-10.40.20-Rec

# Compare output to reference
# Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5
```
