# Unified Pipeline for HD-MEA Data Processing

This module provides a unified pipeline for processing HD-MEA recordings through all analysis steps.

## Quick Start

### Process from CMCR/CMTR

```python
python run_single_from_cmcr.py --cmcr path/to/file.cmcr --cmtr path/to/file.cmtr
```

### Resume from existing HDF5

```python
python run_single_from_hdf5.py path/to/existing.h5
```

## Pipeline Steps

The pipeline consists of 11 steps:

| Step | Name | Description |
|------|------|-------------|
| 1 | load_recording | Load CMCR/CMTR with eimage_sta |
| 2 | add_section_time | Add section timing from playlist |
| 3 | section_spike_times | Section spike times |
| 4 | compute_sta | Compute Spike-Triggered Average |
| 5 | add_metadata | Add CMTR/CMCR metadata |
| 6 | extract_soma_geometry | Extract soma geometry |
| 7 | extract_rf_geometry | Extract RF-STA geometry |
| 8 | add_gsheet | Load Google Sheet metadata |
| 9 | add_cell_type | Add manual cell type labels |
| 10 | compute_ap_tracking | Compute AP tracking |
| 11 | section_by_direction | Section by direction (DSGC) |

## Architecture

```
Projects/unified_pipeline/
├── __init__.py           # Package docstring
├── config.py             # Configuration constants
├── run_single_from_cmcr.py  # Example: process from raw files
├── run_single_from_hdf5.py  # Example: resume from HDF5
├── steps/                # Step wrapper functions
│   ├── __init__.py       # Step exports
│   ├── load_recording.py # Step 1
│   ├── section_time.py   # Steps 2-4
│   ├── metadata.py       # Step 5
│   ├── geometry.py       # Steps 6-7
│   ├── gsheet.py         # Step 8
│   ├── cell_type.py      # Step 9
│   ├── ap_tracking.py    # Step 10
│   ├── dsgc.py           # Step 11
│   └── template.py       # Template for new steps
└── test_output/          # Test output directory
```

## Adding New Steps

1. Copy `steps/template.py` to a new file
2. Update `STEP_NAME` constant
3. Implement step logic
4. Add to `steps/__init__.py`:
   ```python
   from .my_step import my_step_function
   __all__.append("my_step_function")
   ```
5. Add to example scripts as needed

### Step Pattern

```python
def my_step(
    *,
    param: type,
    session: PipelineSession,
) -> PipelineSession:
    """Step description."""
    
    # Skip if already done
    if STEP_NAME in session.completed_steps:
        logger.info(f"Skipping {STEP_NAME}")
        return session
    
    logger.info("Running my step...")
    
    try:
        # Call existing implementation
        # Store results in session
        session.mark_step_complete(STEP_NAME)
    except Exception as e:
        logger.warning(red_warning(f"Step failed: {e}"))
        session.completed_steps.add(f"{STEP_NAME}:failed")
    
    return session
```

## Key Features

### Deferred Saving

Data is processed in memory and only saved when you call `session.save()`:

```python
session = create_session(dataset_id="...")
session = step1(session=session)
session = step2(session=session)
session = step3(session=session)
session.save(output_path)  # One write for all steps
```

### Checkpointing

Save intermediate state without ending the session:

```python
session = step1(session=session)
session.checkpoint("checkpoint1.h5")  # Save progress
session = step2(session=session)
session.checkpoint("checkpoint2.h5")  # Save more progress
session = step3(session=session)
session.save(output_path)  # Final save
```

### Resume from Checkpoint

```python
session = load_session_from_hdf5("checkpoint2.h5")
# Completed steps are restored - step3 will run, step1/step2 will skip
session = step1(session=session)  # Skipped
session = step2(session=session)  # Skipped
session = step3(session=session)  # Runs
session.save(output_path)
```

### Safe Overwrite Protection

```python
# Default: raises error if file exists
session.save(output_path)  # FileExistsError if file exists

# Explicit overwrite required
session.save(output_path, overwrite=True)
```

## Configuration

Edit `config.py` to change:

- File paths (CSV mapping, manual labels, etc.)
- Default parameters for each step
- Output directories

## Testing

Run integration tests:

```bash
pytest tests/integration/test_unified_pipeline.py -v
```

Test with reference file:

```python
python run_single_from_cmcr.py --dataset 2024.08.08-10.40.20-Rec
# Compare output to Projects/dsgc_section/export_dsgc_section_20251226/
```

## Debugging Individual Steps

Import and call steps directly:

```python
from hdmea.pipeline import load_session_from_hdf5
from Projects.unified_pipeline.steps import compute_ap_tracking_step

session = load_session_from_hdf5("test.h5")
session = compute_ap_tracking_step(session=session)
```

