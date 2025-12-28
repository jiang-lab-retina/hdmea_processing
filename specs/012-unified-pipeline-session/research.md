# Research: Unified Pipeline Session

**Branch**: `012-unified-pipeline-session` | **Date**: 2025-12-28

## Overview

This document consolidates research findings for implementing the unified pipeline session feature. All technical decisions are based on analysis of existing codebase patterns and the project constitution.

---

## 1. Universal HDF5 Loader Design

### Decision
Implement `load_session_from_hdf5()` function that recursively loads all HDF5 groups/datasets into the session, with optional feature filtering via `load_features` parameter.

### Rationale
- Existing `PipelineSession.load()` method has limited scope (only restores session metadata, not full data model)
- Analysis of existing batch scripts shows each module implements its own `load_hdf5_to_session()` - consolidation needed
- Universal loader reduces code duplication and ensures consistent data loading

### Alternatives Considered
| Alternative | Why Rejected |
|-------------|--------------|
| Lazy loading | Adds complexity; recordings are small enough (< 8GB) for eager loading |
| Separate loaders per data type | Already exists and causes duplication; universal loader consolidates |
| Extend existing `PipelineSession.load()` | Better to create new function to avoid breaking existing behavior |

### Implementation Details
```python
def load_session_from_hdf5(
    hdf5_path: Path,
    *,
    dataset_id: Optional[str] = None,
    load_features: Optional[List[str]] = None,  # None = load all
) -> PipelineSession:
    """Universal HDF5 loader with optional feature filtering."""
```

---

## 2. Pipeline Step Pattern

### Decision
Each pipeline step is a thin wrapper function in `Projects/unified_pipeline/steps/` that:
1. Imports and calls the existing implementation from `hdmea` or project modules
2. Follows the `session=session` pattern
3. Returns the updated session

### Rationale
- Existing implementations are scattered across `hdmea.io`, `hdmea.features`, and various `Projects/` folders
- Wrapper pattern allows debugging individual steps without modifying core code
- Maintains backward compatibility with existing batch scripts

### Step Mapping

| Step | Wrapper | Existing Implementation |
|------|---------|------------------------|
| 1. Load recording | `steps/load_recording.py` | `hdmea.pipeline.load_recording_with_eimage_sta` |
| 2. Add section time | `steps/section_time.py` | `hdmea.io.add_section_time` |
| 3. Section spike times | `steps/section_time.py` | `hdmea.io.section_spike_times` |
| 4. Compute STA | `steps/section_time.py` | `hdmea.features.compute_sta` |
| 5. Add CMTR/CMCR meta | `steps/metadata.py` | `hdmea.io.cmtr.add_cmtr_unit_info`, `hdmea.io.cmcr.add_sys_meta_info` |
| 6. Extract soma geometry | `steps/geometry.py` | `Projects/sta_quantification/ap_sta.extract_eimage_sta_geometry` |
| 7. Extract RF geometry | `steps/geometry.py` | `Projects/rf_sta_measure/rf_session.*` |
| 8. Load gsheet | `steps/gsheet.py` | `Projects/load_gsheet/load_gsheet.*` |
| 9. Add cell type | `steps/cell_type.py` | `Projects/add_manual_label_cell_type/*` |
| 10. AP tracking | `steps/ap_tracking.py` | `hdmea.features.ap_tracking.compute_ap_tracking` |
| 11. DSGC section | `steps/dsgc.py` | `hdmea.features.section_by_direction` |

---

## 3. Progress Reporting

### Decision
- Use Python `logging` module for step-level progress (INFO level)
- Use `tqdm` for unit iteration progress bars
- Warnings displayed in red using ANSI color codes or colorama

### Rationale
- Consistent with existing batch processing scripts
- `tqdm` is already a dependency (used in AP tracking)
- Logger allows filtering/redirecting output

### Implementation
```python
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def step_with_progress(session: PipelineSession) -> PipelineSession:
    logger.info("Starting step X...")
    for unit_id in tqdm(session.units, desc="Processing units"):
        # process unit
    logger.info("Step X complete")
    return session
```

For red warnings:
```python
# Using colorama for cross-platform color support
from colorama import Fore, Style
logger.warning(f"{Fore.RED}Google Sheet unavailable, skipping...{Style.RESET_ALL}")
```

---

## 4. Overwrite Protection

### Decision
- Default `overwrite=False` on `session.save()` 
- Raise `FileExistsError` if file exists and `overwrite=False`
- Log warning when overwriting with `overwrite=True`

### Rationale
- Prevents accidental data loss during development/testing
- Existing `PipelineSession.save()` already has `overwrite` parameter - just change default
- Matches spec requirement FR-009

### Current vs. New Behavior
| | Current | New |
|--|---------|-----|
| `save(path)` | Overwrites silently | Raises `FileExistsError` if exists |
| `save(path, overwrite=True)` | Overwrites with warning | Same (no change) |
| `save(path, overwrite=False)` | Raises error | Same (no change) |

---

## 5. External Dependency Failure Handling

### Decision
When Google Sheet is unavailable:
1. Log a RED warning message
2. Skip the gsheet loading step
3. Continue pipeline execution
4. Mark step as "skipped" in `session.completed_steps`

### Rationale
- Pipeline should complete even if optional external services fail
- Matches existing `batch_load_gsheet.py` pattern of skipping files without matches
- Red warning ensures visibility without stopping execution

### Implementation Pattern
```python
def add_gsheet_metadata(session: PipelineSession, ...) -> PipelineSession:
    try:
        gsheet_df = import_gsheet_v2(...)
    except Exception as e:
        logger.warning(f"{Fore.RED}Google Sheet unavailable: {e}. Skipping...{Style.RESET_ALL}")
        session.completed_steps.add("add_gsheet_metadata:skipped")
        return session
    
    # Normal processing...
    session.completed_steps.add("add_gsheet_metadata")
    return session
```

---

## 6. Checkpoint and Recovery

### Decision
- `session.checkpoint(path)` saves current state without changing session mode
- Recovery via `load_session_from_hdf5(checkpoint_path)`
- Checkpoints include `completed_steps` for tracking progress

### Rationale
- Enables recovery from long-running pipeline interruptions
- Existing `PipelineSession.checkpoint()` already implemented
- Universal loader makes checkpoint restoration trivial

---

## 7. Testing Strategy

### Decision
- Use `2024.08.08-10.40.20-Rec` as the primary test recording
- Compare output HDF5 structure against reference file at `Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5`
- Do NOT overwrite existing files during tests

### Verification Approach
```python
def verify_output_matches_reference(output_path: Path, reference_path: Path) -> bool:
    """Compare HDF5 group/dataset structure (not byte-for-byte)."""
    with h5py.File(output_path, 'r') as out, h5py.File(reference_path, 'r') as ref:
        return _compare_hdf5_structure(out, ref)
```

### Test File Locations
- **Input CMCR**: Derived from mapping CSV
- **Input CMTR**: Derived from mapping CSV  
- **Reference Output**: `Projects/dsgc_section/export_dsgc_section_20251226/2024.08.08-10.40.20-Rec.h5`
- **Test Output**: `Projects/unified_pipeline/test_output/2024.08.08-10.40.20-Rec.h5` (new file, not overwrite)

---

## Summary

All NEEDS CLARIFICATION items from the spec have been resolved:

| Clarification | Resolution |
|---------------|------------|
| Interrupted processing | Manual checkpoints via `session.checkpoint()` |
| Universal loader selective loading | Default load all, optional `load_features` filter |
| Testing mode / overwrite protection | `overwrite=False` default on save |
| Progress reporting | Logger + tqdm |
| External dependency failure | Skip with red warning, continue pipeline |

