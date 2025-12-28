# Implementation Plan: Unified Pipeline Session

**Branch**: `012-unified-pipeline-session` | **Date**: 2025-12-28 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/012-unified-pipeline-session/spec.md`

## Summary

Create a unified pipeline processing system using the `PipelineSession` class that enables:
1. Starting pipelines from CMCR/CMTR files or existing HDF5 files
2. Universal HDF5 loader with optional selective feature loading
3. Concise, chainable pipeline code pattern
4. Processing 11 sequential steps from raw recording to final DSGC sectioning
5. Deferred saving for performance with explicit checkpoint/save operations

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: h5py, numpy, pandas, tqdm (progress bars), existing hdmea package  
**Storage**: HDF5 files (.h5) for all artifacts  
**Testing**: pytest with synthetic test data and reference file comparison  
**Target Platform**: Windows 10+ (local workstation processing)  
**Project Type**: Single package extension (src/hdmea/)  
**Performance Goals**: Process single recording through 11 steps in < 10 minutes  
**Constraints**: Memory < 8GB per recording, no concurrent file access required  
**Scale/Scope**: ~85-150 units per recording, 11 processing steps

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | All pipeline code in `src/hdmea/pipeline/` |
| II. Modular Subpackage Layout | ✅ PASS | Uses existing subpackage structure |
| III. Explicit I/O and Pure Functions | ✅ PASS | Session parameter makes data flow explicit |
| IV. Single HDF5 Artifact Per Recording | ✅ PASS | Deferred save produces single HDF5 |
| IV.B. Deferred Save Mode | ✅ PASS | Core feature being implemented |
| V. Data Format Standards | ✅ PASS | HDF5 for recordings, proper metadata |
| VI. No Hidden Global State | ✅ PASS | All state in PipelineSession object |
| VII. Independence from Legacy Code | ✅ PASS | No legacy imports |
| Notebook & Jupytext Rules | N/A | Feature is library code, not notebooks |
| Artifact Versioning | ✅ PASS | Session tracks created_at, code_version |

**Gate Result**: ✅ PASSED - No violations

## Project Structure

### Documentation (this feature)

```text
specs/012-unified-pipeline-session/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0: Research findings
├── data-model.md        # Phase 1: Data model
├── quickstart.md        # Phase 1: Quick start guide
├── contracts/           # Phase 1: API contracts
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
src/hdmea/
├── pipeline/
│   ├── __init__.py           # Exports: PipelineSession, create_session
│   ├── session.py            # Existing PipelineSession class (enhance)
│   ├── loader.py             # NEW: Universal HDF5 loader
│   └── steps.py              # NEW: Pipeline step registry/orchestration
├── io/
│   ├── cmtr.py               # Existing: add_cmtr_unit_info
│   └── cmcr.py               # Existing: add_sys_meta_info
├── features/
│   ├── sta_geometry.py       # Existing: extract_eimage_sta_geometry
│   ├── rf_geometry.py        # Existing: extract_rf_geometry_session
│   ├── ap_tracking.py        # Existing: compute_ap_tracking
│   └── dsgc_section.py       # Existing: section_by_direction

Projects/unified_pipeline/              # NEW: Examples and utilities
├── __init__.py
├── run_single_from_cmcr.py            # Example: Start from CMCR/CMTR
├── run_single_from_hdf5.py            # Example: Start from HDF5
├── steps/                             # Pipeline step wrappers
│   ├── __init__.py
│   ├── load_recording.py
│   ├── section_time.py
│   ├── metadata.py
│   ├── geometry.py
│   ├── gsheet.py
│   ├── cell_type.py
│   ├── ap_tracking.py
│   └── dsgc.py
└── config.py                          # Configuration constants

tests/
├── unit/
│   └── test_pipeline_loader.py        # Test universal loader
├── integration/
│   └── test_unified_pipeline.py       # End-to-end test
└── fixtures/
    └── reference_hdf5/                # Reference output for comparison
```

**Structure Decision**: Single package extension following existing `src/hdmea/` structure with example scripts in `Projects/unified_pipeline/`. This maintains the constitution's package-first architecture while providing runnable examples.

## Architecture Overview

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Entry Points                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  From CMCR/CMTR:                    From HDF5:                              │
│  session = create_session()         session = load_session_from_hdf5()      │
│  session = load_recording(...)      (universal loader with filters)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Processing Steps (11 total)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. load_recording_with_eimage_sta    │  7. extract_rf_geometry             │
│  2. add_section_time                  │  8. add_gsheet_metadata             │
│  3. section_spike_times               │  9. add_cell_type_labels            │
│  4. compute_sta                       │ 10. compute_ap_tracking              │
│  5. add_cmtr_unit_info                │ 11. section_by_direction            │
│  6. extract_eimage_sta_geometry       │                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Output Options                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  session.save(output_path)           # Full save, overwrite=False default   │
│  session.save(output_path,           # Explicit overwrite                   │
│               overwrite=True)                                                │
│  session.checkpoint(path)            # Intermediate checkpoint              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `PipelineSession` | `src/hdmea/pipeline/session.py` | In-memory data container, save/checkpoint |
| `load_session_from_hdf5` | `src/hdmea/pipeline/loader.py` | Universal HDF5 → session loader |
| Step wrappers | `Projects/unified_pipeline/steps/` | Thin wrappers calling existing functions |
| Example scripts | `Projects/unified_pipeline/` | Runnable examples for both entry points |

## Complexity Tracking

> No Constitution violations to justify.

| Item | Notes |
|------|-------|
| N/A | All complexity is necessary for the 11-step pipeline |

