# Implementation Plan: Deferred HDF5 Save Pipeline

**Branch**: `001-deferred-hdf5-save` | **Date**: 2024-12-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-deferred-hdf5-save/spec.md`

## Summary

Modify the HD-MEA pipeline to support deferred HDF5 saving, allowing users to run multiple pipeline steps in memory without intermediate disk writes. The solution introduces a `PipelineSession` class that accumulates data across pipeline steps and provides explicit `save()` and `checkpoint()` methods. Default behavior remains backwards-compatible (immediate save).

## Technical Context

**Language/Version**: Python 3.10+ (as per pyproject.toml)  
**Primary Dependencies**: h5py>=3.0.0, numpy>=1.24.0, pandas>=2.0.0  
**Storage**: HDF5 files (.h5) via h5py  
**Testing**: pytest>=7.4.0  
**Target Platform**: Windows/Linux workstations, high-memory servers  
**Project Type**: Single Python package (src/hdmea/)  
**Performance Goals**: 20% faster batch processing (10 recordings) vs intermediate saves  
**Constraints**: Support up to 50 GB in-memory representation, < 2x memory overhead  
**Scale/Scope**: Existing pipeline with ~6 main functions to modify

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | All changes in `src/hdmea/pipeline/` |
| II. Modular Subpackage Layout | ✅ PASS | New code in `pipeline/` subpackage, follows dependency flow |
| III. Explicit I/O and Pure Functions | ✅ PASS | PipelineSession makes state explicit; save/checkpoint are clearly named |
| IV. Single HDF5 Artifact Per Recording | ✅ PASS | Still produces one HDF5 per recording; just deferred |
| V. Data Format Standards | ✅ PASS | Uses HDF5 for hierarchical data as required |
| VI. No Hidden Global State | ✅ PASS | Session object holds state explicitly, not in globals |
| VII. Independence from Legacy Code | ✅ PASS | No legacy imports |

**Gate Result**: PASS - No violations requiring justification.

## Project Structure

### Documentation (this feature)

```text
specs/001-deferred-hdf5-save/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API contracts)
└── checklists/          # Quality checklists
    └── requirements.md
```

### Source Code (repository root)

```text
src/hdmea/
├── pipeline/
│   ├── __init__.py      # Update exports
│   ├── runner.py        # Modify existing functions
│   ├── session.py       # NEW: PipelineSession class
│   └── flows.py         # Existing (no changes needed)
├── io/
│   └── hdf5_store.py    # May need minor updates for batch writes
└── utils/
    └── exceptions.py    # Add new exception types if needed

tests/
├── unit/
│   └── pipeline/
│       └── test_session.py   # NEW: PipelineSession tests
├── integration/
│   └── test_deferred_save.py # NEW: End-to-end deferred save tests
└── fixtures/
    └── synthetic_session.py  # NEW: Fixtures for session testing
```

**Structure Decision**: Single project layout (Option 1). All changes are within the existing `src/hdmea/pipeline/` module with one new file (`session.py`).

## Complexity Tracking

> No Constitution violations - table not required.

---

## Phase 0: Research

See [research.md](./research.md) for detailed findings.

## Phase 1: Design

See [data-model.md](./data-model.md) for entity definitions.
See [contracts/](./contracts/) for API specifications.
See [quickstart.md](./quickstart.md) for usage examples.
