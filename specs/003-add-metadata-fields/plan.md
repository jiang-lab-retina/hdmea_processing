# Implementation Plan: Add frame_time and acquisition_rate to Metadata

**Branch**: `003-add-metadata-fields` | **Date**: 2024-12-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-add-metadata-fields/spec.md`

## Summary

Add `acquisition_rate` and `frame_time` fields to the Zarr metadata group during Stage 1 (data loading). The `acquisition_rate` is extracted using a priority chain (CMCR → CMTR → default 20kHz), and `frame_time` is computed as `1/acquisition_rate`. This enables downstream analyses to access timing parameters without recalculation.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: zarr, numpy, McsPy (for CMCR/CMTR reading)  
**Storage**: Zarr archives (metadata stored as group attributes)  
**Testing**: pytest with synthetic fixtures  
**Target Platform**: Cross-platform (Windows/Linux/macOS)  
**Project Type**: Single Python package (src/hdmea/)  
**Performance Goals**: N/A (metadata extraction is I/O-bound, not compute-bound)  
**Constraints**: Must work with existing pipeline Stage 1 flow; no breaking changes  
**Scale/Scope**: Applies to all recordings processed through the pipeline

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Changes in `src/hdmea/io/` and `src/hdmea/pipeline/` |
| II. Modular Subpackage Layout | ✅ PASS | io/ handles extraction, pipeline/ orchestrates |
| III. Explicit I/O and Pure Functions | ✅ PASS | Extraction functions take explicit paths, return dicts |
| IV. Single Zarr Artifact Per Recording | ✅ PASS | Metadata stored in existing Zarr structure |
| V. Data Format Standards | ✅ PASS | Metadata as Zarr attributes (float values) |
| VI. No Hidden Global State | ✅ PASS | No global state involved |
| VII. Independence from Legacy Code | ✅ PASS | No legacy imports |

**Gate Result**: ✅ All gates pass. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/003-add-metadata-fields/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── metadata_api.py  # API contract for metadata fields
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/hdmea/
├── io/
│   ├── cmcr.py          # CMCR loading - already extracts acquisition_rate
│   ├── cmtr.py          # CMTR loading - add acquisition_rate extraction
│   └── zarr_store.py    # write_metadata() - already handles metadata storage
├── pipeline/
│   └── runner.py        # load_recording() - orchestrates extraction and storage
└── utils/
    └── exceptions.py    # Existing exception types

tests/
├── unit/
│   └── test_metadata_fields.py  # New: unit tests for metadata extraction
└── fixtures/
    └── synthetic_zarr.py        # Existing: may need updates for test fixtures
```

**Structure Decision**: Single project structure. Changes are localized to `io/` (extraction) and `pipeline/` (orchestration). No new subpackages required.

## Complexity Tracking

> No constitution violations requiring justification.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | N/A | N/A |

