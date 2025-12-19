# Implementation Plan: STA Computation

**Branch**: `009-sta-computation` | **Date**: 2025-12-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/009-sta-computation/spec.md`

## Summary

Implement Spike Triggered Average (STA) computation for noise movies. The system will:
1. Automatically detect the noise movie by searching for "noise" in movie names
2. Convert spike times from sampling indices to movie frame numbers
3. Compute STA using vectorized numpy operations with shared memory for multiprocessing
4. Save results to HDF5 under `features/{noise_movie_name}/sta`

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: numpy, h5py, tqdm, multiprocessing (stdlib)  
**Storage**: HDF5 (existing artifact format per constitution)  
**Testing**: pytest with synthetic fixtures  
**Target Platform**: Windows (primary), Linux (compatible)  
**Project Type**: Single Python package (hdmea)  
**Performance Goals**: 100+ units in <5 minutes with multiprocessing  
**Constraints**: 80% CPU cores for parallel processing, shared memory for stimulus array  
**Scale/Scope**: Typical recordings have 100-500 units, noise movies ~10k-100k frames

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Implementation goes in `src/hdmea/features/` |
| II. Modular Subpackage Layout | ✅ PASS | STA extractor in `features/`, uses `io/` for HDF5 |
| III. Explicit I/O and Pure Functions | ✅ PASS | Pure function for STA computation, explicit params |
| IV. Single HDF5 Artifact | ✅ PASS | Saves to existing HDF5 under `features/` group |
| V. Data Format Standards | ✅ PASS | HDF5 for nested data, numpy arrays |
| VI. No Hidden Global State | ✅ PASS | All config passed as parameters |
| VII. Independence from Legacy Code | ✅ PASS | No legacy imports, clean implementation |

**All gates pass. Proceeding to Phase 0.**

## Project Structure

### Documentation (this feature)

```text
specs/009-sta-computation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── api.md
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/hdmea/
├── features/
│   ├── __init__.py
│   ├── sta.py           # NEW: STA computation module
│   └── registry.py      # Existing: feature registry
├── io/
│   └── hdf5_store.py    # Existing: HDF5 read/write
└── utils/
    └── conversion.py    # May need: spike-to-frame conversion

tests/
├── unit/
│   └── test_sta.py      # NEW: STA unit tests
└── fixtures/
    └── sta_fixtures.py  # NEW: Synthetic data generators
```

**Structure Decision**: Single package structure following existing hdmea layout. STA computation goes in `features/sta.py` as a feature extractor.

## Complexity Tracking

No violations to justify - design follows constitution patterns.
