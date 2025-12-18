# Implementation Plan: Replace Zarr Format with HDF5

**Branch**: `007-hdf5-replace-zarr` | **Date**: 2025-12-17 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/007-hdf5-replace-zarr/spec.md`

## Summary

Replace Zarr storage format with HDF5 throughout the HD-MEA pipeline. This involves creating a new `hdf5_store.py` module with equivalent functions to `zarr_store.py`, updating all consumers (pipeline runner, feature extractors, visualization), and modifying file extensions from `.zarr` to `.h5`. The change provides better cross-platform tooling compatibility (HDFView, MATLAB) while maintaining the same logical data hierarchy.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: h5py (HDF5 interface), numpy, existing hdmea package  
**Storage**: HDF5 single-file archives (`.h5`)  
**Testing**: pytest with synthetic test fixtures  
**Target Platform**: Windows (primary), Linux (CI)  
**Project Type**: Single Python package  
**Performance Goals**: I/O performance within 20% of current Zarr implementation  
**Constraints**: No compression (fastest I/O), single-writer access model  
**Scale/Scope**: ~1000 units per recording, ~100MB typical file size

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | New module in `src/hdmea/io/hdf5_store.py` |
| II. Modular Subpackage Layout | ✅ PASS | Follows existing `io/` module pattern |
| III. Explicit I/O and Pure Functions | ✅ PASS | Same function signatures as zarr_store |
| **IV. Single Zarr Artifact Per Recording** | ⚠️ VIOLATION | **Zarr replaced with HDF5** - see justification below |
| **V. Data Format Standards** | ⚠️ VIOLATION | **HDF5 replaces Zarr for hierarchical data** - see justification below |
| VI. No Hidden Global State | ✅ PASS | No globals introduced |
| VII. Independence from Legacy Code | ✅ PASS | No legacy imports |

### Constitution Violation Justification

**Principles IV & V explicitly specify Zarr** for hierarchical recording data. This feature intentionally replaces Zarr with HDF5 for the following reasons:

1. **User Request**: This change was explicitly requested by the project owner
2. **Tooling Compatibility**: HDF5 has broader tool support (HDFView, MATLAB, many languages)
3. **Single-File Simplicity**: `.h5` is a single file vs `.zarr` directory (easier to manage, share, backup)
4. **Maintained Semantics**: The logical data structure (groups, datasets, attributes) is preserved
5. **Constitution Amendment**: After this feature is complete, the constitution should be amended to reflect HDF5 as the standard

**Recommendation**: Update constitution Sections IV and V post-implementation to replace "Zarr" with "HDF5".

## Project Structure

### Documentation (this feature)

```text
specs/007-hdf5-replace-zarr/
├── plan.md              # This file
├── research.md          # h5py best practices and patterns
├── data-model.md        # HDF5 structure documentation
├── quickstart.md        # Usage examples
├── contracts/           # API contracts
│   ├── api.md           # Function signatures
│   └── README.md        # Contracts overview
└── tasks.md             # Implementation tasks (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/hdmea/
├── io/
│   ├── __init__.py          # Updated exports
│   ├── hdf5_store.py        # NEW: HDF5 operations (replaces zarr_store.py)
│   ├── zarr_store.py        # DEPRECATED: Keep for reference, mark deprecated
│   ├── section_time.py      # UPDATE: Use hdf5_store instead of zarr_store
│   └── spike_sectioning.py  # UPDATE: Use hdf5_store instead of zarr_store
├── pipeline/
│   ├── runner.py            # UPDATE: Use hdf5_store for artifact creation
│   └── flows.py             # UPDATE: HDF5 file paths
├── features/
│   └── *.py                 # UPDATE: All feature extractors use hdf5_store
└── viz/
    └── zarr_viz/            # RENAME/UPDATE: Support HDF5 files
        ├── app.py           # UPDATE: Load HDF5 instead of Zarr
        ├── tree.py          # UPDATE: Navigate HDF5 groups
        └── utils.py         # UPDATE: HDF5 utilities

tests/
├── unit/
│   └── test_hdf5_store.py   # NEW: Unit tests for HDF5 operations
└── integration/
    └── test_pipeline_hdf5.py # NEW: End-to-end with HDF5
```

**Structure Decision**: Follows existing single-package structure under `src/hdmea/`. New module `hdf5_store.py` parallels existing `zarr_store.py` for clean migration.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Constitution IV & V (Zarr → HDF5) | User explicitly requested format change for better tooling compatibility | Keeping Zarr doesn't meet user requirements; HDF5 provides same capabilities with better external tool support |

## Implementation Approach

### Phase 1: Core HDF5 Module

1. Create `hdf5_store.py` with all functions from `zarr_store.py`
2. Map Zarr concepts to HDF5:
   - `zarr.Group` → `h5py.Group`
   - `zarr.Dataset` → `h5py.Dataset`
   - `.zattrs` → HDF5 attributes (`.attrs`)
3. Implement single-writer locking check

### Phase 2: Pipeline Integration

1. Update `runner.py` to use `hdf5_store` instead of `zarr_store`
2. Update `section_time.py` and `spike_sectioning.py`
3. Update all feature extractors

### Phase 3: Visualization Update

1. Update `zarr_viz` module to read HDF5 files
2. Consider renaming module to `hdf5_viz` or generic `data_viz`

### Phase 4: Testing & Cleanup

1. Create comprehensive tests
2. Mark `zarr_store.py` as deprecated
3. Update documentation

## Dependencies

- `h5py>=3.0.0` - HDF5 file operations
- Existing: `numpy`, `logging`

## Risks

| Risk | Mitigation |
|------|------------|
| h5py API differences from zarr | Research phase documents mapping; comprehensive tests |
| Concurrent access issues | Enforce single-writer model with explicit error messages |
| Existing Zarr files unusable | Document in release notes; files can be regenerated from raw data |
| Performance regression | Benchmark before/after; no compression for speed |

