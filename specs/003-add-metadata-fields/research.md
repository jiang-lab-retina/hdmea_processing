# Research: Add frame_time and acquisition_rate to Metadata

**Feature Branch**: `003-add-metadata-fields`  
**Date**: 2024-12-15

## Research Tasks

### 1. Current acquisition_rate Extraction (CMCR)

**Question**: How is acquisition_rate currently extracted from CMCR files?

**Findings**:
- Location: `src/hdmea/io/cmcr.py`, function `load_cmcr_data()`
- Current behavior (lines 74-100):
  - Default value: 20000 Hz
  - Estimation method: `len(light_data) / (recording_duration_us / 1e6)` when recording duration is available
  - Falls back to default 20000 Hz if estimation not possible
- Return value: `acquisition_rate` is already returned in the result dict (line 149)

**Decision**: CMCR extraction is already implemented. No changes needed to `cmcr.py`.

**Rationale**: The existing code correctly extracts or estimates acquisition_rate from CMCR files.

---

### 2. CMTR acquisition_rate Extraction (Fallback)

**Question**: Can acquisition_rate be extracted from CMTR files?

**Findings**:
- Location: `src/hdmea/io/cmtr.py`, function `load_cmtr_data()`
- Current behavior: Extracts metadata from file attributes but does NOT specifically extract acquisition_rate
- CMTR files contain spike-sorted data which may include timing metadata in file attributes
- McsPy `McsCMOSMEAData` class provides access to file attributes via `cmtr_data.attrs`

**Decision**: Add acquisition_rate extraction to CMTR as fallback.

**Rationale**: FR-006 requires CMTR fallback. The file attributes may contain sampling rate info that can be extracted.

**Alternatives considered**:
- Skip CMTR extraction entirely → Rejected: violates FR-006
- Require CMCR for all recordings → Rejected: too restrictive for users with only CMTR files

---

### 3. frame_time Computation

**Question**: Where should frame_time be computed and stored?

**Findings**:
- `frame_time = 1 / acquisition_rate` (seconds per sample)
- Should be computed in `runner.py` after acquisition_rate is finalized
- Stored alongside acquisition_rate in metadata dict before calling `write_metadata()`

**Decision**: Compute frame_time in `load_recording()` after determining final acquisition_rate.

**Rationale**: Centralizes the computation in the orchestration layer, ensuring consistency.

---

### 4. Validation Requirements

**Question**: What validation is needed for acquisition_rate?

**Findings**:
- FR-004: Must be positive number
- Typical range: 1000 Hz to 100000 Hz for HD-MEA systems
- MaxOne/MaxTwo typically use 20000 Hz

**Decision**: Validate `acquisition_rate > 0` before storing. Log warning if outside typical range (1000-100000 Hz) but don't reject.

**Rationale**: Strict positivity check catches invalid data; range warning catches potential data issues without blocking legitimate edge cases.

---

### 5. Existing Metadata Storage

**Question**: How is metadata currently stored in Zarr?

**Findings**:
- Location: `src/hdmea/io/zarr_store.py`, function `write_metadata()`
- Scalars (int, float) stored as group attributes (line 251-252)
- Strings stored as attributes (line 254-255)
- Already handles the types we need (acquisition_rate: float, frame_time: float)

**Decision**: No changes needed to `write_metadata()`. Existing implementation handles float attributes correctly.

**Rationale**: The function already stores scalar floats as attributes, which is the desired behavior.

---

## Summary of Changes Required

| File | Change | Type |
|------|--------|------|
| `src/hdmea/io/cmcr.py` | None | No change |
| `src/hdmea/io/cmtr.py` | Add acquisition_rate extraction to metadata | Addition |
| `src/hdmea/pipeline/runner.py` | Add frame_time computation, ensure acquisition_rate in metadata | Modification |
| `src/hdmea/io/zarr_store.py` | None | No change |
| `tests/unit/test_metadata_fields.py` | New unit tests | New file |

## Resolved Clarifications

All technical questions resolved. No NEEDS CLARIFICATION markers remain.

