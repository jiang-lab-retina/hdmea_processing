# Research: JSON-Based Spike Sectioning

**Branch**: `008-json-spike-sectioning` | **Date**: 2024-12-18

## Research Questions

### 1. How should JSON config files be discovered and loaded?

**Decision**: Use explicit config directory path with default fallback to `config/stimuli/`.

**Rationale**: 
- Constitution requires explicit I/O - no hidden global state
- Default path (`config/stimuli/`) provides convenience while maintaining explicitness
- Path can be overridden via parameter for testing and different deployments

**Alternatives Considered**:
- Environment variable: Rejected - Constitution forbids reading env vars in library code
- Hardcoded path only: Rejected - insufficient flexibility for testing
- Auto-discovery from package location: Rejected - fragile, depends on installation method

### 2. How to handle frame-to-sample conversion accurately?

**Decision**: Use existing `_convert_frame_to_sample_index()` function from `section_time.py`.

**Rationale**:
- Function already exists and handles edge cases (clipping to valid range)
- Maintains consistency with existing codebase
- Well-tested in current section_time operations

**Alternatives Considered**:
- Direct array indexing (`frame_timestamps[frame]`): Rejected - lacks bounds checking
- New conversion function: Rejected - would duplicate existing logic

### 3. Where should JSON loading logic be placed?

**Decision**: Create new helper function `_load_stimuli_config()` in `spike_sectioning.py`.

**Rationale**:
- Keeps JSON loading co-located with spike sectioning logic
- Single responsibility - loads config for sectioning purposes
- Could be moved to `io/` subpackage later if reuse needed elsewhere

**Alternatives Considered**:
- Add to `section_time.py`: Rejected - that module handles section_time creation, not consumption
- Create new `config.py` module: Rejected - over-engineering for single use case
- Load in `__init__.py`: Rejected - violates explicit I/O principle

### 4. How to validate JSON config files upfront?

**Decision**: Validate all movie configs before processing any units (fail-fast approach).

**Rationale**:
- Clarified requirement: fail if ANY movie lacks JSON config
- Validates `section_kwargs` structure and required fields
- Reports ALL missing/invalid configs in single error message for better UX

**Alternatives Considered**:
- Validate per-movie during processing: Rejected - could process some units before failure
- Skip validation: Rejected - silent failures are unacceptable per clarification

### 5. How to calculate trial boundaries from JSON parameters?

**Decision**: Calculate as `section_frame_start + PRE_MARGIN_FRAME_NUM + start_frame` for first trial, then add `trial_length_frame` for subsequent trials.

**Rationale**:
- Matches spec requirement exactly
- `section_frame_start` comes from converting HDF5 section_time[0, 0] back to frame
- `PRE_MARGIN_FRAME_NUM` (60) accounts for pre-stimulus padding
- `start_frame` from JSON is relative to movie content start
- All frame calculations done first, then converted to samples at the end

**Formula**:
```
For trial n (0-indexed):
  trial_start_frame = section_frame_start + PRE_MARGIN_FRAME_NUM + start_frame + (n * trial_length_frame)
  trial_end_frame = trial_start_frame + trial_length_frame
  trial_start_sample = frame_timestamps[trial_start_frame]
  trial_end_sample = frame_timestamps[trial_end_frame]
```

## Technical Dependencies

| Dependency | Purpose | Version |
|------------|---------|---------|
| `json` | Parse JSON config files | stdlib |
| `pathlib` | Path handling | stdlib |
| `h5py` | HDF5 file access (existing) | >=3.0 |
| `numpy` | Array operations (existing) | >=1.20 |

## Key Findings

1. **Existing Infrastructure**: The codebase already has `_convert_frame_to_sample_index()` and `_convert_sample_index_to_frame()` functions that handle the frame↔sample conversion accurately.

2. **PRE_MARGIN_FRAME_NUM**: Already defined as constant (60) in `section_time.py` and imported in `spike_sectioning.py`.

3. **Current section_time structure**: HDF5 stores `[start_sample, end_sample]` pairs. The start_sample includes the PRE_MARGIN, so we need to:
   - Convert start_sample back to frame number
   - That gives us `section_frame_start` (which already includes pre-margin from original creation)
   - But JSON's `start_frame` is relative to movie content, so formula needs adjustment

4. **JSON Config Format**: All existing configs have consistent structure:
   ```json
   {
     "section_kwargs": {
       "start_frame": int,
       "trial_length_frame": int,
       "repeat": int
     }
   }
   ```

## Resolved Unknowns

- ✅ Config loading location: `config/stimuli/` with override parameter
- ✅ Frame conversion: Use existing helper functions
- ✅ Validation strategy: Fail-fast with comprehensive error listing
- ✅ Trial boundary formula: Documented above

