# Research: Analog Section Time Detection

**Feature**: 005-analog-section-time  
**Date**: 2025-12-17

## Summary

This document consolidates research findings for implementing analog section time detection with unified acquisition sample indices.

---

## R1: Peak Detection Algorithm

**Decision**: Use `scipy.signal.find_peaks()` on the derivative of raw_ch1

**Rationale**: 
- Already proven in existing `_detect_analog_peaks()` implementation
- Works well for step-function light pulses (sharp transitions create large derivative peaks)
- Threshold parameter allows tuning for different signal amplitudes
- Fast: O(n) complexity, handles 20M+ samples in < 1 second

**Alternatives Considered**:

| Alternative | Rejected Because |
|-------------|------------------|
| Template matching | More complex, not needed for simple step transitions |
| Wavelet detection | Overkill for step functions, adds dependency |
| Zero-crossing detection | Less robust to noise, harder to threshold |
| Schmitt trigger emulation | More complex state machine, no clear benefit |

**Implementation**:
```python
from scipy.signal import find_peaks

# Detect rising edges in raw_ch1
diff_signal = np.diff(raw_ch1)
peaks, _ = find_peaks(diff_signal, height=threshold_value)
# peaks contains indices into raw_ch1 where transitions occur
```

---

## R2: Unit Conversion for Playlist Section Time

**Decision**: Use `frame_timestamps[frame_index]` lookup to convert display frames to acquisition samples

**Rationale**:
- `frame_timestamps` already maps display frame indices to acquisition sample indices
- This is the existing mechanism used for light template extraction in current code
- Simple O(1) array indexing per conversion
- No interpolation needed - direct lookup

**Formula**:
```python
# frame_timestamps: Array where index = display frame, value = acquisition sample
# Convert display frame boundaries to sample boundaries:
start_sample = int(frame_timestamps[start_frame])
end_sample = int(frame_timestamps[end_frame])
```

**Edge Cases**:
- Frame index out of bounds: Clip to valid range (0, len(frame_timestamps)-1)
- Empty frame_timestamps: Return error (required for playlist-based)

---

## R3: Existing Implementation Analysis

**Current `add_section_time_analog()` Flow**:
1. Load `raw_ch1` (or previously `1khz_ch1`) ✓
2. Detect peaks using `_detect_analog_peaks()` ✓
3. Convert 1kHz indices to time in seconds ✗ (unnecessary step)
4. Convert time to acquisition samples ✗ (adds complexity)
5. Convert samples to display frame indices via `frame_timestamps` ✗ (REMOVE)
6. Filter peaks beyond frame_timestamps range ✗ (REMOVE)
7. Store as display frame indices ✗ (CHANGE to sample indices)

**New `add_section_time_analog()` Flow**:
1. Load `raw_ch1` 
2. Detect peaks using `_detect_analog_peaks()` on raw_ch1
3. Peaks ARE acquisition sample indices (direct from raw_ch1 indexing)
4. Compute end_sample = onset_sample + (plot_duration × acquisition_rate)
5. Store as acquisition sample indices

**Simplification**: Removes frame_timestamps dependency entirely for analog section time.

---

## R4: Test Impact Analysis

**Tests to Remove** (no longer applicable):
- `test_raises_on_missing_frame_timestamps` - frame_timestamps not required for analog
- `test_returns_false_peaks_beyond_frame_range` - no frame range filtering

**Tests to Update** (expected values change):
- `test_basic_detection` - section_time values should be sample indices
- `test_plot_duration_affects_section_length` - spans measured in samples
- All other analog tests - verify sample index expectations

**Tests to Add**:
- `test_stores_acquisition_sample_indices` - explicit unit verification
- `test_no_frame_timestamps_required` - analog works without frame_timestamps

**Playlist Section Time Tests**:
- `test_happy_path` - update expected output unit
- `test_computes_frame_boundaries` - rename to `test_computes_sample_boundaries`
- Add conversion verification test

---

## R5: Backward Compatibility

**Decision**: No migration of existing data

**Rationale**:
- User explicitly chose to leave existing data as-is
- Existing zarr files with display frame indices remain valid
- Downstream code not currently consuming section_time
- Future runs will produce consistent sample indices

**Documentation**:
- Spec notes this in "Out of Scope"
- No migration function needed
- Users re-run section_time if they need unified units

---

## Dependencies Identified

| Dependency | Version | Purpose |
|------------|---------|---------|
| scipy | ≥1.7 | `find_peaks()` for peak detection |
| numpy | ≥1.20 | Array operations, derivative |
| zarr | ≥2.0 | Storage backend |

All dependencies already in project requirements.
