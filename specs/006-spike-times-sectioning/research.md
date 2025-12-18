# Research: Spike Times Unit Conversion and Stimulation Sectioning

**Date**: 2025-12-17  
**Spec**: [spec.md](./spec.md)  
**Plan**: [plan.md](./plan.md)

---

## Research Questions

### R1: What unit do raw timestamps from CMTR use?

**Question**: What is the native timestamp unit from `get_peaks_timestamps()` in McsPy?

**Finding**: Raw timestamps are in **nanoseconds** (10^-9 seconds).

**Evidence**:
1. MaxWell/MaxTwo documentation specifies nanosecond precision for spike timing
2. Legacy code (`feature_analysis.py` line 1748) performs conversion:
   ```python
   all_raw_spike_ns * acquisition_rate / 1_000_000
   ```
   This converts ns×Hz/10^6 which is incorrect (should be /10^9), but confirms ns input
3. Current `cmtr.py` stores as `uint64` without conversion - typical ns values for a 1-hour recording would be ~3.6×10^12, fitting in uint64

**Decision**: Timestamps are nanoseconds; convert using `ns × acquisition_rate / 10^9`

---

### R2: Where should the conversion happen?

**Question**: Should we convert in `load_cmtr_data()`, `load_recording()`, or `write_units()`?

**Options**:
| Location | Pros | Cons |
|----------|------|------|
| `load_cmtr_data()` | Single point of truth | Needs acquisition_rate passed in; breaks pure I/O |
| `load_recording()` | Has acquisition_rate; visible in orchestration | Conversion not reusable |
| `write_units()` | Consistent storage | Hides transformation; breaks separation |

**Decision**: Convert in `load_recording()` just before calling `write_units()`.

**Rationale**:
- `load_recording()` already has `acquisition_rate` from metadata extraction
- Keeps `load_cmtr_data()` as a pure data loader (returns what's in file)
- Transformation is visible in the pipeline orchestration code
- Follows existing pattern where runner.py transforms data (e.g., `compute_firing_rate()`)

---

### R3: How should sectioned spike times be stored in Zarr?

**Question**: What Zarr structure best handles sectioned spike data?

**Options**:
| Structure | Description | Pros | Cons |
|-----------|-------------|------|------|
| Per-trial arrays | `{movie}/trial_{N}` | Per-trial access | Complex navigation; many small arrays |
| Single array per movie | `{movie}` with all spikes | Simple; matches legacy | Need to filter for per-trial analysis |
| 2D padded array | `(N_trials, max_spikes)` | Matrix operations | Wastes space; sentinel values needed |

**Decision**: Use single array per movie: `spike_times_sectioned/{movie_name}` containing ALL spikes from all trials combined.

**Rationale**:
- Simpler structure - one array per movie
- Matches legacy pattern from `add_section_time_analog_auto_accurate()` which used `all_traces_full.extend(trace)`
- Easy to filter by trial boundaries if per-trial analysis needed
- Avoids proliferation of small arrays in zarr

---

### R4: Naming conflict - can zarr have both array and group at same path?

**Question**: Can we have `units/{unit_id}/spike_times` (array) AND `units/{unit_id}/spike_times/{movie}` (group)?

**Finding**: **NO** - Zarr cannot have both an array and a group with the same name.

**Evidence**: Zarr stores arrays as directories with `.zarray` metadata. A group is also a directory. You cannot have both at the same path.

**Decision**: Use `spike_times_sectioned/` for the sectioned data group.

**Final Structure** (simplified - single array per movie):
```
units/{unit_id}/
├── spike_times              # Array: full recording spikes (sample indices)
└── spike_times_sectioned/   # Group: sectioned data
    ├── movie_A              # Array: ALL spikes from all trials
    └── movie_B              # Array: ALL spikes from all trials
```

---

### R5: Storage format decision - per-trial vs combined

**Question**: Should spikes be stored per-trial or combined in a single array per movie?

**User Clarification (Updated)**: Store BOTH formats:
- `full_spike_times` - all trials combined
- `trials_spike_times/{trial_idx}` - per-trial split using section_time boundaries

**Decision**: Store BOTH combined and per-trial formats.

**Rationale**:
- `full_spike_times` provides quick access to all relevant spikes
- `trials_spike_times` enables direct per-trial analysis without runtime filtering
- Both formats useful for different analysis workflows
- New `trial_repeats` parameter (default=3) controls how many trials to process

---

### R6: How does existing section_time.py handle FileExistsError?

**Question**: What pattern does `add_section_time_analog()` use for overwrite protection?

**Finding**: Uses explicit FileExistsError with force parameter.

**Evidence** (`section_time.py` lines 594-598):
```python
if movie_name in st_group and not force:
    raise FileExistsError(
        f"section_time/{movie_name} already exists in {zarr_path}. "
        "Use force=True to overwrite."
    )
```

**Decision**: Follow same pattern for `section_spike_times()`:
```python
if unit_has_sectioned_data and not force:
    raise FileExistsError(
        f"spike_times_sectioned already exists for {unit_id} in {zarr_path}. "
        "Use force=True to overwrite."
    )
```

---

### R7: Padding parameter design

**Question**: How should the padding parameter be structured for trial boundary extension?

**User Clarification**: Use a tuple `(pre_margin, post_margin)` in seconds with default `(2.0, 0.0)`.

**Decision**: Parameter `pad_margin: Tuple[float, float] = (2.0, 0.0)`

**Conversion**:
```python
pre_samples = int(pad_margin[0] * acquisition_rate)   # e.g., 2.0 * 20000 = 40000
post_samples = int(pad_margin[1] * acquisition_rate)  # e.g., 0.0 * 20000 = 0
```

**Boundary Calculation**:
```python
padded_start = max(0, trial_start - pre_samples)  # Clamp to 0
padded_end = trial_end + post_samples             # Clamp to max if needed
```

**Rationale**:
- Tuple provides independent control of pre/post margins
- Default `(2.0, 0.0)` provides 2-second pre-stimulus baseline, no post-stimulus extension
- Unit in seconds is intuitive; conversion to samples is internal

---

## Summary of Decisions

| Decision | Choice | Key Rationale |
|----------|--------|---------------|
| Timestamp unit | Nanoseconds (10^-9 s) | MaxWell spec + legacy code evidence |
| Conversion location | `load_recording()` | Has acquisition_rate; visible transformation |
| Conversion formula | `ns × rate / 10^9` | Standard ns→samples conversion |
| Sectioned storage | BOTH combined AND per-trial | Supports different analysis workflows |
| Path for combined | `spike_times_sectioned/{movie}/full_spike_times` | All trials combined |
| Path for per-trial | `spike_times_sectioned/{movie}/trials_spike_times/{idx}` | Per-trial split |
| Trial count parameter | `trial_repeats=3` | Controls number of trials to process |
| Padding parameter | `pad_margin=(2.0, 0.0)` seconds | Tuple of (pre_margin, post_margin) |
| Padding conversion | `pre_samples = int(pad_margin[0] × acquisition_rate)` | Unit consistency with section_time |
| Boundary clamping | Clamp start >= 0 | Prevents negative sample indices |
| Overwrite protection | FileExistsError pattern | Matches existing `add_section_time_analog()` |
| Spike times unit | Absolute (sample indices) | Same coordinate system as spike_times |
