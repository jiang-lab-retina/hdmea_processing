# Implementation Plan: JSON-Based Spike Sectioning

**Branch**: `008-json-spike-sectioning` | **Date**: 2024-12-18 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/008-json-spike-sectioning/spec.md`

## Summary

Modify the spike sectioning module to load trial boundaries from JSON configuration files in `config/stimuli/` instead of calculating them from a `trial_repeats` parameter. Each stimulus type has its own JSON file defining `start_frame`, `trial_length_frame`, and `repeat` values. Frame numbers from JSON are converted to sample indices using `frame_timestamps` for accurate spike extraction.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: h5py, numpy, json (stdlib), pathlib (stdlib)  
**Storage**: HDF5 files (read/write), JSON config files (read-only)  
**Testing**: pytest  
**Target Platform**: Desktop/workstation (Windows, Linux, macOS)  
**Project Type**: Single project (hdmea package)  
**Performance Goals**: Process 1000+ units in <5 minutes (existing performance maintained)  
**Constraints**: Backward-compatible function signatures, preserve HDF5 output format  
**Scale/Scope**: 6 stimulus types currently, extensible to 50+

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Code in `src/hdmea/io/spike_sectioning.py` |
| II. Modular Subpackage Layout | ✅ PASS | `io/` module appropriate for data I/O |
| III. Explicit I/O and Pure Functions | ✅ PASS | `config_dir` passed explicitly |
| IV. Single HDF5 Artifact | ✅ PASS | Reads from existing HDF5, writes to same |
| V. Data Format Standards | ✅ PASS | JSON for config (human-readable), HDF5 for data |
| VI. No Hidden Global State | ✅ PASS | No module-level mutable state |
| VII. Independence from Legacy Code | ✅ PASS | No legacy imports |

**Post-Design Re-check**: All gates still passing after Phase 1 design.

## Project Structure

### Documentation (this feature)

```text
specs/008-json-spike-sectioning/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0: Research findings
├── data-model.md        # Phase 1: Data entities and flow
├── quickstart.md        # Phase 1: Usage guide
├── contracts/           # Phase 1: API contracts
│   ├── api.md          # Function signatures
│   └── README.md
└── checklists/
    └── requirements.md  # Validation checklist
```

### Source Code (repository root)

```text
src/hdmea/
├── io/
│   ├── spike_sectioning.py   # MODIFIED: Main changes here
│   └── section_time.py       # UNCHANGED: Provides conversion functions
└── ...

config/
└── stimuli/                   # EXISTING: JSON config files
    ├── moving_h_bar_s5_d8_3x.json
    ├── step_up_5s_5i_3x.json
    ├── perfect_dense_noise_15x15_15hz_r42_3min.json
    └── ...

tests/
├── unit/
│   └── io/
│       └── test_spike_sectioning.py  # MODIFIED: New test cases
└── fixtures/
    └── stimuli_configs/              # NEW: Test config fixtures
```

**Structure Decision**: Single project layout. All changes confined to `src/hdmea/io/spike_sectioning.py` with new test fixtures.

## Implementation Approach

### Phase 1: Add JSON Config Loading

1. Add `_load_stimuli_config()` helper function
2. Add `_validate_all_configs()` for fail-fast validation
3. Add `config_dir` parameter to `section_spike_times()`

### Phase 2: Modify Trial Boundary Calculation

1. Add `_calculate_trial_boundaries()` function
2. Modify `_section_unit_spikes()` to accept pre-computed boundaries
3. Update main loop to use JSON-derived boundaries

### Phase 3: Integration and Deprecation

1. Mark `trial_repeats` parameter as deprecated
2. Update logging to show config file usage
3. Add comprehensive error messages for config issues

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Config location | Default `config/stimuli/` with override | Explicit but convenient |
| Validation timing | Upfront (all configs) | Fail-fast, comprehensive errors |
| Frame conversion | Use existing helpers | Code reuse, consistency |
| Backward compat | Preserve signatures | Minimize downstream impact |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| JSON file missing | Comprehensive validation with all missing files listed |
| Invalid JSON format | Detailed error messages per field |
| Frame calculation error | Unit tests with known input/output pairs |
| Performance regression | Benchmark before/after on realistic dataset |

## Complexity Tracking

No constitution violations requiring justification.

