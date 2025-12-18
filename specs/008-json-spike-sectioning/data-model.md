# Data Model: JSON-Based Spike Sectioning

**Branch**: `008-json-spike-sectioning` | **Date**: 2024-12-18

## Entities

### StimuliConfig

Configuration loaded from JSON files in `config/stimuli/`.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `name` | `str` | Stimulus identifier (matches filename) | Must match HDF5 movie name |
| `start_frame` | `int` | First trial start frame (relative to movie content) | >= 0 |
| `trial_length_frame` | `int` | Duration of each trial in frames | > 0 |
| `repeat` | `int` | Number of trial repetitions | >= 1 |

**Source**: `config/stimuli/{movie_name}.json` → `section_kwargs` object

### SectionTimeEntry

Existing HDF5 dataset representing movie boundaries.

| Field | Type | Description |
|-------|------|-------------|
| `start_sample` | `int64` | Movie start in acquisition samples |
| `end_sample` | `int64` | Movie end in acquisition samples |

**Source**: HDF5 at `stimulus/section_time/{movie_name}` with shape `(N, 2)`

### TrialBoundary

Computed trial boundaries for spike extraction.

| Field | Type | Description |
|-------|------|-------------|
| `trial_idx` | `int` | Trial index (0-based) |
| `start_sample` | `int64` | Trial start in acquisition samples |
| `end_sample` | `int64` | Trial end in acquisition samples |
| `start_frame` | `int` | Trial start in display frames |
| `end_frame` | `int` | Trial end in display frames |

**Computed from**: StimuliConfig + SectionTimeEntry + frame_timestamps

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT SOURCES                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  config/stimuli/{movie}.json         HDF5 File                          │
│  ┌───────────────────────┐          ┌─────────────────────────────┐    │
│  │ section_kwargs:       │          │ stimulus/section_time/{movie}│    │
│  │   start_frame: 60     │          │   [[start_sample, end_sample]]   │
│  │   trial_length: 4520  │          │                             │    │
│  │   repeat: 3           │          │ metadata/frame_timestamps   │    │
│  └───────────────────────┘          │   [sample_idx_per_frame]    │    │
│           │                          └─────────────────────────────┘    │
│           │                                    │                        │
└───────────┼────────────────────────────────────┼────────────────────────┘
            │                                    │
            v                                    v
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMPUTATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Load & Validate Config                                              │
│     ├─ Check all movies have JSON configs                               │
│     └─ Validate section_kwargs fields                                   │
│                                                                         │
│  2. Convert section_time to frame                                       │
│     └─ section_frame_start = sample_to_frame(section_time[0,0])        │
│                                                                         │
│  3. Calculate trial boundaries (in frames)                              │
│     └─ For n in 0..repeat-1:                                           │
│          trial_start = section_frame_start + PRE_MARGIN + start_frame  │
│                        + (n * trial_length_frame)                       │
│          trial_end = trial_start + trial_length_frame                  │
│                                                                         │
│  4. Convert to sample indices                                           │
│     └─ trial_start_sample = frame_timestamps[trial_start_frame]        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
            │
            v
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT (unchanged)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  HDF5: units/{unit_id}/spike_times_sectioned/{movie_name}/              │
│    ├─ full_spike_times: int64[]      # All trials combined             │
│    ├─ trials_spike_times/                                               │
│    │    ├─ 0: int64[]                # Trial 0 spikes                  │
│    │    ├─ 1: int64[]                # Trial 1 spikes                  │
│    │    └─ ...                                                          │
│    └─ attrs:                                                            │
│         ├─ n_trials: int                                                │
│         ├─ trial_repeats: int                                           │
│         ├─ pad_margin: [float, float]                                   │
│         └─ created_at: str                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Validation Rules

### JSON Config Validation

1. File must exist at `config/stimuli/{movie_name}.json`
2. Must contain `section_kwargs` object
3. `section_kwargs` must have all required fields:
   - `start_frame`: integer >= 0
   - `trial_length_frame`: integer > 0
   - `repeat`: integer >= 1

### Runtime Validation

1. `section_frame_start + PRE_MARGIN_FRAME_NUM + start_frame` must be < `len(frame_timestamps)`
2. Final trial end frame must not exceed available frames (warn if truncated)
3. All movies in `section_time` group must have corresponding JSON config

## Constants

| Constant | Value | Source |
|----------|-------|--------|
| `PRE_MARGIN_FRAME_NUM` | 60 | `section_time.py` |
| `POST_MARGIN_FRAME_NUM` | 120 | `section_time.py` |
| `DEFAULT_CONFIG_DIR` | `config/stimuli/` | New constant |

