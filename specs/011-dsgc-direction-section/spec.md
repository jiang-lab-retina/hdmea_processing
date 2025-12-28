# Feature Specification: DSGC Direction Sectioning

**Feature Branch**: `011-dsgc-direction-section`  
**Created**: 2025-12-27  
**Status**: Draft  
**Input**: User description: "Separate spikes for each unit when a moving bar crosses its receptive field center, organized by direction (8 directions × 3 repetitions)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Section Spikes by Direction at Cell Center (Priority: P1)

A researcher wants to extract spike responses for each direction of a moving bar stimulus, where responses are defined as spikes occurring when the bar crosses the cell's receptive field center. The system uses the per-pixel on/off timing dictionary to determine the exact frame window for each direction trial.

**Why this priority**: This is the core functionality that enables direction selectivity analysis for DSGC cells.

**Independent Test**: Can be tested by running sectioning on a unit with known RF center and verifying spike counts per direction match expected values.

**Acceptance Scenarios**:

1. **Given** an HDF5 recording with `full_spike_times` under `spike_times_sectioned/moving_h_bar_s5_d8_3x` and a cell center location, **When** direction sectioning is requested, **Then** the system extracts spikes for all 24 trials (8 directions × 3 reps) and saves results under `direction_section`.

2. **Given** a unit with an RF center at pixel (row, col), **When** sectioning is performed, **Then** spikes within the on/off time window (with padding) at that pixel are extracted for each trial.

3. **Given** sectioned spike data, **When** results are saved, **Then** data is organized by direction (0, 45, 90, 135, 180, 225, 270, 315) with 3 trials per direction.

---

### User Story 2 - Convert Cell Center from 15×15 to 300×300 Grid (Priority: P1)

A researcher's cell center is stored in STA geometry features using a 15×15 grid coordinate system. The system must convert this to the 300×300 pixel space used by the on/off timing dictionary.

**Why this priority**: Correct coordinate conversion is essential for accurate alignment between spike data and stimulus timing.

**Independent Test**: Can be tested by verifying a known 15×15 center maps to the expected 300×300 coordinate.

**Acceptance Scenarios**:

1. **Given** a cell center stored as `center_row=7, center_col=7` in 15×15 grid, **When** converted to 300×300 space, **Then** the result is approximately `(140, 140)` (scaling factor = 20).

2. **Given** an HDF5 file with STA geometry data under `features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/`, **When** cell center is requested, **Then** `center_row` and `center_col` are read and converted to 300×300 coordinates.

---

### User Story 3 - Apply Padding to Trial Windows (Priority: P2)

A researcher wants to capture spike activity slightly before and after the bar crosses the cell center, using a configurable padding frame count.

**Why this priority**: Padding ensures complete response capture including latency effects, but the core sectioning works without it.

**Independent Test**: Can be tested by comparing spike counts with padding=0 vs padding=10.

**Acceptance Scenarios**:

1. **Given** default padding of 10 frames, **When** trial windows are computed, **Then** `start_frame = on_time - 10` and `end_frame = off_time + 10`.

2. **Given** a custom padding value, **When** sectioning is performed, **Then** the padding is applied symmetrically to all trial windows.

---

### User Story 4 - Save Results Without Overwriting Source (Priority: P1)

A researcher wants to save sectioned results to the HDF5 file without modifying the original `full_spike_times` data. Results should be stored under a new group `direction_section`.

**Why this priority**: Data integrity is critical - source data must never be modified.

**Independent Test**: Can be tested by verifying `full_spike_times` is unchanged after sectioning and `direction_section` group exists.

**Acceptance Scenarios**:

1. **Given** an HDF5 file with `full_spike_times`, **When** direction sectioning completes, **Then** original data is unchanged and new data is under `spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section/`.

2. **Given** sectioned data for direction 45°, **When** saved, **Then** data is stored under `direction_section/45/trials/{0,1,2}` with spike times in sampling indices.

---

### Edge Cases

- What happens when a cell center pixel is outside the 300×300 range?
  - System should clip to valid range (0-299) and log a warning.
- What happens when a trial window has no spikes?
  - System should save an empty array for that trial.
- What happens when the on/off dictionary file is not found?
  - System should raise a clear FileNotFoundError with the expected path.
- What happens when STA geometry data is missing for a unit?
  - System should skip that unit and log a warning.

## Clarifications

### Session 2025-12-27

- Q: What is the exact path for the on/off dictionary? → A: `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl`
- Q: What is the direction sequence? → A: `[0, 45, 90, 135, 180, 225, 270, 315]` degrees
- Q: How are the 24 trials indexed in the on/off dict? → A: Indices 0-7 = Dir 1-8 Rep 1, 8-15 = Dir 1-8 Rep 2, 16-23 = Dir 1-8 Rep 3
- Q: What is the default padding? → A: 10 frames before and after the on/off window
- Q: Where is cell center stored? → A: `units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/center_row` and `center_col`
- Q: What coordinate system is cell center in? → A: 15×15 grid, must be scaled by 20 to convert to 300×300
- Q: Where should results be saved? → A: Under `units/{unit_id}/spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section/`
- Q: What unit should spike times be saved in? → A: Sampling indices (same as source data)
- Q: Where is frame_timestamps for conversion? → A: `metadata/frame_timestamps` in the HDF5 file
- Q: What is PRE_MARGIN_FRAME_NUM for alignment? → A: 60 frames, must be added when computing movie-relative frame from section_time
- Q: What happens if direction_section already exists? → A: Skip by default; overwrite only if `force=True` parameter is set.
- Q: In-place or separate output file? → A: In-place modification by default; for testing, read source but save to export folder to protect test data.
- Q: Process all units or specific units? → A: Process all by default, but allow `unit_ids` parameter to specify a subset.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load the on/off timing dictionary from `M:\Python_Project\Data_Processing_2025\Design_Stimulation_Pattern\Data\Stimulations\moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl`.

- **FR-002**: System MUST read cell center from HDF5 path `units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/center_row` and `center_col`.

- **FR-003**: System MUST convert cell center from 15×15 grid to 300×300 pixel space using scaling factor of 20 (formula: `coord_300 = coord_15 * 20`).

- **FR-004**: System MUST clip converted cell center coordinates to valid range [0, 299].

- **FR-005**: System MUST read `full_spike_times` from `units/{unit_id}/spike_times_sectioned/moving_h_bar_s5_d8_3x/full_spike_times`.

- **FR-006**: System MUST convert spike times from sampling indices to movie-relative frame numbers using `frame_timestamps` from `metadata/frame_timestamps`.

- **FR-007**: System MUST compute movie start frame using: `movie_start_frame = convert_sample_to_frame(section_time[0,0]) + PRE_MARGIN_FRAME_NUM` where `PRE_MARGIN_FRAME_NUM = 60`.

- **FR-008**: System MUST apply configurable padding (default: 10 frames) to trial windows: `start = on_time - padding`, `end = off_time + padding`.

- **FR-009**: System MUST extract spikes where `start_frame <= spike_frame <= end_frame` for each of the 24 trials.

- **FR-010**: System MUST organize output by direction using keys from `direction_list = [0, 45, 90, 135, 180, 225, 270, 315]`.

- **FR-011**: System MUST map trial indices to directions: `direction_index = trial_index % 8`, `repetition = trial_index // 8`.

- **FR-012**: System MUST save sectioned spike times in sampling index units (same as source).

- **FR-013**: System MUST save section start/end times (in sampling indices) as metadata for each trial.

- **FR-014**: System MUST save results under HDF5 path `units/{unit_id}/spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section/{direction}/trials/{rep_index}`.

- **FR-015**: System MUST NOT modify the original `full_spike_times` dataset.

- **FR-016**: System MUST process all units in the HDF5 file that have both `full_spike_times` and STA geometry data, by default.

- **FR-017**: System MUST skip units missing required data (spike times or STA geometry) and log a warning.

- **FR-022**: System MUST support optional `unit_ids` parameter (list of unit IDs) to process only specified units instead of all.

- **FR-023**: When `unit_ids` is provided, system MUST validate that each specified unit exists and has required data, logging warnings for invalid entries.

- **FR-018**: System MUST skip units where `direction_section` already exists, unless `force=True` parameter is provided.

- **FR-019**: When `force=True`, system MUST delete existing `direction_section` group before writing new data.

- **FR-020**: System MUST support in-place modification of HDF5 file by default (write to same file).

- **FR-021**: System MUST support optional `output_path` parameter to write results to a different file (copy source first, then modify copy).

### Key Entities

- **Direction Section**: A grouping of spike times by motion direction (8 directions) and repetition (3 per direction).

- **On/Off Dictionary**: A per-pixel dictionary containing frame indices when the moving bar covers (`on_peak_location`) and leaves (`off_peak_location`) each pixel for all 24 trials.

- **Cell Center**: The receptive field center of a unit in pixel coordinates, used to look up the timing for when the bar crosses the cell's RF.

- **PRE_MARGIN_FRAME_NUM**: A constant (60 frames) representing the recording margin before actual movie content, required for alignment between section_time and movie-relative coordinates.

- **Direction List**: The 8 motion directions in degrees: `[0, 45, 90, 135, 180, 225, 270, 315]`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Direction sectioning completes for all units with valid data in a typical recording within 60 seconds.

- **SC-002**: Sectioned data structure contains exactly 8 direction groups, each with exactly 3 trial datasets.

- **SC-003**: Sum of all sectioned spikes across all trials equals or is less than total spikes in `full_spike_times` (spikes outside all windows are excluded).

- **SC-004**: Section start/end metadata correctly reflects the on/off times plus padding for each trial.

- **SC-005**: Original `full_spike_times` data is unchanged after processing (verified by checksum or direct comparison).

- **SC-006**: Cell center conversion from 15×15 to 300×300 is within 10 pixels of expected value.

## Assumptions

- The on/off dictionary file exists at the specified path with the expected structure.
- The movie name in HDF5 is exactly `moving_h_bar_s5_d8_3x`.
- STA geometry data uses the noise movie name `sta_perfect_dense_noise_15x15_15hz_r42_3min`.
- The on/off dictionary uses 0-indexed pixel coordinates (row, col) from 0 to 299.
- The frame rate is 60 fps and sampling rate is 20 kHz.
- Test file for validation (source, read-only): `M:\Python_Project\Data_Processing_2027\Projects\ap_trace_hdf5\export_ap_tracking_20251226\2024.08.08-10.40.20-Rec.h5`
- Test output location: `M:\Python_Project\Data_Processing_2027\Projects\dsgc_section\export\` (copy source here before modifying)

## Data Model

### Input Data

```
HDF5 File
├── metadata/
│   └── frame_timestamps          # int64[] - sample indices for each frame
├── stimulus/
│   └── section_time/
│       └── moving_h_bar_s5_d8_3x # int64[N,2] - [start, end] sample indices
└── units/
    └── {unit_id}/
        ├── spike_times_sectioned/
        │   └── moving_h_bar_s5_d8_3x/
        │       └── full_spike_times  # int64[] - spike sample indices
        └── features/
            └── sta_perfect_dense_noise.../
                └── sta_geometry/
                    ├── center_row    # float - in 15x15 grid
                    └── center_col    # float - in 15x15 grid
```

### On/Off Dictionary (Pickle)

```python
{
    (row, col): {  # 0-299 pixel coordinates
        'on_peak_location': [frame_idx × 24],   # when bar covers pixel
        'off_peak_location': [frame_idx × 24],  # when bar leaves pixel
    },
    ...  # 90,000 entries (300×300)
}
```

### Output Data

```
HDF5 File (updated)
└── units/
    └── {unit_id}/
        └── spike_times_sectioned/
            └── moving_h_bar_s5_d8_3x/
                ├── full_spike_times          # UNCHANGED
                └── direction_section/
                    ├── 0/                    # Direction 0°
                    │   ├── trials/
                    │   │   ├── 0             # int64[] spike samples
                    │   │   ├── 1             # int64[] spike samples
                    │   │   └── 2             # int64[] spike samples
                    │   └── section_bounds    # int64[3,2] [start,end] per trial
                    ├── 45/                   # Direction 45°
                    │   └── ...
                    ├── 90/
                    │   └── ...
                    └── ... (315)
```

