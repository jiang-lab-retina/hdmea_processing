# How Preferred Direction Is Calculated

## Overview

The `preferred_direction` feature represents the angular direction of stimulus
motion that elicits the strongest response from a retinal ganglion cell. It is
computed from firing-rate traces recorded during a moving horizontal bar
stimulus (`moving_h_bar_s5_d8_3x`) presented at 8 directions in 45-degree
steps, after applying a retinal angle correction to account for the physical
orientation of each recording relative to the retina.

---

## Pipeline Stages

### Stage 1: Extract firing-rate traces from HDF5

**Source file:** `dataframe_compare/pipeline_compare.py` (function
`process_h5_files`)
**Input:** Spike-sorted HDF5 files (`{dataset_id}.h5`)

Each HDF5 file stores per-unit data under
`units/{unit_id}/spike_times_sectioned/moving_h_bar_s5_d8_3x/direction_section/`.
Within that group, each direction (e.g., `0`, `45`, `90`, ..., `315`) contains:

- `section_bounds` -- start/end frame indices for each repetition
- `trials/{rep_idx}` -- spike times within each repetition

For each unit, direction, and repetition the pipeline:

1. Extracts the frame timestamps spanning that section.
2. Bins spike times into inter-frame intervals.
3. Converts spike counts to instantaneous firing rate
   ($\text{FR} = \text{count} / \Delta t$).
4. Stores the resulting 1-D trace as
   `moving_h_bar_s5_d8_3x_{direction}_{rep_idx}` (e.g.,
   `moving_h_bar_s5_d8_3x_90_0` for direction 90, repetition 0).

### Stage 2: Reshape trials into array columns

**Source file:** `dataframe_phase/load_traces/pipeline_firing_rate.py`
(functions `parse_column_groups`, `reshape_to_movies`)

Individual trial columns for each direction are grouped and stacked into a
single array column per direction:

- `moving_h_bar_s5_d8_3x_0` contains a `(3, T)` array (3 trials, T time bins)
- `moving_h_bar_s5_d8_3x_45` similarly, and so on for all 8 directions.

This grouping uses the `MOVING_BAR_PREFIX = "moving_h_bar_s5_d8_3x"` to
identify direction columns, splits on the last `_` to separate the trial
index, and groups by `{prefix}_{direction}`.

### Stage 3: Load angle correction from HDF5

**Source file:** `dataframe_compare/pipeline_compare.py` (function
`load_hdf5_features`)
**Config:** `compare_config.py`

```python
FEATURE_PATHS = {
    "angle_correction_applied":
        "features/ap_tracking/soma_polar_coordinates/angle_correction_applied",
    ...
}
```

`angle_correction_applied` is a scalar (in degrees) stored per recording in the
HDF5 under `features/ap_tracking/soma_polar_coordinates/`. It represents the
angular offset between the stimulus coordinate system and the retinal
coordinate system for that particular recording, derived from soma polar
coordinate tracking.

### Stage 4: Remap to corrected direction columns

**Source file:** `dataframe_phase/extract_feature/extract_feature_dsgc.py`
(functions `compute_corrected_angle`, `remap_direction_columns`)

The raw stimulus directions (0, 45, ..., 315) are shifted by the
recording-specific correction angle to produce retina-referenced direction
labels:

```python
corrected_angle = round((raw_angle + angle_correction_applied) / 45) * 45
corrected_angle = corrected_angle % 360
```

Each raw column `moving_h_bar_s5_d8_3x_{raw_angle}` is reassigned to a
corrected column `corrected_moving_h_bar_s5_d8_3x_{corrected_angle:03d}`.

This ensures that, for example, a bar moving "rightward" on the stimulus screen
maps to the same retinal direction label across all recordings regardless of
how the tissue was oriented. The rounding to the nearest 45 degrees keeps the
8-bin structure.

If `angle_correction_applied` is NaN (correction unavailable), all corrected
columns and downstream features (`dsi`, `osi`, `preferred_direction`) are set
to NaN.

### Stage 5: Compute DSI, OSI, and preferred direction

**Source file:** `dataframe_phase/extract_feature/extract_feature_dsgc.py`
(functions `process_unit`, `calculate_direction_index`,
`calculate_orientation_index`, `compute_permutation_p_value`)

For each unit with valid corrected columns:

1. **Total firing rate per trial per direction.** For each of the 8 corrected
   directions, the trial-level firing-rate traces are summed over time to yield
   a single scalar per trial.

2. **Mean response per direction.** The 3 trial scalars are averaged, producing
   an 8-element vector $R = [R_0, R_{45}, \ldots, R_{315}]$.

3. **Direction Selectivity Index (DSI).** Computed via the vector-sum method:

$$\text{DSI} = \frac{\left|\sum_i R_i \, e^{j\theta_i}\right|}{\max\left(|R_i|\right)}$$

   where $\theta_i$ are the 8 direction angles in radians.

4. **Preferred direction.** The angle of the complex vector sum:

$$\theta_{\text{pref}} = \arg\!\left(\sum_i R_i \, e^{j\theta_i}\right) \mod 360^\circ$$

   This is a continuous angle (not restricted to the 8 stimulus directions).

5. **Orientation Selectivity Index (OSI).** Uses doubled angles to collapse
   opposite directions:

$$\text{OSI} = \frac{\left|\sum_i R_i \, e^{2j\theta_i}\right|}{\sum_i R_i}$$

6. **Permutation p-values** (`ds_p_value`, `os_p_value`). The 24 individual
   trial values ($8 \text{ directions} \times 3 \text{ trials}$) are shuffled
   across direction labels 2000 times. For each permutation, the DSI (or OSI)
   is recomputed from the shuffled mean-per-direction vector. The p-value is:

$$p = 1 - \Phi\!\left(\frac{\text{DSI}_{\text{real}} - \mu_{\text{shuffled}}}{\sigma_{\text{shuffled}}}\right)$$

   where $\Phi$ is the standard normal CDF.

### Stage 6: Prefix and merge into compared dataframe

**Source file:** `dataframe_compare/pipeline_compare.py`

The features computed above are added to the before- and after-blocker
DataFrames. When the two are merged into the final
`compared_dataframe_v2_labeled_spatial.parquet`, every column receives a
`before_` or `after_` prefix:

| Column | Description |
|--------|-------------|
| `before_preferred_direction` | Preferred direction (control, corrected) |
| `after_preferred_direction` | Preferred direction (after blocker, corrected) |
| `before_dsi` / `after_dsi` | Direction Selectivity Index |
| `before_osi` / `after_osi` | Orientation Selectivity Index |
| `before_ds_p_value` / `after_ds_p_value` | Permutation p-value for DSI |
| `before_os_p_value` / `after_os_p_value` | Permutation p-value for OSI |
| `before_corrected_moving_h_bar_s5_d8_3x_000` ... `_315` | 8 corrected direction response arrays |
| `before_angle_correction_applied` | The correction angle itself |

---

## Key Source Files

| File | Role |
|------|------|
| `dataframe_compare/pipeline_compare.py` | Orchestrates the full pipeline: HDF5 loading, feature extraction, merging |
| `dataframe_compare/compare_config.py` | Configuration: HDF5 feature paths, movie prefixes |
| `dataframe_phase/extract_feature/extract_feature_dsgc.py` | Core DSI/OSI/preferred direction computation and angle correction |
| `dataframe_phase/load_traces/pipeline_firing_rate.py` | Firing-rate extraction and trial grouping from HDF5 |
