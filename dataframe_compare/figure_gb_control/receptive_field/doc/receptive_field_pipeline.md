# Receptive Field Calculation Pipeline

This document describes how receptive fields (RFs) are computed, from raw spike data to the final parametric fits stored in the export HDF5 files.

## Pipeline Overview

```
Raw spikes + noise movie .npy
        |
        v
  1. STA computation           (src/hdmea/features/sta.py)
        |
        v
  2. STA preprocessing         (Projects/rf_sta_measure/rf_sta_measure.py)
        |
        v
  3. Center detection
        |
        v
  4. Spatial model fitting     (Gaussian, DoG, ON/OFF)
        |
        v
  5. LNL model fitting
        |
        v
  6. Save to HDF5              (Projects/rf_sta_measure/rf_session.py)
        |
        v
  7. Dataframe export          (dataframe_compare/compare_config.py)
```

---

## Stage 1: Spike-Triggered Average (STA)

**Source:** `src/hdmea/features/sta.py` -- `compute_sta()`

### Input

| Item | Description |
|------|-------------|
| Spike times | Per-unit spike sample indices from `spike_times_sectioned/{movie}/trials_spike_times/0` |
| Noise movie | Binary dense noise stored as `{movie_name}.npy` with shape $(T, H, W)$, typically `uint8` (0 or 255) |
| Frame timestamps | Array mapping sample indices to frame numbers (`stimulus/frame_time/default`) |
| `cover_range` | Frame window relative to each spike, default $(-60, 0)$ -- i.e. 60 frames before the spike |

### Computation

For each spike at frame $f$, extract the stimulus window $\mathbf{S}[f + \text{start} : f + \text{end}]$ with shape $(L, H, W)$ where $L = |\text{cover\_range}|$. The STA is the simple mean over all valid spike windows:

$$\text{STA}[\tau, y, x] = \frac{1}{N} \sum_{i=1}^{N} \mathbf{S}[f_i + \tau,\; y,\; x]$$

where $N$ is the number of spikes whose window falls entirely within the movie bounds.

### Output

3D array of shape $(L, H, W)$, e.g. $(60, 15, 15)$, stored in the HDF5 at:

```
units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/data
```

---

## Stage 2: STA Preprocessing

**Source:** `rf_sta_measure.py` -- `preprocess_sta()`

The raw STA undergoes four sequential steps. The order matters:

### Step 2a: Baseline Subtraction

Compute the per-pixel mean of the first $n$ frames (default $n = 10$) and subtract it from every frame:

$$\text{baseline}[y, x] = \frac{1}{n} \sum_{t=0}^{n-1} \text{STA}[t, y, x]$$

$$\text{STA}'[t, y, x] = \text{STA}[t, y, x] - \text{baseline}[y, x]$$

This removes the stimulus mean (approximately 127.5 for binary 0/255 noise) so that the STA becomes zero-centered.

### Step 2b: Spatial Padding

Add a border of $p = 5$ pixels on all sides, filled with zeros (the data is already baseline-subtracted). This prevents edge artifacts during Gaussian blurring and fitting. The padded shape becomes $(L,\; H + 2p,\; W + 2p)$.

### Step 2c: 2D Gaussian Blur

Apply a 2D Gaussian filter independently to each frame with $\sigma_{\text{spatial}} = 1.5$ pixels. This smooths spatial noise while preserving the RF structure.

### Step 2d: Temporal Smoothing

Apply a 1D Gaussian filter along the time axis with $\sigma_t = 2.0$ frames (mode = `nearest`). This smooths frame-to-frame noise in the temporal profile.

### Configuration Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `baseline_frames` | 10 | Frames 0..9 used for baseline mean |
| `padding` | 5 px | Zero-padding border |
| `gaussian_sigma` | 1.5 px | Spatial blur sigma |
| `sigma_t` | 2.0 frames | Temporal smoothing sigma |

---

## Stage 3: Center Detection

**Source:** `rf_sta_measure.py` -- `find_center_maxmin()`

The RF center is found using the **max-min difference map with temporal filtering**. For each pixel $(y, x)$, a Savitzky-Golay filter (window = 7, order = 3) is applied along the temporal dimension. The smoothed peak and trough values are found, and the difference map is:

$$D[y, x] = \text{peak}[y, x] - \text{trough}[y, x]$$

The pixel with the largest $D$ value is taken as the RF center. This approach is more robust to noise than the simpler "extreme absolute value" method because the Savitzky-Golay filter suppresses transient noise spikes.

An alternative center-detection function `find_center_extreme()` is also available. It finds the pixel with the largest absolute value across all frames (max of positive vs. most negative), but is not used in the default pipeline.

---

## Stage 4: Spatial Model Fitting

All fits are performed using `scipy.optimize.curve_fit` with bounded parameters. Two key constraints apply to every fit:

| Constraint | Value | Meaning |
|------------|-------|---------|
| `center_fit_radius` | 5 px | Fitted center must stay within 5 px of the detected center |
| `max_sigma` | 7.5 px | $\sigma$ cannot exceed 7.5 px (corresponding to a 15 px diameter) |

### 4a. 2D Elliptical Gaussian

**Input:** The max-min difference map $D[y, x]$.

**Model:**

$$G(x, y) = \text{offset} + A \exp\!\Bigl(-\bigl(a(x - x_0)^2 + 2b(x - x_0)(y - y_0) + c(y - y_0)^2\bigr)\Bigr)$$

where the coefficients encode rotation by angle $\theta$:

$$a = \frac{\cos^2\theta}{2\sigma_x^2} + \frac{\sin^2\theta}{2\sigma_y^2}, \quad
b = \frac{-\sin 2\theta}{4\sigma_x^2} + \frac{\sin 2\theta}{4\sigma_y^2}, \quad
c = \frac{\sin^2\theta}{2\sigma_x^2} + \frac{\cos^2\theta}{2\sigma_y^2}$$

**Parameters (7):** $A$, $x_0$, $y_0$, $\sigma_x$, $\sigma_y$, $\theta$, offset.

**Quality metric:** $R^2 = 1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$

**Output fields:** `center_x`, `center_y`, `sigma_x`, `sigma_y`, `amplitude`, `theta`, `offset`, `r_squared`.

### 4b. Difference of Gaussians (DoG) -- Center-Surround

**Input:** The same max-min difference map $D[y, x]$.

**Model:**

$$\text{DoG}(x, y) = \text{offset} + A_{\text{exc}} \exp\!\left(-\frac{r^2}{2\sigma_{\text{exc}}^2}\right) - A_{\text{inh}} \exp\!\left(-\frac{r^2}{2\sigma_{\text{inh}}^2}\right)$$

where $r^2 = (x - x_0)^2 + (y - y_0)^2$. The excitatory component models the RF center and the inhibitory component models the surround.

**Parameters (7):** $A_{\text{exc}}$, $A_{\text{inh}}$, $x_0$, $y_0$, $\sigma_{\text{exc}}$, $\sigma_{\text{inh}}$, offset.

**Bounds:** $A_{\text{exc}} \geq 0$, $A_{\text{inh}} \geq 0$, $\sigma_{\text{exc}} \geq 0.3$, $\sigma_{\text{inh}} \geq 0.5$.

**Output fields:** `center_x`, `center_y`, `sigma_exc`, `sigma_inh`, `amp_exc`, `amp_inh`, `offset`, `r_squared`.

### 4c. ON/OFF Separate Gaussians

**Input:** A **robust extreme map** -- for each pixel, the Savitzky-Golay-filtered peak or trough value with larger absolute magnitude (sign preserved).

**Procedure:**

1. Split the extreme map into positive (ON) and negative (OFF) components.
2. Fit a simple (non-rotated) 2D Gaussian to each component independently:

$$G_{\text{simple}}(x, y) = \text{offset} + A \exp\!\left(-\frac{(x - x_0)^2}{2\sigma_x^2} - \frac{(y - y_0)^2}{2\sigma_y^2}\right)$$

3. Each fit is centered on the peak of its respective component (not the global RF center).

**Parameters per component (6):** $A$, $x_0$, $y_0$, $\sigma_x$, $\sigma_y$, offset.

**Output fields:** `on_center_x`, `on_center_y`, `on_sigma_x`, `on_sigma_y`, `on_amplitude`, `on_r_squared`, and the corresponding `off_*` fields.

### 4d. RF Size Estimation

In addition to the parametric fits, the RF size is estimated directly from a thresholded connected component:

1. Compute max - min across frames 40..60 of the preprocessed STA.
2. Smooth with a small Gaussian ($\sigma = 0.5$).
3. Threshold at 50% of the peak value.
4. Find the connected component containing the RF center.
5. Measure bounding-box width/height, pixel area, and equivalent circular diameter:

$$d_{\text{eq}} = 2\sqrt{\text{area} / \pi}$$

---

## Stage 5: Linear-Nonlinear (LNL) Model

**Source:** `rf_sta_measure.py` -- `fit_lnl_model()`

The LNL model assesses how well the STA predicts the neuron's spiking. It requires the raw stimulus movie and spike times (not just the STA array).

### 5a. Generator Signal

The generator signal $g_t$ is the projection of the stimulus onto the STA filter at each frame:

$$g_t = \sum_{\tau, y, x} \bigl(\text{STA}[\tau, y, x] - \overline{\text{STA}}\bigr) \cdot \bigl(s_{t+\tau}[y, x] - \bar{s}\bigr)$$

Mean-subtraction of both STA and stimulus is critical -- for binary stimuli the raw STA mean is approximately 127.5 and would produce a dominant DC offset with no discriminative power.

### 5b. Histogram Nonlinearity (Bayes Method)

The empirical nonlinearity $f(g)$ is estimated via Bayes' rule:

$$P(\text{spike} \mid g) = \frac{P(g \mid \text{spike}) \cdot P(\text{spike})}{P(g)}$$

$$\lambda(g) = \frac{P(\text{spike} \mid g)}{\Delta t}$$

where $P(g)$ and $P(g \mid \text{spike})$ are estimated from histograms (50 bins), and $\Delta t = 1 / f_{\text{rate}}$. The resulting curve is smoothed with a 1D Gaussian ($\sigma = 1$).

### 5c. Parametric LNP Fit

A parametric exponential nonlinearity $\lambda(g) = \exp(b + a \cdot g)$ is fit by maximizing the Poisson log-likelihood:

$$\log \mathcal{L} = \sum_t \bigl[ y_t \log \lambda_t - \lambda_t \Delta t \bigr]$$

where $y_t$ is the spike count at frame $t$. Optimization uses L-BFGS-B on the z-scored generator signal for numerical stability. The gain parameter $a_{\text{norm}}$ (in normalized space) represents the effect of a 1-std change in $g$ on the log firing rate.

### 5d. Derived Metrics

| Metric | Definition |
|--------|------------|
| `bits_per_spike` | $(\log\mathcal{L} - \log\mathcal{L}_{\text{null}}) / (N_{\text{spikes}} \cdot \ln 2)$. Information gain over constant-rate null model. |
| `r_squared` | Squared Pearson correlation between predicted rate and observed spike counts. |
| `rectification_index` | $(r_+ - r_-) / (r_+ + r_-)$ where $r_\pm$ are mean rates for positive/negative $g$. Ranges from $-1$ (OFF) to $+1$ (ON), 0 = symmetric. |
| `nonlinearity_index` | $1 - R^2_{\text{linear}}$ of a linear fit to the histogram nonlinearity. 0 = perfectly linear, 1 = highly nonlinear. |
| `threshold_g` | Generator value (in std units) where firing rate crosses its mean. |

---

## Stage 6: HDF5 Storage

**Source:** `rf_session.py` -- `save_rf_geometry_to_hdf5()`

All fitted parameters are written under a structured group hierarchy:

```
units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/
    center_row          (scalar)
    center_col          (scalar)
    size_x              (scalar)
    size_y              (scalar)
    area                (scalar)
    equivalent_diameter (scalar)
    peak_frame          (scalar)
    sta_time_course     (1D array, length = n_frames)
    diff_map            (2D array, H x W)
    gaussian_fit/
        center_x, center_y, sigma_x, sigma_y, amplitude, theta, offset, r_squared
    DoG/
        center_x, center_y, sigma_exc, sigma_inh, amp_exc, amp_inh, offset, r_squared
    ONOFF_model/
        on_center_x, on_center_y, on_sigma_x, on_sigma_y, on_amplitude, on_r_squared
        off_center_x, off_center_y, off_sigma_x, off_sigma_y, off_amplitude, off_r_squared
    lnl/
        a, b, a_norm
        log_likelihood, null_log_likelihood, bits_per_spike, r_squared
        rectification_index, nonlinearity_index, threshold_g
        n_frames, n_spikes
        g_bin_centers     (1D array)
        rate_vs_g         (1D array)
```

All coordinates in the output are in the **original (unpadded) frame**. The padding offset is subtracted before saving.

A sentinel value of $-5$ for `center_row` / `center_col` indicates a failed fit (used by downstream code such as `dsgc_direction.get_cell_center`).

---

## Stage 7: Downstream Dataframe Integration

**Source:** `dataframe_compare/compare_config.py` -- `FEATURE_PATHS`

The `pipeline_compare.py` script reads a subset of these HDF5 fields into Parquet dataframes. The column-to-path mapping includes:

| Dataframe column | HDF5 path (relative to unit) |
|------------------|------------------------------|
| `gaussian_sigma_x` | `.../sta_geometry/gaussian_fit/sigma_x` |
| `gaussian_sigma_y` | `.../sta_geometry/gaussian_fit/sigma_y` |
| `gaussian_amp` | `.../sta_geometry/gaussian_fit/amplitude` |
| `gaussian_r2` | `.../sta_geometry/gaussian_fit/r_squared` |
| `dog_sigma_exc` | `.../sta_geometry/DoG/sigma_exc` |
| `dog_sigma_inh` | `.../sta_geometry/DoG/sigma_inh` |
| `dog_amp_exc` | `.../sta_geometry/DoG/amp_exc` |
| `dog_amp_inh` | `.../sta_geometry/DoG/amp_inh` |
| `dog_r2` | `.../sta_geometry/DoG/r_squared` |
| `lnl_bits_per_spike` | `.../sta_geometry/lnl/bits_per_spike` |
| `lnl_rectification_index` | `.../sta_geometry/lnl/rectification_index` |
| `lnl_nonlinearity_index` | `.../sta_geometry/lnl/nonlinearity_index` |

The dense noise movie itself is listed in `EXCLUDED_MOVIES` so it is not treated as a normal stimulus block for response extraction.

---

## Source File Summary

| File | Role |
|------|------|
| `src/hdmea/features/sta.py` | STA computation from raw spikes + noise movie |
| `Projects/rf_sta_measure/rf_sta_measure.py` | Preprocessing, center detection, Gaussian/DoG/ON-OFF/LNL fitting |
| `Projects/rf_sta_measure/rf_session.py` | Session-based batch workflow and HDF5 save |
| `Projects/rf_sta_measure/batch_rf.py` | Batch driver across multiple HDF5 files |
| `dataframe_compare/compare_config.py` | Maps HDF5 paths to dataframe columns |
