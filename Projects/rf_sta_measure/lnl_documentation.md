# Linear-Nonlinear (LNL) Model Fitting Documentation

## Overview

The Linear-Nonlinear (LN) model is a standard framework for characterizing neural encoding in sensory neurons, particularly retinal ganglion cells (RGCs). It describes how a neuron transforms visual stimuli into spike responses through two stages:

1. **Linear Stage**: The stimulus is filtered by a linear spatiotemporal filter (the STA)
2. **Nonlinear Stage**: The filter output passes through a static nonlinearity to produce firing rate

$$
r(t) = f\left( \sum_{\tau, x, y} \text{STA}[\tau, x, y] \cdot s_{t-\tau}[x, y] \right) = f(g_t)
$$

where:
- $s_{t}[x, y]$ is the stimulus at time $t$ and spatial location $(x, y)$
- $\text{STA}[\tau, x, y]$ is the spike-triggered average (linear filter)
- $g_t$ is the **generator signal** (output of linear filter)
- $f(\cdot)$ is the static **nonlinearity**
- $r(t)$ is the predicted firing rate

---

## Computational Pipeline

### Step 1: Compute Generator Signal

The generator signal $g_t$ is computed by convolving the STA with the stimulus:

```python
g_t = sum_{tau,x,y} (STA[tau,x,y] - STA_mean) * (s_{t-tau}[x,y] - s_mean)
```

**Critical**: Both STA and stimulus are **mean-subtracted** before computing the dot product. For binary noise stimuli (0/255), the raw STA has mean ~127.5, which would create a large DC component with no predictive power if not removed.

**Implementation**: `rf_sta_measure.compute_generator_signal()`

### Step 2: Estimate Histogram Nonlinearity (Bayes Rule)

The nonlinearity $f(g)$ is estimated non-parametrically using Bayes' rule:

$$
P(\text{spike} | g) = \frac{P(g | \text{spike}) \cdot P(\text{spike})}{P(g)}
$$

**Steps**:
1. Bin the generator signal values into 50 bins
2. Compute $P(g)$: histogram of all generator signal values
3. Compute $P(g | \text{spike})$: histogram of generator signal values at spike times
4. Apply Bayes' rule to get $P(\text{spike} | g)$
5. Convert to firing rate: $r(g) = P(\text{spike} | g) / dt$

**Implementation**: `rf_sta_measure.estimate_histogram_nonlinearity()`

### Step 3: Fit Parametric LNP Model

A parametric exponential nonlinearity is fit via Poisson maximum likelihood:

$$
f(g) = \exp(b + a \cdot g)
$$

**Optimization**: Maximize Poisson log-likelihood:

$$
\log L = \sum_t \left[ y_t \log(\lambda_t) - \lambda_t \cdot dt \right]
$$

where $\lambda_t = \exp(b + a \cdot g_t)$ is the predicted firing rate and $y_t$ is the spike count at frame $t$.

**Normalization**: The generator signal is z-scored before optimization for numerical stability. Parameters are then transformed back to original scale.

**Implementation**: `rf_sta_measure.fit_lnp_parametric()`

### Step 4: Compute Quality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `bits_per_spike` | $(LL - LL_{null}) / (n_{spikes} \cdot \ln 2)$ | Information gain over constant-rate model |
| `a_norm` | Gain in normalized space | Effect of 1 std change in $g$ on log-rate |
| `r_squared` | $\text{Corr}(\hat{r}, y)^2$ | Frame-by-frame correlation (typically low for sparse data) |

**Implementation**: `rf_sta_measure.fit_lnl_model()`

---

## Data Stored in HDF5

### Location
```
units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_geometry/lnl/
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `a` | float64 | Gain parameter (original scale, very small) |
| `b` | float64 | Baseline parameter (log of baseline rate) |
| `a_norm` | float64 | **Normalized gain** (interpretable: effect per std of g) |
| `bits_per_spike` | float64 | **Information gain** over null model (bits) |
| `r_squared` | float64 | Correlation-based R² (low for sparse spike trains) |
| `rectification_index` | float64 | **ON/OFF asymmetry** (-1=OFF, 0=symmetric, +1=ON) |
| `nonlinearity_index` | float64 | **Curvature measure** (0=linear, 1=highly nonlinear) |
| `threshold_g` | float64 | Generator signal threshold (in std units) |
| `log_likelihood` | float64 | Fitted model log-likelihood |
| `null_log_likelihood` | float64 | Constant-rate model log-likelihood |
| `n_frames` | int64 | Number of frames used for fitting |
| `n_spikes` | int64 | Number of spikes in valid range |
| `g_bin_centers` | float32[50] | Generator signal bin centers |
| `rate_vs_g` | float32[50] | Firing rate at each bin (Hz) |

---

## Interpretation Guide

### bits_per_spike (Primary Metric)

| Value | Interpretation |
|-------|----------------|
| < 0 | Model is worse than null (fitting failed or very noisy) |
| 0 - 0.3 | Poor fit, stimulus has weak influence |
| 0.3 - 0.7 | Moderate fit, typical for many RGCs |
| 0.7 - 1.5 | Good fit, strong stimulus modulation |
| > 1.5 | Excellent fit, highly stimulus-driven |

### a_norm (Normalized Gain)

| Value | Interpretation |
|-------|----------------|
| ~0 | Generator signal has no effect on firing rate |
| 0.5 - 1.0 | Moderate modulation (typical) |
| > 1.0 | Strong modulation by stimulus |
| < 0 | Inverted response (unusual) |

### Nonlinearity Index (NL)

Measures how curved (nonlinear) the input-output function is:

$$
\text{NL} = 1 - R^2_{\text{linear}}
$$

where $R^2_{\text{linear}}$ is the R² of a linear fit to the histogram nonlinearity.

| Value | Interpretation |
|-------|----------------|
| 0 - 0.1 | Nearly linear response |
| 0.1 - 0.3 | Mild nonlinearity |
| 0.3 - 0.6 | Moderate nonlinearity (typical) |
| > 0.6 | Strong nonlinearity (highly rectifying) |

### Rectification Index (RI)

Measures asymmetry between responses to positive vs negative generator signals:

$$
\text{RI} = \frac{r_{g>0} - r_{g<0}}{r_{g>0} + r_{g<0}}
$$

| Value | Interpretation |
|-------|----------------|
| -1 to -0.2 | OFF-type cell (responds to decrements) |
| -0.2 to +0.2 | Symmetric response |
| +0.2 to +1 | ON-type cell (responds to increments) |

### Threshold (threshold_g)

The generator signal value (in std units) at which the firing rate crosses the mean rate. Indicates the "activation threshold" of the neuron.

### Why R² is Low

The frame-by-frame R² is expected to be very low (< 0.1) for sparse spike trains because:
- Spike counts are mostly 0s with occasional 1s
- Even a perfect model cannot predict exact spike timing
- Use `bits_per_spike` as the primary quality metric instead

---

## Visualization

The visualization script `viz_lnl.py` generates:

1. **lnl_population_summary.png**: Distribution of metrics across all units
2. **lnl_best_fits.png**: Top 12 units by bits/spike
3. **lnl_best_vs_worst.png**: Comparison of best vs worst fits
4. **lnl_random_examples.png**: Random sample for unbiased view

### Normalization in Plots

- **X-axis**: Generator signal is z-scored (mean=0, std=1)
- **Y-axis**: Firing rate is normalized to mean (1.0 = mean rate)
- **Filtering**: Bins outside 10th-90th percentile are excluded (sparse data)

---

## Code Files

| File | Description |
|------|-------------|
| `rf_sta_measure.py` | Core LNL fitting functions |
| `rf_session.py` | Session-based workflow (loads data, runs fitting, saves to HDF5) |
| `viz_lnl.py` | Visualization and validation plots |
| `geometry.py` | Pipeline step wrapper (calls rf_session functions) |

### Key Functions

```python
# Compute generator signal from STA and movie
g_all, first_valid, last_valid = compute_generator_signal(sta, movie, cover_range)

# Estimate histogram-based nonlinearity
g_bins, rate_vs_g = estimate_histogram_nonlinearity(g_all, spike_frames, first_valid, dt)

# Fit parametric LNP and compute metrics
lnl_fit = fit_lnl_model(sta, movie, spike_frames, cover_range, frame_rate)
```

---

## References

1. Chichilnisky, E. J. (2001). A simple white noise analysis of neuronal light responses. *Network: Computation in Neural Systems*, 12(2), 199-213.

2. Pillow, J. W. (2007). Likelihood-based approaches to modeling the neural code. *Bayesian Brain*, 53-70.

3. Schwartz, O., et al. (2006). Spike-triggered neural characterization. *Journal of Vision*, 6(4), 13.

---

## Example Usage

```python
from rf_sta_measure import fit_lnl_model

# Fit LNL model for a unit
lnl_fit = fit_lnl_model(
    sta=sta_data,           # Shape: (60, 15, 15)
    movie_array=movie,       # Shape: (10800, 15, 15)
    spike_frames=spikes,     # Spike frame indices
    cover_range=(-60, 0),    # 60 frames before spike
    frame_rate=15.0,         # Hz
)

# Access results
print(f"Bits/spike: {lnl_fit.deviance_explained:.3f}")
print(f"Gain (normalized): {lnl_fit.a_norm:.3f}")
print(f"Baseline log-rate: {lnl_fit.b:.3f}")
```

