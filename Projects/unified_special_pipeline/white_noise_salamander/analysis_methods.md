# Mathematical Methods: RF Geometry Analysis

This document describes the mathematics behind each analysis step in the
white-noise STA pipeline and the RF geometry comparison.

---

## 1. Spike-Triggered Average (STA)

Given a stimulus movie $S(t, x, y)$ and a set of spike times
$\{t_1, t_2, \ldots, t_N\}$, the STA is the average stimulus preceding
each spike:

$$\text{STA}(\tau, x, y) = \frac{1}{N} \sum_{i=1}^{N} S(t_i + \tau,\; x,\; y)$$

where $\tau \in [\tau_{\min}, \tau_{\max})$ is the time lag relative to the
spike. In this pipeline $\tau_{\min} = -60$ frames and $\tau_{\max} = 0$,
so the STA captures the 60 stimulus frames immediately before each spike.

The result is a 3D array of shape $(L, H, W)$ where $L = |\tau_{\max} - \tau_{\min}|$
is the number of lag frames and $(H, W)$ is the stimulus spatial resolution
(e.g. 15 x 15 pixels).

### Section time and frame alignment

Spikes are restricted to a valid stimulus window defined by
`section_time_frame_num = (184, inf)`. Frame 184 corresponds to the onset of
the stimulus presentation after a 184-frame pre-stimulus period. An internal
offset of `PRE_MARGIN_FRAME_NUM = 60` frames is handled by the `compute_sta`
function to ensure correct alignment between spike acquisition samples and
stimulus frame indices.

---

## 2. STA Preprocessing

Before fitting, the raw STA undergoes several preprocessing steps:

1. **Baseline subtraction**: The mean of frames 0-10 (earliest lag frames,
   far from the spike) is subtracted to remove the DC component:
   $$\text{STA}'(\tau, x, y) = \text{STA}(\tau, x, y) - \frac{1}{10}\sum_{\tau=0}^{9} \text{STA}(\tau, x, y)$$

2. **Zero padding**: 5 pixels of zeros are added around the spatial border to
   prevent edge effects during fitting.

3. **Gaussian blur**: A spatial Gaussian filter ($\sigma = 1.5$ px) smooths
   each frame to reduce pixel noise.

4. **Temporal smoothing**: Used in the robust extreme-map computation to reduce
   temporal noise when identifying ON and OFF regions.

---

## 3. Difference Map and Center Finding

The **difference map** (diff_map) summarizes the spatial structure of the RF
by collapsing the temporal dimension. For each pixel, it computes the
difference between the maximum and minimum response across the temporally
smoothed STA:

$$D(x, y) = \max_\tau \tilde{S}(\tau, x, y) - \min_\tau \tilde{S}(\tau, x, y)$$

where $\tilde{S}$ is the temporally smoothed STA. The RF center is located at
the pixel with the largest difference:

$$(x_c, y_c) = \arg\max_{x,y} D(x, y)$$

The **peak frame** is the time lag with the strongest overall spatial response:

$$\tau^* = \arg\max_\tau \max_{x,y} |\text{STA}'(\tau, x, y)|$$

---

## 4. RF Size Estimation

A binary mask is created from the diff_map by thresholding at a fraction of
its maximum:

$$M(x, y) = \begin{cases} 1 & \text{if } D(x, y) \geq f \cdot \max(D) \\ 0 & \text{otherwise} \end{cases}$$

where $f = 0.5$ (the `threshold_fraction`). From this mask:

- **Area** $A = \sum_{x,y} M(x,y)$ (in pixels)
- **Size** along each axis: bounding box width and height of $M$
- **Equivalent diameter**: $d = 2\sqrt{A / \pi}$

---

## 5. 2D Gaussian Fit

A rotated 2D Gaussian is fitted to the diff_map by nonlinear least squares
(Levenberg-Marquardt):

$$G(x, y) = B + A \exp\!\Bigl(-\bigl(a(x - x_0)^2 + 2b(x - x_0)(y - y_0) + c(y - y_0)^2\bigr)\Bigr)$$

where the rotation is parameterised by angle $\theta$:

$$a = \frac{\cos^2\theta}{2\sigma_x^2} + \frac{\sin^2\theta}{2\sigma_y^2}, \quad
b = \frac{-\sin 2\theta}{4\sigma_x^2} + \frac{\sin 2\theta}{4\sigma_y^2}, \quad
c = \frac{\sin^2\theta}{2\sigma_x^2} + \frac{\cos^2\theta}{2\sigma_y^2}$$

**Fitted parameters**: center $(x_0, y_0)$, widths $\sigma_x, \sigma_y$,
amplitude $A$, rotation $\theta$, offset $B$.

**Geometric mean sigma** (used for comparison):

$$\sigma_{\text{geo}} = \sqrt{\sigma_x \cdot \sigma_y}$$

**Goodness of fit** is measured by the coefficient of determination:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

This $R^2$ is the primary quality gate for unit inclusion.

---

## 6. Difference-of-Gaussians (DoG) Center-Surround Model

The DoG model captures the classic center-surround organisation of retinal
ganglion cells:

$$\text{DoG}(x, y) = B + A_{\text{exc}} \exp\!\left(-\frac{r^2}{2\sigma_{\text{exc}}^2}\right) - A_{\text{inh}} \exp\!\left(-\frac{r^2}{2\sigma_{\text{inh}}^2}\right)$$

where $r^2 = (x - x_0)^2 + (y - y_0)^2$.

**Parameters**: center $(x_0, y_0)$, excitatory amplitude $A_{\text{exc}}$
and width $\sigma_{\text{exc}}$ (center), inhibitory amplitude $A_{\text{inh}}$
and width $\sigma_{\text{inh}}$ (surround), offset $B$.

**Surround strength** quantifies the relative strength of surround inhibition:

$$S_{\text{surround}} = \frac{|A_{\text{inh}}|}{|A_{\text{exc}}|}$$

Values near 0 indicate a pure center response; values near 1 indicate strong
surround inhibition balancing the center.

---

## 7. ON/OFF Subfield Model

Separate 2D Gaussians (without rotation) are fitted independently to the
positive (ON) and negative (OFF) components of a robust extreme map:

$$G_{\text{ON}}(x, y) = A_{\text{on}} \exp\!\left(-\frac{(x-x_{\text{on}})^2}{2\sigma_{x,\text{on}}^2} - \frac{(y-y_{\text{on}})^2}{2\sigma_{y,\text{on}}^2}\right) + B_{\text{on}}$$

and analogously for $G_{\text{OFF}}$.

**ON/OFF ratio** measures the balance between subfields:

$$\text{ON/OFF ratio} = \frac{|A_{\text{on}}|}{|A_{\text{on}}| + |A_{\text{off}}|}$$

Values near 0.5 indicate balanced ON and OFF subfields; values near 0 or 1
indicate dominance of one polarity.

---

## 8. LNL (Linear-Nonlinear) Model

The LNL model describes spike generation as a cascade of a linear filter
followed by a static nonlinearity.

### 8.1 Generator signal

The generator signal is the inner product of the STA filter with the stimulus
at each time step:

$$g(t) = \sum_{\tau, x, y} \text{STA}(\tau, x, y) \cdot S(t + \tau, x, y)$$

This is normalised to zero mean and unit variance for comparison across cells.

### 8.2 Histogram-based nonlinearity

The nonlinearity $f(g)$ is estimated via Bayes' rule. The generator signal
is binned into $K = 50$ histogram bins. For each bin $k$:

$$f(g_k) = \frac{P(g_k \mid \text{spike})}{P(g_k)} \cdot \bar{r}$$

where $P(g_k \mid \text{spike})$ is the distribution of $g$ at spike times,
$P(g_k)$ is the unconditional distribution, and $\bar{r}$ is the mean firing
rate. This gives a nonparametric estimate of the firing rate as a function of
generator signal.

### 8.3 Parametric LNP fit

An exponential nonlinearity is fitted by maximum likelihood under a Poisson
spike model:

$$r(t) = \exp(a \cdot g(t) + b) \cdot \Delta t$$

where $a$ and $b$ are fitted by maximising the Poisson log-likelihood:

$$\mathcal{L} = \sum_{t} \left[ n(t) \log r(t) - r(t) \right]$$

with $n(t) \in \{0, 1\}$ indicating spikes. The null model uses a constant
rate $r_0 = N_{\text{spikes}} / T$.

### 8.4 Quality metrics

- **Bits per spike** (deviance explained):
  $$\text{bits/spike} = \frac{\mathcal{L}_{\text{model}} - \mathcal{L}_{\text{null}}}{N_{\text{spikes}} \cdot \ln 2}$$
  Measures the information gain of the LNL model over a homogeneous Poisson
  process, in bits per spike.

- **$R^2$** of the LNL model:
  $$R^2 = 1 - \frac{\mathcal{L}_{\text{null}} - \mathcal{L}_{\text{model}}}{\mathcal{L}_{\text{null}}}$$

- **Rectification index**: Fraction of the nonlinearity curve $f(g)$ that is
  above the mean firing rate. Values near 1 indicate strong rectification
  (cell only fires for positive generator values).

- **Nonlinearity index**: Normalised variance of $f(g)$ relative to a flat
  (linear) response, capturing how much the nonlinearity deviates from a
  constant.

---

## 9. Statistical Comparison

### Quality gating

Units are included only if their 2D Gaussian fit $R^2$ exceeds a threshold.
The analysis is repeated at three thresholds (0.5, 0.7, 0.9) to assess
robustness.

### Group comparison

For each metric, the two age groups (young larval vs old larval) are compared
using the **Mann-Whitney U test** (two-sided), a nonparametric rank-based test
that does not assume normality:

$$H_0: P(X > Y) = P(Y > X)$$

Significance levels:

| Symbol | Criterion |
|---|---|
| `***` | $p < 0.001$ |
| `**` | $p < 0.01$ |
| `*` | $p < 0.05$ |
| `n.s.` | $p \geq 0.05$ |

### Bar chart display

Bar charts show the **mean** of each metric per group, with error bars
representing the **standard error of the mean** (SEM):

$$\text{SEM} = \frac{s}{\sqrt{n}}$$

where $s$ is the sample standard deviation and $n$ is the number of units.
