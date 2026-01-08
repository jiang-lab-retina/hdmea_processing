# Linear and Nonlinear Fitting for Retinal Ganglion Cell STA (Dense Noise)

This note describes a practical workflow to go from a **spike-triggered average (STA)** to:

- a **linear** spatiotemporal filter estimate, and
- a **nonlinear** spike-rate mapping (LN / LNP / GLM-style) fit,

with an emphasis on what you can (and cannot) fit given the data currently available in this repo, and how to integrate a robust fit into the existing pipeline.

## Key references (open links)

- **White-noise / STA analysis**: Chichilnisky (2001), *A simple white noise analysis of neuronal light responses* (PDF)  
  `https://sites.stat.columbia.edu/liam/teaching/neurostat-fall14/papers/Chichilnisky-2001.pdf`

- **Retina population GLM**: Pillow, Shlens, Paninski et al. (2008), *Spatio-temporal correlations and visual signalling in a complete neuronal population* (Nature)  
  `https://www.nature.com/articles/nature07140`  
  (high-level abstract/notes) `https://sites.stat.columbia.edu/liam/research/abstracts/pillow-nature08.html`

- **LNP / point-process cascade MLE**: Paninski (2004), *Maximum likelihood estimation of cascade point-process neural encoding models* (PDF)  
  `https://www.cns.nyu.edu/pub/eero/paninski04b.pdf`

- **STA & STC estimation behavior**: Paninski (2003), *Convergence properties of three spike-triggered analysis techniques* (PDF)  
  `https://www.cns.nyu.edu/pub/eero/paninski03c-reprint.pdf`

- **STC overview & LN context** (review/tutorial style):  
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC3558678/`

## What “linear fit” means for STA

### Stimulus and response notation

- Stimulus frames: $s_t \in \mathbb{R}^{H \times W}$ (here $H=W=15$ for dense noise).
- Spike count per frame bin: $y_t \in \{0,1,2,\dots\}$ (often 0/1 if bins are small).
- Spatiotemporal linear filter: $k \in \mathbb{R}^{L \times H \times W}$ where $L$ is the time window length (e.g., 60 frames).

The **linear drive** (generator signal) at time $t$:

$$g_t=\sum_{\tau=0}^{L-1}\sum_{x,y} k[\tau,y,x]\; s_{t-\tau}[y,x].$$

### STA as a linear filter estimate

Under standard assumptions (white stimulus, weakly nonlinear spike generation), the STA is proportional to the “best” linear filter:

$$\mathrm{STA}(\tau,x,y)=\mathbb{E}[s_{t-\tau}(x,y)\mid \text{spike at }t] - \mathbb{E}[s].$$

For **Gaussian white noise**, $k$ is (up to a scale factor) the STA.  
For **binary/checkerboard noise** and/or **correlated stimulus**, you often need:

- mean subtraction,
- stimulus covariance correction (“whitening”), or
- regularization (smoothness / low-rank / separability constraints).

## What “nonlinear fit” means (LN / LNP / GLM)

### LN model

An LN model maps the linear drive $g_t$ through a static nonlinearity $f(\cdot)$:

$$\hat{r}_t=f(g_t)$$

where $\hat{r}_t$ is predicted firing rate (or spike probability per bin).

Common choices for $f$:

- **Exponential (LNP / Poisson)**: $f(g)=\exp(b+g)$
- **Softplus**: $f(g)=\log(1+\exp(b+g))$
- **Logistic (Bernoulli)**: $f(g)=\sigma(b+g)=\frac{1}{1+\exp(-(b+g))}$
- **Nonparametric**: histogram / spline fit of $\mathbb{E}[y\mid g]$

### LNP / GLM fitting objective

For Poisson spiking in frame bins of width $\Delta t$:

$$\lambda_t=f(g_t),\qquad y_t\sim \mathrm{Poisson}(\lambda_t \Delta t).$$

The (negative) log-likelihood (up to constants) is:

$$\mathcal{L}=-\sum_t \big[y_t\log(\lambda_t)-\lambda_t\Delta t\big].$$

“GLM” usually adds extra terms (e.g., spike history filter, coupling filters across neurons). In a single-cell setting you can start with LNP and later extend.

## Practical fitting workflow (dense-noise STA → LN)

### Step A — Estimate a stable linear filter

Options (in increasing sophistication):

- **Use raw STA** (fast, but noisy).
- **Denoise STA** by temporal smoothing and spatial smoothing (acts like regularization).
- **Separable fit**: $k[\tau,y,x]\approx k_t[\tau]\cdot k_s[y,x]$
  - $k_s$ from 2D Gaussian fit on a summary map (e.g., max-min map).
  - $k_t$ from time course at the RF center (or projection onto $k_s$).

### Step B — Compute generator signal $g_t$

You need the **stimulus movie** and the **frame-aligned spikes**.

- For each frame time $t$, extract a stimulus window of length $L$.
- Compute dot product with the filter $k$.

Implementation notes:

- A full 3D dot product is expensive; separable filters make it much faster:
  - Spatial projection: $u_t=\langle k_s, s_t\rangle$
  - Temporal filtering: $g_t=\sum_{\tau} k_t[\tau]\; u_{t-\tau}$

### Step C — Fit the nonlinearity $f$

Two good starters:

- **Histogram nonlinearity**:
  - bin $g_t$ into quantiles
  - compute average spike count per bin
  - optionally smooth the curve (monotone spline)
- **Parametric LNP** (recommended for robust exportable parameters):
  - choose $f(g)=\exp(b + a g)$ or softplus
  - fit $a,b$ by maximizing Poisson likelihood

### Step D — Validate

Use held-out frames (or trials):

- correlation between predicted and observed PSTH
- Poisson deviance explained
- log-likelihood improvement over baseline

## How this fits into the current repo

### What you already have

1. **STA computation** in `src/hdmea/features/sta.py` (`compute_sta` in session mode):
   - re-loads the dense-noise stimulus movie
   - converts sectioned spike times to frame indices
   - stores `sta_{movie_name}` in session

2. **Geometry extraction** in `Projects/rf_sta_measure/rf_sta_measure.py`:
   - finds robust center, fits 2D Gaussian/DoG/ONOFF on summary maps
   - stores `sta_geometry` under `features/sta_perfect_dense_noise_15x15_15hz_r42_3min/`

### What you need for true LN/LNP fitting

To fit $f(g)$, you must compute $g_t$, which requires:

- the stimulus movie frames $s_t$ (available by re-loading the movie), and
- spike counts per frame (available from `spike_times_sectioned` + frame timestamps).

So LN fitting can be integrated without changing the data model, by following the same approach as `compute_sta`:

- find the noise movie name from `session.stimulus["section_time"]`
- load the movie via the same stimulus loader
- convert spike samples to spike frames

### Proposed integration plan (minimal, consistent with existing code)

Add a new feature computation function (new module) that runs **after `compute_sta_step`** and writes under the same feature name as your RF geometry:

- **New module**: `src/hdmea/features/sta_ln.py` (or `Projects/rf_sta_measure/sta_ln.py` if you want it project-scoped)
- **New pipeline step**: `Projects/unified_pipeline/steps/sta_ln.py` (wrapper similar to Step 7 geometry)

Output path (per unit):

- `units/{unit_id}/features/sta_perfect_dense_noise_15x15_15hz_r42_3min/sta_ln/`
  - `linear_filter` (either full $L\times15\times15$ or separable pieces)
  - `generator_signal` (optional; large)
  - `nonlinearity` parameters (e.g., `a`, `b`, model type)
  - fit metrics (`loglik`, deviance explained, etc.)

### Recommended “fast + robust” model for this codebase

Given you already compute and save:

- `sta_geometry/gaussian_fit/center_x, center_y`
- `sta_geometry/sta_time_course` (recently added)

a very practical LN approximation is:

1. **Spatial filter**: 2D Gaussian evaluated on the 15×15 grid (normalized).
2. **Temporal filter**: `sta_time_course` (normalized / sign-adjusted).
3. **Separable spatiotemporal filter**: outer product of temporal × spatial.
4. **Nonlinearity**: Poisson LNP with $f(g)=\exp(b+a g)$ fit by likelihood.

This aligns with your current geometry workflow and avoids expensive full 3D projection.

## Notes on coordinate conventions (important in this repo)

Your STA arrays follow:

- `sta[t, row, col]` where `row=y` and `col=x`.

So when projecting stimulus onto a spatial filter:

$$u_t=\sum_{y,x} k_s[y,x]\; s_t[y,x].$$

and `center_x` corresponds to `col`, while `center_y` corresponds to `row`.

## Next steps (if you want me to implement LN fitting)

If you want, I can implement a first-pass LN fitter that:

- reuses the movie+spike alignment logic from `src/hdmea/features/sta.py`
- uses the separable filter described above
- fits a 2-parameter LNP nonlinearity (using `scipy.optimize`)
- exports results into each unit’s feature tree (same pattern as `rf_session.py`)


