# Linear and Nonlinear Fitting for a Retinal Ganglion Cell STA (Spike-Triggered Average)

This note summarizes a standard workflow used in retinal ganglion cell (RGC) analysis to go from a **spike-triggered average (STA)** to a fitted **linear–nonlinear (LN)** or **linear–nonlinear–Poisson (LNP)** model. The emphasis is on:
- **Linear fit**: estimate a stimulus filter / receptive field (RF) from white-noise stimulation (STA or regression).
- **Nonlinear fit**: estimate the static nonlinearity that maps the linear filter output (“generator signal”) to firing rate or spike probability.

The canonical reference for the STA→LN pipeline in retina is Chichilnisky (2001), and the “static nonlinearity” estimation described there is widely reused.

---

## 1) Model assumptions and notation

### 1.1 LN / LNP model (single-filter case)

For a time-binned stimulus representation, an LN model assumes that spike probability (or expected spike count) depends on the stimulus only through a **one-dimensional projection**:

- **Stimulus vector** at time bin *t*:  
  \(\mathbf{s}_t \in \mathbb{R}^D\)  
  (often a concatenation of several preceding stimulus frames: a *spatio-temporal embedding*).

- **Linear filter** (receptive field in stimulus space):  
  \(\mathbf{k} \in \mathbb{R}^D\)

- **Generator signal** (scalar projection):  
  \(g_t = \mathbf{k}^\top \mathbf{s}_t\)

- **Static nonlinearity** mapping \(g_t\) to firing rate or spike probability:  
  \(\lambda_t = f(g_t)\)

A common choice is **LNP**: spikes are drawn from a Poisson process with rate \(\lambda_t\) (or Poisson spike counts per bin). A generalized linear model (GLM) is an extension that adds spike-history and coupling filters (useful in retina populations).

### 1.2 When does STA estimate the “true” linear filter?

The STA recovers the linear filter (up to a scaling factor) under stimulus symmetry assumptions (e.g., Gaussian white noise / radially symmetric distributions) and an LN/LNP-type encoding model. If the stimulus is strongly correlated (e.g., natural movies) or the cell depends on multiple stimulus dimensions, the STA can be biased or incomplete—then whitening, STC, MID, or GLM-based estimation is typically used.

---

## 2) Prepare the data

### 2.1 Inputs you need

1. **Stimulus movie / time series**  
   Example: spatio-temporal white-noise checkerboard shown at a fixed frame rate.

2. **Spike times** (or spike counts per time bin)

3. **Time alignment** between stimulus frames and spikes  
   - Choose a time bin \(\Delta t\) (often the stimulus frame duration).
   - Convert spikes into binned counts \(y_t\) (0/1 for spikes, or counts if multiple spikes can occur per bin).

### 2.2 Build a time-lagged stimulus design matrix

To capture temporal dependence, you build a “design matrix” \(\mathbf{X}\) where each row is the stimulus at time *t* and its history:

\[
\mathbf{x}_t = [\mathbf{s}_t,\; \mathbf{s}_{t-1},\; \dots,\; \mathbf{s}_{t-L+1}] \in \mathbb{R}^{D\cdot L}
\]

where \(L\) is the number of time lags (e.g., 10–30 frames).

**Practical tip:** For early time bins where history is unavailable, common practice is zero-padding.

---

## 3) Linear fit: estimate the spatio-temporal filter from STA (or regression)

### 3.1 Compute the STA

The STA is the average stimulus preceding a spike. In a binned representation (spike count \(f_t\) at time bin *t*):

\[
\mathrm{STA} = \mathbf{a} = \frac{\sum_{t=1}^T \mathbf{s}_t f_t}{\sum_{t=1}^T f_t}
\]

Often you subtract the **mean stimulus** \(\langle \mathbf{s} \rangle\) so the STA reflects deviations from mean luminance/contrast:

\[
\mathrm{STA} = \frac{1}{N_{\text{spike}}} \sum_{\text{spikes}} (\mathbf{s}_{t_{\text{spike}}} - \langle \mathbf{s} \rangle)
\]

**Key point:** Under standard white-noise assumptions, the STA is proportional to the true linear filter \(\mathbf{k}\) (scaling is absorbed into the nonlinearity).

### 3.2 STA as a “linear regression” estimator

If you assemble \(\mathbf{X}\) with time lags and have binned spike counts \(\mathbf{y}\), a simple identity is:

\[
\mathrm{STA} = \frac{\mathbf{X}^\top \mathbf{y}}{\sum_t y_t}
\]

This highlights STA as a special case of regression-based estimation when the stimulus is white/no correlations.

### 3.3 Optional: denoise / parameterize the STA (recommended)

Raw STA estimates can be noisy, especially with limited spikes. Common denoising / parameterization steps:

#### A) Fit a 2D Gaussian to the spatial RF (at peak latency)
1. Pick the time lag where the STA has maximal absolute amplitude (peak frame).
2. Fit an **elliptical 2D Gaussian** to the spatial STA slice to estimate RF center, size, and orientation.
3. Use the Gaussian fit to define an RF mask/ROI for visualization or to reduce dimensionality.

(Example usage in retina population work: Gaussian ellipses are often used to summarize RF centers.)

#### B) Fit a parametric temporal kernel
Once you have an RF center/ROI, extract the temporal STA at the center and fit a low-dimensional temporal basis (e.g., biphasic filter). This reduces noise and helps interpret latency and biphasic/monophasic structure.

#### C) Regularized regression / GLM (for correlated stimuli or many parameters)
If the stimulus has correlations or you want a maximum-likelihood fit of \(\mathbf{k}\), you can fit \(\mathbf{k}\) in an LNP/GLM framework with regularization (e.g., ridge, smoothness priors). This is especially helpful for natural stimuli.

---

## 4) Nonlinear fit: estimate the static nonlinearity

Once you have a linear filter \(\mathbf{k}\) (from STA or regression), compute the generator signal:

\[
g_t = \mathbf{k}^\top \mathbf{x}_t
\]

Then fit \(f(\cdot)\) in \(\lambda_t = f(g_t)\).

Two standard approaches are (A) **nonparametric histogram/Bayes**, and (B) **parametric curve fitting / likelihood**.

### 4.1 Nonparametric: “histogram” nonlinearity via Bayes’ rule

This is a classic way to estimate the LN nonlinearity after STA:

1. Compute \(g_t\) for **all** time bins → estimate the prior density \(p(g)\).
2. Compute \(g_t\) only at bins where spikes occurred (or weight by spike count) → estimate \(p(g\mid\text{spike})\).
3. Use Bayes’ rule to get spike probability given generator signal:

\[
p(\text{spike}\mid g) = \frac{p(g\mid\text{spike})\,p(\text{spike})}{p(g)}
\]

4. Convert to firing rate (if using time bins) via \(\lambda(g) \approx p(\text{spike}\mid g)/\Delta t\) for small \(\Delta t\).

**Implementation detail:** you estimate the densities with histograms or kernel density estimation (KDE), then smooth the resulting \(\lambda(g)\) curve.

### 4.2 Parametric: fit a smooth nonlinearity curve

A practical alternative is to pick a parametric form and fit it to the binned averages \(\mathbb{E}[y\mid g]\).

Common choices:
- **Cumulative normal (probit) / sigmoid-like**:  
  \(f(g) = \alpha\,\Phi(\beta g + \gamma)\)  
  where \(\Phi\) is the standard normal CDF.  
  Parameters interpret naturally as max rate (\(\alpha\)), gain (\(\beta\)), and bias/threshold (\(\gamma\)).

- **Logistic sigmoid**: \(f(g) = \alpha / (1 + \exp(-\beta(g-\theta)))\)

- **Exponential (LNP / Poisson GLM)**: \(\lambda_t = \exp(b + g_t)\)

### 4.3 Maximum-likelihood fit in an LNP / GLM framework

If you model spike counts \(y_t\) as Poisson with rate \(\lambda_t\), you can fit parameters by maximizing the log-likelihood:

\[
\log \mathcal{L} = \sum_t \left( y_t \log \lambda_t - \lambda_t \Delta t \right) + \text{const}
\]

For an exponential nonlinearity \(\lambda_t = \exp(b + \mathbf{k}^\top \mathbf{x}_t)\), this becomes a convex optimization problem (often solved with standard numerical optimizers). GLMs extend this by adding spike-history terms and (in population recordings) coupling terms.

---

## 5) Validation and diagnostics

### 5.1 Hold-out prediction
Split data into train/test:
- Fit \(\mathbf{k}\) and \(f\) on training data.
- Predict \(\hat\lambda_t\) on held-out stimulus and compare to observed spiking.

### 5.2 Metrics
Common metrics:
- Correlation between predicted and measured PSTH (if repeated trials exist).
- Log-likelihood on test data (Poisson log-likelihood for LNP/GLM).
- Time-resolved error (e.g., RMS error in binned spike counts vs predicted rate).

### 5.3 When STA fails or is insufficient
- **ON–OFF cells / symmetric nonlinearities**: A single STA can be near zero even if the cell is strongly stimulus-driven; spike-triggered covariance (STC) or multi-filter models are more appropriate.
- **Correlated stimuli**: STA is biased; use whitening (covariance correction) or likelihood-based GLM estimation.
- **Strong spike-history effects**: incorporate a post-spike filter (GLM) to model refractoriness/bursting.

---

## 6) Minimal Python-style pseudocode (STA → LN)

```python
# Inputs:
# stim[t]          : stimulus frame at time t (flattened pixels), shape (T, P)
# spikes[t]        : spike count in bin t, shape (T,)
# L                : number of time lags (frames) in the filter
# dt               : bin width (seconds)

# 1) Build time-lagged design matrix X, shape (T, P*L)
X = build_time_embedded_matrix(stim, L)  # each row stacks stim[t], stim[t-1], ... stim[t-L+1]

# 2) Compute STA (linear filter estimate), shape (P*L,)
k_sta = (X.T @ spikes) / spikes.sum()

# Optional: mean-subtract stimulus before embedding, or subtract mean from STA

# 3) Compute generator signal for all bins
g = X @ k_sta

# 4A) Nonparametric nonlinearity via histograms (Bayes)
p_g = hist_density(g)
p_g_given_spike = hist_density(g[spikes > 0], weights=spikes[spikes > 0])
p_spike = spikes.sum() / len(spikes)
p_spike_given_g = (p_g_given_spike * p_spike) / p_g
rate_of_g = p_spike_given_g / dt

# 4B) Parametric nonlinearity: fit sigmoid/probit to E[spikes|g] (binned)
bins = bin_edges(g)
g_centers, mean_spikes = bin_average(g, spikes, bins)
params = fit_sigmoid(g_centers, mean_spikes)  # e.g., least squares or MLE
```

---

## 7) References (starting points)

- E.J. Chichilnisky (2001). *A simple white noise analysis of neuronal light responses.* Network: Computation in Neural Systems.  
  (Defines STA, shows STA ∝ linear filter under radially symmetric white noise, and describes estimating the static nonlinearity by binning generator signal values.)

- J.W. Pillow et al. (2008). *Spatio-temporal correlations and visual signalling in a complete neuronal population.* Nature.  
  (Retinal ganglion cell GLM; stimulus filter + spike-history + coupling filters; exponential nonlinearity.)

- L. Paninski (2003). *Convergence properties of three spike-triggered analysis techniques.* Network: Computation in Neural Systems.  
  (Theoretical conditions under which STA/STC estimators converge; highlights issues with natural/correlated stimuli.)

- Neuromatch Academy tutorial: *GLMs for Encoding*  
  (Practical design-matrix construction and the identity STA = Xᵀy / sum(y) for white noise stimuli.)

- Reverse correlation notes (UCSD neurophysics course materials)  
  (Clear statement of estimating the nonlinearity using Bayes’ rule: p(spike|g) = p(g|spike)p(spike)/p(g).)

- Cantrell et al. (2010). *Non-centered spike-triggered covariance analysis…* PLOS Computational Biology.  
  (Discusses LN models for RGCs, limitations of STA for symmetric nonlinearities, and recovering static nonlinearities from projections.)

- Huth et al. (2018). *Convis: A Toolbox to Fit and Simulate Filter-Based Models of Early Visual Processing.*  
  (Software-oriented discussion; STA/STC as RF estimates feeding into LN models.)

