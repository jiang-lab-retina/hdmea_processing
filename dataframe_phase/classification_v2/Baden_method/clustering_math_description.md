# Mathematical Description of the Clustering and Labeling Method

This document provides a formal mathematical description of the clustering pipeline, from raw traces to final cluster labels.

---

## 1. Input: Raw Traces

Let the input data be a collection of $N$ cells, each with multiple stimulus responses.

For cell $i$, we have traces:

$$\mathbf{R}_i^{(s)} \in \mathbb{R}^{M_s \times T_s}$$

where:
- $s$ = stimulus type index (frequency sections, color, moving bar, RF)
- $M_s$ = number of trials for stimulus $s$
- $T_s$ = number of time samples for stimulus $s$

---

## 2. Signal Conditioning

### 2.1 Trial Averaging

For each stimulus, compute the mean response across trials:

$$\bar{r}_i^{(s)}(t) = \frac{1}{M_s} \sum_{m=1}^{M_s} R_i^{(s)}(m, t)$$

**Output**: $\bar{\mathbf{r}}_i^{(s)} \in \mathbb{R}^{T_s}$

### 2.2 Low-Pass Filtering

Apply a 4th-order Butterworth low-pass filter with cutoff frequency $f_c = 10$ Hz:

$$\tilde{\mathbf{r}}_i^{(s)} = \mathcal{H}_{LP}(\bar{\mathbf{r}}_i^{(s)})$$

The filter transfer function:

$$H(s) = \frac{1}{\displaystyle\prod_{k=1}^{4} \left(1 - \frac{s}{p_k}\right)}$$

where $p_k$ are the Butterworth poles. Zero-phase filtering is achieved via forward-backward application:

$$\tilde{\mathbf{r}} = \text{flip}\left(\mathcal{H}\left(\text{flip}\left(\mathcal{H}(\bar{\mathbf{r}})\right)\right)\right)$$

### 2.3 Downsampling

Reduce sampling rate from $f_s = 60$ Hz to $f_t = 10$ Hz:

$$\mathbf{x}_i^{(s)}[n] = \tilde{\mathbf{r}}_i^{(s)}[D \cdot n], \quad D = 6$$

**Output**: $\mathbf{x}_i^{(s)} \in \mathbb{R}^{T_s / D}$

---

## 3. Feature Extraction

### 3.1 Sparse PCA

For stimulus traces $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N]^T \in \mathbb{R}^{N \times T}$, Sparse PCA solves:

$$\min_{\mathbf{D}, \mathbf{A}} \frac{1}{2} \|\mathbf{X} - \mathbf{D}\mathbf{A}^T\|_F^2 + \lambda \|\mathbf{A}\|_1$$

where:
- $\mathbf{D} \in \mathbb{R}^{N \times K}$ = dictionary (loadings)
- $\mathbf{A} \in \mathbb{R}^{T \times K}$ = sparse components
- $K$ = number of components
- $\lambda$ = sparsity regularization parameter

**Hard Sparsity Enforcement**: For each component $\mathbf{a}_k$:

1. Identify indices $\mathcal{I}_k = \{j : |a_{k,j}| \text{ is in top-}q\}$
2. Set $a_{k,j} = 0$ for $j \notin \mathcal{I}_k$
3. Renormalize: $\mathbf{a}_k \leftarrow \mathbf{a}_k / \|\mathbf{a}_k\|_2$

**Feature projection**: $\mathbf{f}_i^{(s)} = \mathbf{x}_i^{(s)} \cdot \mathbf{A} \in \mathbb{R}^{K}$

### 3.2 SVD for Moving Bar Responses

For 8-direction moving bar responses, construct per-cell matrix:

$$\mathbf{B}_i = \begin{bmatrix} \mathbf{x}_i^{(0°)} & \mathbf{x}_i^{(45°)} & \cdots & \mathbf{x}_i^{(315°)} \end{bmatrix} \in \mathbb{R}^{T \times 8}$$

Per-cell normalization:

$$\hat{\mathbf{B}}_i = \frac{\mathbf{B}_i}{\max_{j,t} |B_i(t,j)|}$$

SVD decomposition:

$$\hat{\mathbf{B}}_i = \mathbf{U}_i \mathbf{\Sigma}_i \mathbf{V}_i^T$$

Extract dominant time course:

$$\mathbf{tc}_i = \sigma_{i,1} \cdot \mathbf{u}_{i,1}$$

Then apply Sparse PCA to the collection $\{\mathbf{tc}_i\}_{i=1}^N$.

### 3.3 Feature Concatenation

Concatenate all stimulus features into a single vector:

$$\mathbf{f}_i = \left[ \mathbf{f}_i^{(\text{freq})} \,\|\, \mathbf{f}_i^{(\text{color})} \,\|\, \mathbf{f}_i^{(\text{bar})} \,\|\, \mathbf{f}_i^{(\text{RF})} \right] \in \mathbb{R}^{P}$$

where $P = 40$ (total feature dimensions).

---

## 4. Feature Standardization

Apply Z-score normalization across all cells:

$$z_{ij} = \frac{f_{ij} - \mu_j}{\sigma_j}$$

where:
- $\mu_j = \frac{1}{N} \sum_{i=1}^{N} f_{ij}$ (mean of feature $j$)
- $\sigma_j = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (f_{ij} - \mu_j)^2}$ (standard deviation)

**Output**: Standardized feature matrix $\mathbf{Z} \in \mathbb{R}^{N \times P}$

---

## 5. Gaussian Mixture Model Clustering

### 5.1 Model Definition

Assume data is generated from a mixture of $K$ Gaussians with diagonal covariance:

$$p(\mathbf{z}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}_k^2))$$

where:
- $\pi_k$ = mixing coefficient for cluster $k$, with $\sum_{k=1}^{K} \pi_k = 1$
- $\boldsymbol{\mu}_k \in \mathbb{R}^P$ = mean of cluster $k$
- $\boldsymbol{\sigma}_k^2 \in \mathbb{R}^P$ = diagonal variance of cluster $k$

The diagonal Gaussian density:

$$\mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}_k^2)) = \prod_{j=1}^{P} \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left(-\frac{(z_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)$$

### 5.2 Parameter Estimation (EM Algorithm)

The parameters $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\sigma}_k^2\}_{k=1}^K$ are estimated via Expectation-Maximization (EM).

**E-Step**: Compute responsibilities (posterior probabilities):

$$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{z}_i \mid \boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}_k^2))}{\displaystyle\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\mathbf{z}_i \mid \boldsymbol{\mu}_j, \text{diag}(\boldsymbol{\sigma}_j^2))}$$

**M-Step**: Update parameters:

$$N_k = \sum_{i=1}^{N} \gamma_{ik}$$

$$\pi_k^{\text{new}} = \frac{N_k}{N}$$

$$\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \, \mathbf{z}_i$$

$$\sigma_{kj}^{2,\text{new}} = \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (z_{ij} - \mu_{kj}^{\text{new}})^2 + \epsilon$$

where $\epsilon$ is a regularization term for numerical stability.

**Log-Likelihood**:

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{z}_i \mid \boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}_k^2)) \right)$$

---

## 6. Model Selection via BIC

### 6.1 Bayesian Information Criterion

For each candidate $K$, compute:

$$\text{BIC}(K) = -2 \mathcal{L}(\hat{\boldsymbol{\theta}}_K) + d_K \log N$$

where:
- $\mathcal{L}(\hat{\boldsymbol{\theta}}_K)$ = maximized log-likelihood for $K$ components
- $d_K$ = number of free parameters
- $N$ = number of samples

**Parameter count for diagonal GMM**:

$$d_K = K \cdot P + K \cdot P + (K - 1) = K(2P + 1) - 1$$

- $K \cdot P$ parameters for means
- $K \cdot P$ parameters for diagonal variances
- $K - 1$ parameters for mixing coefficients (sum-to-one constraint)

### 6.2 Log Bayes Factor

The log Bayes factor comparing model $K+1$ to model $K$:

$$\log \text{BF}(K+1, K) \approx -\frac{1}{2}\left(\text{BIC}(K+1) - \text{BIC}(K)\right)$$

**Interpretation**:
- $\log \text{BF} > 6$: Strong evidence for $K+1$ over $K$
- $\log \text{BF} \in (2, 6]$: Moderate evidence
- $\log \text{BF} \in (0, 2]$: Weak evidence
- $\log \text{BF} \leq 0$: No evidence for more clusters

### 6.3 Optimal K Selection

Select $K^*$ as the smallest $K$ where:

$$\log \text{BF}(K+1, K) < \tau$$

where $\tau$ is the threshold (default: $\tau = 6$).

If no such $K$ exists, use $K^* = \arg\min_K \text{BIC}(K)$.

---

## 7. Cluster Labeling

### 7.1 Hard Labels

Assign each cell to the cluster with maximum posterior probability:

$$\ell_i = \arg\max_{k \in \{1, \ldots, K^*\}} \gamma_{ik}$$

### 7.2 Posterior Probabilities

The full posterior distribution for cell $i$:

$$\mathbf{p}_i = [\gamma_{i1}, \gamma_{i2}, \ldots, \gamma_{iK^*}]^T$$

where $\sum_{k=1}^{K^*} \gamma_{ik} = 1$.

### 7.3 Assignment Confidence

The confidence of assignment for cell $i$:

$$c_i = \max_k \gamma_{ik}$$

High $c_i$ indicates unambiguous cluster membership; low $c_i$ suggests the cell lies near cluster boundaries.

---

## 8. Summary: Complete Pipeline

$$\boxed{
\begin{aligned}
&\text{Input: } \{\mathbf{R}_i^{(s)}\}_{i=1}^N & &\text{(Raw traces)} \\[6pt]
&\downarrow \\[6pt]
&\bar{\mathbf{r}}_i^{(s)} = \frac{1}{M_s}\sum_{m=1}^{M_s} \mathbf{R}_i^{(s)}(m, :) & &\text{(Trial averaging)} \\[6pt]
&\downarrow \\[6pt]
&\tilde{\mathbf{r}}_i^{(s)} = \mathcal{H}_{LP}(\bar{\mathbf{r}}_i^{(s)}) & &\text{(Low-pass filter)} \\[6pt]
&\downarrow \\[6pt]
&\mathbf{x}_i^{(s)} = \text{Downsample}(\tilde{\mathbf{r}}_i^{(s)}, D) & &\text{(Downsample)} \\[6pt]
&\downarrow \\[6pt]
&\mathbf{f}_i = [\text{SparsePCA}(\mathbf{x}_i) \,\|\, \text{SVD}(\mathbf{B}_i) \,\|\, \text{PCA}(\mathbf{x}_i^{(\text{RF})})] & &\text{(Feature extraction)} \\[6pt]
&\downarrow \\[6pt]
&\mathbf{z}_i = \frac{\mathbf{f}_i - \boldsymbol{\mu}}{\boldsymbol{\sigma}} & &\text{(Z-score standardization)} \\[6pt]
&\downarrow \\[6pt]
&K^* = \arg\min_K \text{BIC}(K) \text{ s.t. } \log\text{BF}(K+1,K) < \tau & &\text{(Model selection)} \\[6pt]
&\downarrow \\[6pt]
&\text{GMM}_{K^*}: p(\mathbf{z}) = \sum_{k=1}^{K^*} \pi_k \mathcal{N}(\mathbf{z}|\boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}_k^2)) & &\text{(Fit final model)} \\[6pt]
&\downarrow \\[6pt]
&\ell_i = \arg\max_k \, p(k|\mathbf{z}_i) & &\text{(Assign labels)} \\[6pt]
&\downarrow \\[6pt]
&\text{Output: } \{\ell_i, \mathbf{p}_i\}_{i=1}^N & &\text{(Labels + posteriors)}
\end{aligned}
}$$

---

## 9. Notation Reference

| Symbol | Description |
|--------|-------------|
| $N$ | Number of cells |
| $P$ | Feature dimension (40) |
| $K$ | Number of clusters |
| $\mathbf{R}_i^{(s)}$ | Raw trace matrix for cell $i$, stimulus $s$ |
| $\bar{\mathbf{r}}_i^{(s)}$ | Trial-averaged trace |
| $\mathbf{z}_i$ | Standardized feature vector |
| $\gamma_{ik}$ | Posterior probability (responsibility) |
| $\ell_i$ | Cluster label for cell $i$ |
| $\pi_k$ | Mixing coefficient for cluster $k$ |
| $\boldsymbol{\mu}_k$ | Mean of cluster $k$ |
| $\boldsymbol{\sigma}_k^2$ | Diagonal variance of cluster $k$ |
| $\text{BIC}(K)$ | Bayesian Information Criterion for $K$ clusters |
| $\log\text{BF}$ | Log Bayes Factor |
