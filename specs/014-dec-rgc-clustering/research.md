# Research Notes: DEC-Refined RGC Subtype Clustering

## Core Algorithms

### 1. Baden-Style Clustering (GMM + BIC)

**Reference**: Baden et al. (2016) - "The functional diversity of retinal ganglion cells in the mouse retina"

#### Approach
- Use **diagonal-covariance Gaussian Mixture Models** (GMM) for computational efficiency
- Select cluster count k* by minimizing **Bayesian Information Criterion (BIC)**

#### BIC Formula
$$\text{BIC}(k) = -2 \ln \hat{L}_k + p_k \ln N$$

Where:
- $\hat{L}_k$ = maximized likelihood for k clusters
- $p_k$ = number of free parameters
- $N$ = number of data points

For diagonal GMM:
$$p_k = k \cdot d + k \cdot d + (k-1) = k(2d+1) - 1$$

Where $d$ = embedding dimension (49).

#### Implementation Notes
- Use multiple random restarts (n_init ≥ 20) to avoid local minima
- Add small regularization to covariance diagonal (reg_covar = 1e-3)
- Run GMM fitting per functional group (DSGC, OSGC, Other)

---

### 2. Deep Embedded Clustering (DEC)

**Reference**: Xie et al. (2016) - "Unsupervised Deep Embedding for Clustering Analysis"

#### Core Idea
DEC jointly optimizes feature representations and cluster assignments by:
1. Using a soft clustering layer on top of embeddings
2. Iteratively sharpening assignments using a self-training target distribution

#### Soft Assignment (Student-t Kernel)
$$q_{ij} = \frac{(1 + \|z_i - \mu_j\|^2/\alpha)^{-(\alpha+1)/2}}{\sum_{j'}(1 + \|z_i - \mu_{j'}\|^2/\alpha)^{-(\alpha+1)/2}}$$

Where:
- $z_i$ = embedding of cell i
- $\mu_j$ = cluster center j
- $\alpha$ = degrees of freedom (typically 1)

#### Target Distribution
$$p_{ij} = \frac{q_{ij}^2 / f_j}{\sum_{j'} q_{ij'}^2 / f_{j'}}$$

Where $f_j = \sum_i q_{ij}$ is the soft cluster frequency.

This sharpens confident assignments and normalizes by cluster frequency.

#### DEC Loss
$$\mathcal{L}_{\text{DEC}} = \text{KL}(P \| Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

#### Training Loop
1. Compute soft assignments Q
2. Compute target distribution P (update periodically, e.g., every 10 iterations)
3. Minimize KL(P || Q) via gradient descent
4. Update encoder weights and cluster centers
5. Stop when assignment changes < threshold

---

### 3. IDEC (Improved DEC)

**Reference**: Guo et al. (2017) - "Improved Deep Embedded Clustering with Local Structure Preservation"

#### Problem with Pure DEC
DEC can distort the embedding space, losing local structure and reconstruction quality.

#### IDEC Solution
Add reconstruction loss to preserve local structure:

$$\mathcal{L}_{\text{IDEC}} = \mathcal{L}_{\text{DEC}} + \gamma \cdot \mathcal{L}_{\text{rec}}$$

Where:
- $\gamma$ = reconstruction weight (typically 0.1)
- $\mathcal{L}_{\text{rec}}$ = autoencoder reconstruction loss

#### Benefits
- Preserves embedding interpretability
- Prevents cluster collapse
- Maintains reconstruction quality for prototype visualization

---

## Key Differences from Autoencoder_method

| Aspect | Autoencoder_method | divide_conquer_method |
|--------|-------------------|----------------------|
| Supervision | Weak supervision (SupCon + Purity loss) | None (pure unsupervised) |
| Grouping | AC > ipRGC > DS > nonDS | DS > OS > Other (RGC only) |
| k Selection | Log Bayes factor threshold | Minimum BIC |
| Refinement | None | DEC/IDEC |
| Validation | Cross-validation purity | ipRGC enrichment |
| Cluster Basis | Coarse group labels | Learned from data |

---

## ipRGC Validation Rationale

### Why ipRGC Enrichment?

1. **Known ground truth**: ipRGC cells have distinctive physiological properties measurable by `iprgc_2hz_QI > 0.8`
2. **Clear functional signature**: Slow, sustained responses due to melanopsin
3. **Relatively rare**: ~5% of RGCs, making enrichment meaningful
4. **Independent of clustering**: Not used in group assignment or training

### Enrichment Metric

$$E_c = \frac{P(\text{ipRGC} \mid \text{cluster}=c)}{P(\text{ipRGC})}$$

Interpretation:
- $E_c = 1$: Cluster has baseline ipRGC rate (no enrichment)
- $E_c > 1$: Cluster is enriched for ipRGCs
- $E_c > 2$: Strong enrichment (2× baseline)

### Purity Metric

$$\text{Purity}_{\text{ipRGC}} = \frac{1}{N} \sum_{c=1}^{K} \max(n_{c,1}, n_{c,0})$$

Where:
- $n_{c,1}$ = number of ipRGCs in cluster c
- $n_{c,0}$ = number of non-ipRGCs in cluster c

This measures how well clusters separate ipRGC vs non-ipRGC cells.

---

## Preprocessing Notes

### ipRGC Test Trace

The user specified: "low-pass 4 Hz, resample to 2 Hz"

**Technical issue**: Resampling to 2 Hz implies Nyquist = 1 Hz. Any valid resampler will effectively band-limit to ≤1 Hz.

**Solution**: 
- Apply 4 Hz low-pass as specified (removes most noise)
- Use proper anti-aliasing resampler (e.g., `scipy.signal.resample_poly`)
- Effective final bandwidth will be ≤1 Hz

### Frequency Section 10 Hz

**Rationale**: Preserve high-frequency dynamics that are informative for fast-responding cells.

**Approach**:
- No low-pass filter
- Keep at 60 Hz
- Edge crop to remove transients (first/last 1 second)
- Ensure fixed length across all cells

---

## Implementation Recommendations

### 1. Initialization

- Pre-train autoencoder for reconstruction only
- Extract embeddings
- Run k-means or GMM to get initial cluster centers
- Initialize DEC cluster layer with these centers

### 2. DEC Convergence

- Update target P every 10-20 iterations (not every step)
- Monitor assignment change rate: $\Delta = \frac{1}{N}\sum_i \mathbb{1}[\arg\max_j q_{ij}^{(t)} \neq \arg\max_j q_{ij}^{(t-1)}]$
- Stop when $\Delta < 0.001$ or max iterations reached

### 3. Comparison Strategy

Save both initial GMM and final DEC results to enable direct comparison:
- Same cells, same embeddings (initial)
- Different cluster assignments
- Same validation metrics computed for both

This allows quantifying "does DEC help?" without re-running the expensive AE training.

---

## References

1. Baden, T., et al. (2016). "The functional diversity of retinal ganglion cells in the mouse retina." Nature.
2. Xie, J., Girshick, R., & Farhadi, A. (2016). "Unsupervised deep embedding for clustering analysis." ICML.
3. Guo, X., et al. (2017). "Improved deep embedded clustering with local structure preservation." IJCAI.
4. Schwarz, G. (1978). "Estimating the dimension of a model." Annals of Statistics.
