# Deep Research Query: Improving Retinal Cell Type Clustering with Autoencoders

## Context and Goal

We are developing an **autoencoder-based clustering pipeline** for classifying subtypes of retinal ganglion cells (RGCs) and amacrine cells (ACs) based on their electrophysiological response traces. The goal is to discover meaningful cell type clusters that:

1. **Generalize well** — clusters should be pure with respect to known biological labels
2. **Capture functional diversity** — reveal cell types within each coarse group
3. **Are reproducible** — stable across different data subsets

### Data Description

- **~17,000 cells** recorded via multi-electrode arrays
- **10 stimulus response segments** per cell:
  - 5 frequency flicker responses (0.5, 1, 2, 4, 10 Hz)
  - Color stimulus response (green/blue)
  - 8 moving bar directions (concatenated)
  - Receptive field time course (STA)
  - ipRGC test response
  - Step-up stimulus response

- **Known coarse labels** (used as weak supervision):
  - `axon_type`: AC (amacrine cell) vs RGC (ganglion cell)
  - `ds_cell`: Direction-selective vs non-DS (based on p-value < 0.05)
  - `iprgc`: Intrinsically photosensitive RGC vs non-ipRGC (based on QI > 0.8)

- **Coarse groups** (mutually exclusive, priority-based):
  - AC → ipRGC → DS-RGC → nonDS-RGC

### Pipeline Architecture

```
Raw traces → Preprocessing → Multi-segment Autoencoder → 49D Embeddings → Per-group GMM Clustering
                                     ↓
                            [Supervision Losses]
```

**Autoencoder design:**
- 10 separate encoder-decoder pairs (one per stimulus segment)
- Latent dimensions: 4-12 per segment, total = 49D
- Architecture: MLP with hidden dims [32, 64, 128]

**Loss function:**
$$\mathcal{L} = \mathcal{L}_{\text{reconstruction}} + \beta \cdot \mathcal{L}_{\text{supervision}}$$

---

## Problem 1: Poor Group Separation in Embedding Space

### Observation

Despite using supervised contrastive loss (SupCon) with coarse group labels, the learned embeddings show **no separation** between coarse groups:

- **Silhouette score: -0.08** (range: -1 to 1, higher = better)
- Negative score means cells are closer to OTHER groups than their OWN group
- UMAP visualization shows complete overlap of all 4 coarse groups

### What We Tried

| Approach | Parameters | Result |
|----------|------------|--------|
| SupCon loss | β = 0.1 to 100, τ = 0.05 to 0.1 | No improvement, silhouette stays negative |
| Increased epochs | 100 → 150 | Reconstruction improves, separation doesn't |
| Batch size tuning | 64 to 256 | Minimal effect |

### Hypothesis

The coarse groups (AC, ipRGC, DS-RGC, nonDS-RGC) may share very similar response dynamics to the stimuli used. The reconstruction loss dominates because it captures variance that is **common across groups**, while group-distinguishing features may be subtle.

---

## Problem 2: Purity Optimization Approach

### Current Attempt

We implemented a **cluster purity loss** that directly minimizes conditional entropy $H(Y|C)$:

$$\mathcal{L}_{\text{purity}} = -\sum_c p(c) \sum_y p(y|c) \log p(y|c)$$

Where:
- $C$ = soft cluster assignments (from learnable SoftKMeans)
- $Y$ = binary labels (axon_type, ds_cell, iprgc)
- Lower entropy = purer clusters

### Open Questions

1. **Is conditional entropy the right objective?** Are there better differentiable purity metrics?
2. **How should soft clusters be modeled?** K-means vs GMM vs other?
3. **How to balance reconstruction vs purity?** What loss weighting strategies exist?

---

## Problem 3: Contrastive Learning Mismatch

### The Issue

Supervised contrastive loss uses **4 coarse groups** as positive/negative pairs, but our evaluation uses **3 binary labels** (axon_type, ds_cell, iprgc).

This creates a mismatch:
- **Training**: Separate AC from ipRGC from DS-RGC from nonDS-RGC
- **Evaluation**: Measure purity for AC/RGC, DS/nonDS, ipRGC/non-ipRGC

### Alternative Approaches to Consider

1. **Multi-label contrastive loss** — separate losses for each binary attribute
2. **Hierarchical contrastive loss** — first AC/RGC, then DS/nonDS within RGC
3. **Remove contrastive loss entirely** — let reconstruction + GMM find natural structure

---

## Problem 4: Cross-Validation Method and Generalization Testing

### Our Cross-Validation Approach

To test whether clusters generalize beyond the labels used during training, we implemented a **leave-one-label-out cross-validation**:

```
For each label L in {axon_type, ds_cell, iprgc}:
    1. Remove label L from the supervision loss (but keep it for evaluation)
    2. Retrain autoencoder from scratch with remaining labels
    3. Re-cluster using GMM
    4. Measure purity of resulting clusters against the OMITTED label L
```

**Rationale**: If clusters are biologically meaningful, they should still separate the omitted label even without explicit supervision.

### Cross-Validation Results

| Omitted Label | Active Labels | Purity on Omitted | Interpretation |
|---------------|---------------|-------------------|----------------|
| `axon_type` | ds_cell, iprgc | 71% | Clusters don't naturally separate AC/RGC |
| `ds_cell` | axon_type, iprgc | 70% | DS cells not well-captured without supervision |
| `iprgc` | axon_type, ds_cell | 87% | ipRGC separation is more robust |

**CV Score** (average of above): **76%** — below our target of 85%

### Observations

1. **axon_type and ds_cell are hard to generalize** — when these labels are removed from training, the model fails to naturally separate them
2. **iprgc generalizes better** — possibly because ipRGC has distinctive response patterns (prolonged response to light offset)
3. **The coarse group structure may be leaking information** — even when a label is "omitted", the remaining labels may partially encode it

### Open Questions on Cross-Validation

1. **Is leave-one-label-out the right CV strategy?** Should we use k-fold on cells instead?
2. **How to prevent label leakage?** The coarse groups are derived from all three labels, creating dependencies
3. **Should CV be done on embedding space or on final clusters?**
4. **What CV metrics are standard for unsupervised/semi-supervised clustering?**

---

## Research Questions for Literature Search

### Q1: Weak Supervision in Autoencoders

What are state-of-the-art methods for incorporating weak/partial labels into autoencoder training? Specifically:
- Semi-supervised autoencoders (VAE-based or otherwise)
- Contrastive learning with noisy/partial labels
- Multi-task learning combining reconstruction with classification

### Q2: Cluster-Aware Representation Learning

How can we learn representations that are optimized for downstream clustering? Relevant topics:
- Deep Embedded Clustering (DEC) and variants
- Differentiable clustering losses
- Joint representation learning and clustering

### Q3: Disentangled Representations for Cell Types

Can we learn representations where different latent dimensions correspond to different biological attributes?
- β-VAE and disentanglement
- Supervised disentanglement methods
- Attribute-guided representation learning

### Q4: Multi-View/Multi-Segment Learning

Our data has 10 different stimulus responses per cell. How to best combine these?
- Multi-view representation learning
- Attention mechanisms for segment weighting
- Which segments carry cell-type-discriminative information?

### Q5: Biological Prior Knowledge

Can we incorporate known biological structure from retinal neuroscience?
- **Direction selectivity**: DS-RGCs have tuning curves across 8 directions — can we encode this as a constraint?
- **Temporal response profiles**: ON cells respond to light onset, OFF cells to offset, ON-OFF to both
- **Sustained vs transient**: Some cells have sustained responses, others are transient
- **ipRGC melanopsin response**: Intrinsic photosensitivity creates distinctive slow, sustained responses
- **Hierarchical taxonomy**: Major divisions (AC/RGC, ON/OFF, sustained/transient) before subtypes
- **Expected number of types**: Literature suggests ~30-40 RGC types in mouse — can we use this as a prior?
- **Transfer learning**: Can we use embeddings or cluster centers from published cell atlases (e.g., Baden et al.)?

### Q6: Cross-Validation for Semi-Supervised Clustering

What are best practices for evaluating generalization in semi-supervised clustering?
- Leave-one-label-out vs k-fold vs bootstrap approaches
- How to handle label dependencies (our 3 binary labels are not independent)
- Metrics for evaluating cluster quality without ground truth
- Stability-based validation methods (e.g., consensus clustering)

---

## Specific Requests for Deep Research

### 1. Literature Review

#### A. Neuroscience & Biology (Primary Focus)

Please prioritize papers from neuroscience and cell biology:
- "retinal ganglion cell classification" + "machine learning"
- "RGC cell type" + "clustering" + "electrophysiology"
- "Baden retinal cell types" (key reference: Baden et al., Nature 2016)
- "single-cell transcriptomics" + "cell type clustering" (methods may transfer)
- "functional cell type classification" + "calcium imaging"
- "direction-selective ganglion cells" + "classification"
- "amacrine cell types" + "electrophysiology"
- "retinal cell atlas" + "unsupervised learning"

#### B. Computational Neuroscience Methods

- "spike sorting" + "clustering" (related problem with temporal signals)
- "neural population analysis" + "dimensionality reduction"
- "latent variable models" + "neural data"
- "Gaussian process factor analysis" (GPFA) for neural time series

#### C. Machine Learning & Deep Learning

- "deep clustering autoencoder" + "weak supervision"
- "contrastive learning" + "cell type classification"
- "differentiable clustering loss"
- "multi-modal autoencoder" + "representation learning"
- "Deep Embedded Clustering" (DEC) and variants

#### D. Cross-Disciplinary Ideas (Search Other Fields)

Look for transferable ideas from:
- **Single-cell genomics**: scRNA-seq clustering methods (Seurat, Scanpy) — how do they handle weak labels?
- **Medical imaging**: Multi-modal fusion for diagnosis — combining different "views" of same patient
- **Speech recognition**: Speaker diarization — clustering temporal signals by identity
- **Computer vision**: Few-shot learning and metric learning — learning with limited labels
- **Natural language processing**: Topic modeling — unsupervised discovery of latent categories
- **Astronomy**: Galaxy classification — clustering objects with continuous properties
- **Chemistry**: Molecular property prediction — multi-task learning with partial labels

### 2. Mathematical Analysis

Analyze our loss function design:
- Is supervised contrastive loss appropriate when groups have overlapping feature distributions?
- What happens when the ratio of reconstruction loss to supervision loss is mismatched?
- Can you derive conditions under which contrastive loss would fail to separate groups?

### 3. Alternative Architectures

Propose alternative architectures that might work better:
- Variational autoencoders (VAE) with supervised latent priors
- Transformer-based encoders for temporal traces
- Graph neural networks if cell-cell relationships exist

### 4. Practical Recommendations

Based on your analysis, recommend:
1. **Top 3 methods** most likely to improve our pipeline
2. **Diagnostic tests** to understand why separation is failing
3. **Ablation experiments** to identify the bottleneck

### 5. Cross-Validation Strategy

Evaluate and suggest improvements to our cross-validation approach:
1. **Is leave-one-label-out appropriate?** What are alternatives for multi-label semi-supervised settings?
2. **How to handle dependent labels?** Our labels are derived hierarchically (AC → remaining → DS/nonDS/ipRGC)
3. **What CV metrics are used in neuroscience cell type discovery?** Are there field-specific standards?
4. **Bootstrap stability analysis** — how to properly implement and interpret cluster stability?
5. **Should we validate on held-out cells or held-out labels?** Trade-offs between these approaches

---

## Constraints and Preferences

- **Framework**: PyTorch (current implementation)
- **Compute**: Single GPU (RTX 3080 or similar)
- **Training time budget**: ~30 minutes acceptable, hours undesirable
- **Preference for interpretability**: We want to understand WHY clusters form, not just achieve high purity

---

## Summary of Key Metrics

| Metric | Current Value | Target |
|--------|---------------|--------|
| Silhouette score (group separation) | -0.08 | > 0.25 |
| Post-hoc purity (axon_type) | 100% | (already perfect due to AC/RGC split) |
| Post-hoc purity (ds_cell) | 88% | > 90% |
| Post-hoc purity (iprgc) | 97% | > 95% |
| **CV purity (omit axon_type)** | 71% | > 85% |
| **CV purity (omit ds_cell)** | 70% | > 85% |
| **CV purity (omit iprgc)** | 87% | > 90% |
| **CV Score (average)** | 76% | > 85% |

---

## Key References to Start From

These papers are foundational to our work — please build upon them:

1. **Baden et al. (2016)** — "The functional diversity of retinal ganglion cells in the mouse retina" — Nature
   - Established functional classification of mouse RGCs using calcium imaging
   - Used hierarchical clustering on response features
   - Our goal is similar but with electrophysiology data

2. **Rheaume et al. (2018)** — "Single cell transcriptome profiling of retinal ganglion cells" — Nat Commun
   - Combined transcriptomics with functional properties
   - Shows that cell types have both molecular and functional signatures

3. **Tran et al. (2019)** — "Single-cell profiles of retinal ganglion cells" — Neuron
   - Comprehensive RGC atlas with ~40 types
   - Ground truth for what diversity we might expect

4. **Xie et al. (2016)** — "Unsupervised Deep Embedding for Clustering Analysis" (DEC)
   - Foundational deep clustering method
   - Joint representation learning and clustering

---

## Instructions for AI Research Assistant

1. **Prioritize neuroscience literature** — solutions should be grounded in how the field approaches cell type discovery
2. **Understand the problem deeply** before suggesting solutions
3. **Cite specific papers** with methods that address similar challenges — include DOI or arXiv links when possible
4. **Explain trade-offs** between different approaches
5. **Provide pseudocode or mathematical formulations** where helpful
6. **Consider biological interpretability** — cell biologists need to understand WHY cells cluster together
7. **Be critical** — if our approach is fundamentally flawed, say so and explain why
8. **Look beyond ML** — biological insights about cell types may suggest better features or priors
9. **Cross-disciplinary transfer** — explicitly identify which ideas from other fields could apply here

### Biological Context to Keep in Mind

- **Direction-selective cells** respond maximally to motion in one direction — the 8-direction bar responses should capture this
- **ipRGCs** have intrinsic photosensitivity and show sustained responses — the ipRGC test protocol is designed for this
- **Amacrine cells** are interneurons with no axon — they process signals differently than RGCs
- **Cell types are not arbitrary** — they reflect developmental programs and circuit roles
- **Good clusters should be biologically interpretable** — not just statistically optimal

The goal is not just to find papers, but to synthesize insights that lead to actionable improvements in our pipeline, grounded in biological understanding of retinal cell types.
