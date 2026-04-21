# How ipRGC Subtypes Are Classified

## Overview

ipRGC (intrinsically photosensitive retinal ganglion cell) classification in
this pipeline is a two-stage process:

1. **Coarse grouping** -- a cell is assigned to the functional group **ipRGC**
   (vs DSGC / OSGC / Other) based on a quality-index threshold applied to its
   melanopsin-mediated response.
2. **Fine subtyping** -- cells within the ipRGC group are subdivided into
   numbered subtypes (`ipRGC_0`, `ipRGC_1`, ...) by unsupervised deep
   clustering (autoencoder + DEC) applied to multi-stimulus response traces.

The fine subtypes are data-driven cluster IDs, not a direct mapping to the
biological M1--M6 nomenclature from the literature.

---

## Stage 1: Coarse Group Assignment

### Feature used: `iprgc_2hz_QI`

The coarse grouping relies on `iprgc_2hz_QI`, a Pearson-correlation-based
quality index that measures how reproducibly a cell responds to the
`iprgc_test` stimulus across trials.

**Source file:** `dataframe_phase/extract_feature/extract_feature_step_iprgc.py`
(function `compute_iprgc_qi`)

#### Computation steps

1. **Stack trials.** The `iprgc_test` column stores a list of firing-rate
   traces (one per trial, sampled at 60 Hz). Trials are stacked into a 2-D
   array of shape `(n_trials, n_timepoints)`.

2. **Baseline subtraction.** For each trial, the mean of the **last 1 second**
   (60 samples) is subtracted from the entire trace, zeroing the baseline.

3. **2 Hz Bessel lowpass filter.** A 5th-order Bessel filter at 2 Hz is
   applied to each trial via `scipy.signal.filtfilt`.

4. **Trim to response window.** Only the portion from **2 s onward**
   (sample 120+) is kept, discarding the stimulus onset transient.

5. **Hard gates (set QI = 0).** If any of the following are true in the
   trimmed window, `iprgc_2hz_QI` is forced to 0.0:
   - Any trial has **mean raw firing rate < 1 Hz** (too little spiking).
   - Any trial has **zero standard deviation** after filtering (constant
     trace).
   - The **maximum absolute value** of all filtered traces is < 1.0 (all near
     zero).

6. **Pearson QI.** Provided no gate triggered, the QI is computed as:

$$\text{QI} = \frac{1}{N}\sum_{i=1}^{N} r\bigl(\mathbf{x}_i,\;\bar{\mathbf{x}}\bigr)$$

   where $\mathbf{x}_i$ is the filtered trace for trial $i$,
   $\bar{\mathbf{x}}$ is the mean trace across trials, and $r$ denotes the
   Pearson correlation coefficient. Trials with zero variance are excluded
   from the average.

A companion metric `iprgc_20hz_QI` is computed identically except: 20 Hz
lowpass filter, window trimmed to **2 s -- 10 s** (samples 120--600). It is
stored but not used for group assignment.

### Threshold and priority rule

**Source file:** `dataframe_phase/classification_v2/divide_conquer_method/config.py`

| Parameter | Value | Column |
|-----------|-------|--------|
| `IPRGC_QI_THRESHOLD` | 0.8 | `iprgc_2hz_QI` |
| `DS_P_THRESHOLD` | 0.05 | `ds_p_value` |
| `OS_P_THRESHOLD` | 0.05 | `os_p_value` |

**Source file:** `dataframe_phase/classification_v2/divide_conquer_method/grouping.py`
(function `assign_groups`)

Assignment uses **priority order: ipRGC > DSGC > OSGC > Other**. Labels are
written in reverse-priority order so that higher-priority groups overwrite
lower ones:

```python
group = pd.Series("Other", index=df.index)
group[is_os] = "OSGC"     # os_p_value < 0.05
group[is_ds] = "DSGC"     # ds_p_value < 0.05  (overwrites OSGC)
group[is_iprgc] = "ipRGC"  # iprgc_2hz_QI > 0.8 (overwrites DS/OS)
```

A cell that meets multiple criteria (e.g., high ipRGC QI **and** significant
DS p-value) is classified as **ipRGC** because it has the highest priority.

In the Autoencoder pipeline variant
(`dataframe_phase/classification_v2/Autoencoder_method/grouping.py`), an
additional constraint excludes amacrine cells:

```python
iprgc_mask = (iprgc_qi > threshold) & (df["axon_type"] != "ac")
```

### Overlap diagnostics

The grouping function logs overlap counts:

- How many cells satisfy both DS and OS thresholds.
- How many ipRGC cells also pass the DS or OS threshold.
- Final group sizes and percentages.

---

## Stage 2: Fine Subtype Assignment (DEC Clustering)

Once cells are partitioned into coarse groups, each group is passed through a
**multi-segment autoencoder** followed by **Deep Embedded Clustering (DEC)** to
discover fine subtypes.

### Input traces

The autoencoder takes 10 stimulus segments per cell, including the
`iprgc_test` trace:

| Segment | Description |
|---------|-------------|
| `freq_section_0p5hz` | Frequency stimulus at 0.5 Hz |
| `freq_section_1hz` | Frequency stimulus at 1 Hz |
| `freq_section_2hz` | Frequency stimulus at 2 Hz |
| `freq_section_4hz` | Frequency stimulus at 4 Hz |
| `freq_section_10hz` | Frequency stimulus at 10 Hz |
| `green_blue_3s_3i_3x` | Green-blue chromatic stimulus |
| `bar_concat` | Concatenation of 8 moving-bar directions |
| `sta_time_course` | Spike-triggered average time course |
| `iprgc_test` | Melanopsin-mediated sustained response |
| `step_up_5s_5i_b0_3x` | Step-up luminance stimulus |

Each segment is encoded by a dedicated TCN (temporal convolutional network)
encoder into a low-dimensional latent vector. The per-segment latent
dimensions are:

```
freq_section_*: 4 each (x5 = 20)
green_blue:     6
bar_concat:     12
sta_time:       3
iprgc_test:     4
step_up:        4
--------------------------
Total latent:   49
```

**Source file:** `dataframe_phase/classification_v2/divide_conquer_method/config.py`

### Clustering procedure

1. **Autoencoder pretraining.** The multi-segment autoencoder is trained to
   reconstruct all 10 stimulus traces, learning a 49-dimensional latent
   representation per cell.

2. **GMM/BIC k-selection.** Gaussian Mixture Models are fit to the latent
   space for a range of cluster counts ($k \in [1, 20]$ for ipRGC). The
   optimal $k$ is chosen by BIC elbow detection with a 3% relative
   improvement threshold.

3. **DEC refinement.** Starting from GMM-initialized cluster centers, DEC
   jointly optimizes the latent embedding and cluster assignments by
   minimizing KL divergence between soft assignments and a sharpened target
   distribution. Convergence is declared when fewer than 0.1% of cells change
   assignment between iterations.

**Expected ipRGC cluster range:** 6--10 subtypes (biologically motivated
prior, not a hard constraint).

**Source file:** `dataframe_phase/classification/subgroup_clustering/config.py`

### Subtype naming

The resulting cluster IDs are combined with the group name to produce the
final subtype label:

```python
subtype = f"{group_name}_{cluster_id}"   # e.g., "ipRGC_0", "ipRGC_4"
```

**Source file:** `dataframe_compare/classify_blocker.py` (line ~565)

These labels are stored in the `subtype` and `cluster_id` columns of the
output parquet. Downstream figure code extracts the parent group from the
subtype string by prefix matching:

```python
def parent_group(subtype: str) -> str:
    for g in GROUP_ORDER:
        if subtype.startswith(g):
            return g
    return "Unknown"
```

**Source file:**
`dataframe_compare/figure_gb_control/spatial/paper_comparison/_common.py`

---

## Blocker Experiment: ipRGC Column Sourcing

In the GB-control (gabazine/strychnine blocker) comparison pipeline, ipRGC-
related columns are sourced from the **after-blocker** recording rather than
the before-blocker control. This is because the `iprgc_test` stimulus is only
run after blocker application.

```python
IPRGC_SOURCE_COLUMNS = {"iprgc_test", "iprgc_2hz_QI", "iprgc_20hz_QI"}
```

All other traces and metadata (DS/OS features, moving bar, frequency
responses) are sourced from the **before-blocker** recording.

**Source file:** `dataframe_compare/classify_blocker.py`

---

## Relationship to Biological M1--M6 Types

The cluster IDs produced by this pipeline (`ipRGC_0` through `ipRGC_n`) are
**unsupervised, data-driven** groupings. They are **not** explicitly mapped
to the canonical M1--M6 subtypes described in the literature. Any
correspondence between cluster IDs and biological subtypes requires
post-hoc interpretation based on response profiles, morphology, or other
independent evidence.

---

## Key Source Files

| File | Role |
|------|------|
| `dataframe_phase/extract_feature/extract_feature_step_iprgc.py` | Computes `iprgc_2hz_QI` and `iprgc_20hz_QI` from raw traces |
| `dataframe_phase/classification_v2/divide_conquer_method/config.py` | Thresholds (`IPRGC_QI_THRESHOLD = 0.8`), latent dims, DEC params |
| `dataframe_phase/classification_v2/divide_conquer_method/grouping.py` | Coarse group assignment with ipRGC > DS > OS priority |
| `dataframe_phase/classification_v2/Autoencoder_method/grouping.py` | Variant with explicit amacrine cell exclusion |
| `dataframe_phase/classification/subgroup_clustering/config.py` | Expected cluster ranges, Optuna-optimized hyperparameters |
| `dataframe_compare/classify_blocker.py` | Applies trained models to blocker data; produces `subtype` labels |
| `dataframe_phase/classification_v2/divide_conquer_method/validation/iprgc_metrics.py` | Binary ipRGC label for validation |
| `dataframe_compare/figure_gb_control/spatial/paper_comparison/_common.py` | `parent_group()` helper for downstream figures |
