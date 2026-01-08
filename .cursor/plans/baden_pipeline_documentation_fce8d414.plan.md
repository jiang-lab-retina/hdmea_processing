---
name: Baden Pipeline Documentation
overview: Create a comprehensive Markdown documentation file explaining the modifications made to the Baden-method clustering pipeline, aligned with current config.py settings, including preprocessing differences for electrophysiology data, mathematical mechanisms of SVD/GMM, and data cleaning thresholds.
todos:
  - id: create-md-doc
    content: Create baden_pipeline_modification_explained.md aligned with current config.py
    status: completed
---

# Baden Pipeline Modification Documentation

Create a new file `dataframe_phase/classification_v2/Baden_method/baden_pipeline_modification_explained.md` documenting all pipeline modifications and mathematical foundations. **Document actual current settings from config.py.**

## Document Structure

### 1. Introduction and Overview

- Brief summary of Baden's original paper approach (calcium imaging data from mouse retina)
- Purpose of this modified pipeline (electrophysiology spike rate data)
- Key differences in data characteristics (60 Hz sampling vs ~15.6 Hz calcium imaging)

### 2. Preprocessing Modifications for Electrophysiology

Document the signal processing chain from [`preprocessing.py`](dataframe_phase/classification_v2/Baden_method/preprocessing.py), aligned with current config:

**Current Configuration:**

- `APPLY_BASELINE_ZEROING = False` (disabled, but available)
- `APPLY_MAX_ABS_NORMALIZATION = False` (disabled, but available)

| Step | Original Baden | Modified Pipeline | Current Status |
|------|---------------|-------------------|----------------|
| Sampling | ~15.6 Hz calcium | 60 Hz spike rate | Active |
| Filtering | None needed | 10 Hz low-pass Butterworth | Active |
| Downsampling | None | 6x (60→10 Hz) | Active |
| Baseline subtraction | Per-stimulus | Global per-cell | **Disabled** (configurable) |
| Normalization | Max-abs | Max-abs | **Disabled** (configurable) |

Include flowchart showing current active steps vs optional steps.

### 3. SVD Mathematical Mechanism and Interpretation

Document the SVD decomposition from [`features.py`](dataframe_phase/classification_v2/Baden_method/features.py) lines 321-370:

- Matrix construction: $(T \times 8)$ where T=time, 8=directions
- SVD decomposition: $X = U \Sigma V^T$
- Time course extraction: $\mathbf{tc} = \mathbf{u}_1 \cdot \sigma_1$
- Why time course (not direction tuning) is used for clustering
- Per-cell normalization to preserve directional amplitude differences

### 4. GMM Mathematical Mechanism

Document from [`clustering.py`](dataframe_phase/classification_v2/Baden_method/clustering.py):

- Diagonal covariance GMM model
- Log-likelihood and BIC formula
- Multiple initialization strategy (`GMM_N_INIT = 20`)
- GPU acceleration option (`USE_GPU = True` by default)

### 5. GMM_REG_COVAR and LOG_BF_THRESHOLD

Document current settings:

- `GMM_REG_COVAR = 1e-3` (more conservative than Baden's 1e-5)
- Purpose: Regularization added to covariance diagonal
- Effect: Prevents singularity, controls cluster tightness

- `LOG_BF_THRESHOLD = 6.0` (Kass & Raftery "strong evidence")
- Formula: $\log \text{BF}(k+1, k) \approx -0.5 \times (\text{BIC}_{k+1} - \text{BIC}_k)$
- Stops when evidence for adding clusters becomes weak

### 6. Frequency Section Data Preprocessing

Document the `freq_section_*` columns with current settings:

| Column | Frequency | Frame Range | Filtering | Downsampling |
|--------|-----------|-------------|-----------|--------------|
| freq_section_0p5hz | 0.5 Hz | 30-270 | Yes (10Hz LP) | Yes (6x) |
| freq_section_1hz | 1 Hz | 330-570 | Yes | Yes |
| freq_section_2hz | 2 Hz | 630-870 | Yes | Yes |
| freq_section_4hz | 4 Hz | 930-1170 | Yes | Yes |
| freq_section_10hz | 10 Hz | 1230-1470 | **No** | **No** |

Document `FREQ_SECTION_FILTER` dict and special `[60:-60]` slice for 10Hz.

### 7. Data Cleaning Thresholds

Document current settings from config.py:

| Parameter | Current Value | Purpose |
|-----------|---------------|---------|
| `QI_THRESHOLD` | **0.7** | Min step_up quality index |
| `MIN_BATCH_GOOD_CELLS` | **25** | Min cells per recording batch |
| `BASELINE_MAX_THRESHOLD` | **200.0** | Max baseline firing rate (Hz) |
| `DS_P_THRESHOLD` | **0.05** | Direction selectivity p-value |
| `VALID_AXON_TYPES` | **["rgc", "ac"]** | Included cell types |

## Output

Single comprehensive Markdown file (~250-350 lines) with proper LaTeX math formatting using `$...$` for inline and `$$...$$` for display equations. All settings documented with their **current values** from config.py.