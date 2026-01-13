# Implementation Plan: DEC-Refined RGC Subtype Clustering

**Branch**: `014-dec-rgc-clustering` | **Date**: 2026-01-13 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/014-dec-rgc-clustering/spec.md`

---

## Summary

Implement a two-phase RGC subtype clustering pipeline:
1. **Phase 1**: Train CNN autoencoder (reconstruction-only), select k* via diagonal GMM + BIC
2. **Phase 2**: Refine clusters using DEC/IDEC with GMM-initialized centers

Key technical approach: Reuse preprocessing and model architecture from `Autoencoder_method`, but remove weak supervision (SupCon, purity loss) and add DEC refinement stage. Validate via ipRGC enrichment.

---

## Technical Context

**Language/Version**: Python 3.10+ (matching Autoencoder_method)  
**Primary Dependencies**:
- PyTorch 2.0+ (autoencoder, DEC)
- scikit-learn (GMM, BIC, StandardScaler)
- NumPy, Pandas, SciPy (data manipulation)
- UMAP-learn (visualization)
- tqdm (progress bars)

**Storage**: 
- Input: Parquet (feature tables)
- Output: Parquet (embeddings, assignments), JSON (metrics, BIC curves), PNG (plots)
- Checkpoints: PyTorch .pt files

**Testing**: pytest with synthetic data fixtures  
**Target Platform**: Linux/Windows workstation with CUDA GPU  
**Project Type**: Single Python package (analysis pipeline)  
**Performance Goals**: < 30 minutes per group on GPU; 2000+ cells without memory errors  
**Constraints**: Must be compatible with Autoencoder_method preprocessing; GPU memory < 8GB  
**Scale/Scope**: ~5000 cells total across 3 groups (DSGC, OSGC, Other)

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ⚠️ DEVIATION | Code in `dataframe_phase/classification_v2/divide_conquer_method/` not `src/hdmea/` - follows Autoencoder_method precedent |
| II. Modular Subpackage Layout | ✅ PASS | Will use same structure as Autoencoder_method |
| III. Explicit I/O and Pure Functions | ✅ PASS | Config-driven, explicit parameters |
| IV. Single HDF5 Artifact Per Recording | N/A | Consumes Parquet, not raw recordings |
| V. Data Format Standards | ✅ PASS | Parquet for tables, JSON for config |
| VI. No Hidden Global State | ✅ PASS | Config object pattern from Autoencoder_method |
| VII. Independence from Legacy Code | ✅ PASS | No legacy imports |

**Deviation Justification**: This is an analysis pipeline in `dataframe_phase/classification_v2/` alongside `Autoencoder_method/` and `Baden_method/`. These analysis methods operate on extracted feature parquet files, not raw recordings, and are not part of the core `src/hdmea/` processing pipeline.

---

## Project Structure

### Documentation (this feature)

```text
specs/014-dec-rgc-clustering/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output (already exists)
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (pipeline API)
├── checklists/          # Quality checklists
│   └── requirements.md
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
dataframe_phase/classification_v2/divide_conquer_method/
├── __init__.py              # Package init
├── config.py                # Configuration parameters
├── run_pipeline.py          # Main entry point / CLI
├── data_loader.py           # Load and filter parquet data
├── preprocessing.py         # Segment-specific signal processing
├── grouping.py              # DS/OS group assignment
├── models/
│   ├── __init__.py
│   ├── autoencoder.py       # MultiSegmentAutoencoder (reconstruction-only)
│   ├── encoders.py          # Segment encoders (reuse from AE_method)
│   ├── decoders.py          # Segment decoders (reuse from AE_method)
│   └── dec.py               # DEC/IDEC implementation
├── clustering/
│   ├── __init__.py
│   ├── gmm_bic.py           # GMM fitting + BIC selection
│   └── dec_refine.py        # DEC refinement loop
├── validation/
│   ├── __init__.py
│   └── iprgc_metrics.py     # ipRGC purity and enrichment
├── train.py                 # AE training loop (reconstruction-only)
├── embed.py                 # Embedding extraction
├── evaluation.py            # Comparison metrics
├── visualization.py         # UMAP, BIC curves, prototypes
├── results/                 # Output data files
├── plots/                   # Generated visualizations
└── models_saved/            # Trained model checkpoints
```

**Structure Decision**: Follows Autoencoder_method layout for consistency, adding `clustering/` and `validation/` submodules for the new GMM/DEC and ipRGC validation logic.

---

## Complexity Tracking

| Deviation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Code in dataframe_phase/ not src/hdmea/ | Analysis pipelines for classification live here alongside existing methods | Moving to hdmea would break existing Autoencoder_method patterns |

---

## Phase 0: Research Complete

See [research.md](./research.md) for algorithm details on:
- Baden-style GMM + BIC selection
- DEC/IDEC refinement
- ipRGC enrichment metrics

**Key Decisions**:
1. Use **minimum BIC** for k selection (not log Bayes factor threshold)
2. Use **IDEC** (DEC + reconstruction) to prevent embedding collapse
3. Initialize DEC centers from **GMM means** (not k-means)
4. **Student-t kernel** with α=1 for soft assignments

---

## Phase 1: Design & Contracts

### Artifacts Generated

1. **data-model.md**: Input/output schemas (already exists)
2. **contracts/pipeline_api.md**: CLI and Python API specifications
3. **quickstart.md**: Usage examples and quick start guide

See individual files for details.

---

## Implementation Notes

### Code Reuse from Autoencoder_method

| Component | Reuse | Modification |
|-----------|-------|--------------|
| `preprocessing.py` | Full reuse | None - same segment processing |
| `models/encoders.py` | Full reuse | None - same CNN architecture |
| `models/decoders.py` | Full reuse | None - same CNN architecture |
| `models/autoencoder.py` | Partial | Remove SupCon/purity loss integration |
| `train.py` | Partial | Simplify to reconstruction-only |
| `clustering.py` | Partial | Extract GMM/BIC, add DEC |
| `visualization.py` | Partial | Add comparison plots, ipRGC highlights |
| `config.py` | Template | New parameters for DEC, DS/OS thresholds |

### New Components to Implement

1. **`models/dec.py`**: DEC/IDEC layer with Student-t soft assignments
2. **`clustering/dec_refine.py`**: DEC training loop with convergence check
3. **`validation/iprgc_metrics.py`**: Enrichment and purity calculations
4. **`grouping.py`**: DS > OS priority assignment (new logic)

### Testing Strategy

1. **Unit tests**: Synthetic embeddings for GMM/DEC, mock traces for preprocessing
2. **Integration test**: Full pipeline on subset (500 cells) with known ipRGC labels
3. **Regression test**: Compare BIC curve shape to known-good output
