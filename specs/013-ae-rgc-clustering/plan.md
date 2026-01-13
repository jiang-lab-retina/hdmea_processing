# Implementation Plan: Weakly Supervised RGC Clustering with Autoencoder Features

**Branch**: `013-ae-rgc-clustering` | **Date**: 2026-01-11 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/013-ae-rgc-clustering/spec.md`

## Summary

Build a weakly supervised RGC subtype clustering pipeline that replaces sPCA/SVD/PCA feature extraction with autoencoder-derived 49D latent embeddings while maintaining Baden-style GMM clustering, bootstrap stability testing, and adding cross-validation via omitted-label purity scoring. All code is self-contained within `dataframe_phase/classification_v2/Autoencoder_method/`.

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: PyTorch (autoencoder), scikit-learn (GMM), scipy (signal processing), pandas/numpy (data), umap-learn (visualization), matplotlib (plotting)  
**Storage**: Parquet (input), HDF5/PKL (models), Parquet (results)  
**Testing**: pytest with synthetic data fixtures  
**Target Platform**: Linux/Windows workstation with optional CUDA GPU  
**Project Type**: Single analysis pipeline (standalone module)  
**Performance Goals**: Full pipeline completes in <4 hours for 10,000 cells  
**Constraints**: <32GB RAM, optional GPU acceleration for AE training  
**Scale/Scope**: Datasets up to 10,000 cells, 49D embeddings, up to 100 clusters total

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Code in `dataframe_phase/classification_v2/Autoencoder_method/` - standalone module, not in hdmea package (acceptable for analysis code) |
| II. Modular Subpackage Layout | ✅ PASS | Module follows clear internal structure (config, preprocessing, models, clustering, etc.) |
| III. Explicit I/O and Pure Functions | ✅ PASS | All parameters via config.py, no hidden global state |
| IV. Single HDF5 Artifact Per Recording | N/A | This pipeline consumes existing parquet, produces parquet/pkl outputs |
| V. Data Format Standards | ✅ PASS | Parquet for tables, PKL for trained models (documented) |
| VI. No Hidden Global State | ✅ PASS | Config passed explicitly, logging via logger |
| VII. Independence from Legacy Code | ✅ PASS | No imports from Legacy_code/ or Baden_method/ |

**Gate Status**: ✅ PASSED - Ready for Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/013-ae-rgc-clustering/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output - design decisions
├── data-model.md        # Phase 1 output - entity schemas
├── quickstart.md        # Phase 1 output - usage guide
├── contracts/           # Phase 1 output - API contracts
│   └── pipeline_api.md
└── checklists/
    └── requirements.md  # Validation checklist
```

### Source Code (repository root)

```text
dataframe_phase/classification_v2/Autoencoder_method/
├── __init__.py              # Package marker
├── config.py                # All configurable parameters
├── data_loader.py           # Load parquet, extract arrays, filtering
├── preprocessing.py         # Signal conditioning (filter, resample, concat)
├── grouping.py              # Coarse label computation + precedence
├── models/
│   ├── __init__.py
│   ├── encoders.py          # Per-segment encoder architectures
│   ├── decoders.py          # Per-segment decoder architectures
│   ├── autoencoder.py       # Multi-encoder/decoder AE model
│   └── losses.py            # Reconstruction + supervised contrastive loss
├── train.py                 # AE training loop + checkpointing
├── embed.py                 # Extract 49D embeddings from trained AE
├── clustering.py            # Diagonal GMM, BIC selection, group-constrained clustering
├── stability.py             # Bootstrap subsampling stability testing
├── crossval.py              # Omitted-label CV turns + purity computation
├── visualization.py         # UMAP, BIC curves, response prototypes
├── evaluation.py            # Metrics computation (purity, silhouette, stability)
├── run_pipeline.py          # Single entry point for full pipeline
├── models_saved/            # Trained autoencoder checkpoints
├── results/                 # Clustering outputs, embeddings, CV tables
└── plots/                   # Generated visualizations
```

**Structure Decision**: Single-module structure within `dataframe_phase/classification_v2/`, parallel to existing `Baden_method/` folder. This maintains project organization while keeping all AE-specific code self-contained.

## Architecture Overview

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PREPROCESSING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Load parquet → 2. Filter cells → 3. Compute group labels                │
│  4. Preprocess traces (filter/downsample) → 5. Build segment map            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUTOENCODER TRAINING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  6. Build segment-wise encoders (10 segments) → 7. Train with:              │
│     - Reconstruction loss (weighted per segment)                             │
│     - Supervised contrastive loss (using active group labels)               │
│  8. Checkpoint best model                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING EXTRACTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  9. Load trained AE → 10. Encode all cells → 11. Concatenate to 49D         │
│  12. Z-score standardize embeddings                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BADEN-STYLE CLUSTERING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  13. For each disjoint group:                                                │
│      - Fit diagonal GMM for k=1..k_max                                       │
│      - Select k* via BIC/logBF threshold                                     │
│      - Assign cluster labels                                                 │
│  14. Merge results with group prefixes                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VALIDATION                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  15. Bootstrap stability (90% subsampling × 20 iterations)                  │
│  16. Cross-validation turns (omit each label, measure purity)               │
│  17. Compute aggregate CVScore                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VISUALIZATION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  18. UMAP embeddings (colored by group/cluster)                              │
│  19. BIC curves per group                                                    │
│  20. Response prototypes per cluster                                         │
│  21. CV summary tables                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Segment-Wise Autoencoder Design

```
Input Traces (preprocessed)
     │
     ├── freq_section_0p5hz ──► Encoder_0 ──► z_0 (4D) ──► Decoder_0 ──► recon_0
     ├── freq_section_1hz   ──► Encoder_1 ──► z_1 (4D) ──► Decoder_1 ──► recon_1
     ├── freq_section_2hz   ──► Encoder_2 ──► z_2 (4D) ──► Decoder_2 ──► recon_2
     ├── freq_section_4hz   ──► Encoder_3 ──► z_3 (4D) ──► Decoder_3 ──► recon_3
     ├── freq_section_10hz  ──► Encoder_4 ──► z_4 (4D) ──► Decoder_4 ──► recon_4
     ├── green_blue         ──► Encoder_5 ──► z_5 (6D) ──► Decoder_5 ──► recon_5
     ├── bar_concat         ──► Encoder_6 ──► z_6 (12D)──► Decoder_6 ──► recon_6
     ├── sta_time_course    ──► Encoder_7 ──► z_7 (3D) ──► Decoder_7 ──► recon_7
     ├── iprgc_test         ──► Encoder_8 ──► z_8 (4D) ──► Decoder_8 ──► recon_8
     └── step_up            ──► Encoder_9 ──► z_9 (4D) ──► Decoder_9 ──► recon_9
                                    │
                                    ▼
                          z = concat(z_0..z_9) = 49D
                                    │
                                    ▼
                      Supervised Contrastive Loss
                      (using active group labels)
```

### Loss Function

```
L_total = L_reconstruction + β × L_supcon

L_reconstruction = Σ_s (λ_s × MSE(x_s, x̂_s))
    where λ_s ∝ 1/len(x_s)  (inverse length weighting)

L_supcon = SupConLoss(z, group_labels)
    - Pulls embeddings of same group closer
    - Pushes embeddings of different groups apart
    - Temperature τ = 0.1 (configurable)
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Encoder architecture | 1D CNN per segment | Captures temporal patterns, handles variable lengths |
| Weak supervision loss | Supervised contrastive | Better manifold separation than center loss |
| Clustering | Diagonal GMM | Matches Baden methodology, computationally efficient |
| Group constraint | Cluster per group separately | Guaranteed group purity, interpretable results |
| CV strategy | Omit-one-label turns | Tests generalization without subtype labels |
| Stability test | 90% bootstrap × 20 | Matches Baden validation approach |

## Complexity Tracking

No constitution violations requiring justification.

## Dependencies

### Python Packages (requirements.txt)

```
torch>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
umap-learn>=0.5.4
matplotlib>=3.7.0
seaborn>=0.13.0
tqdm>=4.66.0
```

### Optional (GPU acceleration)

```
# CUDA 11.8+ for PyTorch GPU support
```

## Phase Outputs

- **Phase 0**: [research.md](./research.md) - All design decisions documented
- **Phase 1**: [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)
- **Phase 2**: [tasks.md](./tasks.md) - Implementation tasks (via `/speckit.tasks`)
