# Implementation Plan: HD-MEA Data Analysis Pipeline v1

**Branch**: `001-hdmea-modular-pipeline` | **Date**: 2025-12-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/001-hdmea-modular-pipeline/spec.md`

---

## Summary

Build a modular Python pipeline for processing HD-MEA recordings from external `.cmcr`/`.cmtr` files through a two-stage architecture:

1. **Stage 1 (Data Loading)**: External raw files → single Zarr artifact per recording
2. **Stage 2 (Feature Extraction)**: Zarr → Zarr with features stored under each unit

The pipeline uses a registry pattern for ~100 extensible feature extractors, produces self-describing Zarr artifacts with versioned metadata, and optionally exports to Parquet for cross-recording analysis. The system is fully independent of legacy code (reference only).

---

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**:
- `zarr` - Hierarchical data storage (Zarr format)
- `pyarrow` / `pandas` - Parquet I/O
- `numpy` - Numerical operations
- `McsPy.McsCMOSMEA` - Raw file reading (assumed available per spec)
- `scipy` - Signal processing, statistical tests
- `scikit-image` - Gaussian fitting for RF analysis

**Storage**: Zarr archives (local filesystem), Parquet files (optional export), JSON configs  
**Testing**: pytest + fixtures with synthetic data  
**Target Platform**: Windows 10+ (primary), Linux (secondary)  
**Project Type**: Single Python package (`src/hdmea/`)  
**Performance Goals**:
- Stage 1: <10 minutes for typical recording (<30 min, <200 units)
- Stage 2: <5 minutes for all core features
- Feature access: <1 second from Zarr

**Constraints**:
- Must run on single machine (no distributed processing)
- Must work with network paths (UNC paths like `//server/share/`)
- Must NOT import from `./Legacy_code/`

**Scale/Scope**:
- ~100 feature extractors (v1 supports core "set6a" workflow)
- Recordings typically <1 hour, <500 units
- ~7 stimulus types in v1 (configurable)

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | All logic in `src/hdmea/`, notebooks orchestrate only |
| II. Modular Subpackage Layout | ✅ PASS | Structure follows io/preprocess/features/analysis/viz/pipeline/utils |
| III. Explicit I/O and Pure Functions | ✅ PASS | Explicit config passing, seeded randomness |
| IV. Single Zarr Per Recording | ✅ PASS | Stage 1 produces exactly one Zarr; Stage 2 writes to same Zarr |
| V. Data Format Standards | ✅ PASS | Zarr for hierarchical, Parquet for tabular, JSON for config |
| VI. No Hidden Global State | ✅ PASS | Logger via `__name__`, config passed explicitly |
| VII. Independence from Legacy | ✅ PASS | No imports from Legacy_code; reference only |
| Notebook Jupytext Pairing | ✅ PASS | Notebooks paired with `.py` percent format |
| Feature Registry Pattern | ✅ PASS | Decorator-based registry, no `eval()` dispatch |
| Artifact Versioning | ✅ PASS | All artifacts include dataset_id, params_hash, code_version, timestamp |
| Spec vs Plan Separation | ✅ PASS | spec.md contains WHAT/WHY; plan.md contains HOW |

**Gate Result**: ✅ ALL GATES PASS - Proceed to Phase 0

---

## Project Structure

### Documentation (this feature)

```text
specs/001-hdmea-modular-pipeline/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output (research findings)
├── data-model.md        # Phase 1 output (entity schemas)
├── quickstart.md        # Phase 1 output (getting started)
└── checklists/          # Validation checklists
    └── requirements.md  # Requirements checklist
```

### Source Code (repository root)

```text
src/hdmea/
├── __init__.py           # Package init, version
├── io/                   # Raw file I/O and Zarr operations
│   ├── __init__.py
│   ├── cmcr.py           # CMCR file reading via McsPy
│   ├── cmtr.py           # CMTR file reading via McsPy
│   ├── zarr_store.py     # Zarr read/write operations
│   └── parquet_export.py # Optional Parquet export
├── preprocess/           # Data cleaning and alignment
│   ├── __init__.py
│   ├── alignment.py      # Stimulus timing alignment
│   ├── filtering.py      # Signal filtering
│   └── quality.py        # Quality checks and flags
├── features/             # Feature extractors and registry
│   ├── __init__.py
│   ├── registry.py       # FeatureRegistry class
│   ├── base.py           # FeatureExtractor base class
│   ├── on_off/           # ON/OFF response features
│   ├── receptive_field/  # STA and RF features
│   ├── direction/        # DSI, OSI features
│   ├── chromatic/        # Color response features
│   ├── frequency/        # Chirp/frequency response
│   ├── baseline/         # Baseline firing features
│   └── cell_type/        # Cell classification
├── analysis/             # Downstream analyses
│   ├── __init__.py
│   └── summary.py        # Cross-recording summaries
├── viz/                  # Visualization utilities
│   ├── __init__.py
│   ├── styles.py         # Plot styles and themes
│   ├── raster.py         # Raster plots
│   ├── rf_plots.py       # Receptive field visualizations
│   └── tuning.py         # Tuning curve plots
├── pipeline/             # Pipeline orchestration
│   ├── __init__.py
│   ├── runner.py         # Step runner with caching
│   ├── flows.py          # Named flow definitions
│   └── config.py         # Configuration loading
└── utils/                # Shared utilities
    ├── __init__.py
    ├── logging.py        # Logging setup
    ├── hashing.py        # Config/params hashing
    └── validation.py     # Input validation

tests/
├── fixtures/             # Synthetic test data
│   ├── synthetic_spikes.py
│   └── minimal_zarr/     # Minimal Zarr for testing
├── smoke/                # Import smoke tests
├── unit/                 # Unit tests per module
│   ├── test_registry.py
│   ├── test_on_off.py
│   └── ...
├── integration/          # Integration tests
│   └── test_pipeline.py
└── conftest.py           # Pytest fixtures

notebooks/
├── dev/                  # Exploration notebooks
├── validation/           # QA and regression notebooks
├── reports/              # Shareable reports
├── templates/            # Parameterized notebooks
├── flows/                # Auto-generated flow indexes
└── _scratch/             # Temporary work (gitignored)

config/
├── flows/                # Flow configurations
│   └── set6a_full.json
├── stimuli/              # Stimulus definitions (configurable)
│   ├── step_up_5s_5i_3x.json
│   ├── moving_h_bar_s5_d8_3x.json
│   └── ...
└── defaults.json         # Default parameters

artifacts/                # Gitignored output directory
├── {dataset_id}.zarr/    # Per-recording Zarr archives
└── ...

exports/                  # Gitignored Parquet exports
└── {dataset_id}_features.parquet
```

**Structure Decision**: Single Python package under `src/hdmea/` following constitution-mandated modular layout. Dependencies flow `io → preprocess → features → analysis → viz` with `pipeline/` as orchestrator.

---

## Complexity Tracking

> **No violations requiring justification**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | — | — |

---

## Phase 0: Research Summary

See [research.md](./research.md) for detailed findings.

### Key Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Zarr library | `zarr-python` v2.x | Mature, chunked, supports `.zattrs` metadata |
| Parquet library | `pyarrow` | Fast, full Parquet support, pandas integration |
| Registry pattern | Class-based decorator | Declarative, supports metadata, no `eval()` |
| Config format | JSON with schema validation | Human-readable, versionable, `pydantic` for validation |
| Hashing | SHA256 of JSON-serialized config | Deterministic, standard |
| Testing | pytest + hypothesis (optional) | Standard Python testing, property-based for numerics |

---

## Phase 1: Design Artifacts

### Data Model

See [data-model.md](./data-model.md) for complete entity schemas.

### Contracts

See [contracts/](./contracts/) for API contracts.

### Quickstart

See [quickstart.md](./quickstart.md) for getting started guide.

---

## Post-Design Constitution Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | All entities in `src/hdmea/` |
| II. Modular Subpackage Layout | ✅ PASS | Subpackages match constitution |
| III. Explicit I/O | ✅ PASS | Config objects passed explicitly |
| IV. Single Zarr Per Recording | ✅ PASS | Data model reflects single artifact |
| V. Data Format Standards | ✅ PASS | Zarr, Parquet, JSON only |
| VI. No Hidden Global State | ✅ PASS | No module-level mutable state |
| VII. Independence from Legacy | ✅ PASS | No legacy references in design |

**Final Gate Result**: ✅ ALL GATES PASS

---

## Next Steps

1. Run `/speckit.tasks` to generate implementation tasks
2. Begin implementation following task order
3. Run tests after each module
4. Validate against legacy outputs
