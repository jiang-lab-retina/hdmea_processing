<!--
  Sync Impact Report
  ==================
  Version change: 0.0.0 → 1.0.0
  Modified principles: N/A (initial creation)
  Added sections: Purpose, Core Principles (I-VII), Architecture Rules,
    Notebook & Jupytext Rules, Artifacts/Caching/Reproducibility,
    Feature Extraction Standards, Analysis/Viz Standards, Testing/Quality Gates,
    Documentation Rules, Governance, Amendment/Update Checklist
  Removed sections: All template placeholders
  Templates requiring updates: ✅ All templates compatible
  Follow-up TODOs: None
-->

# HD-MEA Data Analysis Pipeline Constitution

## Purpose

This constitution governs the **HD-MEA Data Analysis Pipeline** project: a Python-based system for
processing high-density multi-electrode array (HD-MEA) recordings from `.cmcr` and `.cmtr` files,
extracting ~100 physiological features (on/off responses, receptive fields, nonlinearity indices,
direction selectivity, etc.), and supporting extensible downstream analyses and visualizations.

The project is **independent of legacy code** located in `Legacy_code/`. Legacy code serves as
reference only and MUST NOT be imported or executed by the new pipeline.

**Primary Goals**:

1. Reproducible, cacheable pipeline from raw recordings to feature tables
2. Modular, registry-based feature extraction that scales to 100+ features
3. Clear separation between library code (reusable) and notebooks (orchestration/visualization)
4. Long-term maintainability through strict architectural boundaries

---

## Core Principles

### I. Package-First Architecture (NON-NEGOTIABLE)

The **single source of truth** for all production logic is the Python package under `src/hdmea/`.

- **Production logic** (loading, preprocessing, feature extraction, analysis algorithms) MUST reside
  in the package, never in notebooks or standalone scripts.
- Notebooks and scripts MUST only **orchestrate, visualize, and validate**—they call package APIs.
- The package MUST be installable via `pip install -e .` for development.
- All imports from notebooks/scripts MUST use the package namespace (e.g., `from hdmea.io import ...`).

**Rationale**: Centralizing logic in an installable package ensures testability, reusability, and
prevents "notebook drift" where critical code becomes trapped in untestable notebooks.

---

### II. Modular Subpackage Layout (NON-NEGOTIABLE)

The package MUST follow this subpackage structure:

```
src/hdmea/
├── __init__.py
├── io/              # CMCR/CMTR loading, artifact read/write, Zarr operations
├── preprocess/      # Alignment, filtering, QC, data cleaning
├── features/        # Feature extractors, registry, feature schemas
├── analysis/        # Downstream analyses (clustering, statistics, etc.)
├── viz/             # Plotting, report generation utilities
├── pipeline/        # Step runner, caching, reproducibility orchestration
└── utils/           # Shared utilities (logging, config, helpers)
```

**Dependency Flow** (strictly enforced):

```
io → preprocess → features → analysis → viz
         ↑                       ↑
         └───────── pipeline ────┘ (orchestrates all)
```

- **Circular imports are FORBIDDEN**. If module A imports from module B, module B MUST NOT import
  from module A (directly or transitively).
- `pipeline/` is the orchestrator and MAY import from all other subpackages.
- `utils/` MUST have no internal dependencies (leaf module).

**Rationale**: Clear dependency direction prevents import hell and makes the codebase navigable.

---

### III. Explicit I/O and Pure Functions

- Functions SHOULD prefer explicit inputs and outputs over hidden global state.
- Side effects (file I/O, caching) MUST be isolated to clearly named functions or the `pipeline/`
  module.
- Feature extractors MUST be pure functions: given the same input data and parameters, they MUST
  produce identical outputs.
- Configuration MUST be passed explicitly (via config objects or parameters), not read from
  environment variables or global singletons inside library code.

**Rationale**: Explicit data flow enables testing, debugging, and reproducibility.

---

### IV. Single Zarr Artifact Per Recording

In the preprocessing stage:

- One `.cmcr` and/or `.cmtr` file MUST produce exactly **one Zarr archive** containing all
  information needed for downstream processing.
- The Zarr archive MUST include:
  - Raw or minimally processed spike data
  - Stimulus information and timing
  - Metadata (recording parameters, electrode layout, etc.)
  - QC flags and alignment results
- Downstream stages (features, analysis) MUST consume Zarr artifacts, NOT raw `.cmcr`/`.cmtr` files.

**Rationale**: A single, self-contained artifact per recording simplifies caching, sharing, and
ensures preprocessing runs exactly once.

---

### V. Data Format Standards

| Data Type | Format | Notes |
|-----------|--------|-------|
| Tabular data (feature tables, metadata) | **Parquet** | Portable, typed, efficient |
| Nested/hierarchical data (spike trains, recordings) | **Zarr** | Chunked, lazy-loadable |
| Configuration and parameters | **JSON** | Human-readable, versionable |
| Internal intermediates (temporary) | PKL | Only when Zarr/Parquet impractical; must be documented |

- PKL files MUST NOT be the primary artifact format for any pipeline stage.
- All artifact files MUST include version metadata in their structure or companion JSON.

---

### VI. No Hidden Global State

- Library code MUST NOT use module-level mutable state that affects function behavior.
- Logging configuration SHOULD be set at the application/notebook entry point, not inside library
  modules.
- Library code MUST use `logging.getLogger(__name__)`, never `print()` statements.

---

### VII. Independence from Legacy Code

- The new pipeline MUST NOT import from `Legacy_code/`.
- Legacy code is reference material only.
- If functionality from legacy code is needed, it MUST be reimplemented in the new package with
  proper tests and documentation.

---

## Architecture Rules

### Package Installation

```bash
# Development installation (from project root)
pip install -e ".[dev]"
```

The project MUST have a `pyproject.toml` with:

- Package metadata and version
- Dependencies (core and optional groups: dev, test, docs)
- Entry points for CLI commands (if any)

### Import Conventions

```python
# CORRECT: Import from package
from hdmea.io import load_cmcr, load_cmtr
from hdmea.features import FeatureRegistry
from hdmea.pipeline import run_pipeline

# FORBIDDEN: Relative imports that bypass package structure
from ...io import load_cmcr  # Never in notebooks

# FORBIDDEN: Legacy imports
from Legacy_code.Processing_2024 import ...  # Never allowed
```

### Logging Standards

```python
# In library modules:
import logging
logger = logging.getLogger(__name__)

# Use appropriate levels:
logger.debug("Detailed tracing info")
logger.info("Progress updates")
logger.warning("Recoverable issues")
logger.error("Failures")

# FORBIDDEN in library code:
print("...")  # Use logger instead
```

---

## Notebook & Jupytext Rules

### Jupytext Pairing (NON-NEGOTIABLE)

- Every notebook (`.ipynb`) MUST have a synced text representation via Jupytext.
- **Paired format**: percent-format `.py` (e.g., `notebook.ipynb` ↔ `notebook.py`)
- The **text file (`.py`) is the primary artifact** for:
  - Code review (clean diffs)
  - Refactoring
  - Search/grep operations
- The `.ipynb` file is for interactive execution only.

**Configuration** (in `pyproject.toml` or `jupytext.toml`):

```toml
[tool.jupytext]
formats = "ipynb,py:percent"
```

### Notebook Directory Structure

```
notebooks/
├── dev/           # Exploration, experiments, prototyping
├── validation/    # Sanity checks, regression tests, QC
├── reports/       # Shareable narrative notebooks (polished)
├── templates/     # Parameterized, re-runnable notebooks
├── flows/         # Auto-generated index files per analysis flow
└── _scratch/      # Temporary work (gitignored)
```

### Notebook Metadata Tagging

Every notebook MUST include YAML front matter (in the paired `.py` file) or notebook metadata with:

```yaml
# ---
# hdmea:
#   flows: ["set6a", "rf_v2"]           # Which analysis flows this belongs to
#   stage: "preprocess"                  # preprocess|features|analysis|viz|report
#   status: "draft"                      # draft|stable|deprecated
#   datasets: ["JIANG009", "JIANG010"]   # Optional: datasets used
# ---
```

### Flow Index Generation

- A script MUST exist to scan notebook tags and generate `notebooks/flows/<flow>.md` index files.
- These index files are the **official navigation** for each analysis flow.
- Index files MUST be regenerated on CI or pre-commit.

### Notebook Content Rules

- Notebooks MUST NOT contain function definitions longer than 10 lines (extract to package).
- Notebooks MUST NOT contain class definitions (extract to package).
- Notebooks SHOULD import and call package functions, not reimplement logic.
- Each notebook SHOULD have a clear purpose stated in the first cell.

---

## Artifacts, Caching, and Reproducibility

### Artifact Storage

```
artifacts/                    # Gitignored
├── raw/                      # Downloaded/copied raw data (.cmcr, .cmtr)
├── preprocessed/             # Zarr archives from preprocessing
├── features/                 # Parquet feature tables
├── analysis/                 # Analysis outputs (Parquet, Zarr)
├── viz/                      # Generated figures (PNG, SVG, PDF)
└── cache/                    # Intermediate cache objects
```

### Artifact Versioning

Every pipeline artifact MUST include metadata with:

| Field | Description | Example |
|-------|-------------|---------|
| `dataset_id` | Unique recording identifier | `"JIANG009_2024-01-15"` |
| `params_hash` | SHA256 of parameters/config | `"a1b2c3d4..."` |
| `code_version` | Package version or git hash | `"0.3.1"` or `"abc1234"` |
| `created_at` | ISO 8601 timestamp | `"2025-12-14T15:30:00Z"` |
| `stage` | Pipeline stage that produced it | `"preprocess"` |

Metadata MUST be stored:

- For Zarr: in `.zattrs` at the root level
- For Parquet: in file metadata or companion `.meta.json`
- For JSON configs: inline in the JSON structure

### Caching Policy

- Large data loads MUST be cached to disk in `artifacts/`.
- Cache invalidation occurs when `params_hash` or `code_version` changes.
- Analyses MUST consume artifacts, NOT re-run raw loading.
- Artifacts MUST be readable without running the entire pipeline (self-describing).

### Reproducibility Requirements

- Every pipeline run MUST log:
  - Input artifact paths and their metadata
  - Parameters used
  - Output artifact paths
  - Code version
- Pipeline SHOULD support "dry run" mode to show what would be executed.
- Random operations MUST use explicit seeds passed as parameters.

---

## Feature Extraction Standards

### Registry Pattern (NON-NEGOTIABLE)

Feature extraction MUST use a **registry pattern**:

```python
# In hdmea/features/registry.py
class FeatureRegistry:
    _registry: dict[str, FeatureExtractor] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(extractor_class):
            cls._registry[name] = extractor_class
            return extractor_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> FeatureExtractor:
        return cls._registry[name]
    
    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._registry.keys())
```

- **No giant "master function"** that switches on feature names.
- **No `eval()`-based dispatch**.
- New features are added by creating a new extractor class with the `@register` decorator.

### Feature Extractor Requirements

Every feature extractor MUST declare:

```python
@FeatureRegistry.register("on_off_index")
class OnOffIndexExtractor(FeatureExtractor):
    # Required metadata
    name: str = "on_off_index"
    version: str = "1.2.0"  # Version changes invalidate cache
    runtime_class: Literal["fast", "slow"] = "fast"
    
    # Required I/O declaration
    required_inputs: list[str] = ["spike_trains", "stimulus_timing"]
    output_columns: list[str] = ["on_index", "off_index", "on_off_ratio"]
    output_schema: dict = {
        "on_index": {"dtype": "float64", "unit": "spikes/s", "range": [0, None]},
        "off_index": {"dtype": "float64", "unit": "spikes/s", "range": [0, None]},
        "on_off_ratio": {"dtype": "float64", "unit": "dimensionless", "range": [-1, 1]},
    }
    
    def extract(self, data: FeatureInput) -> pd.DataFrame:
        """Extract on/off response indices."""
        ...
```

### Feature Documentation

- Every output column MUST have documented:
  - Data type
  - Unit (or "dimensionless")
  - Expected range (where applicable)
  - Definition/formula (in docstring or separate docs)
- Feature schemas MUST be exportable to JSON for external validation.

### Feature Output Policy

- Feature extraction MUST NOT silently overwrite prior outputs.
- If outputs exist with same `dataset_id` but different `params_hash` or `version`:
  - Raise an error by default
  - Allow overwrite only with explicit `force=True` parameter
- Feature tables MUST include a `_feature_version` column tracking extractor versions.

---

## Analysis and Visualization Standards

### Analysis Requirements

- Analyses MUST consume standardized feature tables (Parquet) + metadata artifacts.
- Analyses MUST NOT directly load raw recordings (consume preprocessed Zarr).
- Analyses MUST be deterministic: given fixed inputs and explicit random seed, outputs are identical.
- Random seeds MUST be passed as explicit parameters, not set globally.

### Visualization Standards

- All plotting code MUST reside in `hdmea/viz/`.
- Plotting functions MUST be callable from both notebooks and scripts.
- Plotting functions MUST return figure objects (not just display them).
- Plots SHOULD accept an optional `ax` parameter for subplot integration.
- Default styles SHOULD be defined in `hdmea/viz/styles.py` and applied consistently.

### Visualization Function Signature Pattern

```python
def plot_receptive_field(
    rf_data: pd.DataFrame,
    cell_id: str,
    *,
    ax: Optional[plt.Axes] = None,
    style: str = "default",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot receptive field for a single cell."""
    ...
```

---

## Testing and Quality Gates

### Required Test Categories

| Category | Location | Purpose | Required Coverage |
|----------|----------|---------|-------------------|
| Smoke tests | `tests/smoke/` | Every module imports without error | 100% of modules |
| Unit tests | `tests/unit/` | Critical feature math on synthetic data | All feature extractors |
| Integration tests | `tests/integration/` | End-to-end pipeline on tiny dataset | At least 1 full pipeline |
| Contract tests | `tests/contract/` | API stability, schema validation | Public APIs |

### Synthetic Test Data

- Unit tests MUST use small, synthetic inputs (not real recordings).
- A `tests/fixtures/` directory MUST contain:
  - Synthetic spike train generators
  - Minimal Zarr archives for testing
  - Expected output fixtures for regression testing

### Minimum Test Requirements

1. **Per-module smoke test**: Import succeeds, no syntax errors
2. **Per-feature extractor**: Unit test with synthetic input, verified output
3. **End-to-end test**: Full pipeline on a "tiny dataset" fixture (< 1MB)

### Definition of Done

A feature/analysis is complete when:

- [ ] Implementation in `src/hdmea/` (not in notebook)
- [ ] Type hints on all public functions
- [ ] Docstring with Args, Returns, Raises
- [ ] Unit test with synthetic data
- [ ] Artifact schema documented (if new artifact type)
- [ ] Registered in appropriate registry (if feature extractor)
- [ ] Version string updated (if modifying existing)

---

## Documentation Rules

### Spec vs Plan Separation (NON-NEGOTIABLE)

| Document | Purpose | Content | FORBIDDEN Content |
|----------|---------|---------|-------------------|
| `spec.md` | WHAT and WHY | Requirements, user stories, success criteria | Library names, implementation details, performance tuning |
| `plan.md` | HOW | Architecture, libraries, data structures, performance | Product requirements, user stories, business logic |

- If **technical details leak into spec.md**, they MUST be moved to plan.md immediately.
- If **product requirements leak into plan.md**, they MUST be moved to spec.md immediately.
- PR reviews MUST check for this separation.

### Code Documentation

- **Public functions**: MUST have docstrings with Args, Returns, Raises
- **Private functions**: SHOULD have docstrings if logic is non-obvious
- **Modules**: MUST have module-level docstring explaining purpose
- **Classes**: MUST have class-level docstring explaining responsibility

### Docstring Format

Use Google-style docstrings:

```python
def extract_on_off_index(
    spike_trains: dict[str, np.ndarray],
    stim_onset: float,
    stim_offset: float,
    window_ms: float = 200.0,
) -> pd.DataFrame:
    """Extract ON and OFF response indices from spike trains.
    
    Args:
        spike_trains: Dict mapping cell_id to spike time arrays (seconds)
        stim_onset: Stimulus onset time (seconds)
        stim_offset: Stimulus offset time (seconds)
        window_ms: Analysis window duration (milliseconds)
    
    Returns:
        DataFrame with columns: cell_id, on_index, off_index, on_off_ratio
    
    Raises:
        ValueError: If window_ms <= 0 or stim_offset <= stim_onset
    """
    ...
```

---

## Governance

### Constitution Authority

- This constitution **supersedes all other practices** when conflicts arise.
- All PRs and code reviews MUST verify compliance with this constitution.
- Violations MUST be corrected before merge.

### Amendment Process

1. Propose amendment in a dedicated PR with `docs: constitution amendment` prefix
2. Document rationale for the change
3. Update version number according to semantic versioning:
   - **MAJOR**: Backward-incompatible principle changes or removals
   - **MINOR**: New principles or significant expansions
   - **PATCH**: Clarifications, typos, non-semantic refinements
4. Update `LAST_AMENDED_DATE`
5. Require approval from project maintainer(s)
6. Update dependent templates if affected

### Compliance Review

- Quarterly review of codebase against constitution principles
- New team members MUST read constitution before contributing
- Constitution violations in existing code SHOULD be tracked as technical debt

---

## Amendment / Constitution Update Checklist

When amending this constitution:

- [ ] **Rationale documented**: Why is this change needed?
- [ ] **Version bumped**: MAJOR/MINOR/PATCH as appropriate
- [ ] **Date updated**: `LAST_AMENDED_DATE` set to today
- [ ] **Templates checked**: Review and update if needed:
  - [ ] `.specify/templates/plan-template.md`
  - [ ] `.specify/templates/spec-template.md`
  - [ ] `.specify/templates/tasks-template.md`
- [ ] **README updated**: If amendment affects getting-started instructions
- [ ] **No placeholder tokens**: All `[BRACKETS]` resolved
- [ ] **MUST/SHOULD language**: Principles are declarative and testable
- [ ] **Sync Impact Report**: HTML comment at top of file updated
- [ ] **PR approved**: By project maintainer(s)

---

**Version**: 1.0.0 | **Ratified**: 2025-12-14 | **Last Amended**: 2025-12-14
