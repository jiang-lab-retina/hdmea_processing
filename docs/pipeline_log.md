# HD-MEA Pipeline Changelog

This document logs all major changes to the HD-MEA data analysis pipeline.
Entries are in reverse chronological order (newest first).

---

## [2025-12-16] Add Section Time Loading

**Change**: Added `add_section_time()` function to load movie section timing from playlist and movie_length CSV configuration files. Computes frame boundaries for each movie and stores them in Zarr under `stimulus/section_time/`.

**Affected**:
- `hdmea.io.section_time` (new module)
- `hdmea.pipeline.__init__` (exports new function)
- Zarr structure: new `stimulus/section_time/` and `stimulus/light_template/` groups

**Migration**: No migration needed - this is a new optional feature.

**PR/Branch**: `004-load-section-time`

---

## [2025-12-16] Pipeline Documentation Requirements

**Change**: Added constitution requirement for pipeline documentation files (`pipeline_explained.md` and `pipeline_log.md`). All major pipeline changes MUST be logged.

**Affected**:
- `.specify/memory/constitution.md` (new Pipeline Documentation section)
- `docs/pipeline_explained.md` (new file)
- `docs/pipeline_log.md` (this file)

**Migration**: No migration needed - documentation is additive.

**PR/Branch**: `004-load-section-time`

---

## [2025-12-14] Initial Pipeline Implementation

**Change**: Initial implementation of the HD-MEA data analysis pipeline with two-stage architecture:
- Stage 1: `load_recording()` - Load CMCR/CMTR files into Zarr
- Stage 2: `extract_features()` - Extract registered features from Zarr

**Affected**:
- `hdmea.pipeline.runner` - Pipeline runner with caching
- `hdmea.io.cmcr` - CMCR file loading
- `hdmea.io.cmtr` - CMTR file loading
- `hdmea.io.zarr_store` - Zarr read/write operations
- `hdmea.features.registry` - Feature extractor registry

**Migration**: N/A - initial implementation.

**PR/Branch**: `001-hdmea-modular-pipeline`

---

## Template for New Entries

```markdown
## [YYYY-MM-DD] Brief Title

**Change**: Description of what changed

**Affected**: List of affected modules/components

**Migration**: Steps needed to update existing code (if applicable)

**PR/Branch**: Reference to PR or branch (if applicable)
```

