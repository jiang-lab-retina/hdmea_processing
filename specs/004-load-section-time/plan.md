# Implementation Plan: Load Section Time Metadata

**Branch**: `004-load-section-time` | **Date**: 2025-12-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-load-section-time/spec.md`

## Summary

Add `add_section_time()` function to compute and store movie section timing boundaries in Zarr archives. The function reads playlist and movie_length CSV configurations, computes frame boundaries for each movie using the legacy algorithm, extracts/averages light templates, and writes results to `stimulus/section_time/` and `stimulus/light_template/` groups in the Zarr.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: pandas, numpy, zarr, itertools (stdlib)
**Storage**: Zarr archives (existing infrastructure in `hdmea.io.zarr_store`)
**Testing**: pytest with synthetic fixtures
**Target Platform**: Windows/Linux workstations, network file access
**Project Type**: Single Python package (hdmea)
**Performance Goals**: < 5 seconds for typical playlists (< 20 movies)
**Constraints**: Must work with UNC network paths (//Jiangfs1/...)
**Scale/Scope**: Single recording at a time, ~5-30 movies per playlist

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First Architecture | ✅ PASS | Implementation in `src/hdmea/io/section_time.py` |
| II. Modular Subpackage Layout | ✅ PASS | New module in `io/` subpackage (correct location for I/O operations) |
| III. Explicit I/O and Pure Functions | ✅ PASS | All parameters explicit; file I/O isolated to named functions |
| IV. Single Zarr Artifact Per Recording | ✅ PASS | Writes to existing Zarr, doesn't create new artifacts |
| V. Data Format Standards | ✅ PASS | Uses Zarr for hierarchical data, reads CSV configuration |
| VI. No Hidden Global State | ✅ PASS | Uses `logging.getLogger(__name__)`, no module-level state |
| VII. Independence from Legacy Code | ✅ PASS | Reimplements algorithm from scratch, no legacy imports |
| Feature Output Policy | ✅ PASS | Requires `force=True` to overwrite (per clarification) |
| Pipeline Documentation | ✅ PASS | Will update `docs/pipeline_explained.md` and `docs/pipeline_log.md` |

**Gate Result**: ✅ ALL GATES PASS - Proceed to implementation

## Project Structure

### Documentation (this feature)

```text
specs/004-load-section-time/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── checklists/
    └── requirements.md  # Spec validation checklist
```

### Source Code (repository root)

```text
src/hdmea/
├── io/
│   ├── __init__.py
│   ├── section_time.py    # NEW: Main implementation
│   └── zarr_store.py      # Existing: Zarr operations (used by section_time)
├── pipeline/
│   └── __init__.py        # UPDATE: Export add_section_time
└── utils/
    └── exceptions.py      # Existing: Custom exceptions

tests/
├── unit/
│   └── test_section_time.py  # NEW: Unit tests with synthetic data
└── fixtures/
    └── section_time/         # NEW: Test fixtures (CSV files, minimal Zarr)
```

**Structure Decision**: Single project structure matching existing hdmea package layout. New module `section_time.py` placed in `io/` subpackage per constitution (I/O operations).

## Complexity Tracking

No constitution violations requiring justification.

---

## Phase 0: Research

See [research.md](./research.md) for detailed findings.

### Key Decisions

1. **Algorithm Implementation**: Direct port of legacy algorithm with explicit parameters
2. **Error Handling**: Return `False` for file errors; raise `FileExistsError` for overwrite protection
3. **Light Template Averaging**: Use `itertools.zip_longest` with `fillvalue=np.nan` then `np.nanmean`
4. **Zarr Integration**: Use existing `open_recording_zarr()` from `zarr_store.py`

---

## Phase 1: Design

See [data-model.md](./data-model.md) for entity definitions.

### API Design

```python
def add_section_time(
    zarr_path: Union[str, Path],
    playlist_name: str,
    *,
    playlist_file_path: Optional[Union[str, Path]] = None,
    movie_length_file_path: Optional[Union[str, Path]] = None,
    repeats: int = 1,
    pad_frame: int = 180,
    pre_margin_frame_num: int = 60,
    post_margin_frame_num: int = 120,
    force: bool = False,
) -> bool:
    """Add section time metadata to a Zarr recording."""
```

### Data Flow

```
playlist.csv ──┐
               ├──> _get_movie_start_end_frame() ──> section_time_auto
movie_length.csv ─┤                                   template_auto
                  │
Zarr (read) ──────┘
    ├── stimulus/light_reference/
    └── metadata/frame_time

                          ↓

Zarr (write) ─────────────┘
    ├── stimulus/section_time/{movie_name}
    └── stimulus/light_template/{movie_name}
```

### Error States

| Condition | Behavior |
|-----------|----------|
| Playlist file not found | Log warning, return `False` |
| Movie length file not found | Log warning, return `False` |
| Playlist name not in CSV | Log error with available names, return `False` |
| Movie not in movie_length | Log warning, skip movie, continue |
| Zarr not found | Log error, return `False` |
| No frame_time in Zarr | Log error, return `False` |
| section_time exists, force=False | Raise `FileExistsError` |
| section_time exists, force=True | Delete and overwrite |

---

## Post-Design Constitution Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Package-First | ✅ | All logic in `hdmea.io.section_time` |
| II. Modular Layout | ✅ | Correct subpackage, no circular imports |
| III. Explicit I/O | ✅ | All parameters explicit, side effects isolated |
| IV. Single Zarr | ✅ | Modifies existing Zarr, no new artifacts |
| V. Data Formats | ✅ | Zarr for hierarchical, CSV for config |
| VI. No Global State | ✅ | Uses `logging.getLogger(__name__)` |
| VII. No Legacy | ✅ | Clean reimplementation |
| Pipeline Docs | ✅ | Will update after implementation |

**Final Gate Result**: ✅ READY FOR IMPLEMENTATION

