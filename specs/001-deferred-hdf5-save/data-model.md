# Data Model: Deferred HDF5 Save Pipeline

**Feature**: 001-deferred-hdf5-save  
**Date**: 2024-12-20

## Entities

### PipelineSession

The central container for accumulated pipeline data. Mirrors the HDF5 structure in memory.

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np

class SaveState(Enum):
    """Session persistence state."""
    DEFERRED = "deferred"  # Data in memory only
    SAVED = "saved"        # Data persisted to HDF5


@dataclass
class PipelineSession:
    """
    Container for in-memory pipeline data accumulated across steps.
    
    Attributes:
        dataset_id: Unique identifier for the recording (e.g., "2025.04.10-11.12.57-Rec")
        save_state: Current persistence state (DEFERRED or SAVED)
        hdf5_path: Path to HDF5 file (set after save(), None while deferred)
        output_dir: Default directory for save operations
        
        units: Unit data keyed by unit_id
        metadata: Recording metadata (acquisition_rate, timing, etc.)
        stimulus: Stimulus data (light_reference, frame_times, section_time)
        source_files: Paths to original CMCR/CMTR files
        
        completed_steps: Set of pipeline steps that have been run
        warnings: Accumulated warnings from pipeline operations
    """
    # Identity
    dataset_id: str
    save_state: SaveState = SaveState.DEFERRED
    hdf5_path: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))
    
    # Data containers (mirror HDF5 structure)
    units: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    stimulus: Dict[str, Any] = field(default_factory=dict)
    source_files: Dict[str, Optional[Path]] = field(default_factory=dict)
    
    # Pipeline tracking
    completed_steps: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: Optional[str] = None
    saved_at: Optional[str] = None
```

### Unit Data Structure

Each unit in `session.units` contains:

| Field | Type | Description |
|-------|------|-------------|
| `spike_times` | `np.ndarray[uint64]` | Spike timestamps as sample indices |
| `firing_rate_10hz` | `np.ndarray[float64]` | Binned firing rate at 10 Hz |
| `features` | `Dict[str, Any]` | Extracted features keyed by feature name |
| `eimage_sta` | `Optional[np.ndarray]` | Electrical image STA if computed |
| `sta` | `Optional[Dict]` | STA data if computed |
| `sectioned_spikes` | `Optional[Dict]` | Sectioned spike times by movie/trial |

### Metadata Structure

The `session.metadata` dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `acquisition_rate` | `float` | Sampling rate in Hz |
| `sample_interval` | `float` | Time between samples (1/rate) |
| `dataset_id` | `str` | Recording identifier |
| `sys_meta` | `Dict` | System metadata from CMCR/CMTR |
| `frame_timestamps` | `Optional[np.ndarray]` | Frame boundary timestamps |
| `frame_time` | `Optional[np.ndarray]` | Frame times in seconds |

### Stimulus Structure

The `session.stimulus` dict contains:

| Field | Type | Description |
|-------|------|-------------|
| `light_reference` | `Dict[str, np.ndarray]` | Light reference channels |
| `frame_times` | `Dict[str, np.ndarray]` | Frame timestamps per stimulus |
| `section_time` | `Dict[str, np.ndarray]` | Section boundaries per movie |

## State Transitions

```
┌─────────────┐     save()      ┌─────────────┐
│  DEFERRED   │ ───────────────→│    SAVED    │
│             │                 │             │
│ (in-memory) │                 │ (persisted) │
└─────────────┘                 └─────────────┘
       │                               │
       │ checkpoint()                  │ (can continue
       │ (stays DEFERRED)              │  with HDF5 ops)
       ↓                               ↓
  [snapshot.h5]                   [final.h5]
  (not linked)                    (session.hdf5_path)
```

### Transition Rules

1. **DEFERRED → SAVED**: Via `save()`. Session remains valid, `hdf5_path` is set.
2. **DEFERRED → DEFERRED**: Via `checkpoint()`. Snapshot saved, session continues.
3. **DEFERRED → SAVED (auto)**: When HDF5-path function called in deferred mode. Warning logged.
4. **SAVED → SAVED**: Subsequent saves update the same file (with force=True).

## Validation Rules

| Rule | Enforcement |
|------|-------------|
| `dataset_id` must be non-empty | Validated at session creation |
| `units` keys must match `unit_\d{3}` pattern | Validated on add_unit |
| `save_state` must be SAVED before HDF5-path operations | Auto-save with warning |
| `hdf5_path` must not conflict with existing file | Respect `overwrite` parameter |
| Checkpoint path must differ from final save path | Warning if same path |

## Serialization

### To HDF5 (save/checkpoint)

```
recording.h5
├── units/
│   ├── unit_001/
│   │   ├── spike_times         [dataset: uint64]
│   │   ├── firing_rate_10hz    [dataset: float64]
│   │   ├── features/
│   │   │   ├── frif/           [group with feature data]
│   │   │   └── on_off/         [group with feature data]
│   │   └── eimage_sta          [dataset: float64, optional]
│   └── unit_002/
│       └── ...
├── metadata/
│   ├── acquisition_rate        [attr or dataset]
│   ├── dataset_id              [attr]
│   └── sys_meta/               [group]
├── stimulus/
│   ├── light_reference/
│   ├── frame_times/
│   └── section_time/
├── source_files/
│   ├── cmcr_path               [attr]
│   └── cmtr_path               [attr]
└── pipeline/
    ├── stage1_completed        [attr: bool]
    ├── stage1_timestamp        [attr: str]
    └── session_info/           [group, new]
        ├── completed_steps     [dataset: strings]
        ├── saved_at            [attr: str]
        └── warnings            [dataset: strings]
```

### From HDF5 (load)

`PipelineSession.load(path)` reconstructs the session from the HDF5 structure, restoring:
- All data containers (units, metadata, stimulus)
- `completed_steps` from pipeline/session_info
- `save_state = SAVED`
- `hdf5_path = path`

