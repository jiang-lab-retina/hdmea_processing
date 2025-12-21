# Research: Deferred HDF5 Save Pipeline

**Feature**: 001-deferred-hdf5-save  
**Date**: 2024-12-20

## Research Questions

### 1. How to implement deferred writes with h5py?

**Decision**: Use a `PipelineSession` class that holds data in Python dicts/arrays, then writes to HDF5 in a single batch when `save()` is called.

**Rationale**: 
- h5py does not natively support "in-memory files that can later be flushed to disk" in a clean way
- Keeping data in Python native structures (dicts, numpy arrays) is simpler and more flexible
- The existing `write_units()`, `write_metadata()`, `write_stimulus()` functions can be reused at save time
- Memory mapping or h5py's `core` driver with `backing_store=False` is complex and doesn't solve the problem cleanly

**Alternatives considered**:
- h5py core driver with backing_store: Rejected - complex to manage, doesn't integrate well with existing code
- Zarr in-memory store: Rejected - project has moved to HDF5, adding Zarr complexity is undesirable
- Temporary file with rename: Rejected - still has I/O overhead

### 2. How to maintain backwards compatibility?

**Decision**: Add optional `session` parameter to pipeline functions. When `session=None` (default), functions behave exactly as before (immediate save). When `session` is provided, data accumulates in the session.

**Rationale**:
- No changes to function signatures break existing code
- Default behavior is unchanged
- Users opt-in to deferred mode explicitly
- Existing `hdf5_path` parameters still work for immediate-save use cases

**Alternatives considered**:
- Global defer flag: Rejected - violates constitution (no hidden global state)
- Context manager only: Rejected - less flexible, harder to checkpoint
- New function names: Rejected - duplicates API surface, harder to maintain

### 3. How to handle auto-save on mode mixing?

**Decision**: When a function is called that requires an HDF5 path but session is in deferred mode:
1. Log a warning: "Auto-saving deferred session to {path} to satisfy HDF5-path requirement"
2. Call `session.save(default_path)` 
3. Update session state to "saved"
4. Continue with the HDF5-path-based operation

**Rationale**:
- Prevents cryptic errors when users forget to save
- Warning makes behavior transparent
- Session remains usable in saved mode
- Default path derived from session's dataset_id

**Alternatives considered**:
- Raise error: Rejected - too strict, poor UX for exploratory work
- Silent save: Rejected - surprising behavior, could mask bugs

### 4. What data structures for in-memory accumulation?

**Decision**: Mirror the HDF5 structure in Python dicts:

```python
@dataclass
class PipelineSession:
    # Core data
    units: Dict[str, Dict[str, Any]]  # unit_id -> {spike_times, firing_rate, features...}
    metadata: Dict[str, Any]          # acquisition_rate, dataset_id, etc.
    stimulus: Dict[str, Any]          # light_reference, frame_times, section_time
    
    # State tracking
    save_state: SaveState             # DEFERRED or SAVED
    hdf5_path: Optional[Path]         # Set after save()
    dataset_id: str                   # Recording identifier
    
    # Tracking what's been computed
    completed_steps: Set[str]         # {"load", "extract_features", ...}
```

**Rationale**:
- Matches existing HDF5 group structure (units/, metadata/, stimulus/)
- Easy to serialize with existing `write_*` functions
- Flexible for adding new data types
- `completed_steps` enables resume functionality

### 5. How to implement checkpoint vs save?

**Decision**: 
- `save(path)`: Write all data to HDF5, transition to SAVED state, set `hdf5_path`
- `checkpoint(path)`: Write all data to HDF5 but remain in DEFERRED state, don't update `hdf5_path`

**Rationale**:
- Checkpoint is a snapshot, not a state transition
- Save is the final commit that enables HDF5-path operations
- Clear semantic difference

### 6. Memory management for large sessions?

**Decision**: Document that users are responsible for ensuring sufficient RAM. No automatic memory management.

**Rationale**:
- Clarification confirmed 10-50 GB target size
- Automatic memory management (swapping to disk) defeats the purpose of deferred saves
- Users on high-memory servers (128+ GB) can handle this
- Simpler implementation, clearer behavior

**Documentation requirements**:
- Add memory requirements to docstrings
- Add warning in `load_recording` if data size exceeds threshold
- Document in quickstart.md

## Dependencies Identified

| Dependency | Version | Purpose |
|------------|---------|---------|
| h5py | >=3.0.0 | HDF5 read/write |
| numpy | >=1.24.0 | Array storage |
| dataclasses | stdlib | Session class definition |
| enum | stdlib | SaveState enum |
| logging | stdlib | Warning on auto-save |

No new dependencies required.

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory exhaustion | High | Document requirements, add size warning |
| Data loss on crash | Medium | Document clearly, recommend checkpoints |
| API confusion | Low | Clear docstrings, quickstart examples |
| Performance regression | Low | Benchmark before/after |

