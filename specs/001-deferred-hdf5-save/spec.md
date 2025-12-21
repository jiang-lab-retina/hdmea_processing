# Feature Specification: Deferred HDF5 Save Pipeline

**Feature Branch**: `001-deferred-hdf5-save`  
**Created**: 2024-12-20  
**Status**: Draft  
**Input**: User description: "Modify pipeline functions to allow processing without saving intermediate HDF5 files, with option to save at any middle point"

## Clarifications

### Session 2024-12-20

- Q: What is the maximum expected in-memory size for deferred processing? → A: Large (10-50 GB, requires high-memory server)
- Q: What happens to session after save()? → A: Convert to saved mode (session stays valid, can continue with HDF5 path)
- Q: How to handle checkpoint file conflicts? → A: User-controlled via parameter: overwrite with warning (default) or raise error
- Q: How to handle mixing deferred and immediate-save modes? → A: Implicit save (auto-save when HDF5 path is needed, log warning)
- Q: How should resume from checkpoint work? → A: Class method `PipelineSession.load(checkpoint_path)`

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Complete Pipeline Without Intermediate Saves (Priority: P1)

A researcher wants to run the full pipeline (load_recording → extract_features → add_section_time → section_spike_times → compute_sta) on multiple recordings in batch. Currently, each step saves to HDF5, causing significant I/O overhead. The researcher wants to keep all data in memory and only save the final result.

**Why this priority**: This is the primary use case - reducing I/O overhead for batch processing of large datasets. Most users will benefit from faster end-to-end processing.

**Independent Test**: Can be fully tested by running a complete pipeline with `defer_save=True` and verifying no intermediate HDF5 files are created until explicit save is called.

**Acceptance Scenarios**:

1. **Given** a valid CMCR/CMTR file pair, **When** user runs `load_recording(..., defer_save=True)` followed by subsequent pipeline steps, **Then** no HDF5 file is written to disk until `save()` is called
2. **Given** a pipeline session with multiple steps completed in memory, **When** user calls `session.save(output_path)`, **Then** all accumulated data is written to a single HDF5 file
3. **Given** a pipeline session running in deferred mode, **When** the process crashes before save, **Then** user understands data was not persisted (expected behavior)

---

### User Story 2 - Save at Intermediate Checkpoint (Priority: P2)

A researcher is running a long pipeline that takes hours to complete. They want to save progress at specific checkpoints so they can resume later or inspect intermediate results, without committing to saving after every single step.

**Why this priority**: Important for long-running experiments and debugging, but secondary to the core deferred-save functionality.

**Independent Test**: Can be tested by running 3 pipeline steps, saving after step 2, modifying in-memory state in step 3, and verifying the checkpoint contains only steps 1-2.

**Acceptance Scenarios**:

1. **Given** a pipeline session with steps 1-3 completed in memory, **When** user calls `session.checkpoint(path, checkpoint_name="after_step2")`, **Then** current state is saved without interrupting the session
2. **Given** a checkpointed session, **When** user continues processing in memory, **Then** future checkpoints can include additional steps
3. **Given** a checkpoint file exists, **When** user calls `PipelineSession.load(checkpoint_path)`, **Then** a new session is created with all previously accumulated state restored

---

### User Story 3 - Backwards-Compatible Default Behavior (Priority: P1)

Existing scripts that use the current pipeline API must continue to work without modification. The default behavior should remain "save after each step" to ensure backwards compatibility.

**Why this priority**: Critical for not breaking existing workflows - tied with P1.

**Independent Test**: Run existing pipeline_test.py without any modifications and verify it produces the same HDF5 outputs as before.

**Acceptance Scenarios**:

1. **Given** an existing script using `load_recording(cmcr_path, cmtr_path)` without new parameters, **When** script runs, **Then** HDF5 is saved immediately as before
2. **Given** an existing script using `extract_features(hdf5_path, features)`, **When** script runs, **Then** features are written to HDF5 immediately as before

---

### Edge Cases

- What happens when user forgets to call `save()` and the session goes out of scope? (Data is lost - document this clearly)
- How does the system handle memory pressure when deferring saves for large recordings? (Document memory requirements; user must ensure sufficient RAM)
- What happens if user mixes deferred and immediate-save calls in the same session? → System auto-saves when HDF5 path is needed and logs a warning, then continues in saved mode

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support a `defer_save` parameter on pipeline functions that, when True, keeps data in memory instead of writing to disk
- **FR-002**: System MUST provide a unified `PipelineSession` object that holds accumulated in-memory data across multiple pipeline steps
- **FR-003**: System MUST provide a `save()` method on PipelineSession that writes all accumulated data to HDF5
- **FR-004**: System MUST provide a `checkpoint()` method that saves current state without ending the session. When target file exists: default behavior is overwrite with warning logged; user can set `overwrite=False` to raise error instead
- **FR-005**: System MUST maintain 100% backwards compatibility - existing scripts without `defer_save` parameter must work identically
- **FR-006**: System MUST support transitioning from deferred mode to immediate mode by calling `save()` and then passing the HDF5 path to subsequent functions
- **FR-007**: System MUST auto-save and log warning when user calls a function requiring HDF5 path while in deferred mode, then continue in saved mode
- **FR-008**: System MUST provide `PipelineSession.load(checkpoint_path)` class method to resume from a previously saved checkpoint

### Key Entities

- **PipelineSession**: Container that holds in-memory data accumulated across pipeline steps. Contains units data, metadata, stimulus data, features, and section times. After `save()` is called, session converts to "saved" mode and remains valid for continued use with HDF5 path-based operations.
- **SaveState**: Enum or flag indicating whether the session is "deferred" (in-memory only) or "saved" (persisted to HDF5). Transitions from deferred → saved upon `save()` call; session remains usable in saved mode.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Processing 10 recordings in batch with deferred save completes at least 20% faster than with intermediate saves (reduced I/O overhead)
- **SC-002**: All existing pipeline test scripts pass without modification after this change
- **SC-003**: Users can checkpoint long-running pipelines and verify checkpoint files contain expected data
- **SC-004**: Memory usage for a typical recording in deferred mode stays below 2x the data size (no excessive duplication)
- **SC-005**: System supports deferred processing of recordings with in-memory representation up to 50 GB (requires high-memory server)
