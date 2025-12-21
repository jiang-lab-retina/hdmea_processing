# Quickstart: Deferred HDF5 Save Pipeline

This guide shows how to use the new deferred save functionality in the HD-MEA pipeline.

## Installation

No additional installation required. The feature is part of the existing `hdmea` package.

```bash
pip install -e .
```

## Usage Patterns

### Pattern 1: Immediate Save (Default - Unchanged)

Existing code continues to work without modification:

```python
from hdmea.pipeline import load_recording, extract_features, add_section_time

# Each function saves to HDF5 immediately (existing behavior)
result = load_recording(
    cmcr_path="path/to/recording.cmcr",
    cmtr_path="path/to/recording.cmtr",
)

extract_features(
    hdf5_path=result.hdf5_path,
    features=["frif", "on_off"],
)

add_section_time(
    hdf5_path=result.hdf5_path,
    playlist_name="my_playlist",
)
```

### Pattern 2: Deferred Save (New)

Run the entire pipeline in memory, save once at the end:

```python
from hdmea.pipeline import (
    create_session,
    load_recording,
    extract_features,
    add_section_time,
    compute_sta,
)

# Create a session for deferred saving
session = create_session(
    cmcr_path="path/to/recording.cmcr",
    cmtr_path="path/to/recording.cmtr",
)

# All operations accumulate in memory
session = load_recording(
    cmcr_path="path/to/recording.cmcr",
    cmtr_path="path/to/recording.cmtr",
    session=session,
)

session = extract_features(
    features=["frif", "on_off"],
    session=session,
)

session = add_section_time(
    playlist_name="my_playlist",
    session=session,
)

session = compute_sta(
    cover_range=(-60, 0),
    session=session,
)

# Save everything at once
hdf5_path = session.save()
print(f"Saved to: {hdf5_path}")
```

### Pattern 3: Checkpoint During Long Pipeline

Save intermediate checkpoints without interrupting the session:

```python
from hdmea.pipeline import create_session, load_recording, extract_features

session = create_session(cmcr_path="path/to/recording.cmcr")

# Step 1: Load data
session = load_recording(
    cmcr_path="path/to/recording.cmcr",
    cmtr_path="path/to/recording.cmtr",
    session=session,
)

# Checkpoint after loading (in case extraction takes hours)
session.checkpoint("checkpoints/after_load.h5")

# Step 2: Extract features (takes a long time)
session = extract_features(
    features=["dense_noise", "moving_bar", "chirp"],
    session=session,
)

# Checkpoint after extraction
session.checkpoint("checkpoints/after_features.h5")

# Step 3: More processing...
# ...

# Final save
session.save("final_output.h5")
```

### Pattern 4: Resume from Checkpoint

Resume a crashed or interrupted pipeline:

```python
from hdmea.pipeline import PipelineSession, extract_features

# Load from checkpoint
session = PipelineSession.load("checkpoints/after_load.h5")

# Check what steps are complete
print(f"Completed steps: {session.completed_steps}")
# Output: {'load_recording'}

# Continue from where we left off
session = extract_features(
    features=["dense_noise", "moving_bar"],
    session=session,
)

session.save("final_output.h5")
```

### Pattern 5: Batch Processing Multiple Recordings

Process multiple recordings efficiently:

```python
from hdmea.pipeline import create_session, load_recording, extract_features
from pathlib import Path

recordings = [
    ("recording1.cmcr", "recording1.cmtr"),
    ("recording2.cmcr", "recording2.cmtr"),
    ("recording3.cmcr", "recording3.cmtr"),
]

for cmcr, cmtr in recordings:
    # Create fresh session for each recording
    session = create_session(cmcr_path=cmcr, cmtr_path=cmtr)
    
    # Process entirely in memory
    session = load_recording(cmcr_path=cmcr, cmtr_path=cmtr, session=session)
    session = extract_features(features=["frif"], session=session)
    
    # Save and free memory
    session.save()
    del session  # Release memory before next recording
```

## Memory Considerations

Deferred save keeps all data in memory. For large recordings (10-50 GB in-memory):

- Ensure your system has sufficient RAM (recommended: 2x expected data size)
- Use checkpoints for very long pipelines
- Monitor memory usage with `session.memory_estimate_gb`

```python
# Check estimated memory usage
print(f"Session using ~{session.memory_estimate_gb:.1f} GB")
```

## Common Scenarios

### Mixed Mode (Auto-Save)

If you call a function requiring HDF5 while in deferred mode:

```python
session = create_session(...)
session = load_recording(..., session=session)

# This function requires HDF5 path and session is deferred
# System will auto-save with a warning and continue
some_hdf5_only_function(hdf5_path=session.hdf5_path)  # Warning logged
```

### Overwrite Behavior

By default, save/checkpoint overwrites existing files with a warning:

```python
# Overwrite with warning (default)
session.save("output.h5")  # Warning if exists
session.save("output.h5")  # Warning again, overwrites

# Fail if exists
session.save("output.h5", overwrite=False)  # FileExistsError if exists
```

## Error Handling

```python
try:
    session = PipelineSession.load("nonexistent.h5")
except FileNotFoundError:
    print("Checkpoint file not found")

try:
    session.save("output.h5", overwrite=False)
except FileExistsError:
    print("File already exists, use overwrite=True")
```

## Best Practices

1. **Use deferred save for batch processing** - significant I/O savings
2. **Checkpoint long pipelines** - protect against crashes
3. **Monitor memory** - use `memory_estimate_gb` property
4. **Delete sessions after save** - free memory for next recording
5. **Use immediate save for debugging** - easier to inspect intermediate results

