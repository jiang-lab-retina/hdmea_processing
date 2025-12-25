# Pipeline Design Principles

This document describes the design principles for the HD-MEA data processing pipeline, providing guidelines for session-based workflows, data preservation, and batch processing that apply to all analysis and feature extraction modules.

## Overview

The pipeline follows a modular, session-based architecture that enables:
- **Deferred execution**: Accumulate operations in memory, save once at the end
- **Data preservation**: Update HDF5 files without removing existing data
- **Batch processing**: Process multiple files with consistent parameters
- **Extensibility**: Add new analysis steps that integrate with existing data

## Core Design Principles

### 1. Session-Based Workflow

The `PipelineSession` object serves as the central data container that:
- Holds all data in memory during processing
- Tracks completed pipeline steps
- Enables deferred HDF5 saving

```
┌─────────────────────────────────────────────────────────────┐
│                    PipelineSession                          │
├─────────────────────────────────────────────────────────────┤
│  dataset_id: str                                            │
│  units: Dict[unit_id, unit_data]                           │
│  metadata: Dict                                             │
│  completed_steps: Set[str]                                  │
│  hdf5_path: Path (set after save)                          │
└─────────────────────────────────────────────────────────────┘
```

**Standard Workflow Pattern:**

```python
from hdmea.pipeline import create_session, PipelineSession

# 1. Create session
session = create_session(dataset_id="recording_name")

# 2. Chain pipeline steps (data accumulates in memory)
session = step_1(..., session=session)
session = step_2(..., session=session)
session = step_3(..., session=session)

# 3. Save once at the end
session.save()
```

**Benefits:**
- Single disk write reduces I/O overhead
- Easy to add/remove steps without intermediate files
- Clear tracking of what operations were performed
- Memory-efficient when processing subsets of data

### 2. Data Preservation (Non-Destructive Updates)

When adding new analysis results to existing HDF5 files, always:
1. **Copy the source file** to preserve all existing data
2. **Open in append mode** (`'a'`) instead of write mode (`'w'`)
3. **Only update specific groups** without touching other data

```python
import shutil
import h5py

def save_results_to_hdf5(session, output_path):
    """Save new results while preserving all existing data."""
    
    # Step 1: Copy source file to preserve existing data
    if session.hdf5_path and session.hdf5_path.exists():
        shutil.copy2(session.hdf5_path, output_path)
    
    # Step 2: Open in append mode
    with h5py.File(output_path, 'a') as f:
        for unit_id, unit_data in session.units.items():
            # Step 3: Only update the target group
            target_path = f'units/{unit_id}/features/{FEATURE_NAME}'
            
            # Remove old version if exists (to allow updates)
            if target_path in f:
                del f[target_path]
            
            # Create new group with results
            group = f.create_group(target_path)
            for key, value in unit_data['results'].items():
                group.create_dataset(key, data=value)
```

**HDF5 Structure Preservation:**

```
Before:                          After:
units/                           units/
├── unit_001/                    ├── unit_001/
│   ├── spike_times     ✓       │   ├── spike_times     ✓ (preserved)
│   ├── waveform        ✓       │   ├── waveform        ✓ (preserved)
│   ├── unit_meta/      ✓       │   ├── unit_meta/      ✓ (preserved)
│   └── features/                │   └── features/
│       ├── existing/   ✓       │       ├── existing/   ✓ (preserved)
│       └── ...                  │       ├── ...
│                                │       └── new_feature/  ← NEW
│                                │           ├── param_1
│                                │           └── param_2
```

### 3. Function Signature Pattern

All pipeline functions should follow a consistent signature pattern that supports both immediate and deferred modes:

```python
from typing import Union, Optional
from hdmea.pipeline import PipelineSession

def analysis_function(
    # Required parameters
    param1: type,
    param2: type,
    *,
    # Optional session for deferred mode
    session: Optional[PipelineSession] = None,
    # Other optional parameters with defaults
    option1: type = default,
    option2: type = default,
) -> Union[ResultType, PipelineSession]:
    """
    Perform analysis on data.
    
    Args:
        param1: Description
        param2: Description
        session: If provided, operates in deferred mode and returns session.
                 If None, operates in immediate mode and returns result.
        option1: Optional parameter
        option2: Optional parameter
    
    Returns:
        If session provided: Updated PipelineSession
        If session is None: Analysis result
    """
    if session is not None:
        # Deferred mode: process and store results in session
        for unit_id, unit_data in session.units.items():
            result = _process_unit(unit_data, param1, param2)
            unit_data['features'][FEATURE_NAME] = result
        
        session.completed_steps.add('analysis_function')
        return session
    else:
        # Immediate mode: process and return result directly
        return _process_immediate(param1, param2)
```

### 4. Loading Existing HDF5 into Session

When adding new analysis to existing processed data, load only what's needed:

```python
def load_hdf5_to_session(
    hdf5_path: Path,
    dataset_id: str = None,
    feature_path: str = None,
) -> PipelineSession:
    """
    Load existing HDF5 data into a session for further processing.
    
    Args:
        hdf5_path: Path to existing HDF5 file
        dataset_id: Optional ID (uses filename stem if not provided)
        feature_path: Path to feature data within each unit (e.g., 'features/sta/data')
    
    Returns:
        PipelineSession with loaded data
    """
    if dataset_id is None:
        dataset_id = hdf5_path.stem
    
    session = create_session(dataset_id=dataset_id)
    session.hdf5_path = hdf5_path  # Track source for later saving
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'units' not in f:
            raise ValueError(f"No 'units' group found in {hdf5_path}")
        
        for unit_id in f['units'].keys():
            unit_group = f[f'units/{unit_id}']
            unit_data = {'features': {}}
            
            # Load required feature data
            if feature_path and feature_path in unit_group:
                unit_data['features'][FEATURE_NAME] = {
                    'data': unit_group[feature_path][:]
                }
            
            if unit_data['features']:
                session.units[unit_id] = unit_data
    
    session.completed_steps.add('load_hdf5')
    return session
```

### 5. Batch Processing Pattern

Batch processing scripts should follow this consistent structure:

```python
from pathlib import Path
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration at module level
INPUT_DIR = Path(__file__).parent / "input_folder"
OUTPUT_DIR = Path(__file__).parent / "output_folder"

def process_single_file(input_path: Path, output_path: Path) -> bool:
    """
    Process one file.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load
        session = load_hdf5_to_session(input_path)
        
        if len(session.units) == 0:
            logger.warning(f"No units in {input_path.name}, skipping")
            return False
        
        # Process
        session = analysis_function(session=session)
        
        # Save
        save_results_to_hdf5(session, output_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_process():
    """Process all files in input directory."""
    print("=" * 70)
    print("Batch Processing")
    print("=" * 70)
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validate input
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return
    
    hdf5_files = sorted(INPUT_DIR.glob("*.h5"))
    if not hdf5_files:
        print(f"Error: No HDF5 files found")
        return
    
    print(f"\nFound {len(hdf5_files)} files to process")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process with tracking
    successful, failed, skipped = 0, 0, 0
    start_time = time.time()
    
    for i, input_path in enumerate(hdf5_files, 1):
        print(f"\n[{i}/{len(hdf5_files)}] {input_path.name}")
        
        output_path = OUTPUT_DIR / input_path.name
        
        # Skip if already processed (allows resume)
        if output_path.exists():
            logger.info("  Already exists, skipping")
            skipped += 1
            continue
        
        file_start = time.time()
        
        if process_single_file(input_path, output_path):
            successful += 1
            logger.info(f"  Time: {time.time() - file_start:.1f}s")
        else:
            failed += 1
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Skipped:    {skipped}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Write processing log
    write_processing_log(hdf5_files, successful, failed, skipped, total_time)


def write_processing_log(files, successful, failed, skipped, total_time):
    """Write a log file summarizing the batch processing."""
    log_path = OUTPUT_DIR / "processing_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Batch Processing Log\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Input:  {INPUT_DIR}\n")
        f.write(f"Output: {OUTPUT_DIR}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Successful: {successful}\n")
        f.write(f"  Failed:     {failed}\n")
        f.write(f"  Skipped:    {skipped}\n")
        f.write(f"  Total time: {total_time:.1f}s\n")
        f.write(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nFiles:\n")
        for path in files:
            output_path = OUTPUT_DIR / path.name
            status = "OK" if output_path.exists() else "FAILED"
            f.write(f"  [{status}] {path.name}\n")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    batch_process()
```

**Batch Processing Features:**
- Progress tracking (`[1/99]`, `[2/99]`, etc.)
- Skip existing outputs (allows resuming interrupted runs)
- Graceful error handling (continue with remaining files)
- Processing log with summary statistics
- Timing information per file and total

### 6. Structured Output Organization

Results should be organized hierarchically with individual datasets for each parameter:

```
feature_name/
├── scalar_param_1      # Scalar: float64, shape=()
├── scalar_param_2      # Scalar: float64, shape=()
├── array_param         # Array: float64, shape=(n,)
├── subgroup_1/         # Group for related parameters
│   ├── param_a         # Scalar: float64
│   ├── param_b         # Scalar: float64
│   └── quality_metric  # Scalar: float64 (e.g., r_squared)
└── subgroup_2/         # Another group
    ├── param_x
    └── param_y
```

**Guidelines:**
- Use individual datasets for each parameter (not combined arrays)
- Group related parameters in subgroups
- Include quality metrics (e.g., `r_squared`, `confidence`)
- Use descriptive names that are self-documenting

### 7. Configuration Pattern

Use module-level constants for configuration:

```python
# =============================================================================
# Configuration
# =============================================================================

# Paths
INPUT_DIR = Path(__file__).parent / "input_data"
OUTPUT_DIR = Path(__file__).parent / "output_data"

# Feature settings
FEATURE_NAME = "my_feature"
FEATURE_DATA_PATH = f"features/{FEATURE_NAME}/data"

# Analysis parameters
PARAM_1 = 10
PARAM_2 = 0.5

# Dataclass for complex configuration
@dataclass
class AnalysisConfig:
    param_a: float = 1.0
    param_b: int = 10
    param_c: str = "default"

CONFIG = AnalysisConfig()
```

## Integration with Existing Pipeline

New analysis modules integrate with the existing pipeline by:

1. **Importing from the core module:**
   ```python
   from hdmea.pipeline import create_session, PipelineSession
   ```

2. **Following the session pattern:**
   ```python
   session = create_session(dataset_id="...")
   session = existing_step(..., session=session)
   session = new_analysis(..., session=session)  # New step
   session.save()
   ```

3. **Using consistent HDF5 paths:**
   - Units: `units/{unit_id}/`
   - Features: `units/{unit_id}/features/{feature_name}/`
   - Metadata: `metadata/`
   - Stimulus: `stimulus/`

4. **Tracking completed steps:**
   ```python
   session.completed_steps.add('step_name')
   ```

## File Organization

```
Projects/
└── my_analysis/
    ├── my_analysis.py       # Core analysis functions
    ├── my_session.py        # Session-based workflow
    ├── batch_my_analysis.py # Batch processing script
    ├── README.md            # Documentation
    └── output_folder/       # Generated outputs
        ├── file1.h5
        ├── file2.h5
        └── processing_log.txt
```

## Summary Table

| Principle | Implementation |
|-----------|---------------|
| Deferred execution | `PipelineSession` accumulates data, single `save()` call |
| Data preservation | Copy source file, open in append mode (`'a'`), update only target groups |
| Consistent interface | `session=session` parameter enables deferred mode |
| Batch processing | Skip existing, log progress, handle errors gracefully |
| Structured output | Hierarchical groups with individual datasets |
| Configuration | Module-level constants and dataclasses |
| Integration | Follow HDF5 path conventions, track completed steps |

## Checklist for New Analysis Modules

- [ ] Create core analysis function with `session` parameter
- [ ] Implement `load_hdf5_to_session()` for loading existing data
- [ ] Implement `save_results_to_hdf5()` that preserves existing data
- [ ] Create batch processing script with progress tracking
- [ ] Add processing log generation
- [ ] Write README documentation
- [ ] Follow consistent HDF5 path structure
- [ ] Track completed steps in session

