# Research: HD-MEA Data Analysis Pipeline v1

**Date**: 2025-12-14  
**Plan**: [plan.md](./plan.md)

---

## Overview

This document consolidates research findings for implementing the HD-MEA Data Analysis Pipeline. All "NEEDS CLARIFICATION" items from Technical Context have been resolved.

---

## 1. Zarr Library Selection

### Decision: `zarr-python` v2.x

### Rationale

- **Mature and stable**: v2.x is production-ready with extensive documentation
- **Chunked storage**: Efficient for large spike train arrays
- **Metadata support**: `.zattrs` provides JSON-serializable metadata at any level
- **Lazy loading**: Only loads chunks when accessed, critical for large recordings
- **Python-native**: No external dependencies beyond numpy

### Alternatives Considered

| Alternative | Reason Rejected |
|-------------|-----------------|
| `zarr` v3 (beta) | Not stable enough for production; API still evolving |
| HDF5 (`h5py`) | Less flexible metadata; harder to version; monolithic files |
| NetCDF | Primarily for climate data; less Python ecosystem support |
| Raw directory + numpy | No metadata standard; manual serialization |

### Implementation Notes

```python
import zarr

# Create Zarr store
store = zarr.DirectoryStore(f"artifacts/{dataset_id}.zarr")
root = zarr.group(store=store)

# Store with metadata
units = root.create_group("units")
unit = units.create_group(unit_id)
unit.create_dataset("spike_times", data=spike_times, dtype="uint64")
unit.attrs["row"] = row
unit.attrs["col"] = col

# Read back
root = zarr.open(store, mode="r")
spikes = root["units"][unit_id]["spike_times"][:]
```

---

## 2. Parquet Library Selection

### Decision: `pyarrow` with pandas integration

### Rationale

- **Performance**: Fastest Python Parquet implementation
- **Full Parquet support**: Supports all Parquet features including metadata
- **Pandas integration**: `pd.read_parquet()` / `df.to_parquet()` use pyarrow by default
- **Schema enforcement**: Strong typing for feature tables
- **Compression**: Built-in snappy/zstd compression

### Alternatives Considered

| Alternative | Reason Rejected |
|-------------|-----------------|
| `fastparquet` | Slower; less complete Parquet support |
| CSV | No types; larger files; no metadata |
| Feather | Less portable; Parquet is more standard |

### Implementation Notes

```python
import pandas as pd
import pyarrow.parquet as pq

# Write with metadata
table = pa.Table.from_pandas(df)
table = table.replace_schema_metadata({
    "dataset_id": dataset_id,
    "feature_version": "1.0.0",
    "created_at": timestamp
})
pq.write_table(table, f"exports/{dataset_id}_features.parquet")

# Read
df = pd.read_parquet(path)
metadata = pq.read_metadata(path).schema.pandas_metadata
```

---

## 3. Feature Registry Pattern

### Decision: Class-based decorator registry

### Rationale

- **Declarative**: Metadata (name, version, inputs) declared as class attributes
- **No `eval()`**: Registration via decorator, not string dispatch
- **Extensible**: New features added without editing existing code
- **Introspectable**: Registry can list all features, their versions, and requirements
- **Type-safe**: Base class enforces interface

### Alternatives Considered

| Alternative | Reason Rejected |
|-------------|-----------------|
| Function registry | Less structured metadata; no inheritance |
| Plugin discovery (`entry_points`) | Overkill for single package |
| `eval()`-based dispatch | Security concern; forbidden by constitution |
| Giant switch/if-else | Not extensible; violates constitution |

### Implementation Notes

```python
# hdmea/features/registry.py
from typing import Type
from .base import FeatureExtractor

class FeatureRegistry:
    _registry: dict[str, Type[FeatureExtractor]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(extractor_class: Type[FeatureExtractor]):
            if name in cls._registry:
                raise ValueError(f"Feature '{name}' already registered")
            extractor_class.name = name
            cls._registry[name] = extractor_class
            return extractor_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[FeatureExtractor]:
        if name not in cls._registry:
            raise KeyError(f"Unknown feature: {name}")
        return cls._registry[name]
    
    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._registry.keys())
    
    @classmethod
    def get_metadata(cls, name: str) -> dict:
        extractor = cls.get(name)
        return {
            "name": extractor.name,
            "version": extractor.version,
            "required_inputs": extractor.required_inputs,
            "output_schema": extractor.output_schema,
            "runtime_class": extractor.runtime_class,
        }

# Usage in feature module
@FeatureRegistry.register("step_up_5s_5i_3x")
class StepUpFeatureExtractor(FeatureExtractor):
    version = "1.0.0"
    required_inputs = ["spike_times", "stimulus/light_reference"]
    output_schema = {
        "on_response_flag": {"dtype": "bool"},
        "off_response_flag": {"dtype": "bool"},
        "on_peak_value": {"dtype": "float64", "unit": "spikes/s"},
        # ...
    }
    runtime_class = "fast"
    
    def extract(self, unit_data: zarr.Group, stimulus_data: zarr.Group) -> dict:
        # Implementation
        ...
```

---

## 4. Configuration Format and Validation

### Decision: JSON with Pydantic validation

### Rationale

- **Human-readable**: Easily edited and version-controlled
- **Standard format**: Universal support, no custom parsers
- **Schema validation**: Pydantic provides type checking and defaults
- **Serializable**: Easy to hash for cache invalidation

### Alternatives Considered

| Alternative | Reason Rejected |
|-------------|-----------------|
| YAML | More error-prone (indentation); less universal |
| TOML | Less flexible for nested structures |
| Python dicts | Not versionable as files |
| INI | Too limited for complex configs |

### Implementation Notes

```python
# hdmea/pipeline/config.py
from pydantic import BaseModel
from pathlib import Path
import json

class StimulusConfig(BaseModel):
    name: str
    movie_length_frames: int
    frame_rate_hz: float
    num_repeats: int
    section_mapping: dict[str, tuple[int, int]]

class FlowConfig(BaseModel):
    name: str
    features: list[str]
    stimuli: list[str]
    
def load_flow_config(path: Path) -> FlowConfig:
    with open(path) as f:
        data = json.load(f)
    return FlowConfig(**data)

# config/flows/set6a_full.json
{
    "name": "set6a_full",
    "features": [
        "baseline_127",
        "step_up_5s_5i_3x",
        "moving_h_bar_s5_d8_3x",
        "perfect_dense_noise_15x15_15hz_r42_3min"
    ],
    "stimuli": ["step_up_5s_5i_3x", "moving_h_bar_s5_d8_3x", ...]
}
```

---

## 5. Configuration Hashing for Cache Invalidation

### Decision: SHA256 of JSON-serialized config

### Rationale

- **Deterministic**: Same config always produces same hash
- **Standard**: SHA256 is universally supported
- **Collision-resistant**: Sufficient for cache keys
- **Debuggable**: Can compare JSON to see differences

### Implementation Notes

```python
# hdmea/utils/hashing.py
import hashlib
import json

def hash_config(config: dict) -> str:
    """Produce deterministic hash of configuration dict."""
    # Sort keys for determinism
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()[:16]}"
```

---

## 6. McsPy Integration

### Decision: Wrap McsPy with abstraction layer

### Rationale

- **Isolation**: Changes to McsPy API don't propagate
- **Testability**: Can mock the wrapper for unit tests
- **Error handling**: Consistent error messages

### Implementation Notes

```python
# hdmea/io/cmtr.py
from McsPy import McsCMOSMEA
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_cmtr_data(cmtr_path: Path) -> dict:
    """Load spike-sorted data from CMTR file.
    
    Args:
        cmtr_path: Path to .cmtr file (may be UNC path)
        
    Returns:
        Dict with keys: units, metadata
        
    Raises:
        FileNotFoundError: If cmtr_path does not exist
        IOError: If file cannot be read
    """
    if not cmtr_path.exists():
        raise FileNotFoundError(f"CMTR file not found: {cmtr_path}")
    
    try:
        cmtr_data = McsCMOSMEA.McsCMOSSpikeStream(str(cmtr_path))
        # Extract data...
        return {"units": units_dict, "metadata": meta_dict}
    except Exception as e:
        logger.error(f"Failed to read CMTR: {cmtr_path}: {e}")
        raise IOError(f"Cannot read CMTR file: {cmtr_path}") from e
```

---

## 7. Testing Strategy

### Decision: pytest with synthetic fixtures

### Rationale

- **Standard tooling**: pytest is the Python testing standard
- **Fixtures**: Reusable synthetic data generators
- **Parametrization**: Test multiple scenarios efficiently
- **No real data in tests**: Synthetic data only (as per constitution)

### Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Smoke | Imports work | `tests/smoke/` |
| Unit | Feature math on synthetic | `tests/unit/` |
| Integration | End-to-end pipeline | `tests/integration/` |
| Contract | API stability | `tests/contract/` |

### Synthetic Data Approach

```python
# tests/fixtures/synthetic_spikes.py
import numpy as np

def generate_poisson_spikes(
    duration_s: float,
    rate_hz: float,
    seed: int = 42
) -> np.ndarray:
    """Generate Poisson spike train for testing."""
    rng = np.random.default_rng(seed)
    n_expected = int(duration_s * rate_hz * 1.5)  # oversample
    isi = rng.exponential(1.0 / rate_hz, n_expected)
    times = np.cumsum(isi)
    return times[times < duration_s]

def generate_on_off_response(
    baseline_rate: float = 5.0,
    on_rate: float = 50.0,
    off_rate: float = 30.0,
    stim_onset: float = 1.0,
    stim_offset: float = 2.0,
    duration: float = 3.0,
    seed: int = 42
) -> np.ndarray:
    """Generate spike train with ON/OFF response for testing extractors."""
    # Implementation...
```

---

## 8. Logging Standards

### Decision: Standard library logging with `__name__`

### Rationale

- **No external dependencies**: Standard library only
- **Hierarchical**: `hdmea.io.cmtr` logs can be filtered
- **Configurable**: Level set at entry point, not library

### Implementation Notes

```python
# hdmea/utils/logging.py
import logging

def setup_logging(level: str = "INFO") -> None:
    """Configure logging for pipeline execution."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# In library modules:
import logging
logger = logging.getLogger(__name__)

# NEVER use print() in library code
```

---

## 9. Error Handling Strategy

### Decision: Descriptive exceptions with context

### Rationale

- **User-friendly**: Errors explain what went wrong and suggest fixes
- **Debuggable**: Include file paths, parameter values
- **Consistent**: Standard exception hierarchy

### Exception Hierarchy

```python
# hdmea/utils/exceptions.py

class HDMEAError(Exception):
    """Base exception for HDMEA pipeline."""
    pass

class DataLoadError(HDMEAError):
    """Error loading raw data files."""
    pass

class FeatureExtractionError(HDMEAError):
    """Error during feature extraction."""
    pass

class ConfigurationError(HDMEAError):
    """Invalid configuration."""
    pass

class MissingInputError(HDMEAError):
    """Required input not found."""
    pass
```

---

## 10. Network Path Handling

### Decision: Use `pathlib.Path` with string conversion for McsPy

### Rationale

- **Windows UNC paths**: `Path("//server/share/file.cmcr")` works
- **Cross-platform**: Works on Windows and Linux
- **McsPy compatibility**: Convert to string when passing to McsPy

### Implementation Notes

```python
from pathlib import Path

def validate_external_path(path: str | Path) -> Path:
    """Validate and normalize external file path."""
    path = Path(path)
    
    # UNC paths start with \\
    if str(path).startswith("\\\\") or str(path).startswith("//"):
        # Network path - existence check may be slow
        pass
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return path
```

---

## Summary

All technical decisions have been made and documented. No remaining "NEEDS CLARIFICATION" items. Ready for Phase 1 design artifacts.

