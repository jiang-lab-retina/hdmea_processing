# Quickstart: HD-MEA Data Analysis Pipeline v1

**Date**: 2025-12-14  
**Plan**: [plan.md](./plan.md)

---

## Prerequisites

- Python 3.10+
- Access to `.cmcr` and/or `.cmtr` files (external paths)
- McsPy library installed (for reading MaxWell files)

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Data_Processing_2027

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install in development mode
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Load Recording (Stage 1)

```python
from hdmea.pipeline import load_recording

# Provide external paths to raw files
zarr_path = load_recording(
    cmcr_path="//server/data/recordings/JIANG009.cmcr",
    cmtr_path="//server/data/recordings/JIANG009.cmtr",
    dataset_id="JIANG009_2024-01-15"
)

print(f"Created: {zarr_path}")
# Output: Created: artifacts/JIANG009_2024-01-15.zarr
```

### 2. Extract Features (Stage 2)

```python
from hdmea.pipeline import extract_features

# Run specific feature extractors
extract_features(
    zarr_path="artifacts/JIANG009_2024-01-15.zarr",
    features=["step_up_5s_5i_3x", "moving_h_bar_s5_d8_3x"]
)
```

### 3. Run Complete Flow

```python
from hdmea.pipeline import run_flow

# Run a named flow (Stage 1 + Stage 2)
run_flow(
    flow_name="set6a_full",
    cmcr_path="//server/data/recordings/JIANG009.cmcr",
    cmtr_path="//server/data/recordings/JIANG009.cmtr",
    dataset_id="JIANG009_2024-01-15"
)
```

---

## Configuration

### Flow Configuration

Create flow configs in `config/flows/`:

```json
// config/flows/set6a_full.json
{
  "name": "set6a_full",
  "stages": {
    "features": {
      "feature_sets": [
        "baseline_127",
        "step_up_5s_5i_3x",
        "moving_h_bar_s5_d8_3x",
        "perfect_dense_noise_15x15_15hz_r42_3min"
      ]
    }
  }
}
```

### Stimulus Configuration

Define stimulus types in `config/stimuli/`:

```json
// config/stimuli/step_up_5s_5i_3x.json
{
  "name": "step_up_5s_5i_3x",
  "movie_length_frames": 300,
  "frame_rate_hz": 30.0,
  "num_repeats": 3,
  "sections": {
    "baseline": [0, 150],
    "step_up": [150, 300]
  }
}
```

---

## Accessing Data

### Read Zarr Artifact

```python
import zarr

# Open Zarr store
store = zarr.open("artifacts/JIANG009_2024-01-15.zarr", mode="r")

# Access metadata
print(store.attrs["dataset_id"])
print(store.attrs["features_extracted"])

# Access unit data
for unit_id in store["units"]:
    unit = store["units"][unit_id]
    spikes = unit["spike_times"][:]
    print(f"Unit {unit_id}: {len(spikes)} spikes")
    
    # Access features
    if "features" in unit:
        for feature_name in unit["features"]:
            feature = unit["features"][feature_name]
            print(f"  Feature: {feature_name}, version: {feature.attrs['extractor_version']}")
```

### Export to Parquet

```python
from hdmea.io import export_features_to_parquet

export_features_to_parquet(
    zarr_paths=["artifacts/JIANG009_2024-01-15.zarr"],
    output_path="exports/features.parquet",
    features=["step_up_5s_5i_3x", "moving_h_bar_s5_d8_3x"]
)
```

---

## Adding New Feature Extractors

Create a new file in `src/hdmea/features/`:

```python
# src/hdmea/features/my_feature/extractor.py
from hdmea.features.registry import FeatureRegistry
from hdmea.features.base import FeatureExtractor
import zarr

@FeatureRegistry.register("my_new_feature")
class MyNewFeatureExtractor(FeatureExtractor):
    version = "1.0.0"
    required_inputs = ["spike_times", "stimulus/light_reference"]
    output_schema = {
        "my_value": {"dtype": "float64", "unit": "spikes/s"}
    }
    runtime_class = "fast"
    
    def extract(self, unit_data: zarr.Group, stimulus_data: zarr.Group) -> dict:
        """Extract my new feature."""
        spikes = unit_data["spike_times"][:]
        # ... compute feature ...
        return {"my_value": computed_value}
```

The feature is automatically available:

```python
from hdmea.features import FeatureRegistry

print("my_new_feature" in FeatureRegistry.list_all())  # True

# Use in extraction
extract_features(
    zarr_path="artifacts/JIANG009_2024-01-15.zarr",
    features=["my_new_feature"]
)
```

---

## Project Layout

```
Data_Processing_2027/
├── src/hdmea/              # Package source
│   ├── io/                 # File I/O
│   ├── preprocess/         # Data preprocessing
│   ├── features/           # Feature extractors
│   ├── analysis/           # Downstream analyses
│   ├── viz/                # Visualization
│   └── pipeline/           # Orchestration
├── config/                 # Configuration files
│   ├── flows/              # Flow definitions
│   └── stimuli/            # Stimulus definitions
├── artifacts/              # Output Zarr files (gitignored)
├── exports/                # Parquet exports (gitignored)
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test suite
└── Legacy_code/            # Reference only (DO NOT IMPORT)
```

---

## Troubleshooting

### "Feature already extracted" error

```python
# Force re-extraction
extract_features(
    zarr_path="...",
    features=["step_up_5s_5i_3x"],
    force=True  # Overwrite existing
)
```

### "Missing input" error

Check that the Zarr has required data:

```python
import zarr
store = zarr.open("artifacts/JIANG009_2024-01-15.zarr", mode="r")
print(list(store["stimulus"].keys()))  # Check available stimulus data
```

### "File not found" for external paths

- Verify the path exists and is accessible
- UNC paths (e.g., `//server/share/file.cmcr`) require network access
- Stage 1 requires access to raw files; Stage 2 only needs the Zarr

---

## Next Steps

1. [Data Model](./data-model.md) - Entity schemas and Zarr structure
2. [API Contracts](./contracts/) - Programmatic interfaces
3. [Research](./research.md) - Technical decisions and rationale

