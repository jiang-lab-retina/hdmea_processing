# Feature Extractors

This directory contains all feature extractors for the HD-MEA pipeline.

## Quick Start

1. Copy `example/example_feature.py` to a new file
2. Rename the class and change the `@register` decorator name
3. Modify `required_inputs`, `output_schema`, and `extract()` method
4. Import your module in the parent `__init__.py`

## Directory Structure

```
features/
├── __init__.py          # Imports all extractors to trigger registration
├── base.py              # FeatureExtractor base class
├── registry.py          # FeatureRegistry for registration
├── README.md            # This file
├── example/             # Example/template extractor
├── on_off/              # ON/OFF response features
├── baseline/            # Baseline firing statistics
├── direction/           # Direction selectivity (DSI, OSI)
├── receptive_field/     # STA and RF features
├── chromatic/           # Color response features
├── frequency/           # Temporal frequency tuning
└── cell_type/           # Cell type classification
```

## Creating a New Extractor

### Step 1: Create New File

```python
# src/hdmea/features/myfeature/my_extractor.py

from hdmea.features.base import FeatureExtractor
from hdmea.features.registry import FeatureRegistry

@FeatureRegistry.register("my_feature_name")
class MyFeatureExtractor(FeatureExtractor):
    name = "my_feature_name"
    version = "1.0.0"
    required_inputs = ["spike_times", "stimulus/light_reference"]
    output_schema = {
        "my_value": {"dtype": "float64", "unit": "Hz"}
    }
    runtime_class = "fast"
    
    def extract(self, unit_data, stimulus_data, config=None):
        spikes = unit_data["spike_times"][:]
        # ... compute feature ...
        return {"my_value": computed_value}
```

### Step 2: Update Package Init

```python
# src/hdmea/features/myfeature/__init__.py

from hdmea.features.myfeature.my_extractor import MyFeatureExtractor
```

### Step 3: Add to Features Init

```python
# src/hdmea/features/__init__.py

# Add import to trigger registration
from hdmea.features.myfeature import MyFeatureExtractor
```

### Step 4: Create Stimulus Config (if needed)

```json
// config/stimuli/my_stimulus.json
{
    "name": "my_stimulus",
    "movie_length_frames": 300,
    "frame_rate_hz": 30.0,
    "num_repeats": 3
}
```

## Required Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Unique identifier (set by @register) |
| `version` | str | Semver string (increment on algorithm change) |
| `required_inputs` | list | Zarr paths this extractor needs |
| `output_schema` | dict | Documentation of output columns |
| `runtime_class` | str | "fast" or "slow" |

## Output Schema Format

```python
output_schema = {
    "column_name": {
        "dtype": "float64",  # numpy dtype
        "unit": "Hz",        # physical unit
        "range": [0, 100],   # optional: expected range
        "description": "..."  # optional: longer description
    }
}
```

## Best Practices

1. **Pure Functions**: `extract()` must be deterministic
2. **Handle Empty Data**: Always check for empty spike arrays
3. **Use Explicit Seeds**: Pass random seeds via config
4. **Document Outputs**: Fill in output_schema completely
5. **Version Updates**: Increment version when algorithm changes
6. **Validate Inputs**: Use `validate_inputs()` before extraction

## Testing Your Extractor

```python
# tests/unit/test_my_feature.py

import pytest
import numpy as np
from unittest.mock import MagicMock
from hdmea.features.myfeature.my_extractor import MyFeatureExtractor

def test_my_extractor():
    extractor = MyFeatureExtractor()
    
    # Create mock Zarr groups
    unit_data = MagicMock()
    unit_data.__getitem__ = lambda s, k: np.array([1000, 2000, 3000])
    
    stimulus_data = MagicMock()
    
    result = extractor.extract(unit_data, stimulus_data)
    
    assert "my_value" in result
    assert result["my_value"] > 0
```

## Adding to Flows

Add your feature to a flow configuration:

```json
// config/flows/my_flow.json
{
    "name": "my_flow",
    "stages": {
        "features": {
            "feature_sets": [
                "baseline_127",
                "my_feature_name"  // Add here
            ]
        }
    }
}
```

