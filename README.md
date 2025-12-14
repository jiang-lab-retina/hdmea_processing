# HD-MEA Data Analysis Pipeline

A modular Python package for processing high-density multi-electrode array (HD-MEA) recordings, extracting physiological features, and supporting extensible analyses.

## Features

- **Two-Stage Pipeline**: Load recordings → Extract features
- **Zarr-Based Storage**: Single artifact per recording with embedded features
- **Registry Pattern**: Add new feature extractors without editing core code
- **7+ Built-in Extractors**: ON/OFF response, baseline, direction selectivity, receptive field, chromatic, frequency, cell type
- **Configurable Flows**: Define processing workflows via JSON configuration

## Installation

```bash
# Development installation
pip install -e ".[dev]"
```

## Quick Start

```python
from hdmea.pipeline import load_recording, extract_features

# Stage 1: Load recording to Zarr
result = load_recording(
    cmcr_path="//server/data/recording.cmcr",
    cmtr_path="//server/data/recording.cmtr",
    dataset_id="JIANG009_2024-01-15"
)

# Stage 2: Extract features
extract_features(
    zarr_path=result.zarr_path,
    features=["baseline_127", "step_up_5s_5i_3x"]
)
```

### Run a Complete Flow

```python
from hdmea.pipeline.flows import run_flow

result = run_flow(
    flow_name="set6a_full",
    cmtr_path="//server/data/recording.cmtr",
    dataset_id="JIANG009"
)
```

## Project Structure

```
src/hdmea/
├── io/          # File I/O (CMCR, CMTR, Zarr, Parquet)
├── preprocess/  # Signal filtering, alignment
├── features/    # Feature extractors with registry
├── analysis/    # Downstream analyses
├── viz/         # Visualization utilities
├── pipeline/    # Orchestration and flows
└── utils/       # Logging, hashing, validation
```

## Available Feature Extractors

| Name | Description |
|------|-------------|
| `baseline_127` | Baseline firing statistics |
| `step_up_5s_5i_3x` | ON/OFF response features |
| `moving_h_bar_s5_d8_3x` | Direction selectivity (DSI, OSI) |
| `perfect_dense_noise_15x15_15hz_r42_3min` | Receptive field via STA |
| `green_blue_3s_3i_3x` | Chromatic response |
| `freq_step_5st_3x` | Temporal frequency tuning |
| `cell_type_classifier` | RGC classification |

## Adding New Features

```python
from hdmea.features import FeatureRegistry, FeatureExtractor

@FeatureRegistry.register("my_feature")
class MyExtractor(FeatureExtractor):
    version = "1.0.0"
    required_inputs = ["spike_times"]
    output_schema = {"value": {"dtype": "float64"}}
    
    def extract(self, unit_data, stimulus_data, config=None):
        spikes = unit_data["spike_times"][:]
        return {"value": len(spikes)}
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT

