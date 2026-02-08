# Step Change Analysis Pipeline

This pipeline analyzes step responses over time, particularly for experiments where an agonist or treatment is applied during the recording session. It tracks units across multiple recordings and measures how responses change before and after treatment.

## Overview

The pipeline:
1. Loads CMCR/CMTR recording files and saves to HDF5 format
2. Detects light step stimuli and extracts response windows
3. Aligns units across multiple recordings using waveform and response matching
4. Analyzes response magnitude changes over time
5. Generates visualization plots

## Directory Structure

```
step_change_analysis/
├── __init__.py              # Package initialization
├── specific_config.py       # Configuration settings
├── data_loader.py           # CMCR/CMTR loading and HDF5 operations
├── unit_alignment.py        # Unit tracking across recordings
├── response_analysis.py     # Response feature extraction
├── visualization.py         # Plotting functions
├── run_pipeline.py          # Main pipeline orchestrator
├── README.md               # This file
├── output/                  # Generated HDF5 files (created at runtime)
└── figures/                 # Generated plots (created at runtime)
```

## Installation

This pipeline is part of the Data_Processing_2027 project. Ensure you have the required dependencies:

```bash
pip install numpy pandas scipy matplotlib h5py
```

The pipeline also requires the `hdmea` package from this project:

```bash
pip install -e /path/to/Data_Processing_2027
```

## Usage

### Command Line

Run the full pipeline:

```bash
python -m Projects.unified_special_pipeline.step_change_analysis.run_pipeline
```

With custom paths:

```bash
python -m Projects.unified_special_pipeline.step_change_analysis.run_pipeline \
    --data-folder "P:/20251002_HttrB_antagonist" \
    --output-dir "./output" \
    --figures-dir "./figures" \
    --overwrite
```

Resume from existing HDF5:

```bash
python -m Projects.unified_special_pipeline.step_change_analysis.run_pipeline \
    --from-hdf5 "./output/aligned_group.h5"
```

### Python API

```python
from Projects.unified_special_pipeline.step_change_analysis import (
    run_full_pipeline,
    PipelineConfig,
)

# Run with default settings
results = run_full_pipeline()

# Or with custom configuration
from pathlib import Path

results = run_full_pipeline(
    data_folder=Path("P:/20251002_HttrB_antagonist"),
    output_dir=Path("./output"),
    figures_dir=Path("./figures"),
    overwrite=True,
)

# Access results
grouped_data = results["grouped_data"]
analysis = results["analysis"]
```

### Step-by-Step Usage

```python
from Projects.unified_special_pipeline.step_change_analysis.data_loader import (
    load_and_save_recording,
    load_recording_from_hdf5,
)
from Projects.unified_special_pipeline.step_change_analysis.unit_alignment import (
    create_aligned_group,
)
from Projects.unified_special_pipeline.step_change_analysis.response_analysis import (
    summarize_response_timecourse,
)
from Projects.unified_special_pipeline.step_change_analysis.visualization import (
    plot_response_timecourse,
)

# Step 1: Load a single recording
data, hdf5_path = load_and_save_recording(
    cmcr_path="path/to/recording.cmcr",
    cmtr_path="path/to/recording-.cmtr",
)

# Step 2: Align multiple recordings
grouped_data, group_path = create_aligned_group(
    [hdf5_path1, hdf5_path2, hdf5_path3]
)

# Step 3: Analyze responses
summary = summarize_response_timecourse(grouped_data, "ON")

# Step 4: Plot
fig = plot_response_timecourse(summary, save_path="response.png")
```

## Configuration

All configuration is in `specific_config.py`. Key settings:

### Data Paths

```python
DATA_FOLDER = Path("P:/20251002_HttrB_antagonist")

TEST_FILES = [
    {"cmcr": "2025.10.01-09.45.32-Rec.cmcr", ...},
    {"cmcr": "2025.10.01-09.55.44-Rec.cmcr", ...},
    {"cmcr": "2025.10.01-10.05.45-Rec.cmcr", ...},
]

AGONIST_START_TIME_S = 300.0  # 5 minutes
```

### Step Detection

```python
@dataclass
class StepDetectionConfig:
    threshold_height: float = 35000.0
    min_peak_distance: int = 3
    repeat_range: Tuple[int, int] = (1, -1)
    pre_margin: int = 10   # 1 second before
    post_margin: int = 50  # 5 seconds after
```

### Quality Filtering

```python
@dataclass
class QualityConfig:
    quality_threshold: float = 0.05
    max_trace_value: float = 400.0
```

### Unit Alignment

```python
@dataclass
class AlignmentConfig:
    waveform_weight: float = 10.0
    iteration_distances: Tuple[int, ...] = (0, 1, 2)
    fixed_ref_index: int = -1  # Last recording
```

### Response Analysis

```python
@dataclass
class ResponseAnalysisConfig:
    baseline_range: Tuple[int, int] = (0, 5)
    on_peak_range: Tuple[int, int] = (10, 20)
    off_peak_range: Tuple[int, int] = (40, 50)
    normalize_mode: str = "first"
```

## Output Files

### Individual Recording HDF5

```
recording.h5
├── units/{unit_id}/
│   ├── spike_times          # Original spike times (microseconds)
│   ├── firing_rate_10hz     # Firing rate at 10Hz
│   ├── waveform             # Mean spike waveform
│   ├── step_responses       # (N_trials x N_timepoints) response array
│   ├── response_signature   # Mean response across trials
│   └── quality_index        # Response quality metric
├── stimulus/
│   ├── light_reference_10hz # Downsampled light signal
│   ├── step_on_times        # Step onset sample indices
│   └── step_off_times       # Step offset sample indices
└── metadata/
    └── acquisition_rate, source files, etc.
```

### Aligned Group HDF5

```
aligned_group.h5
├── alignment/
│   └── chains               # Sequential alignment chains
├── fixed_alignment/
│   └── chains               # Fixed-reference alignment chains
├── connections/
│   └── {rec1}_to_{rec2}     # Pairwise unit matches
├── fixed_connections/
│   └── {ref}_to_{target}    # Fixed-reference matches
└── recordings/
    └── {rec_name}/
        ├── unit_ids
        ├── quality_indices
        └── source_path
```

## Generated Figures

The pipeline generates these plots in the `figures/` directory:

1. **on_response_timecourse.png** - ON response magnitude over time
2. **off_response_timecourse.png** - OFF response magnitude over time
3. **alignment_chains.png** - Unit tracking visualization
4. **on_response_heatmap.png** - Heatmap of all ON responses
5. **off_response_heatmap.png** - Heatmap of all OFF responses
6. **{recording}_summary.png** - Per-recording summary plots

## Quality Index

The quality index measures response consistency across trials:

$$QI = \frac{\text{Var}(\text{mean trace across trials})}{\text{Mean}(\text{Var}(\text{each trial}))}$$

Higher values indicate more consistent, reliable responses. Default threshold is 0.05.

## Unit Alignment Algorithm

Units are matched across recordings using:

1. **Electrode proximity**: Units must be on same or nearby electrodes
2. **Waveform similarity**: Euclidean distance between mean waveforms
3. **Response similarity**: Euclidean distance between mean responses

The algorithm iterates with increasing distance thresholds (0, 1, 2 electrodes) to find matches progressively.

## Treatment Effect Analysis

The pipeline marks the treatment time (default: 5 minutes into first recording) and calculates:

- Pre-treatment baseline (1 minute before treatment)
- Post-treatment response (1 minute after treatment)
- Percent change in response magnitude

## Ported From

This pipeline is based on legacy analysis scripts:

- `Legacy_code/.../low_glucose/A01_load_basic_pkl.py` - Data loading
- `Legacy_code/.../low_glucose/A02_pkl_alignment.py` - Unit alignment
- `Legacy_code/.../low_glucose/A03_label_cell_types.py` - Cell labeling
- `Legacy_code/.../low_glucose/A04_step_analysis_v2.py` - Response analysis

Key improvements:
- Uses HDF5 instead of pickle for better performance and interoperability
- Leverages existing `hdmea.io` modules for data loading
- Modular design with separate concerns
- Configurable via dataclasses
- Comprehensive logging

## Test Data

Default test files from `P:\20251002_HttrB_antagonist`:

| Recording | File | Description |
|-----------|------|-------------|
| 1 | 2025.10.01-09.45.32-Rec | Agonist applied at 5 min |
| 2 | 2025.10.01-09.55.44-Rec | Agonist continues |
| 3 | 2025.10.01-10.05.45-Rec | Agonist continues |

## Troubleshooting

### Files not found

Check that the data folder exists and contains the expected files:

```python
from pathlib import Path
data_folder = Path("P:/20251002_HttrB_antagonist")
print(list(data_folder.glob("*.cmcr")))
```

### No steps detected

Adjust the step detection threshold in config:

```python
config.step_detection.threshold_height = 20000  # Lower threshold
```

### No units aligned

Try relaxing the alignment parameters:

```python
config.alignment.quality_threshold = 0.01  # Lower quality threshold
config.alignment.iteration_distances = (0, 1, 2, 3)  # Larger search radius
```

### Memory issues

Process recordings one at a time or reduce the number loaded:

```python
# Load only first 2 recordings
file_paths = get_all_test_file_paths()[:2]
hdf5_paths = step1_load_recordings(file_paths=file_paths)
```

## License

Part of the Data_Processing_2027 project.
