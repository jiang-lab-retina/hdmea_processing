# Quickstart: Axon Tracking (AP Trace) for HDF5 Pipeline

**Feature Branch**: `010-ap-trace-hdf5`
**Date**: 2025-12-25

## Prerequisites

1. HDF5 file with `eimage_sta` computed for units
2. Trained CNN model file (`CNN_3d_with_velocity_model_from_all_process.pth`)
3. GPU recommended for optimal performance (CPU fallback available)

## Installation

The AP tracking feature is part of the hdmea package:

```bash
# From project root
pip install -e ".[dev]"
```

## Basic Usage

### Single File Processing

```python
from pathlib import Path
from hdmea.features.ap_tracking import compute_ap_tracking

# Define paths
hdf5_path = Path("Projects/load_gsheet/export_gsheet_20251225/2025.03.06-12.38.11-Rec.h5")
model_path = Path("Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth")

# Run AP tracking (writes directly to HDF5)
compute_ap_tracking(
    hdf5_path=hdf5_path,
    model_path=model_path,
)
```

### With Session (Deferred Save)

```python
from hdmea.pipeline import create_session
from hdmea.features.ap_tracking import compute_ap_tracking

# Create session
session = create_session(dataset_id="2025.03.06-12.38.11-Rec")

# Load recording into session
session = load_recording(hdf5_path, session=session)

# Compute AP tracking (accumulates in memory)
session = compute_ap_tracking(
    hdf5_path=hdf5_path,
    model_path=model_path,
    session=session,
)

# Save all at once
session.save()
```

### Batch Processing

```python
from pathlib import Path
from hdmea.features.ap_tracking import compute_ap_tracking_batch

# Find all HDF5 files
hdf5_dir = Path("Projects/load_gsheet/export_gsheet_20251225")
hdf5_files = list(hdf5_dir.glob("*.h5"))

model_path = Path("Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth")

# Process all files
results = compute_ap_tracking_batch(
    hdf5_paths=hdf5_files,
    model_path=model_path,
    skip_existing=True,  # Skip files already processed
)

# Print summary
for path, status in results.items():
    print(f"{Path(path).name}: {status}")
```

## Configuration Options

### Force CPU Processing

```python
compute_ap_tracking(
    hdf5_path=hdf5_path,
    model_path=model_path,
    force_cpu=True,  # Disable GPU, use CPU only
)
```

### Limit Units for Testing

```python
compute_ap_tracking(
    hdf5_path=hdf5_path,
    model_path=model_path,
    max_units=5,  # Process only first 5 units
)
```

### Custom Detection Parameters

```python
compute_ap_tracking(
    hdf5_path=hdf5_path,
    model_path=model_path,
    # Soma detection
    soma_std_threshold=3.0,
    soma_temporal_range=(5, 27),
    soma_refine_radius=5,
    # AIS detection
    ais_search_xy_radius=5,
    ais_search_t_radius=5,
    # Post-processing
    temporal_window_size=5,
    centroid_threshold=0.05,
    max_displacement=5,
    # Pathway fitting
    min_points_for_fit=10,
    r2_threshold=0.8,
)
```

## Reading Results

After processing, results are stored in the HDF5 file. All values are stored as explicit datasets (not attributes):

```python
import h5py

with h5py.File(hdf5_path, "r") as f:
    for unit_id in f["units"].keys():
        ap_tracking = f[f"units/{unit_id}/features/ap_tracking"]
        
        # Read DVNT positions (all stored as datasets)
        dv = ap_tracking["DV_position"][()]  # scalar dataset
        nt = ap_tracking["NT_position"][()]
        lr = ap_tracking["LR_position"][()].decode("utf-8")  # string dataset
        
        # Read refined soma
        soma = ap_tracking["refined_soma"]
        soma_t = soma["t"][()]
        soma_x = soma["x"][()]
        soma_y = soma["y"][()]
        
        # Read prediction data
        prediction = ap_tracking["prediction_sta_data"][:]
        
        # Read polar coordinates (if available)
        if "soma_polar_coordinates" in ap_tracking:
            polar = ap_tracking["soma_polar_coordinates"]
            radius = polar["radius"][()]
            angle = polar["angle"][()]
            quadrant = polar["quadrant"][()].decode("utf-8")
            
        print(f"Unit {unit_id}:")
        print(f"  DVNT: DV={dv}, NT={nt}, LR={lr}")
        print(f"  Soma: t={soma_t}, x={soma_x}, y={soma_y}")
        if "soma_polar_coordinates" in ap_tracking:
            print(f"  Polar: r={radius:.2f}, θ={angle:.2f}, {quadrant}")
```

**Note**: All scalar values use `dataset[()]` syntax to read the value. String datasets need `.decode("utf-8")` for Python 3 compatibility.

## Output Structure

All values are stored as explicit HDF5 datasets (not attributes):

```
units/{unit_id}/features/ap_tracking/
├── DV_position          # (dataset) Dorsal-ventral position
├── NT_position          # (dataset) Nasal-temporal position
├── LR_position          # (dataset) Left/Right eye string
├── refined_soma/        # Refined soma position
│   ├── t                # (dataset) Time index
│   ├── x                # (dataset) Row index
│   └── y                # (dataset) Column index
├── axon_initial_segment/
│   ├── t                # (dataset) Time index
│   ├── x                # (dataset) Row index
│   └── y                # (dataset) Column index
├── prediction_sta_data  # (dataset) CNN prediction array
├── post_processed_data/
│   ├── filtered_prediction  # (dataset) Filtered predictions
│   └── axon_centroids       # (dataset) Nx3 centroid array
├── ap_pathway/          # Line fit to AP direction
│   ├── slope            # (dataset) Line slope
│   ├── intercept        # (dataset) Line intercept
│   ├── r_value          # (dataset) Correlation
│   ├── p_value          # (dataset) P-value
│   └── std_err          # (dataset) Standard error
├── all_ap_intersection/ # Optic disc intersection
│   ├── x                # (dataset) X coordinate
│   └── y                # (dataset) Y coordinate
└── soma_polar_coordinates/
    ├── radius           # (dataset) Distance from intersection
    ├── angle            # (dataset) Angle in radians
    ├── cartesian_x      # (dataset) X distance
    ├── cartesian_y      # (dataset) Y distance
    ├── quadrant         # (dataset) Geometric quadrant string
    └── anatomical_quadrant  # (dataset) Anatomical quadrant string
```

## Troubleshooting

### No eimage_sta Data

```
ValueError: HDF5 has no units with eimage_sta data
```

**Solution**: Run STA computation first:

```python
from hdmea.features.sta import compute_sta

compute_sta(hdf5_path, movie_name="eimage")
```

### Model File Not Found

```
FileNotFoundError: Model file not found: /path/to/model.pth
```

**Solution**: Verify model path:

```python
model_path = Path("Projects/ap_trace_hdf5/model/CNN_3d_with_velocity_model_from_all_process.pth")
assert model_path.exists(), f"Model not found at {model_path}"
```

### GPU Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or force CPU:

```python
compute_ap_tracking(
    hdf5_path=hdf5_path,
    model_path=model_path,
    batch_size=64,  # Reduce from auto
    # or
    force_cpu=True,
)
```

### No Polar Coordinates Calculated

Polar coordinates require at least 2 units with valid AP pathway fits (R² > 0.8). 
Check the log for messages like:

```
Insufficient valid fits (1) for intersection calculation. Need at least 2.
```

**Solution**: This is expected for recordings with few RGCs or noisy data.

## Performance Tips

1. **Use GPU**: 10-20x faster than CPU
2. **Batch processing**: Amortizes model loading overhead
3. **Skip existing**: Set `skip_existing=True` for resume capability
4. **Monitor memory**: Watch GPU memory with `nvidia-smi`

