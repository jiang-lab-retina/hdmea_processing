# Data Model: Add frame_time and acquisition_rate to Metadata

**Feature Branch**: `003-add-metadata-fields`  
**Date**: 2024-12-15

## Entities

### Metadata Fields (New)

These fields are added to the existing `/metadata` group in the Zarr archive.

| Field | Type | Unit | Description | Source |
|-------|------|------|-------------|--------|
| `acquisition_rate` | float64 | Hz | Sampling rate (samples per second) | CMCR (primary) → CMTR (fallback) → default 20000 |
| `frame_time` | float64 | seconds | Duration of single sample | Computed: `1 / acquisition_rate` |

### Validation Rules

| Field | Rule | Error Handling |
|-------|------|----------------|
| `acquisition_rate` | Must be > 0 | Reject and try next source in priority chain |
| `acquisition_rate` | Warn if outside 1000-100000 Hz | Log warning but accept value |
| `frame_time` | Must equal `1 / acquisition_rate` | Computed, not validated |

### Existing Entities (Referenced)

| Entity | Location | Description |
|--------|----------|-------------|
| Zarr Root | `/` | Root group with pipeline metadata |
| Metadata Group | `/metadata` | Group storing recording-level parameters |
| Units Group | `/units` | Group storing spike-sorted unit data |
| Stimulus Group | `/stimulus` | Group storing light reference data |

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  CMCR File  │────▶│ load_cmcr() │────▶│ acquisition_rate│
└─────────────┘     └─────────────┘     │ (primary)       │
                                        └────────┬────────┘
                                                 │
┌─────────────┐     ┌─────────────┐              │ fallback
│  CMTR File  │────▶│ load_cmtr() │──────────────┤
└─────────────┘     └─────────────┘              │
                                                 │
                                        ┌────────▼────────┐
                                        │ Default 20000 Hz│
                                        │ (last resort)   │
                                        └────────┬────────┘
                                                 │
                                        ┌────────▼────────┐
                                        │ validate() > 0  │
                                        └────────┬────────┘
                                                 │
                                        ┌────────▼────────┐
                                        │ frame_time =    │
                                        │ 1 / acq_rate    │
                                        └────────┬────────┘
                                                 │
                                        ┌────────▼────────┐
                                        │ write_metadata()│
                                        │ → Zarr attrs    │
                                        └─────────────────┘
```

## State Transitions

N/A - These are static metadata fields with no state transitions.

## Storage Format

### Zarr Metadata Group Attributes

After this feature, `/metadata` group will contain (among others):

```python
# Accessible via:
root = zarr.open("recording.zarr", mode="r")
metadata = root["metadata"]

# As attributes:
metadata.attrs["acquisition_rate"]  # float64, e.g., 20000.0
metadata.attrs["frame_time"]        # float64, e.g., 0.00005
```

### Example Values

| Recording Type | acquisition_rate | frame_time |
|----------------|------------------|------------|
| MaxOne 20kHz | 20000.0 | 0.00005 |
| MaxTwo 20kHz | 20000.0 | 0.00005 |
| Custom 10kHz | 10000.0 | 0.0001 |

