# Research: DSGC Direction Sectioning

**Feature**: 011-dsgc-direction-section  
**Date**: 2025-12-27

## Research Topics

### 1. Existing Frame Conversion Patterns

**Question**: How does the existing codebase handle spike-to-frame conversion?

**Finding**: The `hdmea.io.section_time` module provides:

```python
def convert_sample_index_to_frame(sample_index: np.ndarray, frame_timestamps: np.ndarray) -> np.ndarray:
    """Convert sample indices to frame numbers using searchsorted."""
    frame = np.searchsorted(frame_timestamps, sample_index, side='right') - 1
    frame = np.clip(frame, 0, len(frame_timestamps) - 1)
    return frame
```

**Decision**: Reuse this function directly.

**Rationale**: Consistent with existing codebase, tested, efficient O(log n) per sample.

---

### 2. On/Off Dictionary Structure

**Question**: What is the exact structure of the on/off timing dictionary?

**Finding**: Verified by inspection:

```python
# File: moving_h_bar_s5_d8_3x_on_off_dict_area_hd.pkl
# Type: Dict[Tuple[int, int], Dict[str, List[int]]]
# Keys: 90,000 (row, col) tuples from (0,0) to (299,299)
# Values: {'on_peak_location': [24 ints], 'off_peak_location': [24 ints]}

# Trial index mapping:
# Indices 0-7:   Directions 0°-315° (8 dirs), Repetition 1
# Indices 8-15:  Directions 0°-315° (8 dirs), Repetition 2
# Indices 16-23: Directions 0°-315° (8 dirs), Repetition 3
```

**Decision**: Load with `pickle.load()`, access via `(row, col)` tuple keys.

**Rationale**: Standard Python pickle, no special handling needed.

---

### 3. Cell Center Coordinate Conversion

**Question**: How to convert 15×15 grid coordinates to 300×300 pixel space?

**Finding**: 
- STA geometry is computed on 15×15 downsampled grid
- Original stimulus is 300×300 pixels
- Scaling factor: 300 / 15 = 20

```python
center_row_300 = int(center_row_15 * 20)
center_col_300 = int(center_col_15 * 20)
# Clip to valid range [0, 299]
```

**Decision**: Multiply by 20, cast to int, clip.

**Rationale**: Simple integer arithmetic, matches grid ratio exactly.

**Alternatives Rejected**:
- Interpolation: Unnecessary complexity for integer grid lookup
- Rounding: int() truncation is appropriate since center may be float

---

### 4. HDF5 Nested Group Creation

**Question**: How to create the nested `direction_section/{dir}/trials/{rep}` structure?

**Finding**: h5py supports recursive group creation:

```python
# Method 1: Create groups step by step
group = f.require_group(f"units/{unit_id}/spike_times_sectioned/{movie}/direction_section")
dir_group = group.require_group(str(direction))
trials_group = dir_group.require_group("trials")
trials_group.create_dataset(str(rep), data=spike_array)

# Method 2: Create full path at once (creates intermediates)
f.create_dataset(
    f"units/{unit_id}/spike_times_sectioned/{movie}/direction_section/{dir}/trials/{rep}",
    data=spike_array
)
```

**Decision**: Use `require_group()` for safety with `force` parameter logic.

**Rationale**: `require_group()` is idempotent, handles existing groups gracefully.

---

### 5. PRE_MARGIN_FRAME_NUM Alignment

**Question**: Why add 60 frames when computing movie start?

**Finding**: From `section_time.py`:

```python
PRE_MARGIN_FRAME_NUM = 60  # Frames before actual movie content
# section_time[0,0] points to frame 60 BEFORE movie frame 0
# on_off_dict frame 0 = first actual movie frame
# Therefore: movie_start_frame = convert(section_time[0,0]) + 60
```

**Decision**: Add `PRE_MARGIN_FRAME_NUM` to section_time-derived frame.

**Rationale**: Aligns recording frame reference with movie/on_off_dict frame reference.

---

### 6. Copy-on-Test Pattern

**Question**: How to protect source test file during development?

**Finding**: Best practice:

```python
import shutil

def copy_for_test(source_path: Path, output_dir: Path) -> Path:
    """Copy source HDF5 to output directory for safe modification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    shutil.copy2(source_path, output_path)
    return output_path
```

**Decision**: Implement `output_path` parameter; copy source if specified.

**Rationale**: Protects original data, enables reproducible testing.

---

### 7. Direction List Ordering

**Question**: What is the exact direction sequence?

**Finding**: From user specification:

```python
direction_list = [0, 45, 90, 135, 180, 225, 270, 315]  # degrees
# Index 0 → 0°, Index 1 → 45°, ..., Index 7 → 315°
```

**Decision**: Use this exact list as constant `DIRECTION_LIST`.

**Rationale**: Matches stimulus generation and on_off_dict indexing.

---

## Summary of Decisions

| Topic | Decision | Key Reason |
|-------|----------|------------|
| Frame conversion | Reuse `convert_sample_index_to_frame` | Consistency, tested |
| Dictionary loading | Standard `pickle.load()` | Simple, sufficient |
| Center scaling | Multiply by 20, int, clip | Matches grid ratio |
| HDF5 groups | Use `require_group()` | Safe with force flag |
| Margin offset | Add 60 frames | Aligns reference systems |
| Test safety | Copy to output path | Protect source data |
| Direction order | `[0, 45, 90, ..., 315]` | Matches stimulus design |

## Open Questions

None - all clarifications resolved.

