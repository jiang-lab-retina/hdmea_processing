# HTR Pipeline Command Line Interface

This document describes how to run the HTR agonist/antagonist analysis pipeline using command line arguments for different datasets.

## Quick Start

### Agonist Data (Default)

```powershell
cd "m:/Python_Project/Data_Processing_2027/Projects/unified_special_pipeline/HTR_agonist_antagonist"

# Run full pipeline with default paths (output/)
python alignment.py
python middle_recordings_alignment.py
python visualize_alignment.py
python middle_visualization.py
```

### Antagonist Data

```powershell
cd "m:/Python_Project/Data_Processing_2027/Projects/unified_special_pipeline/HTR_agonist_antagonist"

# Run full pipeline with custom paths (output_antagonist/)
python alignment.py --h5-folder output_antagonist --output output_antagonist/aligned
python middle_recordings_alignment.py --h5-folder output_antagonist
python visualize_alignment.py --input output_antagonist/aligned --output output_antagonist/figures
python middle_visualization.py --input output_antagonist/aligned_middle --output output_antagonist/figures_middle
```

---

## Pipeline Steps

### Step 0: Process Raw Data (if needed)

Process raw CMCR/CMTR files to HDF5 format.

```powershell
# Agonist data
python batch_pipeline.py --data_folder "O:/20251001_HttrB_agonist" --output output --skip_existing

# Antagonist data
python batch_pipeline.py --data_folder "P:/20251002_HttrB_antagonist" --output output_antagonist --skip_existing
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--data_folder` | Folder containing raw CMCR/CMTR files | `O:/20251001_HttrB_agonist` |
| `--output` | Output directory for H5 files | `output` |
| `--skip_existing` | Skip files that already have H5 output | False |

---

### Step 1: BEFORE→AFTER Alignment

Aligns units between "before" (step response) and "after" (manual) recordings with cell type classification.

```powershell
# Agonist (default)
python alignment.py

# Antagonist
python alignment.py --h5-folder output_antagonist --output output_antagonist/aligned
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--h5-folder` | Folder containing processed H5 files | `output` |
| `--output` | Output directory for aligned files | `{h5-folder}/aligned` |
| `--quality-threshold` | Quality index threshold | `0.05` |
| `--waveform-weight` | Waveform similarity weight | `10.0` |
| `--debug` | Enable debug logging | False |

**Output:** `{output}/*_aligned.h5`

---

### Step 2: Middle Recordings Alignment

Aligns middle recordings (agonist/antagonist application period) using BEFORE→AFTER chain.

```powershell
# Agonist (default)
python middle_recordings_alignment.py

# Antagonist
python middle_recordings_alignment.py --h5-folder output_antagonist
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--h5-folder` | Folder containing processed H5 files | `output` |
| `--aligned-folder` | Folder with BEFORE→AFTER aligned files | `{h5-folder}/aligned` |
| `--output` | Output directory | `{h5-folder}/aligned_middle` |
| `--debug` | Enable debug logging | False |

**Output:** `{output}/*_middle_aligned.h5`

**Note:** First middle recording must have a match (no zero-fill). Subsequent recordings can be zero-filled if match is missing.

---

### Step 3: BEFORE→AFTER Visualization

Generates comparison plots for before/after step responses.

```powershell
# Agonist (default)
python visualize_alignment.py

# Antagonist
python visualize_alignment.py --input output_antagonist/aligned --output output_antagonist/figures
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Folder containing aligned H5 files | `output/aligned` |
| `--output` | Output directory for figures | `{input}/../figures` |
| `--debug` | Enable debug logging | False |

**Output:**
- `summary_table.png` - Overall statistics
- `genotype_summary.png` - By genotype breakdown
- `genotype_mean_traces.png` - Mean PSTH traces
- `genotype_overlay_traces.png` - Before/after overlay
- `genotype_percent_change.png` - Percent change comparison
- `genotype_distribution.png` - Response distributions
- `genotype_paired_changes.png` - Paired change analysis
- `SUMMARY.md` - Statistical summary

---

### Step 4: Middle Recordings Visualization

Generates time course plots showing firing rate changes during agonist/antagonist application.

```powershell
# Agonist (default)
python middle_visualization.py

# Antagonist
python middle_visualization.py --input output_antagonist/aligned_middle --output output_antagonist/figures_middle
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Folder containing aligned middle H5 files | `output/aligned_middle` |
| `--output` | Output directory for figures | `{input}/../figures_middle` |
| `--debug` | Enable debug logging | False |

**Output:**
- `timecourse_by_genotype.png` - Time courses by genotype
- `timecourse_by_cell_type.png` - Time courses by cell type (WT top, KO bottom)
- `genotype_change_comparison.png` - WT vs KO percent change
- `wt_ko_timecourse_comparison.png` - WT/KO overlay by cell type
- `baseline_vs_agonist_scatter.png` - Individual unit responses
- `effect_size_comparison.png` - Cohen's d effect sizes
- `cell_type_*.png` - Cell type specific plots
- `STATISTICAL_SUMMARY.md` - Complete statistical report

---

## Complete Pipeline Commands

### Agonist Dataset (O:/20251001_HttrB_agonist)

```powershell
cd "m:/Python_Project/Data_Processing_2027/Projects/unified_special_pipeline/HTR_agonist_antagonist"

# Process raw data (skip if H5 files exist)
python batch_pipeline.py --skip_existing

# Run alignments
python alignment.py
python middle_recordings_alignment.py

# Generate visualizations
python visualize_alignment.py
python middle_visualization.py
```

### Antagonist Dataset (P:/20251002_HttrB_antagonist)

```powershell
cd "m:/Python_Project/Data_Processing_2027/Projects/unified_special_pipeline/HTR_agonist_antagonist"

# Process raw data
python batch_pipeline.py --data_folder "P:/20251002_HttrB_antagonist" --output output_antagonist --skip_existing

# Run alignments
python alignment.py --h5-folder output_antagonist --output output_antagonist/aligned
python middle_recordings_alignment.py --h5-folder output_antagonist

# Generate visualizations
python visualize_alignment.py --input output_antagonist/aligned --output output_antagonist/figures
python middle_visualization.py --input output_antagonist/aligned_middle --output output_antagonist/figures_middle
```

---

## Output Directory Structure

```
output/                              # Agonist data
├── *.h5                            # Processed recordings
├── aligned/                        # BEFORE→AFTER alignment
│   └── *_to_*_aligned.h5
├── aligned_middle/                 # Middle recordings alignment
│   └── *_to_*_middle_aligned.h5
├── figures/                        # BEFORE→AFTER visualization
│   ├── *.png
│   └── SUMMARY.md
└── figures_middle/                 # Middle recordings visualization
    ├── *.png
    └── STATISTICAL_SUMMARY.md

output_antagonist/                   # Antagonist data (same structure)
├── *.h5
├── aligned/
├── aligned_middle/
├── figures/
└── figures_middle/
```

---

## Alternative Alignment Method

For strict chain matching (requires match in ALL recordings, no zero-fill):

```powershell
python middle_recordings_alignment_strict.py --h5-folder output_antagonist
```

This produces fewer unit chains but ensures complete data for all recordings.

---

## Troubleshooting

### No aligned files found
- Ensure Step 1 (alignment.py) completed successfully
- Check that H5 files exist in the specified `--h5-folder`

### No middle recordings found
- Verify the Google Sheet CSV has the correct condition labels
- Check that middle recording H5 files exist

### Visualization errors
- Ensure alignment steps completed before running visualization
- Check for sufficient units in each cell type category

### Debug mode
Add `--debug` to any command for verbose logging:
```powershell
python alignment.py --h5-folder output_antagonist --debug
```
