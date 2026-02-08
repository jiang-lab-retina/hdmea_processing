# HTR Agonist/Antagonist Alignment Analysis Summary

## Experiment Overview

**Analysis Date:** 2026-02-02  
**Analysis Windows:**
- ON Response: 1-3 seconds after step onset (sustained ON)
- OFF Response: 6-8 seconds (1-3s after step offset at 5s)

---

## Data Summary

| Genotype | Recordings | Total Aligned Pairs | ON Cells | OFF Cells | ON_OFF Cells |
|----------|------------|---------------------|----------|-----------|--------------|
| HttrB KO | 2 | 46 | 16 | 21 | 2 |
| Httr WT | 2 | 149 | 73 | 18 | 53 |
| **Total** | **4** | **195** | **89** | **39** | **55** |

---

## Statistical Findings

### HttrB KO

#### ON Cells (n=16)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| ON | 43.84 ± 10.62 | 38.73 ± 6.35 | -11.7% | 0.7020 | No |

#### OFF Cells (n=21)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| OFF | 25.37 ± 3.51 | 19.31 ± 3.56 | -23.9% | 0.2109 | No |

#### ON_OFF Cells (n=2)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| ON | 13.58 ± 6.89 | 2.50 ± 0.00 | -81.6% | 0.4593 | No |
| OFF | 20.33 ± 10.49 | 24.33 ± 10.96 | +19.7% | 0.1051 | No |

---

### Httr WT

#### ON Cells (n=73)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| ON | 33.64 ± 3.28 | 29.87 ± 3.97 | -11.2% | 0.4386 | No |

#### OFF Cells (n=18)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| OFF | 30.11 ± 3.36 | 41.81 ± 5.75 | **+38.9%** | 0.0247 | **Yes** |

#### ON_OFF Cells (n=53)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| ON | 37.74 ± 3.14 | 28.88 ± 4.73 | -23.5% | 0.0975 | No |
| OFF | 22.88 ± 2.56 | 25.22 ± 3.12 | +10.2% | 0.3669 | No |

---

## Key Observations

### 1. HttrB KO Response Changes
- **ON_OFF cells:** ON unchanged (p=0.46), OFF unchanged (p=0.11)

### 2. Httr WT Response Changes
- **OFF cells:** Increase in OFF response (+39%)
- **ON_OFF cells:** ON unchanged (p=0.10), OFF unchanged (p=0.37)

---

## Figures

| Figure | Description |
|--------|-------------|
| `summary_table.png` | Tabular statistics summary |
| `genotype_summary.png` | Bar plots of before/after responses |
| `genotype_mean_traces.png` | Mean PSTH traces by genotype |
| `genotype_overlay_traces.png` | Before vs After overlay comparison |
| `genotype_percent_change.png` | Percent change between genotypes |
| `genotype_distribution.png` | Response distribution box plots |
| `genotype_paired_changes.png` | Individual cell paired changes |

---

## Methods

### Cell Type Classification
Cells were classified based on sustained step response (1-3s window):
- **ON cells:** Significant ON response, no OFF response
- **OFF cells:** Significant OFF response, no ON response
- **ON_OFF cells:** Both ON and OFF responses significant

### Statistical Tests
- Paired t-test for before/after comparisons (same cells tracked across recordings)
- Significance threshold: p < 0.05
- Values reported as mean ± SEM

### Alignment
Units were matched across recordings using:
- Spatial proximity (electrode position)
- Waveform similarity
- Response signature similarity