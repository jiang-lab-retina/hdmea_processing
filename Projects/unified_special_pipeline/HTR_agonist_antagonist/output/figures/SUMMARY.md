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
| Httr WT | 3 | 363 | 78 | 155 | 116 |
| HttrB KO | 3 | 281 | 91 | 42 | 139 |
| **Total** | **6** | **644** | **169** | **197** | **255** |

---

## Statistical Findings

### Httr WT

#### ON Cells (n=78)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| ON | 5.31 ± 0.65 | 36.06 ± 3.83 | **+578.8%** | <0.0001 | **Yes** |

#### OFF Cells (n=155)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| OFF | 36.09 ± 1.81 | 28.06 ± 1.38 | **-22.3%** | 0.0002 | **Yes** |

#### ON_OFF Cells (n=116)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| ON | 18.73 ± 1.81 | 36.73 ± 3.29 | **+96.1%** | <0.0001 | **Yes** |
| OFF | 17.30 ± 1.91 | 30.88 ± 1.95 | **+78.5%** | <0.0001 | **Yes** |

---

### HttrB KO

#### ON Cells (n=91)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| ON | 29.04 ± 1.87 | 37.51 ± 3.41 | **+29.2%** | 0.0212 | **Yes** |

#### OFF Cells (n=42)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| OFF | 36.30 ± 3.09 | 37.61 ± 2.45 | +3.6% | 0.6412 | No |

#### ON_OFF Cells (n=139)
| Response | Before (Hz) | After (Hz) | Change | p-value | Significant |
|----------|-------------|------------|--------|---------|-------------|
| ON | 24.90 ± 1.70 | 22.80 ± 2.36 | -8.4% | 0.4568 | No |
| OFF | 19.48 ± 1.41 | 25.67 ± 1.54 | **+31.8%** | <0.0001 | **Yes** |

---

## Key Observations

### 1. Httr WT Response Changes
- **ON cells:** Increase in ON response (+579%)
- **OFF cells:** Decrease in OFF response (-22%)
- **ON_OFF cells:** ON increased (+96%), OFF increased (+79%)

### 2. HttrB KO Response Changes
- **ON cells:** Increase in ON response (+29%)
- **ON_OFF cells:** ON unchanged (p=0.46), OFF increased (+32%)

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