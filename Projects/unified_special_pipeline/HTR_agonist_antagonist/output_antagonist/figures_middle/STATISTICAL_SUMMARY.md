# HTR Agonist Effect: Statistical Summary

**Generated:** 2026-02-02 10:08:54

---

## Dataset Overview

| Genotype | Chips | Total Units |
|----------|-------|-------------|
| Httr WT | 2 | 142 |
| HttrB KO | 2 | 41 |

**Total units analyzed:** 183
**Units with zero-filled recordings:** 36 (19.7%)

### Cell Type Distribution

| Cell Type | Count | Percentage |
|-----------|-------|------------|
| ON | 83 | 45.4% |
| OFF | 38 | 20.8% |
| ON_OFF | 51 | 27.9% |
| unknown | 11 | 6.0% |

---

## Agonist Effect by Cell Type

### ON Cells

| Genotype | n | Baseline (Hz) | Agonist (Hz) | % Change | Cohen's d | p-value (paired) |
|----------|---|---------------|--------------|----------|-----------|------------------|
| Httr WT | 70 | 8.7 ± 1.2 | 27.2 ± 1.2 | +211% | 1.81 | <0.001 *** |
| HttrB KO | 13 | 15.7 ± 2.0 | 22.9 ± 2.9 | +46% | 0.81 | 0.0397 * |

### OFF Cells

| Genotype | n | Baseline (Hz) | Agonist (Hz) | % Change | Cohen's d | p-value (paired) |
|----------|---|---------------|--------------|----------|-----------|------------------|
| Httr WT | 18 | 24.5 ± 3.2 | 22.6 ± 2.4 | -8% | -0.16 | 0.5262 |
| HttrB KO | 20 | 23.9 ± 2.7 | 11.0 ± 2.5 | -54% | -1.11 | <0.001 *** |

### ON_OFF Cells

| Genotype | n | Baseline (Hz) | Agonist (Hz) | % Change | Cohen's d | p-value (paired) |
|----------|---|---------------|--------------|----------|-----------|------------------|
| Httr WT | 50 | 13.2 ± 2.1 | 22.1 ± 1.3 | +68% | 0.74 | <0.001 *** |
| HttrB KO | 1 | 18.1 ± 0.0 | 7.9 ± 0.0 | -56% | 0.00 | N/A |

---

## WT vs KO Comparison

Comparing the magnitude of agonist effect (firing rate change) between genotypes.

| Cell Type | WT % Change | KO % Change | p-value (t-test) | p-value (Mann-Whitney) | Significance |
|-----------|-------------|-------------|------------------|------------------------|--------------|
| ON | +211% | +46% | 0.0078 | 0.0060 | ** (p<0.01) |
| OFF | -8% | -54% | 0.0078 | 0.0186 | ** (p<0.01) |
| ON_OFF | +68% | -56% | nan | nan | ns |

---

## Key Findings

- **ON cells** show a significantly larger agonist response in WT (+211%) compared to KO (+46%), p=0.0078
- **OFF cells** show a significantly larger agonist response in KO (-54%) compared to WT (-8%), p=0.0078
- **Httr WT ON cells** show a highly significant increase in firing rate (+211%, Cohen's d=1.81, p<0.001)
- **HttrB KO OFF cells** show a highly significant decrease in firing rate (-54%, Cohen's d=-1.11, p<0.001)
- **Httr WT ON_OFF cells** show a highly significant increase in firing rate (+68%, Cohen's d=0.74, p<0.001)
- Httr WT ON cells: large effect size (Cohen's d = 1.81)
- HttrB KO ON cells: large effect size (Cohen's d = 0.81)
- HttrB KO OFF cells: large effect size (Cohen's d = -1.11)
- Httr WT ON_OFF cells: medium effect size (Cohen's d = 0.74)

---

## Methods

- **Baseline period:** 0-5 minutes (before agonist application)
- **Agonist period:** 5-30 minutes (after agonist application)
- **Bin size:** 30 seconds
- **Statistical tests:**
  - Paired t-test: Within-genotype baseline vs agonist comparison
  - Independent t-test: WT vs KO comparison of agonist effect magnitude
  - Mann-Whitney U test: Non-parametric alternative for WT vs KO comparison
- **Effect size:** Cohen's d (small: 0.2, medium: 0.5, large: 0.8)

---

*Report generated automatically by middle_visualization.py*