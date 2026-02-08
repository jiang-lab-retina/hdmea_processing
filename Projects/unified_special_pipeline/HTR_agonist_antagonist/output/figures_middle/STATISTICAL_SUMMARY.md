# HTR Agonist Effect: Statistical Summary

**Generated:** 2026-02-02 10:17:08

---

## Dataset Overview

| Genotype | Chips | Total Units |
|----------|-------|-------------|
| Httr WT | 3 | 354 |
| HttrB KO | 3 | 204 |

**Total units analyzed:** 558
**Units with zero-filled recordings:** 52 (9.3%)

### Cell Type Distribution

| Cell Type | Count | Percentage |
|-----------|-------|------------|
| ON | 157 | 28.1% |
| OFF | 195 | 34.9% |
| ON_OFF | 184 | 33.0% |
| unknown | 22 | 3.9% |

---

## Agonist Effect by Cell Type

### ON Cells

| Genotype | n | Baseline (Hz) | Agonist (Hz) | % Change | Cohen's d | p-value (paired) |
|----------|---|---------------|--------------|----------|-----------|------------------|
| Httr WT | 77 | 15.3 ± 1.7 | 24.8 ± 1.4 | +62% | 0.69 | <0.001 *** |
| HttrB KO | 80 | 12.0 ± 1.4 | 25.8 ± 1.3 | +115% | 1.13 | <0.001 *** |

### OFF Cells

| Genotype | n | Baseline (Hz) | Agonist (Hz) | % Change | Cohen's d | p-value (paired) |
|----------|---|---------------|--------------|----------|-----------|------------------|
| Httr WT | 153 | 27.3 ± 1.3 | 22.9 ± 1.0 | -16% | -0.31 | <0.001 *** |
| HttrB KO | 42 | 34.8 ± 2.3 | 28.7 ± 1.6 | -17% | -0.46 | 0.0034 ** |

### ON_OFF Cells

| Genotype | n | Baseline (Hz) | Agonist (Hz) | % Change | Cohen's d | p-value (paired) |
|----------|---|---------------|--------------|----------|-----------|------------------|
| Httr WT | 111 | 16.3 ± 1.2 | 23.1 ± 1.2 | +41% | 0.55 | <0.001 *** |
| HttrB KO | 73 | 17.6 ± 1.4 | 20.5 ± 1.4 | +16% | 0.24 | 0.0133 * |

---

## WT vs KO Comparison

Comparing the magnitude of agonist effect (firing rate change) between genotypes.

| Cell Type | WT % Change | KO % Change | p-value (t-test) | p-value (Mann-Whitney) | Significance |
|-----------|-------------|-------------|------------------|------------------------|--------------|
| ON | +62% | +115% | 0.1211 | 0.2402 | ns |
| OFF | -16% | -17% | 0.5562 | 0.5944 | ns |
| ON_OFF | +41% | +16% | 0.0422 | 0.0241 | * (p<0.05) |

---

## Key Findings

- **ON_OFF cells** show a significantly larger agonist response in WT (+41%) compared to KO (+16%), p=0.0422
- **Httr WT ON cells** show a highly significant increase in firing rate (+62%, Cohen's d=0.69, p<0.001)
- **HttrB KO ON cells** show a highly significant increase in firing rate (+115%, Cohen's d=1.13, p<0.001)
- **Httr WT OFF cells** show a highly significant decrease in firing rate (-16%, Cohen's d=-0.31, p<0.001)
- **Httr WT ON_OFF cells** show a highly significant increase in firing rate (+41%, Cohen's d=0.55, p<0.001)
- Httr WT ON cells: medium effect size (Cohen's d = 0.69)
- HttrB KO ON cells: large effect size (Cohen's d = 1.13)
- Httr WT ON_OFF cells: medium effect size (Cohen's d = 0.55)

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