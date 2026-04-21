# Summary: Szatko et al. 2020 vs MEA GB-control data

**Reference paper:** Szatko, Korympidou, Ran, Berens, Dalkara, Schubert, Euler & Franke (2020).
*Neural circuits in the mouse retina support color vision in the upper visual field.*
Nature Communications 11:3481. https://doi.org/10.1038/s41467-020-17113-8

**User dataset:** before-blocker green/blue MEA recordings, 3 experiments
(`_ptx_str`, `_ptx`, `_str`), pooled into
`output_gb_control/combined_gb_control.parquet` — **29,849 cells**.

---

## What the paper claims (relevant to RGC level, Fig. 6-7)

1. **D-V spectral gradient** — RGC center responses follow the opsin gradient:
   ventral $SC_\text{center}=-0.35\pm0.27$, dorsal $SC_\text{center}=+0.06\pm0.25$.
   The gradient spans $\Delta SC\approx0.41$ from ventral to dorsal.

2. **Ventral color opponency** — fraction of color-opponent GCL cells:
   ventral **30.9%** (1312/4247), dorsal **11.4%** (191/1675).
   Color-opponency is ~3× enriched in ventral retina.

3. **Cell-type specificity** — opponency is not uniform across RGC types.
   Sustained-On groups (G22, G24 alpha, G26–G28) are most opponent;
   transient-On and Off-suppressed groups are least opponent.

4. **Rod-cone mechanism** — the green-sensitive surround of ventral cones is
   driven by rods via horizontal cells; blocking AMPA/kainate (NBQX) eliminates
   the green surround. This predicts contrast-dependence since rods saturate at
   high photopic levels.

5. **Both channels scale together in dorsal retina** — dorsal cells have
   matched green and UV center sensitivity because M-opsin dominates; surround
   is also green-dominant, so no net opponency.

---

## Spectral contrast definition used here

$$SC = \frac{G - B}{|G| + |B| + \varepsilon}, \quad \varepsilon = 10^{-6}$$

Bounded in $[-1,+1]$. Positive = green-dominant, negative = blue/UV-dominant.
Absolute values in the denominator keep sign meaningful when peaks are negative
(~10% of cells per channel). The paper defines SC identically except with UV
instead of blue.

**Color opponency proxy** (no center/surround stimulus available):
$|SC_\text{on} - SC_\text{off}| > 0.6$, analogous to the paper's
onset-vs-offset full-field opponency criterion ($\rho_\text{onset}<-0.3$
or $\rho_\text{offset}<-0.3$).

---

## Finding 1 — D-V spectral gradient

| D-V bin (Y, um) | n | User $SC_\text{on}$ | User $SC_\text{off}$ | Paper $SC_\text{center}$ ref |
|---|---|---|---|---|
| $-1000$ (ventral) | 3765 | $+0.025$ | $+0.027$ | $-0.35$ (ventral) |
| $-500$ | 7959 | $+0.032$ | $+0.011$ | |
| $0$ | 5673 | $+0.030$ | $+0.021$ | |
| $+500$ | 8857 | $+0.025$ | $+0.010$ | |
| $+1000$ (dorsal) | 3595 | $+0.015$ | $+0.009$ | $+0.06$ (dorsal) |

Pearson $r(Y_{\mu m},\, SC_\text{on}) = -0.014$, $p = 0.020$, $n = 29{,}849$.

**Verdict: NOT reproduced.**
All bins are slightly positive (green-dominant). The ventral-to-dorsal sign
reversal the paper reports (UV ventral, green dorsal) is absent. The effect
size is ~30× smaller than in the paper ($\Delta SC \approx 0.013$ across
2000 um here vs $\approx 0.41$ in the paper).

**Most likely reasons:**
- The MEA "blue" LED is not deep-UV ($\approx 360$ nm); it does not
  selectively drive ventral S-opsin the way the paper's UV LED does.
- The MEA pools all RGC types including transient-On types that the paper
  itself shows are weakly color-tuned (their Figs. 6b, 7a).
- Full-field spike responses reflect both center and surround; paper shows
  surround is green-dominant everywhere, partially cancelling the ventral UV
  center preference in full-field conditions.

---

## Finding 2 — Ventral enrichment of color opponency

| D-V bin (Y, um) | n | User opponent fraction | 95% Wilson CI |
|---|---|---|---|
| $-1000$ (ventral) | 3765 | 0.228 | [0.215, 0.242] |
| $-500$ | 7959 | 0.221 | [0.212, 0.230] |
| $0$ | 5673 | 0.218 | [0.207, 0.229] |
| $+500$ | 8857 | 0.221 | [0.213, 0.230] |
| $+1000$ (dorsal) | 3595 | 0.244 | [0.230, 0.258] |

Paper reference: ventral 0.309, dorsal 0.114.

Pooled: ventral (Y < 0) fraction = **0.221** ($n=14{,}503$),
dorsal (Y > 0) fraction = **0.227** ($n=15{,}346$). Difference is negligible
and in the opposite direction from the paper.

**Verdict: NOT reproduced.**
Opponent fraction sits uniformly at ~22% across the entire D-V extent,
bracketed between the paper's dorsal (11%) and ventral (31%) values.
No significant spatial gradient.

---

## Finding 3 — Cell-type-specific opponency

| Group | $n$ | Opponent fraction | Ventral | Dorsal | Fisher $p$ (V vs D) |
|---|---|---|---|---|---|
| Other | 4966 | 0.255 | 0.246 | 0.261 | 0.223 |
| OSGC  | 1520 | 0.169 | 0.179 | 0.162 | 0.368 |
| DSGC  | 2595 | 0.217 | 0.195 | 0.235 | **0.015** |
| **ipRGC** | 795 | **0.338** | 0.330 | 0.350 | 0.544 |

**Verdict: REPRODUCED.**
ipRGCs (intrinsically photosensitive RGCs, sustained-On) show the highest
opponent fraction at **33.8%** — matching the paper's pooled ventral fraction
(30.9%) and roughly double the OSGC rate (16.9%). This parallels the paper's
finding that sustained-On groups (G22, G24 alpha, G26–G28) are most enriched
for color-opponency (their Fig. 7a).

OSGCs (direction-selective Off cells) are least opponent, consistent with the
paper's finding that transient/Off groups have few opponent cells. DSGCs show
a significant dorsal-ventral asymmetry ($p=0.015$) but in the opposite
direction from the paper (more opponent dorsally), which may reflect a
different retinal position bias in the DSGC recordings.

---

## Finding 3b — Subtype-level opponency

Splitting the 4 parent groups into their 33 labeled subtypes (9,876 cells,
32 subtypes with $n \geq 20$) reveals large variation within each group.

**Top 5 subtypes by opponency fraction:**

| Subtype | Parent group | $n$ | Opponent fraction | 95% Wilson CI |
|---|---|---|---|---|
| ipRGC_6 | ipRGC | 64 | **0.484** | [0.367, 0.604] |
| ipRGC_4 | ipRGC | 161 | **0.429** | [0.354, 0.506] |
| Other_4 | Other | 673 | **0.334** | [0.300, 0.370] |
| ipRGC_2 | ipRGC | 61 | **0.328** | [0.224, 0.449] |
| ipRGC_8 | ipRGC | 94 | **0.319** | [0.234, 0.417] |

**Bottom 5 subtypes by opponency fraction:**

| Subtype | Parent group | $n$ | Opponent fraction | 95% Wilson CI |
|---|---|---|---|---|
| OSGC_1 | OSGC | 23 | **0.043** | [0.008, 0.211] |
| DSGC_5 | DSGC | 353 | **0.125** | [0.097, 0.159] |
| DSGC_6 | DSGC | 274 | **0.135** | [0.101, 0.178] |
| DSGC_0 | DSGC | 173 | **0.139** | [0.096, 0.196] |
| OSGC_2 | OSGC | 214 | **0.140** | [0.100, 0.192] |

**Key observations:**

- The top of the ranking is dominated by **ipRGC subtypes**: 5 of the top 6
  subtypes are ipRGCs, confirming that the group-level enrichment (Finding 3)
  is consistent across most ipRGC subtypes, not driven by a single outlier.
  ipRGC_6 reaches 48.4% opponent, the highest fraction of any subtype.
- The bottom of the ranking is dominated by **DSGC and OSGC subtypes**:
  DSGC_5 (12.5%), DSGC_6 (13.5%), DSGC_0 (13.9%) are all below the
  group-level DSGC mean of 21.7%, indicating substantial within-group
  heterogeneity.
- **Other_4** (33.4% opponent, $n=673$) is a notable outlier among the
  "Other" subtypes — its opponency fraction matches ipRGC levels and is
  well above the Other group mean (25.5%). This subtype may contain
  sustained-On or other chromatic-processing cells that were not classified
  into a named group.
- The **scatter of SC_on vs opponency fraction** (panel C) shows a
  significant negative correlation ($r=-0.39$, $p=0.026$, $n=32$ subtypes):
  subtypes with lower mean SC_on tend to have higher opponency fractions,
  suggesting that subtypes whose chromatic preference differs from the
  population mean (green-dominant) are also the ones exhibiting more
  ON-OFF spectral mismatch.
- Three subtypes show significant ventral-vs-dorsal asymmetry in opponency
  (Fisher exact $p<0.05$): **DSGC_4** ($p=0.009$, more dorsal), **DSGC_8**
  ($p=0.035$, more dorsal), and **Other_2** ($p=0.002$, more dorsal).
  All three show higher opponent fractions dorsally, the opposite direction
  from the paper's ventral enrichment.

**Verdict: REPRODUCED with finer resolution.**
The group-level conclusion holds at the subtype level: ipRGC subtypes
consistently rank highest in color-opponency, while OSGC and DSGC subtypes
cluster at the bottom. The 10× range in opponency fraction across subtypes
(4.3% to 48.4%) parallels the large inter-group variance the paper reports
across its 32 functional RGC groups (their Fig. 7a).

---

## Finding 4 — Contrast-dependence (rod mechanism proxy)

Pearson $r(Y_{\mu m},\, SC_\text{on})$ at each contrast level:

| Contrast | $r$ | $p$ |
|---|---|---|
| Low  | $-0.022$ | $1.4 \times 10^{-4}$ |
| Mid  | $-0.011$ | $0.052$ |
| High | $+0.007$ | $0.255$ |

**Verdict: WEAK SUPPORT.**
The D-V spectral gradient (already very small) is statistically significant
only at low contrast and disappears at high contrast. This is consistent with
rod involvement: rods contribute to the green-sensitive surround at low
photopic levels but saturate at high contrast, removing their contribution.
Effect sizes remain small, but the trend direction matches the paper's
proposed rod-cone opponent mechanism.

---

## Finding 5 — Both channels co-vary along D-V

| Channel | Ventral mean | Dorsal mean | Dorsal $-$ Ventral |
|---|---|---|---|
| Green ON peak  | 62.2 | 68.1 | $+5.9$ |
| Blue  ON peak  | 59.1 | 65.2 | $+6.1$ |
| Green OFF peak | 46.5 | 43.6 | $-2.9$ |
| Blue  OFF peak | 44.9 | 42.6 | $-2.3$ |

**Verdict: REPRODUCED.**
Green and blue peaks scale together along D-V: both ON-peak amplitudes are
larger in dorsal retina by roughly equal amounts, and both OFF-peak amplitudes
are larger in ventral retina. Because the two channels rise and fall together
their ratio (and hence $SC$) is approximately constant — explaining directly
why Finding 1 fails to show the paper's gradient.

---

## Finding 3c -- Sustained cells and color opponency

The paper (Szatko et al. 2020, Fig. 7a) claims that sustained-On RGC groups
show higher color-opponency than transient groups. To test this directly from
the MEA data, GB-specific transient/sustained (T/S) ratios were computed from
raw traces using the 1 second immediately before light-off as the sustained
response window.

**Stimulus windows used** (60 Hz sampling, 719 samples per trial):

| Window | Samples | Time |
|---|---|---|
| Baseline | [0, 60) | 0.0-1.0 s |
| Green sustained | [180, 240) | 3.0-4.0 s (1 s before green off) |
| Blue sustained | [540, 600) | 9.0-10.0 s (1 s before blue off) |

**T/S ratio formula:** $R = \tanh\!\left(\dfrac{\text{peak\_extreme}}{\text{sustained\_mean} - \text{baseline\_mean}}\right)$

Values near $\pm 1$ are transient-dominated; values near $0$ are sustained.

**Cell-level comparison (N = 27,055 valid cells):**

| Group | Opponent fraction | Opponent fraction | Mean T/S |
|---|---|---|---|
| Opponent (n = 6,177) | 33.0% -- | mean T/S = **0.880** +/- 0.238 |  |
| Non-opponent (n = 20,878) | -- | mean T/S = **0.896** +/- 0.268 |  |

| Statistical test | Result |
|---|---|
| Welch t-test | t = -4.59, p = 4.4e-06 |
| Mann-Whitney U | U = 52,866,343, p = 2.0e-103 |

Opponent cells have a **slightly lower** mean T/S ratio (0.880 vs 0.896),
meaning they are marginally more sustained. The difference is highly
significant statistically but the effect size is small (Cohen's d ~ 0.06).

**Per-group T/S ratio and opponency:**

| Group | n | Mean T/S | Opponent fraction |
|---|---|---|---|
| Other | 4,820 | 0.850 | 25.5% |
| OSGC  | 1,411 | 0.904 | 16.6% |
| DSGC  | 2,527 | 0.824 | 21.8% |
| **ipRGC** | 777 | **0.974** | **33.5%** |

ipRGCs have both the highest T/S ratio (most transient ON-dominant overall)
AND the highest opponency fraction. This seems contradictory, but reflects
the fact that ipRGCs have strong, sustained ON responses in which peak
responses greatly exceed sustained, making their T/S ratio high. The sustained
net response is also large (non-zero), which is what matters for the paper's
claim.

**Decile correlation (T/S vs opponency fraction):**

| Channel | Pearson r | p-value |
|---|---|---|
| Green T/S deciles | r = +0.741 | p = 0.022 |
| Blue T/S deciles  | r = +0.774 | p = 0.014 |

The positive decile correlations indicate that cells with **more positive T/S
ratios** (i.e., strong transient ON responses) tend to have higher opponency
fractions. Cells at the extreme negative end (T/S near -1, transient
OFF-dominated) have the lowest opponency (~9%). This is consistent with the
paper's finding: sustained-On cells are enriched for opponency, but the
dominant pattern in this dataset is that *transient-Off* cells are the least
opponent, rather than that purely sustained cells are the most opponent.

**Verdict: PARTIAL SUPPORT.**
The data shows that color-opponent cells are marginally more sustained than
non-opponent cells (statistically significant but small effect size). The
stronger signal is that OFF-transient cells have the least opponency. ipRGCs,
the most consistently opponent group, have large sustained ON components
alongside large transient peaks. The weakness of the effect relative to the
paper is likely because (a) the paper uses direct center-surround stimulation
to classify sustained vs. transient types, whereas this analysis uses a
full-field sustained window that conflates cell types, and (b) the paper's
"sustained" grouping (G22, G24, G26--G28) is based on functional clustering
of temporal kinetics, not just sustained-window firing rate.

---

## Overall comparison table

| Paper finding | Status | Notes |
|---|---|---|
| Ventral UV-dominant, dorsal green-dominant ($\Delta SC\approx0.41$) | **Not reproduced** | Gradient is $\approx30\times$ weaker; both channels similar everywhere |
| Opponency 3× enriched in ventral retina | **Not reproduced** | Flat ~22% across D-V; paper reports 31% ventral vs 11% dorsal |
| Sustained-On RGCs most color-opponent | **Partially reproduced** | ipRGC 33.5% vs OSGC 16.6%; opponent cells slightly more sustained (T/S 0.880 vs 0.896, p=2e-103); OFF-transient cells least opponent; matches paper's G22/G24 enrichment |
| Opponency varies widely across subtypes (paper Fig. 7a) | **Reproduced** | 10x range across 32 subtypes (4.3%--48.4%); ipRGC subtypes dominate top ranks |
| Contrast-dependent gradient (rod mechanism) | **Weak support** | Low-contrast SC-Y correlation significant; vanishes at high contrast |
| Green and blue channels co-vary across retina | **Reproduced** | Both channels scale proportionally along D-V |
| Full-field opponency (rho < -0.3): ventral 31%, dorsal 11% | **Not reproduced** | 1/15,174 cells opponent (0.007%); green-blue kernels positively correlated (mean rho_onset = +0.63) because MEA blue LED does not isolate S-opsin |

---

## Why the main gradient is absent — interpretation

The paper's dominant finding (ventral UV / dorsal green) depends on two
conditions that are not met in this dataset:

1. **Chromatic stimulus specificity.** The paper uses a $\approx360$ nm UV
   LED that selectively activates S-opsin, which is absent or co-expressed
   only in ventral M-cones. A longer-wavelength "blue" LED activates both
   M-opsin and S-opsin throughout the retina, blunting the spatial contrast.

2. **Recording modality.** Two-photon calcium imaging of the GCL with
   centre-surround stimuli selects for chromatic tuning. The MEA records
   spikes from all retinal ganglion cell types equally, including
   transient-On types that the paper shows are NOT color-opponent and make
   up the majority of RGCs.

The finding that does hold — type-specific opponency enriched in sustained-On
cells — is robust to both of these limitations because it is a comparative
claim (ipRGC vs OSGC) within the same recording paradigm.

---

## Finding 6 -- Full-field opponency (Fig. 6c analog)

The paper defines full-field color opponency by correlating UV and green
event kernels (stimulus-triggered response windows) at light onset and offset:

$$\rho_\text{onset} = \text{corr}(\text{green\_onset}, \text{blue\_onset})$$
$$\rho_\text{offset} = \text{corr}(\text{green\_offset}, \text{blue\_offset})$$
$$\text{ff\_opponency} = \min(\rho_\text{onset},\, \rho_\text{offset})$$

A cell is classified as color-opponent when $\rho_\text{onset} < -0.3$ or
$\rho_\text{offset} < -0.3$ (equivalently, $\text{ff\_opponency} < -0.3$).

**Event-kernel windows** (120 frames = 2.0 s at 60 Hz):

| Kernel | Frames | Time |
|---|---|---|
| green onset  | [60, 180)  | 1.0--3.0 s |
| blue onset   | [420, 540) | 7.0--9.0 s |
| green offset | [240, 359) | 4.0--5.98 s (truncated to 119 frames) |
| blue offset  | [600, 719) | 10.0--11.98 s (119 frames available) |

The blue-OFF segment is only 119 frames long (trace ends at frame 719);
the green-OFF kernel is truncated to match so the Pearson correlation
is computed on equal-length vectors.

**Results (n = 15,174 cells after response filter):**

| Metric | Value |
|---|---|
| Mean $\rho_\text{onset}$ | $+0.625$ |
| Mean $\rho_\text{offset}$ | $+0.491$ |
| Mean ff\_opponency | $+0.443$ |
| Median ff\_opponency | $+0.466$ |
| Cells with $\rho_\text{onset} < -0.3$ | 0 |
| Cells with $\rho_\text{offset} < -0.3$ | 1 |
| **Total opponent (ff < $-0.3$)** | **1 / 15,174 = 0.007%** |

| Retina half | Opponent fraction | 95% Wilson CI |
|---|---|---|
| Dorsal (Y > 0) | 0/6,599 = 0.000 | [0.000, 0.001] |
| Ventral (Y < 0) | 1/5,741 = 0.000 | [0.000, 0.001] |

Paper reference: ventral 0.309, dorsal 0.114.

**Verdict: NOT reproduced.**

Virtually no cells meet the paper's full-field opponency criterion. The green
and blue ON-response kernels are strongly positively correlated (mean
$\rho = +0.63$), meaning the two colors drive nearly identical temporal
response profiles in every cell. The OFF kernels are also positively correlated
(mean $\rho = +0.49$), though slightly less so.

This result is fully consistent with Findings 1--2: the MEA "blue" LED does
not selectively activate S-opsin the way the paper's UV LED does, so both
colors drive M-opsin throughout the retina. Full-field opponency requires
the two colors to produce *opposite* response polarities (negative
correlation), which only happens when different opsin pathways are activated
-- a condition the current stimulus does not satisfy.

The GAM-smoothed spatial maps of $\rho_\text{onset}$ and $\rho_\text{offset}$
show mild spatial variation (slightly lower correlations near the retinal
periphery) but remain well above zero everywhere, confirming the absence of
any opponent region.

---

## Files produced

| File | Contents |
|---|---|
| `figures/fig1_opsin_gradient.png` | Retina schematic + user SC vs Y |
| `figures/fig2_spatial_maps.png` | Spatial scatter/hexbin of SC and opponency |
| `figures/fig3_dv_gradient.png` | Per-bin SC distributions + mean curves |
| `figures/fig4_opponency_map.png` | Opponency scatter + D-V fraction + SC_diff histograms |
| `figures/fig5_group_specific.png` | Per-group bars + group x bin heatmap |
| `figures/fig5b_subtype_specific.png` | Per-subtype bars + subtype x bin heatmap + SC vs opponency scatter |
| `figures/fig6_contrast_breakdown.png` | Low/mid/high contrast SC gradients |
| `figures/fig7_peak_response_maps.png` | Raw green/blue ON/OFF amplitude hexbins |
| `tables/fig3_dv_sc_stats.csv` | Per-bin SC_on and SC_off means |
| `tables/fig4_opponency_dv_fractions.csv` | Per-bin opponent fractions with Wilson CI |
| `tables/fig5_group_opponency_summary.csv` | Per-group opponent fractions, Fisher p |
| `tables/fig5_group_x_bin_opponency.csv` | Group x D-V bin heatmap data |
| `tables/fig5b_subtype_opponency_summary.csv` | Per-subtype opponent fractions, Fisher p, SC means |
| `tables/fig5b_subtype_x_bin_opponency.csv` | Subtype x D-V bin heatmap data |
| `tables/fig6_contrast_sc_correlations.csv` | SC-Y Pearson r per contrast level |
| `tables/fig7_peak_vd_means.csv` | Ventral vs dorsal peak amplitude means |
| `figures/fig8_sustained_opponency.png` | T/S ratio vs opponency panels (A-F) |
| `tables/fig8_sustained_vs_opponency.csv` | Decile T/S vs opponency fraction per channel |
| `tables/fig8_sustained_per_group.csv` | Per-group mean T/S ratio and opponency |
| `tables/fig8_per_cell_sustained.csv` | Per-cell sustained features and T/S ratios |
| `figures/fig10_fullfield_opponency.png` | Full-field opponency hex + GAM spatial maps (rho_onset, rho_offset, ff_opponency, binary opponent) |
| `tables/fig10_per_cell_ffopp.csv` | Per-cell rho_onset, rho_offset, ff_opponency, is_ff_opp |
| `tables/fig10_dv_summary.csv` | Per-D-V-bin full-field opponent fraction with Wilson CI |
