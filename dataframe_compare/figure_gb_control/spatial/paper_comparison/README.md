# Paper comparison: Szatko et al. 2020 vs GB-control RGC data

Reproduces the core spatial/chromatic findings of Szatko, Korympidou, Ran,
Berens, Dalkara, Schubert, Euler and Franke (2020), *Neural circuits in the
mouse retina support color vision in the upper visual field*,
[Nat. Commun. 11:3481](https://doi.org/10.1038/s41467-020-17113-8),
using the before-blocker green/blue MEA dataset pooled in
[`combined_gb_control.parquet`](../../../output_gb_control/combined_gb_control.parquet)
(29,849 cells, 3 experiments).

Pipeline entry point:

```powershell
python run_all.py
```

Each figure is also runnable standalone. PNGs land in `figures/`,
per-figure summary tables in `tables/`.

## Methodological notes

- **Spectral contrast (SC) definition.** The user's preexisting
  `green_blue_on_ratio` / `green_blue_off_ratio` are
  $\tanh(\text{green peak}/\text{blue peak})$, which saturates heavily
  near the extremes and makes paper comparison awkward. Each script
  recomputes a paper-style

  $$SC = \frac{G - B}{|G| + |B| + \varepsilon}$$

  bounded in $[-1,+1]$. The absolute values in the denominator keep the
  sign interpretable even when one of the peaks is negative (i.e. the
  within-window extreme is a decrement, which happens in ~10 percent of
  cells for each channel).
- **Color-opponency proxy.** The user's data has no center-vs-surround
  stimulus. Instead the chromatic preference can differ between ON and
  OFF phases (analogous to the paper's onset-vs-offset full-field
  opponency, their $\rho_\text{onset}/\rho_\text{offset}$ in Fig. 4d).
  A cell is called color-opponent when $|SC_\text{on}-SC_\text{off}|>0.6$.
- **UV vs blue.** The paper's UV LED peaks near 360 nm and selectively
  drives mouse S-opsin. The user's "blue" channel is presumably a
  longer-wavelength LED, so any ventral S-opsin-specific signal will be
  weaker than in the paper.
- **D-V binning.** 0.5 mm bins in $Y_{\mu m}=\text{improved\_ty}\times 16$,
  matching paper's Fig. 4/6 binning. Only bins 1-5 (centers -1000 to +1000 um)
  actually contain data.

## Figure-by-figure results

### Fig. 1 - Opsin gradient vs observed SC gradient

Panel A reproduces the paper's Fig. 1b cartoon. Panel B overlays the user's
mean $SC_\text{on}$ and $SC_\text{off}$ per D-V bin with the paper's
reference means for ventral/dorsal GCL cells.

Per-bin user means (tables/fig3_dv_sc_stats.csv):

| bin center Y um | n | SC_on mean | SC_off mean |
|---|---|---|---|
| -1000 | 3765 | +0.025 | +0.027 |
| -500  | 7959 | +0.032 | +0.011 |
|   0   | 5673 | +0.030 | +0.021 |
| +500  | 8857 | +0.025 | +0.010 |
| +1000 | 3595 | +0.015 | +0.009 |

**Conclusion:** the user's dataset does **not** reproduce the paper's strong
ventral-UV / dorsal-green gradient (paper ventral $SC_\text{center}=-0.35$,
dorsal $+0.06$). Instead all bins sit near zero and slightly positive,
meaning green slightly dominates everywhere and the sign does not flip
between dorsal and ventral retina. Likely reasons: (1) the MEA spikes are
pooled across all RGC types and include transient-On types that the paper
shows are less color-tuned; (2) the "blue" LED here is not deep UV and
poorly drives ventral S-opsin; (3) the recording geometry and stimulus
paradigm differ fundamentally.

### Fig. 2 - Per-cell spatial maps of chromatic preference

Six panels: per-cell scatter and hexbin of $SC_\text{on}$, $SC_\text{off}$,
and opponency flag over the retinal XY plane. This is the direct analog of
Szatko et al. Fig. 6a. Visually the hexbin maps show weak spatial
structure - consistent with Fig. 3 finding that the gradient is small.

### Fig. 3 - D-V gradient of SC (paper Fig. 6b analog)

Panel A reproduces the paper's per-bin SC distribution style; panel B plots
mean $\pm$ SEM per bin against paper reference horizontal lines.
The Pearson correlation $r(Y, SC_\text{on})=-0.014$ (p=0.02, n=29849) is
statistically significant but tiny, two orders of magnitude weaker than
what would be needed to span from paper's ventral $-0.35$ to dorsal $+0.06$.

### Fig. 4 - ON-OFF color opponency (paper Fig. 6c-d analog)

Fraction of cells with $|SC_\text{on}-SC_\text{off}|>0.6$ as a function of
D-V position (tables/fig4_opponency_dv_fractions.csv):

| bin center Y um | n | opponent fraction | 95 percent Wilson CI |
|---|---|---|---|
| -1000 | 3765 | 0.228 | [0.215, 0.242] |
| -500  | 7959 | 0.221 | [0.212, 0.230] |
|   0   | 5673 | 0.218 | [0.207, 0.229] |
| +500  | 8857 | 0.221 | [0.213, 0.230] |
| +1000 | 3595 | 0.244 | [0.230, 0.258] |

Paper-reported fractions: ventral 0.309, dorsal 0.114. The user's data sits
in between (0.22-0.24 everywhere) and **does not show the ventral
enrichment** the paper reports. Instead the dorsal bin at +1000 um is
slightly higher than the ventral bin at -1000 um - the opposite of the
paper's trend, but statistically indistinguishable.

### Fig. 5 - Per-group opponency (paper Fig. 7a analog)

Fraction opponent by functional group (tables/fig5_group_opponency_summary.csv):

| group | n | fraction opponent | fraction ventral | fraction dorsal | Fisher p (V vs D) |
|---|---|---|---|---|---|
| Other | 4966 | 0.255 | 0.246 | 0.261 | 0.223 |
| OSGC  | 1520 | 0.169 | 0.179 | 0.162 | 0.368 |
| DSGC  | 2595 | 0.217 | 0.195 | 0.235 | 0.015 |
| **ipRGC** | **795** | **0.338** | **0.330** | **0.350** | 0.544 |

**Key match with paper:** ipRGCs are sustained-On cells. The paper's Fig. 7
finds that sustained-On groups (G22, G24 alpha, G26, G27, G28) have
significantly higher color-opponency than expected from pure center/surround
SC differences. In our data, **ipRGCs show the highest opponent fraction
(33.8%), roughly double that of OSGCs (16.9%) and matching the paper's
ventral GCL pooled rate (30.9%)**. DSGCs show a small but significant
ventral-vs-dorsal shift (p=0.015), with the *opposite* sign from the paper
(more opponent dorsally).

The group-x-bin heatmap (panel B, tables/fig5_group_x_bin_opponency.csv)
shows the per-bin detail.

### Fig. 5b - Per-subtype opponency (paper Fig. 7a analog, finer-grained)

Splits the 4 parent groups into 33 labeled subtypes (32 with n >= 20).
Opponency fraction ranges from 4.3% (OSGC_1) to 48.4% (ipRGC_6) -- a 10x
range that parallels the paper's large inter-group variance across their 32
functional groups. ipRGC subtypes dominate the top of the ranking (5 of top
6); DSGC and OSGC subtypes cluster at the bottom. The SC_on vs opponency
scatter (panel C) reveals a significant negative correlation (r = -0.39,
p = 0.026): subtypes with lower spectral contrast tend to be more opponent.
Other_4 (33.4%, n=673) is a notable outlier that matches ipRGC-level
opponency despite being in the "Other" group.

See SUMMARY.md, Finding 3b for full tables and interpretation.

### Fig. 6 (extra) - Contrast-dependent SC gradients

SC_on / SC_off computed separately for low / mid / high contrast peaks,
with Pearson r vs Y (tables/fig6_contrast_sc_correlations.csv):

| feature | r(Y) | p | n |
|---|---|---|---|
| SC_on_low   | -0.022 | 1.4e-04 | 29849 |
| SC_off_low  | -0.017 | 3.6e-03 | 29849 |
| SC_on_mid   | -0.011 | 5.2e-02 | 29849 |
| SC_off_mid  | -0.015 | 1.1e-02 | 29849 |
| SC_on_high  | +0.007 | 0.25    | 29849 |
| SC_off_high | -0.001 | 0.84    | 29849 |

The D-V spectral gradient is marginally stronger at **low** contrast than
at high contrast - consistent with rod involvement (rods saturate at high
photopic levels and dominate low-contrast regimes). This is a weak but
coherent hint at the same rod-cone-opponent mechanism the paper
hypothesises, although the effect size remains very small.

### Fig. 7 (extra) - Raw peak amplitude maps

Ventral vs dorsal means of raw peak amplitudes
(tables/fig7_peak_vd_means.csv):

| channel | ventral mean | dorsal mean | dorsal - ventral |
|---|---|---|---|
| green ON peak  | 62.2 | 68.1 | +5.9 |
| blue  ON peak  | 59.1 | 65.2 | +6.1 |
| green OFF peak | 46.5 | 43.6 | -2.9 |
| blue  OFF peak | 44.9 | 42.6 | -2.3 |

Both chromatic channels scale together along D-V: ON amplitudes are larger
in dorsal retina, OFF amplitudes are slightly larger in ventral retina.
Because green and blue rise and fall together, their *ratio* is roughly
constant - which is exactly why SC is near zero everywhere.

## Summary comparison

| Paper finding (Szatko et al.) | Reproduced in user data? |
|---|---|
| Ventral retina UV-dominant, dorsal green-dominant (strong SC gradient) | **No.** Gradient present but two orders of magnitude too weak. |
| Color opponency enriched in ventral retina | **No.** Opponent fraction nearly flat (0.22-0.24) across D-V. |
| Color opponency is RGC-type-dependent; sustained-On alpha cells particularly opponent | **Yes.** ipRGCs (sustained-On) have the highest opponent fraction (33.8%), OSGCs the lowest (16.9%). |
| Rod involvement implies contrast-dependence | **Weak yes.** SC-Y correlation stronger at low contrast than high. |
| Both chromatic channels drive similar center responses across retina | **Yes.** Raw peak maps show green and blue co-vary along D-V. |

Overall, the user's MEA pan-RGC dataset partially reproduces the paper's
cell-type-specific opponency finding but not the paper's striking ventral
spectral dichotomy. The most likely explanation is stimulus + LED
differences (no deep-UV channel, no center-surround paradigm), combined
with the fact that the MEA pools all spiking types - including
transient-On types that the paper itself shows are weakly color-opponent.

## File tree

```
paper_comparison/
  README.md                        # this file
  run_all.py                       # runs fig1..fig7 in order
  _common.py                       # loading, SC, D-V binning, styling
  fig1_opsin_gradient.py
  fig2_spatial_maps.py
  fig3_dv_gradient.py
  fig4_opponency_map.py
  fig5_group_specific.py
  fig5b_subtype_specific.py
  fig6_contrast_breakdown.py
  fig7_peak_response_maps.py
  figures/  fig1..fig7 PNGs
  tables/   per-figure CSV summaries
```
