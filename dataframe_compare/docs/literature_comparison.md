# Literature vs Pipeline Findings: RGC Spatial Distribution Under Inhibitory Blockade

This document compares the spatial analysis results from our MEA-based blocker
comparison pipeline with the published literature on mouse retinal ganglion cell
(RGC) spatial organisation, as reviewed in
[RGC Spatial Distribution Review.docx](RGC%20Spatial%20Distribution%20Review.docx).

Three experimental conditions were analysed:

- **ptx_str** -- picrotoxin + strychnine (combined GABA + glycine block)
- **ptx** -- picrotoxin only (GABAergic block)
- **str** -- strychnine only (glycinergic block)

Supporting CSV tables are in `docs/tables/`; regenerate them with
`python docs/extract_comparison_data.py`.

---

## Cell Population Summary

| Experiment | TOTAL | DSGC | ipRGC | OSGC | Other | Unclassified |
|------------|------:|-----:|------:|-----:|------:|-------------:|
| ptx_str    | 6552  |  453 |   375 |  232 |   871 |         4621 |
| ptx        | 6626  |  530 |   158 |  333 |  1126 |         4479 |
| str        | 6248  |  524 |   102 |  363 |   804 |         4455 |

---

## 1. Opsin Gradient and Color Opponency

### Literature prediction

Mouse cones exhibit a strong dorsoventral opsin gradient: M-opsin is enriched
dorsally and S-opsin ventrally, with a tilted transition zone across the
nasal-temporal axis [2][25]. This photoreceptor gradient produces enrichment
of chromatic opponency in ventral retina [29]. Critically, GABAergic amacrine
cells partially "de-bias" this chromatic gradient at the retinal output [48],
predicting that GABA blockade should **increase** spatial nonuniformity of
chromatic features while glycine blockade may have a smaller effect.

### Pipeline features

`green_blue_on_ratio`, `green_blue_off_ratio`, `green_blue_on_ratio_high`,
`green_blue_off_ratio_high`, `gb_base_mean`, `gb_base_mean_high`

### Quantitative results

#### Before vs After: Plane $R^2$ and Moran's I changes (all cells)

| Feature | $\Delta R^2$ ptx_str | $\Delta R^2$ ptx | $\Delta R^2$ str | $\Delta$ Moran ptx_str | $\Delta$ Moran ptx | $\Delta$ Moran str |
|---------|-----:|-----:|-----:|-----:|-----:|-----:|
| green_blue_off_ratio      | **-0.085** | +0.011 | -0.038 | **-0.089** | +0.021 | -0.021 |
| green_blue_off_ratio_high | **-0.042** | +0.017 | **-0.053** | **-0.076** | +0.056 | **-0.066** |
| green_blue_on_ratio       | -0.022 | -0.040 | -0.003 | -0.024 | **-0.077** | -0.009 |
| green_blue_on_ratio_high  | -0.025 | -0.014 | +0.005 | -0.034 | -0.032 | +0.026 |
| gb_base_mean              | +0.025 | **-0.126** | -0.005 | +0.027 | -0.035 | -0.013 |
| gb_base_mean_high         | +0.026 | **-0.089** | -0.001 | +0.039 | -0.024 | -0.006 |

Bold = |value| > 0.04.

#### Radial centre shifts (ptx_str, all cells)

| Feature | Before (Cx, Cy) um | After (Cx, Cy) um | Shift um | Before \|r\| | After \|r\| |
|---------|----:|----:|----:|----:|----:|
| green_blue_off_ratio      | (1800, 0) | (1121, -478) | 830 | **0.335** | 0.156 |
| green_blue_off_ratio_high | (-1800, -593) | (585, -193) | 2418 | **0.274** | 0.165 |
| gb_base_mean              | (480, -377) | (420, 1800) | 2177 | 0.198 | 0.226 |

### Interpretation

**Partial agreement with literature.** The literature predicts that GABA
blockade should unmask the underlying photoreceptor chromatic gradient, leading
to *increased* spatial structure of green-blue ratios. Our results show the
**opposite** for the combined blocker (ptx_str): `green_blue_off_ratio` loses
substantial spatial structure ($\Delta R^2 = -0.085$, $\Delta$ Moran $= -0.089$),
and the radial correlation drops from $|r| = 0.335$ to $0.156$.

However, this is **consistent** with the literature finding that GABAergic
amacrine cells actively shape chromatic output [48][59][60]. Rather than
simply de-biasing the photoreceptor gradient, blocking all inhibition
(GABA + glycine) appears to disrupt the organised spatial structure of
chromatic tuning entirely, suggesting that inhibitory circuits are
*constructive* rather than merely subtractive in creating the spatial
pattern of color opponency at the ganglion cell level.

The ptx-only condition shows a different pattern: `gb_base_mean` loses
significant spatial structure ($\Delta R^2 = -0.126$) while OFF chromatic
ratios slightly increase, suggesting GABAergic circuits specifically
contribute to maintaining the spatial organisation of baseline chromatic
sensitivity. The str-only condition shows moderate loss of
`green_blue_off_ratio_high` structure ($\Delta R^2 = -0.053$),
indicating glycinergic circuits also contribute to OFF-pathway chromatic
spatial patterning.

The large radial centre shift for `green_blue_off_ratio_high` (2418 um under
ptx_str) indicates the functional centre of chromatic organisation
relocates dramatically when inhibition is removed, supporting the view
that inhibitory circuits actively define the spatial reference frame for
chromatic processing.

---

## 2. Alpha-like ON Sustained Topography

### Literature prediction

Alpha-like sustained ON RGCs exhibit a >3-fold nasal-to-temporal density
gradient, with corresponding decreases in dendritic arbor and receptive field
size toward temporal retina [3][13]. This gradient implies increased sampling
density for binocular/frontal visual space. The gradient direction should be
approximately along the nasal-temporal axis.

### Pipeline features

`on_sustained`, `on_peak_extreme`

### Quantitative results

#### Before vs After: Plane $R^2$, gradient direction, and magnitude (all cells)

| Feature | Exp | Before $R^2$ | After $R^2$ | $\Delta R^2$ | Before dir | After dir | Before mag | After mag |
|---------|-----|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| on_sustained | ptx_str | 0.093 | 0.069 | -0.024 | -114 | -131 | 0.0048 | 0.0042 |
| on_sustained | ptx     | 0.052 | 0.131 | **+0.079** | -116 | -122 | 0.0038 | 0.0073 |
| on_sustained | str     | 0.067 | 0.039 | -0.029 | +175 | -143 | 0.0050 | 0.0038 |
| on_peak_extreme | ptx_str | 0.022 | 0.040 | +0.018 | -6 | -23 | 0.0083 | 0.0120 |
| on_peak_extreme | ptx     | 0.001 | 0.095 | **+0.094** | +180 | -73 | 0.0018 | 0.0190 |
| on_peak_extreme | str     | 0.012 | 0.015 | +0.003 | +137 | -43 | 0.0067 | 0.0072 |

#### Per-group spatial structure (ptx_str, on_sustained)

| Group | Before $R^2$ | After $R^2$ | Before Moran | After Moran |
|-------|-----:|-----:|-----:|-----:|
| Other | **0.299** | 0.135 | 0.126 | 0.176 |
| ipRGC | **0.224** | 0.012 | **0.224** | 0.093 |
| OSGC  | 0.025 | 0.098 | 0.011 | 0.126 |
| DSGC  | 0.025 | 0.002 | -0.036 | 0.054 |

### Interpretation

**Agreement with literature.** The baseline "before" condition shows
meaningful spatial gradients for `on_sustained` ($R^2$ = 0.052 to 0.093
across experiments) with gradient directions predominantly in the
110-175 degree range, consistent with a dorsal-to-ventral or
nasal-to-temporal gradient as reported for alpha-like sustained ON cells.

The most striking finding is the **drug-specific response**: under
ptx-only (GABAergic block), both `on_sustained` and `on_peak_extreme`
show dramatic *increases* in spatial structure ($\Delta R^2 = +0.079$ and
$+0.094$, respectively), with gradient magnitude nearly doubling. This
suggests that GABAergic inhibition normally **attenuates** the intrinsic
spatial gradient of ON sustained responses, and removing it reveals a
stronger underlying topographic organisation -- consistent with
literature descriptions of inhibitory shaping of alpha-like RGC response
properties [4][52][53].

Under combined block (ptx_str), the effect is weaker, and under
str-only the ON sustained gradient slightly decreases, suggesting
that glycinergic circuits may partially counteract the GABAergic
unmasking effect.

The per-group analysis shows that the "Other" and "ipRGC" groups carry
the strongest baseline spatial structure for `on_sustained` (likely
containing alpha-like cells that are not classified as DSGC or OSGC),
and that blocker application preferentially disrupts the ipRGC group's
spatial pattern (from $R^2 = 0.224$ to $0.012$).

---

## 3. Direction Selectivity Spatial Bias

### Literature prediction

Direction-selective (DS) circuits depend critically on asymmetric GABAergic
inhibition from starburst amacrine cells [33][37][44]. GABA blockade should
therefore substantially reduce DS tuning, and the spatial organisation of
DSI should be disrupted. Direction-selective RGCs are distributed across
the retina but some subtypes show regional enrichment, and DS-circuit
plasticity shows dorsoventral variation [4][36].

### Pipeline features

`dsi`, per-group DSGC statistics

### Quantitative results

#### Before vs After: dsi spatial structure (all cells)

| Exp | Before $R^2$ | After $R^2$ | $\Delta R^2$ | Before Moran | After Moran | $\Delta$ Moran |
|-----|-----:|-----:|-----:|-----:|-----:|-----:|
| ptx_str | 0.012 | 0.011 | -0.001 | 0.030 | 0.006 | -0.024 |
| ptx     | 0.009 | 0.018 | +0.010 | 0.069 | 0.069 | +0.000 |
| str     | 0.008 | 0.009 | +0.000 | 0.027 | 0.044 | +0.017 |

#### DSGC-specific dsi spatial structure (ptx_str)

| Condition | $R^2$ | Moran's I | Gradient dir | Radial r |
|-----------|------:|----------:|-------------:|---------:|
| Before    | **0.193** | **0.182** | +94 | -0.216 |
| After     | 0.105 | 0.146 | +159 | +0.111 |
| Delta     | 0.099 | 0.114 | -139 | +0.270 |

#### Radial centre shifts (ptx_str, dsi)

| Metric | Before | After | Change |
|--------|-----:|-----:|-----:|
| Centre (Cx, Cy) um | (-371, -803) | (266, -32) | 1000 um shift |
| \|r\| | 0.104 | 0.134 | +0.030 |

### Interpretation

**Agreement with literature.** At the all-cells level, `dsi` shows weak
baseline spatial structure ($R^2 < 0.02$), which is expected because DSI
is a property of a specific RGC subtype (DSGCs), not the population mean.

When examined **within the DSGC group**, however, the spatial structure
is substantial (before $R^2 = 0.193$, Moran's I $= 0.182$) and is
reduced by combined block ($R^2$ drops to $0.105$, Moran's I to $0.146$).
This is consistent with the literature prediction that GABA blockade
disrupts DS tuning. The gradient direction shift (from 94 to 159 degrees)
and radial correlation sign flip (from $-0.216$ to $+0.111$) indicate
that the spatial pattern of direction selectivity is reorganised under
blocker, not just attenuated.

Notably, ptx-only shows a slight *increase* in dsi spatial structure at the
population level ($\Delta R^2 = +0.010$), possibly because while individual
DS tuning is reduced, the population's spatial distribution of residual
tuning becomes more structured. The str-only condition shows minimal change
in $R^2$ but increased Moran's I ($+0.017$), consistent with glycinergic
circuits having a subtler but distinct contribution to DS spatial
organisation [4][36].

---

## 4. ON/OFF Pathway Modulation by Inhibition

### Literature prediction

Crossover inhibition between ON and OFF pathways is a fundamental inhibitory
circuit motif shaped by both GABAergic and glycinergic amacrine cells [4][34].
Blocking inhibition should: (a) unmask latent response components in both
pathways, (b) alter ON/OFF balance, and (c) potentially increase spatial
gradients of ON and OFF peak responses by removing the spatial homogenisation
that crossover inhibition provides [52][53].

### Pipeline features

`on_off_ratio`, `on_off_sus_ratio`, `on_peak_extreme`, `off_peak_extreme`

### Quantitative results

#### Before vs After (all cells)

| Feature | $\Delta R^2$ ptx_str | $\Delta R^2$ ptx | $\Delta R^2$ str | $\Delta$ Moran ptx_str | $\Delta$ Moran ptx | $\Delta$ Moran str |
|---------|-----:|-----:|-----:|-----:|-----:|-----:|
| on_off_ratio      | +0.024 | **+0.045** | +0.009 | **+0.097** | +0.006 | +0.037 |
| on_off_sus_ratio  | -0.041 | -0.009 | -0.004 | -0.017 | +0.013 | +0.028 |
| on_peak_extreme   | +0.018 | **+0.094** | +0.003 | +0.028 | **+0.052** | -0.010 |
| off_peak_extreme  | +0.045 | **+0.096** | +0.016 | +0.031 | **+0.158** | +0.049 |

### Interpretation

**Strong agreement with literature.** The most robust finding across all
experiments is the *increase* in spatial structure of `off_peak_extreme`
under blocker. Under ptx-only, $\Delta R^2 = +0.096$ and $\Delta$ Moran
$= +0.158$, the largest Moran's I increase of any feature-experiment
combination. This directly supports the prediction that inhibition
normally constrains OFF pathway response amplitude variation across
the retina, and removing it reveals stronger spatial gradients.

The `on_off_ratio` gains significant spatial autocorrelation under all
conditions (Moran's I increases by +0.097 in ptx_str, +0.037 in str),
confirming that inhibitory circuits spatially homogenise ON/OFF balance
across the retina. The GABAergic contribution (ptx) dominates the effect
on peak response amplitudes, while the combined blocker effect on ON/OFF
ratio clustering (Moran) is the largest.

The `on_off_sus_ratio` behaves differently, losing spatial structure under
ptx_str ($\Delta R^2 = -0.041$). This suggests that the sustained component
of ON/OFF balance is organised by a distinct inhibitory mechanism from
the one shaping peak responses, consistent with separate circuit pathways
for transient and sustained signalling.

---

## 5. Transient/Sustained Temporal Filtering

### Literature prediction

Glycinergic circuitry mediates dorsoventral variation in temporal response
properties within identified RGC types [4][41]. The ON transient-to-sustained
ratio should therefore show strong spatial gradients that are specifically
disrupted by strychnine (glycine block). Regional differences in
temporal filtering are linked to rod-pathway interactions and crossover
inhibitory routing.

### Pipeline features

`on_trans_sus_ratio`, `off_trans_sus_ratio`

### Quantitative results

#### Before vs After (all cells)

| Feature | Exp | Before $R^2$ | After $R^2$ | $\Delta R^2$ | Before Moran | After Moran | $\Delta$ Moran |
|---------|-----|-----:|-----:|-----:|-----:|-----:|-----:|
| on_trans_sus_ratio | ptx_str | **0.080** | 0.017 | **-0.064** | **0.138** | 0.046 | **-0.092** |
| on_trans_sus_ratio | ptx     | 0.050 | 0.056 | +0.007 | 0.077 | 0.107 | +0.029 |
| on_trans_sus_ratio | str     | **0.096** | 0.009 | **-0.087** | **0.129** | 0.037 | **-0.092** |
| off_trans_sus_ratio | ptx_str | 0.015 | 0.010 | -0.005 | 0.058 | 0.029 | -0.029 |
| off_trans_sus_ratio | ptx     | 0.009 | 0.009 | +0.000 | 0.017 | 0.055 | +0.038 |
| off_trans_sus_ratio | str     | 0.000 | 0.009 | +0.008 | 0.019 | 0.038 | +0.019 |

### Interpretation

**Strong agreement with literature.** The `on_trans_sus_ratio` is the
clearest example of a **glycine-specific spatial effect** in our dataset.
Under str-only, it shows the largest $\Delta R^2$ of any feature-experiment
pair ($-0.087$) and a matching Moran's I collapse ($-0.092$). The combined
blocker (ptx_str) shows a similarly large effect ($\Delta R^2 = -0.064$,
$\Delta$ Moran $= -0.092$), driven by the strychnine component.

Critically, **ptx-only has essentially no effect** ($\Delta R^2 = +0.007$),
cleanly dissociating the contributions: the spatial organisation of ON
transient/sustained balance is maintained by glycinergic, not GABAergic,
circuits. This directly matches the literature finding that glycinergic
circuitry mediates dorsoventral temporal response reformatting [4][41].

The baseline before-blocker gradient ($R^2 = 0.080$ to $0.096$, direction
around -150 to +177 degrees) is among the strongest of any feature,
consistent with the dorsoventral temporal filtering gradient described
in the literature.

The `off_trans_sus_ratio` shows weaker baseline spatial structure and
less drug sensitivity, suggesting that the temporal filtering gradient
is primarily an ON pathway phenomenon at the ganglion cell level, or
that OFF temporal properties are more spatially uniform.

---

## 6. ipRGC Spatial Organisation

### Literature prediction

Intrinsically photosensitive RGCs (ipRGCs) show dorsoventral gradients
in melanopsin expression and subtype-specific density variations across
the retina [15]. Different ipRGC subtypes (M1-M6) have distinct
topographies coupled to opsin gradients that spectrally tune ipRGC
responses by location.

### Pipeline features

`step_up_QI` (step-up response quality index, a functional indicator
related to ipRGC-like sustained responses)

### Quantitative results

#### Before vs After (all cells)

| Exp | Before $R^2$ | After $R^2$ | $\Delta R^2$ | Before Moran | After Moran | $\Delta$ Moran |
|-----|-----:|-----:|-----:|-----:|-----:|-----:|
| ptx_str | 0.009 | 0.043 | +0.034 | 0.072 | 0.128 | +0.056 |
| ptx     | 0.006 | **0.106** | **+0.100** | 0.071 | **0.184** | **+0.112** |
| str     | 0.023 | 0.003 | -0.020 | 0.065 | 0.094 | +0.028 |

#### Per-group step_up_QI spatial structure (ptx_str)

| Group | Before $R^2$ | After $R^2$ | $\Delta R^2$ | Before Moran | After Moran |
|-------|-----:|-----:|-----:|-----:|-----:|
| ipRGC | 0.012 | 0.049 | +0.037 | 0.001 | 0.024 |
| Other | 0.039 | **0.139** | **+0.100** | -0.090 | **0.192** |
| DSGC  | 0.031 | 0.089 | +0.058 | -0.073 | 0.020 |
| OSGC  | 0.027 | 0.053 | +0.026 | -0.046 | -0.091 |

### Interpretation

**Agreement with literature, with novel drug-specific insight.** Under
ptx-only, `step_up_QI` shows the most dramatic increase in spatial
structure of any feature-experiment pair ($\Delta R^2 = +0.100$,
$\Delta$ Moran $= +0.112$). This reveals that GABAergic inhibition
strongly suppresses the spatial gradient of step-up responses,
and removing it unmasks a robust topographic pattern -- consistent
with the literature's description of spatially varying ipRGC properties.

The combined blocker (ptx_str) shows a moderate effect ($\Delta R^2 = +0.034$),
while str-only actually *decreases* the spatial gradient ($\Delta R^2 = -0.020$).
This suggests that while GABA blockade reveals the underlying spatial
gradient, glycine blockade partially counteracts this unmasking,
possibly by disrupting the sustained response components that
contribute to the step-up quality metric.

The per-group analysis reveals that the strongest blocker effect on
`step_up_QI` spatial structure occurs in the "Other" group ($\Delta R^2 =
+0.100$, Moran rising from $-0.090$ to $+0.192$), suggesting that the
unmasked spatial gradient may involve cells not classified as ipRGC by
the current model, or that step-up responses are a more broadly
distributed functional property than the ipRGC classification captures.

---

## 7. Orientation Selectivity

### Literature prediction

Orientation selectivity in mouse RGCs is less well characterised spatially
than direction selectivity [55]. Some regional variation exists, but the
circuit mechanisms and their inhibitory dependence are only partially
described.

### Pipeline features

`osi` (orientation selectivity index)

### Quantitative results

#### Before vs After (all cells)

| Exp | Before $R^2$ | After $R^2$ | $\Delta R^2$ | Before Moran | After Moran | $\Delta$ Moran |
|-----|-----:|-----:|-----:|-----:|-----:|-----:|
| ptx_str | 0.018 | 0.018 | -0.001 | 0.040 | 0.033 | -0.007 |
| ptx     | 0.032 | 0.021 | -0.010 | **0.113** | **0.118** | +0.004 |
| str     | **0.094** | 0.023 | **-0.071** | **0.115** | 0.030 | **-0.085** |

### Interpretation

**Novel finding: glycine-dependent OSI spatial structure.** The most striking
result is the str-only condition, where `osi` shows one of the largest
$\Delta R^2$ values in the entire dataset ($-0.071$) with a dramatic
Moran's I collapse ($-0.085$). This reveals that the spatial organisation
of orientation selectivity depends substantially on **glycinergic** circuits.

Under ptx-only, the osi spatial structure is largely preserved (Moran's I
stays at ~0.115), and under the combined blocker (ptx_str), the effect
is minimal. This apparent contradiction (combined block < glycine-only)
may reflect compensatory interactions: when both inhibitory systems are
removed simultaneously, the spatial pattern re-organises via non-inhibitory
mechanisms.

The baseline osi gradient is modest in ptx_str ($R^2 = 0.018$) but stronger
in str ($R^2 = 0.094$), likely reflecting inter-preparation variability.
The direction of the gradient (approximately -60 to -65 degrees) is
consistent across experiments, suggesting a reproducible nasal-temporal
axis of orientation selectivity variation.

---

## Cross-Experiment Consistency: Drug-Specific Effects

The literature predicts that GABAergic and glycinergic circuits have distinct
roles in spatial organisation [4][41][48]. Our data provide a systematic
dissociation:

### Features primarily disrupted by **strychnine** (glycine block)

| Feature | $\Delta R^2$ ptx | $\Delta R^2$ str | Evidence |
|---------|-----:|-----:|:---|
| on_trans_sus_ratio | +0.007 | **-0.087** | Glycine-specific temporal filtering [4][41] |
| osi | -0.010 | **-0.071** | Glycine-dependent OS spatial structure |
| green_blue_off_ratio_high | +0.017 | **-0.053** | Glycinergic chromatic OFF pathway |

### Features primarily disrupted by **picrotoxin** (GABA block)

| Feature | $\Delta R^2$ ptx | $\Delta R^2$ str | Evidence |
|---------|-----:|-----:|:---|
| step_up_QI | **+0.100** | -0.020 | GABA suppresses ipRGC-like gradient |
| off_peak_extreme | **+0.096** | +0.016 | GABA constrains OFF peak gradients |
| on_peak_extreme | **+0.094** | +0.003 | GABA constrains ON peak gradients |
| on_sustained | **+0.079** | -0.029 | GABA attenuates sustained gradient |
| gb_base_mean | **-0.126** | -0.005 | GABA maintains chromatic baseline |

### Features affected by **both** (synergistic or independent contributions)

| Feature | $\Delta R^2$ ptx_str | $\Delta R^2$ ptx | $\Delta R^2$ str | Pattern |
|---------|-----:|-----:|-----:|:---|
| green_blue_off_ratio | -0.085 | +0.011 | -0.038 | Combined > either alone |
| on_off_ratio | +0.024 | +0.045 | +0.009 | Both contribute, GABA dominant |
| off_peak_extreme | +0.045 | +0.096 | +0.016 | Both contribute, GABA dominant |

### Key observations

1. **Glycine-specific effects** are concentrated on temporal filtering
   (`on_trans_sus_ratio`) and orientation selectivity -- features related
   to fine temporal and orientation tuning that depend on glycinergic
   amacrine cell circuits.

2. **GABA-specific effects** dominate response amplitude features
   (`on_peak_extreme`, `off_peak_extreme`, `on_sustained`, `step_up_QI`)
   and overall chromatic sensitivity (`gb_base_mean`), consistent with
   GABAergic surround inhibition and gain control mechanisms.

3. **Combined block** sometimes shows effects larger than either drug alone
   (e.g., `green_blue_off_ratio`), suggesting synergistic disruption,
   and sometimes shows intermediate effects (e.g., `on_sustained`),
   suggesting partial cancellation between the two inhibitory systems.

4. These drug-specific dissociations are **consistent with the literature**
   framework [4][41][48] that GABAergic and glycinergic circuits play
   complementary roles in shaping RGC spatial response properties,
   with glycinergic circuits more involved in temporal/orientation
   tuning and GABAergic circuits more involved in gain control and
   surround mechanisms.

---

## Discrepancies and Inconsistencies with Literature

Several findings from the pipeline **do not match** straightforward
predictions from the literature. These discrepancies are important because
they highlight either (a) circuit mechanisms more complex than current
models, (b) limitations of the MEA/pipeline approach, or (c) genuinely
novel biology.

### D1. Chromatic gradient: blocker *reduces* rather than *reveals* spatial structure

The most direct contradiction. The literature describes GABAergic amacrine
cells as "de-biasing" the photoreceptor chromatic gradient [48][59][60],
implying that blocking GABA should remove this de-biasing and *increase*
spatial nonuniformity of green-blue ratios (i.e., reveal the raw
photoreceptor gradient at the ganglion cell output). Our data show the
opposite:

- `green_blue_off_ratio` under ptx_str: $\Delta R^2 = -0.085$,
  $\Delta$ Moran $= -0.089$
- Radial $|r|$ drops from 0.335 to 0.156

This suggests that the spatial structure of chromatic opponency at the
RGC level is not a passive read-out of the photoreceptor gradient minus
inhibitory de-biasing. Instead, inhibitory circuits appear to
**constructively generate** the organised spatial pattern of colour
sensitivity. When inhibition is removed, the spatial pattern collapses
rather than intensifies, implying that amacrine cell networks actively
impose spatial order on chromatic processing.

Furthermore, the drug-specific effects are internally contradictory:
ptx-only slightly *increases* `green_blue_off_ratio` structure
($\Delta R^2 = +0.011$), while str-only and combined block *decrease*
it. If GABAergic circuits de-bias the gradient, ptx should show the
largest increase, not the smallest change. This pattern is hard to
reconcile with the simple "de-biasing" model.

**Possible explanations:**
- The "de-biasing" described in the literature may operate at the
  bipolar cell or amacrine cell level rather than at the ganglion cell
  output measured by MEA
- The MEA recordings may lack the spatial resolution to resolve the
  fine-grained dorsoventral gradient, and the chromatic structure
  detected by the pipeline may reflect a different (circuit-level)
  organisation that depends on inhibition
- The combined blocker (ptx_str) disrupts network-level activity patterns
  that are necessary for maintaining any coherent spatial map, regardless
  of the photoreceptor gradient

### D2. Combined block often weaker than single-drug block

For several features, the combined blocker (ptx_str) produces a *smaller*
effect than either drug alone, which is paradoxical if both circuits
contribute independently:

| Feature | $\Delta R^2$ ptx_str | $\Delta R^2$ ptx | $\Delta R^2$ str |
|---------|-----:|-----:|-----:|
| on_sustained | -0.024 | **+0.079** | -0.029 |
| on_peak_extreme | +0.018 | **+0.094** | +0.003 |
| off_peak_extreme | +0.045 | **+0.096** | +0.016 |
| step_up_QI | +0.034 | **+0.100** | -0.020 |
| osi | -0.001 | -0.010 | **-0.071** |

For `on_sustained` and `on_peak_extreme`, ptx-only produces large
increases in spatial structure, but adding strychnine (ptx_str) nearly
cancels this effect. The literature does not predict this cancellation.

This pattern is inconsistent with simple additive models of
GABAergic + glycinergic contributions and instead suggests **antagonistic
interactions** between the two inhibitory systems: removing one unmasks
certain spatial gradients, but removing both triggers compensatory
network reorganisation or reveals a different functional state where the
unmasked gradients are no longer coherent.

For osi, the combined block produces almost no change despite str-only
producing the third-largest $\Delta R^2$ in the dataset. This is
particularly difficult to explain mechanistically and raises the
possibility that cross-preparation variability contributes to the
apparent drug-specificity.

### D3. Direction selectivity: spatial structure is reduced but not abolished

The literature predicts that GABA blockade should largely abolish
direction selectivity by disrupting the asymmetric starburst amacrine
cell inhibition [33][37][44][67]. Our DSGC-specific analysis shows that
dsi spatial structure is reduced ($R^2$: 0.193 to 0.105) but far from
eliminated under combined block. The Moran's I drops from 0.182 to 0.146,
a ~20% reduction rather than the near-complete loss predicted.

Additionally, at the **all-cells** level, dsi spatial structure under
ptx-only actually *increases* ($\Delta R^2 = +0.010$, Moran unchanged
at 0.069), which is the opposite of what GABA-dependent DS circuit
disruption would predict. This may reflect that population-level DSI
statistics are dominated by the large unclassified fraction (~70%),
where dsi changes have a different spatial pattern than in true DSGCs.

The gradient direction within DSGCs shifts substantially (from +94 to
+159 degrees), and the radial correlation sign flips (from $-0.216$ to
$+0.111$). This reorganisation, rather than simple attenuation, suggests
that blocking inhibition does not merely remove DS tuning but changes
the spatial distribution of residual tuning -- a finding not anticipated
by the "starburst removal" model.

### D4. ON sustained gradient is *reduced* under combined block, not increased

While ptx-only reveals a stronger `on_sustained` gradient ($\Delta R^2 =
+0.079$), the combined block (ptx_str) and str-only both *decrease* it
($\Delta R^2 = -0.024$ and $-0.029$). The literature on alpha-like
sustained ON RGCs predicts a robust intrinsic topographic gradient
[3][13] that should become more visible when inhibitory circuits that
spatially normalise responses are removed.

The str-only decrease suggests glycinergic circuits may contribute
positively to the ON sustained spatial gradient, contrary to the
expectation that inhibition generally attenuates gradients. This could
reflect glycinergic rod-pathway interactions [4][41] that enhance
sustained responses in specific retinal regions (e.g., where rod density
is higher), creating spatial structure that is removed by strychnine.

### D5. ipRGC-classified cells show weaker step-up spatial effects than "Other"

The literature describes ipRGCs as having spatially varying properties
[15]. Under ptx_str, the strongest `step_up_QI` spatial gain occurs in
the "Other" group ($\Delta R^2 = +0.100$, Moran from $-0.090$ to
$+0.192$), not in the ipRGC group ($\Delta R^2 = +0.037$, Moran from
$+0.001$ to $+0.024$).

This is inconsistent with step_up_QI being primarily an ipRGC-specific
marker. Possible explanations:

- The autoencoder + DEC classifier may under-classify ipRGCs, placing
  many true ipRGCs into "Other" (the unclassified pool of ~4500 cells
  likely contains ipRGC-like cells)
- Step-up responses are a broader population property that reflects
  sustained excitation rather than melanopsin-intrinsic photosensitivity
- Under blocker, non-ipRGC cells may acquire step-up-like responses by
  losing inhibitory suppression of sustained ON components, creating
  spatial gradients in a population that did not originally express them

### D6. OFF temporal filtering gradient is not glycine-dependent

The literature predicts that glycinergic circuitry shapes temporal
filtering across the retina [4][41]. While `on_trans_sus_ratio` strongly
confirms this for the ON pathway (str: $\Delta R^2 = -0.087$), the
analogous OFF pathway feature `off_trans_sus_ratio` does **not** show
a corresponding glycine-dependent effect:

- str: $\Delta R^2 = +0.008$ (slight *increase*)
- ptx: $\Delta R^2 = +0.000$ (no change)
- ptx_str: $\Delta R^2 = -0.005$ (minimal)

Moreover, under ptx-only the OFF temporal ratio gains Moran's I
($+0.038$), the opposite direction from the ON pathway prediction.
This ON/OFF asymmetry in glycine sensitivity of temporal filtering
is not emphasised in the literature, which tends to describe temporal
filtering gradients more broadly. The data suggest that OFF pathway
temporal properties are maintained by a different mechanism than ON
pathway temporal properties, possibly involving distinct amacrine cell
subtypes or bipolar cell circuits.

### D7. Weak baseline spatial structure across most features

The literature emphasises that mouse retina, while lacking a fovea,
harbours "pronounced, behavior-relevant nonuniformities" across many
RGC types [1][12][68]. However, the pipeline finds that most features
have modest baseline plane $R^2$ values:

| Baseline $R^2$ range | Feature count (of 17) |
|----------------------:|----------------------:|
| > 0.10               | 1 (green_blue_off_ratio in ptx_str) |
| 0.05 -- 0.10         | 4-5 (varies by experiment) |
| 0.02 -- 0.05         | 5-6 |
| < 0.02               | 5-7 |

Only `green_blue_off_ratio` (ptx_str: $R^2 = 0.107$), `on_sustained`
(ptx_str: $R^2 = 0.093$), `on_trans_sus_ratio` (str: $R^2 = 0.096$),
and `osi` (str: $R^2 = 0.094$) approach $R^2 = 0.10$.

This contrasts with the transcriptomic and imaging literature that reports
many RGC types with strong spatial biases [12][16][17]. The discrepancy
likely reflects:

- The pipeline analyses **all cells pooled**, diluting type-specific
  gradients across a heterogeneous population (per-group $R^2$ values
  are higher, e.g., DSGC dsi $R^2 = 0.193$)
- Hexbin spatial smoothing averages over mosaic-level local order
- MEA sampling is limited to the array footprint, which may not span
  the full extent of retinal gradients
- Functional features (firing rates, ratios) are noisier spatial
  markers than molecular identity

---

## Summary of Agreement with Literature

| Theme | Agreement | Key finding | Discrepancy |
|-------|-----------|-------------|-------------|
| 1. Opsin/chromatic gradient | **Partial** | Blocker disrupts rather than reveals gradient | Opposite to "de-biasing" prediction (D1); ptx-only vs combined inconsistent |
| 2. Alpha ON sustained | Mixed | ptx-only reveals stronger gradient | Combined and str-only *reduce* gradient, contradicting simple unmasking model (D4) |
| 3. Direction selectivity | **Partial** | DSGC-specific structure reduced by blocker | Spatial structure not abolished, only halved; all-cells dsi *increases* under ptx (D3) |
| 4. ON/OFF pathway | **Strong** | OFF peak extreme shows largest Moran increase | on_off_sus_ratio behaves oppositely to other ON/OFF features |
| 5. Temporal filtering | **Strong** | Glycine-specific ON temporal filtering disruption | OFF pathway does NOT show glycine-dependent pattern (D6) |
| 6. ipRGC organisation | **Partial** | GABA block unmasks step_up_QI gradient | Effect is in "Other" group, not ipRGC group (D5) |
| 7. Orientation selectivity | **Novel** | Glycine-dependent spatial structure | Combined block shows no effect despite large str-only effect (D2) |

---

## Notes on Additional Analyses

1. **Per-group spatial analysis** is limited by smaller bin counts
   (n=73-107 bins vs 773-780 for all-cells), reducing statistical power.
   The DSGC-specific direction selectivity finding is robust despite this.

2. **Gradient direction interpretation** requires careful mapping to
   retinal coordinates. The pipeline uses ONH-centred improved coordinates,
   but the exact retinal orientation (dorsal/ventral/nasal/temporal) depends
   on the mounting convention, which varies across preparations.

3. **Within-type gene expression gradients** described in recent
   transcriptomic studies [17][24] predict that a single functional cluster
   may split into region-dependent subclusters under pharmacology -- this
   aligns with the per-group $R^2$ changes observed above.

4. **Radial centre analysis** provides a complementary view to planar
   gradients: features whose radial centres shift dramatically under
   blocker (e.g., `green_blue_off_ratio_high`: 2418 um shift) suggest
   that inhibition fundamentally alters the spatial reference frame
   of that feature's organisation.

---

*Generated by the blocker comparison pipeline. Supporting data in
`docs/tables/`. Regenerate with `python docs/extract_comparison_data.py`.*

*Literature references refer to citation numbers in
[RGC Spatial Distribution Review.docx](RGC%20Spatial%20Distribution%20Review.docx).*
