# Subtype Spatial Distribution Analysis

**Input**: `labeled_dataframe_with_legacy_coords_freq.parquet`
**Coordinates**: legacy_transformed_x/y (electrode units x16 = microns)
**Spatial filter**: |coord| < 100 electrode units
**Total cells after filter**: 10,801
**Valid subtypes**: 30

---

## 1. Subtype Cell Counts

| Group | Subtype | n cells |
|-------|---------|---------|
| DSGC  | DSGC_0  | 194     |
| DSGC  | DSGC_2  | 645     |
| DSGC  | DSGC_3  | 535     |
| DSGC  | DSGC_4  | 277     |
| DSGC  | DSGC_5  | 229     |
| DSGC  | DSGC_6  | 303     |
| DSGC  | DSGC_7  | 163     |
| DSGC  | DSGC_8  | 273     |
| OSGC  | OSGC_0  | 383     |
| OSGC  | OSGC_1  | 210     |
| OSGC  | OSGC_2  | 191     |
| OSGC  | OSGC_3  | 307     |
| OSGC  | OSGC_4  | 59      |
| OSGC  | OSGC_5  | 520     |
| OSGC  | OSGC_6  | 86      |
| Other | Other_0 | 471     |
| Other | Other_1 | 1,611   |
| Other | Other_2 | 968     |
| Other | Other_3 | 724     |
| Other | Other_4 | 152     |
| Other | Other_5 | 361     |
| Other | Other_6 | 290     |
| Other | Other_7 | 468     |
| ipRGC | ipRGC_2 | 244     |
| ipRGC | ipRGC_3 | 266     |
| ipRGC | ipRGC_4 | 92      |
| ipRGC | ipRGC_5 | 199     |
| ipRGC | ipRGC_6 | 189     |
| ipRGC | ipRGC_7 | 89      |
| ipRGC | ipRGC_8 | 173     |

---

## 2. Subtypes Ranked by Average Spatial Structure

### 2a. Spatial Clustering (Mean Moran's I Across All Features)

Moran's I measures spatial autocorrelation: positive values indicate that neighbouring hexbins tend to have similar feature values (spatial clustering); values near zero indicate a spatially random distribution.

| Rank | Subtype | Avg Moran's I | Max Moran's I | Feature at Max |
|------|---------|--------------|--------------|----------------|
| 1    | OSGC_3  | 0.079 | 0.429 | angle_correction_applied |
| 2    | ipRGC_7 | 0.074 | 0.627 | angle_correction_applied |
| 3    | Other_2 | 0.066 | 0.535 | angle_correction_applied |
| 4    | Other_3 | 0.062 | 0.544 | angle_correction_applied |
| 5    | DSGC_8  | 0.060 | 0.443 | angle_correction_applied |
| 6    | Other_5 | 0.057 | 0.336 | angle_correction_applied |
| 7    | DSGC_0  | 0.052 | 0.426 | angle_correction_applied |
| 8    | OSGC_6  | 0.049 | 0.392 | angle_correction_applied |
| 9    | Other_1 | 0.048 | 0.608 | angle_correction_applied |
| 10   | OSGC_2  | 0.048 | 0.320 | angle_correction_applied |

**Key observation**: `angle_correction_applied` (DVNT-based angle correction) dominates as the highest-clustering feature in 29 of 30 subtypes. This is expected -- it reflects the recording geometry rather than a biological property. Among non-geometric features, the next most spatially structured features are `chip_effective_area`, `gaussian_r2`, and `dog_sigma_exc`.

### 2b. Average Gradient Strength

Average magnitude of the linear spatial gradient across all features, per subtype.

| Rank | Subtype | Avg Gradient Mag | Avg Gradient R^2 | n cells |
|------|---------|-----------------|-----------------|---------|
| 1    | OSGC_4  | 0.300 | 0.053 | 59  |
| 2    | OSGC_6  | 0.228 | 0.063 | 86  |
| 3    | Other_4 | 0.196 | 0.039 | 152 |
| 4    | ipRGC_4 | 0.118 | 0.031 | 92  |
| 5    | OSGC_1  | 0.087 | 0.028 | 210 |
| 6    | ipRGC_7 | 0.087 | 0.059 | 89  |
| 7    | OSGC_2  | 0.063 | 0.034 | 191 |
| 8    | ipRGC_6 | 0.050 | 0.029 | 189 |
| 9    | DSGC_4  | 0.047 | 0.024 | 277 |
| 10   | Other_5 | 0.045 | 0.015 | 361 |

**Note**: Subtypes with the strongest apparent gradients (OSGC_4, OSGC_6, Other_4, ipRGC_4) tend to have smaller sample sizes (n < 160). This may inflate gradient estimates due to noise. Subtypes with large populations (DSGC_2, Other_1, Other_2) show weaker average gradients, suggesting their spatial distributions are more uniform.

### 2c. Average Hexbin Unevenness (CV)

Higher CV of hexbin means indicates more spatially heterogeneous feature distributions.

| Rank | Subtype | Avg Hexbin CV | Avg Moran's I | n cells |
|------|---------|--------------|--------------|---------|
| 1    | DSGC_0  | 5.09  | 0.052 | 194 |
| 2    | Other_0 | 2.50  | 0.029 | 471 |
| 3    | OSGC_5  | 2.25  | 0.028 | 520 |
| 4    | OSGC_6  | 2.11  | 0.049 | 86  |
| 5    | ipRGC_6 | 2.06  | 0.025 | 189 |
| 6    | DSGC_2  | 2.04  | 0.036 | 645 |
| 7    | ipRGC_2 | 2.00  | 0.021 | 244 |
| 8    | OSGC_0  | 1.84  | 0.047 | 383 |
| 9    | Other_4 | 1.83  | 0.047 | 152 |
| 10   | Other_7 | 1.79  | 0.026 | 468 |

---

## 3. Feature-Centric View: Cross-Subtype Consistency

Features that show a significant spatial gradient ($R^2 > 0.02$) in the most subtypes, indicating a robust spatial trend that persists across cell types.

| Feature | # Subtypes with sig. gradient | Gradient direction std (deg) | Avg gradient mag |
|---------|-------------------------------|------------------------------|------------------|
| angle_correction_applied | 30 / 30 | 13.2 | 0.071 |
| freq_sinefit_0p5hz_r_squared | 22 / 30 | 139.4 | 0.000044 |
| gaussian_r2 | 22 / 30 | 44.6 | 0.000018 |
| freq_sinefit_1hz_r_squared | 20 / 30 | 138.3 | 0.000047 |
| dog_r2 | 20 / 30 | 40.6 | 0.000018 |
| chip_effective_area | 19 / 30 | 59.9 | 0.191 |
| gaussian_amp | 19 / 30 | 41.3 | 0.00070 |
| freq_sinefit_2hz_r_squared | 18 / 30 | 136.0 | 0.000044 |
| freq_sinefit_0p5hz_amplitude | 18 / 30 | 127.8 | 0.0046 |
| freq_sinefit_1hz_amplitude | 17 / 30 | 125.1 | 0.0048 |
| gaussian_sigma_x | 17 / 30 | 54.7 | 0.00038 |
| gaussian_sigma_y | 17 / 30 | 51.2 | 0.00036 |
| green_off_peak_extreme | 16 / 30 | 18.0 | 0.013 |
| freq_sinefit_4hz_r_squared | 16 / 30 | 96.4 | 0.000040 |
| on_sustained | 16 / 30 | 126.1 | 0.0061 |

**Interpretation**:

- **angle_correction_applied**: Present in all 30 subtypes with very consistent direction (std = 13.2 deg). This reflects the recording geometry (dorsal-ventral / nasal-temporal axis), not a biological feature.
- **gaussian_r2, dog_r2**: Goodness-of-fit of receptive field models shows a consistent spatial gradient (direction std ~40-45 deg) across 20-22 subtypes. This likely reflects variation in recording quality or electrode density across the retina.
- **chip_effective_area**: Significant in 19 subtypes. Reflects the recording array coverage at different retinal positions.
- **gaussian_amp, gaussian_sigma_x/y**: Receptive field parameters show spatial gradients in 17-19 subtypes with moderate directional consistency (~41-55 deg std). This may reflect genuine eccentricity-dependent RF scaling.
- **green_off_peak_extreme**: Significant in 16 subtypes with the tightest non-geometric directional consistency (std = 18.0 deg), suggesting a genuine and reproducible OFF-response spatial gradient.
- **freq_sinefit amplitudes/R^2**: Frequency response features are significant in 16-22 subtypes but with high directional scatter (>96 deg std), suggesting subtype-specific response patterns rather than a universal spatial trend.

---

## 4. Strongest Radial Trends per Subtype

The feature most correlated with distance from the retinal centre for each subtype. Negative radial r means the feature decreases from centre to periphery; positive means it increases.

| Subtype | Top radial feature | Radial r | p-value | C/P ratio |
|---------|--------------------|----------|---------|-----------|
| DSGC_0  | freq_sinefit_0p5hz_amplitude | +0.181 | 1.2e-02 | 0.80 |
| DSGC_2  | angle_correction_applied | -0.236 | 1.3e-09 | -0.64 |
| DSGC_3  | angle_correction_applied | -0.164 | 1.3e-04 | -0.22 |
| DSGC_4  | angle_correction_applied | -0.260 | 1.2e-05 | -0.19 |
| DSGC_5  | angle_correction_applied | -0.218 | 8.7e-04 | 0.06 |
| DSGC_6  | angle_correction_applied | -0.245 | 1.6e-05 | 0.31 |
| DSGC_7  | green_off_peak_extreme | -0.227 | 3.5e-03 | 1.84 |
| DSGC_8  | chip_effective_area | +0.231 | 1.1e-04 | 0.90 |
| OSGC_0  | angle_correction_applied | -0.235 | 3.5e-06 | -1.24 |
| OSGC_1  | freq_sinefit_10hz_r_squared | +0.234 | 6.5e-04 | 0.73 |
| OSGC_2  | freq_sinefit_4hz_phase_deg | +0.203 | 1.3e-02 | 1.34 |
| OSGC_3  | lnl_b | +0.226 | 6.4e-05 | 0.88 |
| OSGC_4  | freq_sinefit_2hz_phase_deg | -0.428 | 9.2e-03 | 0.33 |
| OSGC_5  | freq_sinefit_10hz_phase_deg | -0.177 | 1.5e-02 | 2.04 |
| OSGC_6  | time_to_green_on_peak | +0.347 | 1.1e-03 | 0.84 |
| Other_0 | gb_base_mean | +0.127 | 5.7e-03 | 0.62 |
| Other_1 | lnl_a | +0.113 | 5.4e-06 | 0.92 |
| Other_2 | chip_effective_area | +0.142 | 8.8e-06 | 0.94 |
| Other_3 | lnl_null_log_likelihood | +0.151 | 4.8e-05 | 0.92 |
| Other_4 | freq_sinefit_2hz_phase_deg | -0.369 | 4.1e-05 | -1.43 |
| Other_5 | base_std | +0.169 | 1.2e-03 | 0.80 |
| Other_6 | time_to_on_peak_extreme | +0.195 | 1.6e-03 | 0.82 |
| Other_7 | angle_correction_applied | -0.177 | 1.2e-04 | 0.26 |
| ipRGC_2 | angle_correction_applied | -0.235 | 2.1e-04 | -1.39 |
| ipRGC_3 | angle_correction_applied | -0.235 | 5.1e-04 | -0.17 |
| ipRGC_4 | chip_effective_area | +0.313 | 2.4e-03 | 0.88 |
| ipRGC_5 | angle_correction_applied | -0.276 | 8.0e-05 | -1.01 |
| ipRGC_6 | angle_correction_applied | -0.229 | 1.5e-03 | 0.10 |
| ipRGC_7 | freq_sinefit_10hz_phase_deg | -0.613 | 1.9e-03 | -0.12 |
| ipRGC_8 | freq_sinefit_2hz_amplitude | +0.258 | 6.2e-04 | 0.90 |

**Highlights**:

- **angle_correction_applied** is the top radial feature for 12 of 30 subtypes (always negative r, meaning the angle correction decreases with eccentricity). This is a geometric artefact.
- **Excluding geometric features**, notable biological radial trends include:
  - **OSGC_4**: `freq_sinefit_2hz_phase_deg` (r = -0.43), suggesting phase response shifts with eccentricity.
  - **ipRGC_7**: `freq_sinefit_10hz_phase_deg` (r = -0.61), the strongest radial correlation overall (though n = 89).
  - **Other_4**: `freq_sinefit_2hz_phase_deg` (r = -0.37).
  - **OSGC_6**: `time_to_green_on_peak` (r = +0.35), green ON response latency increases with eccentricity.

---

## 5. Dominant Quadrant Analysis

Distribution of which quadrant (NE/NW/SE/SW) holds the highest mean value, tallied across all 70 features per subtype.

| Subtype | NE | NW | SE | SW | Most frequent |
|---------|----|----|----|----|--------------|
| DSGC_0  | 14 | 23 | 19 | 14 | NW |
| DSGC_2  | 21 | 6  | 26 | 17 | SE |
| DSGC_3  | 8  | 22 | 28 | 12 | SE |
| DSGC_4  | 22 | 6  | 17 | 25 | SW |
| DSGC_5  | 23 | 8  | 26 | 13 | SE |
| DSGC_6  | 15 | 25 | 15 | 15 | NW |
| DSGC_7  | 8  | 13 | 33 | 16 | SE |
| DSGC_8  | 9  | 18 | 23 | 20 | SE |
| OSGC_0  | 19 | 24 | 15 | 12 | NW |
| OSGC_1  | 19 | 18 | 18 | 15 | NE |
| OSGC_2  | 19 | 27 | 21 | 3  | NW |
| OSGC_3  | 14 | 5  | 20 | 31 | SW |
| OSGC_4  | 16 | 28 | 17 | 9  | NW |
| OSGC_5  | 31 | 15 | 17 | 7  | NE |
| OSGC_6  | 19 | 28 | 14 | 9  | NW |
| Other_0 | 21 | 12 | 19 | 18 | NE |
| Other_1 | 9  | 21 | 24 | 16 | SE |
| Other_2 | 11 | 25 | 17 | 17 | NW |
| Other_3 | 11 | 21 | 28 | 10 | SE |
| Other_4 | 14 | 20 | 23 | 13 | SE |
| Other_5 | 8  | 21 | 31 | 10 | SE |
| Other_6 | 12 | 23 | 21 | 14 | NW |
| Other_7 | 13 | 14 | 22 | 21 | SE |
| ipRGC_2 | 13 | 15 | 18 | 24 | SW |
| ipRGC_3 | 11 | 22 | 13 | 24 | SW |
| ipRGC_4 | 8  | 17 | 27 | 18 | SE |
| ipRGC_5 | 27 | 9  | 15 | 19 | NE |
| ipRGC_6 | 25 | 15 | 17 | 13 | NE |
| ipRGC_7 | 19 | 10 | 18 | 23 | SW |
| ipRGC_8 | 23 | 11 | 23 | 13 | NE |

**Patterns by group**:

- **DSGC**: Predominantly SE-biased (DSGC_2, 3, 5, 7, 8), with exceptions in NW (DSGC_0, 6) and SW (DSGC_4).
- **OSGC**: Mixed -- NW-biased (OSGC_0, 2, 4, 6), NE (OSGC_1, 5), and SW (OSGC_3).
- **Other**: SE-biased trend (Other_1, 3, 4, 5, 7), with NW (Other_2, 6) and NE (Other_0).
- **ipRGC**: Diverse -- SW (ipRGC_2, 3, 7), NE (ipRGC_5, 6, 8), SE (ipRGC_4).

---

## 6. DSGC Subtypes -- Detailed Comparison

| Metric | DSGC_0 | DSGC_2 | DSGC_3 | DSGC_4 | DSGC_5 | DSGC_6 | DSGC_7 | DSGC_8 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| n cells | 194 | 645 | 535 | 277 | 229 | 303 | 163 | 273 |
| Avg gradient mag | 0.042 | 0.020 | 0.007 | 0.047 | 0.014 | 0.037 | 0.021 | 0.021 |
| Avg gradient R^2 | 0.043 | 0.014 | 0.015 | 0.024 | 0.012 | 0.029 | 0.028 | 0.025 |
| Avg hexbin CV | 5.09 | 2.04 | 1.04 | 1.13 | 1.20 | 1.15 | 1.41 | 0.87 |
| Avg Moran's I | 0.052 | 0.036 | 0.046 | 0.046 | 0.003 | 0.045 | 0.013 | 0.060 |
| Dominant quad | NW | SE | SE | SW | SE | NW | SE | SE |

- **DSGC_0**: Highest hexbin CV among DSGCs (5.09), meaning its features are the most spatially heterogeneous. Dominant quadrant NW.
- **DSGC_8**: Highest Moran's I (0.060), strongest spatial autocorrelation. Lowest hexbin CV (0.87), suggesting smooth gradients rather than patchy distributions.
- **DSGC_3**: Weakest gradient (0.007) and low hexbin CV (1.04), the most spatially uniform DSGC subtype.
- **DSGC_5**: Near-zero Moran's I (0.003), features are essentially randomly distributed spatially.

---

## 7. ipRGC Subtypes -- Detailed Comparison

| Metric | ipRGC_2 | ipRGC_3 | ipRGC_4 | ipRGC_5 | ipRGC_6 | ipRGC_7 | ipRGC_8 |
|--------|---------|---------|---------|---------|---------|---------|---------|
| n cells | 244 | 266 | 92 | 199 | 189 | 89 | 173 |
| Avg gradient mag | 0.012 | 0.011 | 0.118 | 0.026 | 0.050 | 0.087 | 0.032 |
| Avg gradient R^2 | 0.018 | 0.019 | 0.031 | 0.015 | 0.029 | 0.059 | 0.019 |
| Avg hexbin CV | 2.00 | 1.13 | 1.06 | 1.32 | 2.06 | 1.35 | 1.57 |
| Avg Moran's I | 0.021 | 0.034 | 0.009 | 0.002 | 0.025 | 0.074 | 0.027 |
| Dominant quad | SW | SW | SE | NE | NE | SW | NE |

- **ipRGC_7**: Despite small sample size (n=89), shows the second-highest Moran's I across all subtypes (0.074) and strong gradients ($R^2$=0.059). Its strongest radial feature is `freq_sinefit_10hz_phase_deg` ($r=-0.61$), the strongest single radial correlation in the entire dataset.
- **ipRGC_4**: Elevated gradient magnitude (0.118) driven by small sample size.
- **ipRGC_5**: Nearly zero Moran's I (0.002), spatially random feature distributions.

---

## 8. Key Takeaways

1. **Geometric confound**: `angle_correction_applied` and `chip_effective_area` are the two most spatially structured features across nearly all subtypes. These reflect recording geometry (retinal orientation and electrode array coverage), not cell biology. Analyses of biological spatial patterns should exclude or control for these.

2. **Receptive field scaling**: `gaussian_r2`, `dog_r2`, `gaussian_amp`, `gaussian_sigma_x/y` show consistent spatial gradients in 17-22 subtypes. This may reflect genuine eccentricity-dependent receptive field scaling, a known property of retinal ganglion cells.

3. **Frequency response spatial patterns**: Sine-fit amplitude and R^2 features (0.5-10 Hz) are significant in 16-22 subtypes but with highly variable gradient directions, suggesting cell-type-specific spatial modulation rather than a universal eccentricity effect. Phase features show stronger subtype-specificity.

4. **Subtype-specific spatial structure**: OSGC_3, ipRGC_7, Other_2, and Other_3 show the strongest spatial clustering (Moran's I > 0.06), while DSGC_5, ipRGC_5, and OSGC_4 show essentially random spatial distributions.

5. **Quadrant biases**: DSGC subtypes are predominantly SE-biased, while ipRGC subtypes show more diverse quadrant preferences. This may reflect sampling bias (more recordings from certain retinal regions) or genuine subtype-specific topographic organisation.

6. **Sample size caveat**: Subtypes with the strongest apparent gradients (OSGC_4 n=59, OSGC_6 n=86, ipRGC_7 n=89) all have small populations. These results should be interpreted cautiously and validated with larger samples.

---

## Output Files

- `hexbin_data_all_cells.parquet` (86,160 rows) -- bin centres, raw means, GAM predictions for all cells
- `hexbin_data_per_subtype.parquet` (250,760 rows) -- bin centres, raw means per subtype
- `spatial_metrics.parquet` (2,170 rows x 27 cols) -- quantification metrics (70 all-cells + 2,100 per-subtype)
- `figures_legacy/all_cells/` -- 70 hexbin heatmaps (Raw + GAM) per feature
- `figures_legacy/per_subtype/` -- 70 hexbin heatmaps per feature, 30 subtypes each
