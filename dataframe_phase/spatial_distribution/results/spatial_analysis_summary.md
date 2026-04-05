# Spatial Distribution Analysis Summary

**Input**: `labeled_dataframe_with_legacy_coords_freq.parquet`  
**Coordinate system**: legacy_transformed_x/y (electrode units x16 = microns)  
**Spatial filter**: |coord| < 100 electrode units  
**Cells after filter**: 10801  
**Features analysed**: 70  
**Valid subtypes**: 30  

## Strongest Spatial Gradients (all cells)

Features with the largest linear gradient magnitude (feature ~ x + y).

| Feature | Gradient Mag | Direction (deg) | R^2 |
|---------|-------------|-----------------|-----|
| chip_effective_area | 0.153593 | 137.5 | 0.0211 |
| lnl_log_likelihood | 0.149575 | 99.3 | 0.0001 |
| lnl_null_log_likelihood | 0.117548 | 99.3 | 0.0001 |
| lnl_n_spikes | 0.083967 | 82.0 | 0.0002 |
| angle_correction_applied | 0.068645 | -93.2 | 0.1786 |
| freq_sinefit_2hz_phase_deg | 0.013075 | 4.6 | 0.0130 |
| green_off_peak_extreme | 0.012138 | -20.4 | 0.0213 |
| off_peak_extreme | 0.012012 | -22.6 | 0.0186 |
| blue_off_peak_extreme | 0.011550 | -18.6 | 0.0206 |
| freq_sinefit_1hz_phase_deg | 0.010384 | 111.3 | 0.0066 |
| freq_sinefit_4hz_phase_deg | 0.009305 | 14.5 | 0.0058 |
| freq_sinefit_0p5hz_phase_deg | 0.007963 | 140.2 | 0.0038 |
| dog_amp_inh | 0.005830 | -123.9 | 0.0003 |
| freq_sinefit_10hz_phase_deg | 0.005127 | -40.7 | 0.0032 |
| dog_amp_exc | 0.005059 | -121.8 | 0.0002 |

## Most Spatially Uneven Features (all cells)

Ranked by coefficient of variation (CV) of hexbin means.

| Feature | Hexbin CV | Gini | Moran's I |
|---------|----------|------|-----------|
| freq_sinefit_1hz_phase_deg | 135.8721 | 73.7078 | 0.0570 |
| off_sustained | 8.9196 | 4.6573 | 0.0467 |
| angle_correction_applied | 4.9303 | -2.7924 | 0.6864 |
| on_off_sus_ratio | 4.1627 | -2.3275 | 0.0392 |
| freq_sinefit_2hz_phase_deg | 3.6472 | 2.0237 | 0.0520 |
| off_trans_sus_ratio | 2.7786 | 1.5532 | 0.0553 |
| freq_sinefit_0p5hz_phase_deg | 2.4680 | -1.3399 | 0.0466 |
| dog_amp_inh | 2.3185 | 0.6966 | 0.0228 |
| dog_amp_exc | 2.0638 | 0.6124 | 0.0216 |
| on_sustained | 1.5923 | 0.8833 | 0.0183 |
| freq_sinefit_10hz_phase_deg | 1.5107 | 0.8378 | 0.0263 |
| lnl_bits_per_spike | 1.4706 | 0.4416 | -0.0061 |
| iprgc_2hz_QI | 1.1864 | 0.6031 | 0.0846 |
| freq_sinefit_4hz_phase_deg | 1.0390 | -0.5809 | 0.0644 |
| on_trans_sus_ratio | 0.9589 | 0.5315 | 0.0675 |

## Strongest Radial Trends (all cells)

Features most correlated with distance from retinal centre.

| Feature | Radial r | p-value | Slope (per um) | Centre mean | Periphery mean | C/P ratio |
|---------|---------|---------|----------------|-------------|----------------|-----------|
| angle_correction_applied | -0.1420 | 8.72e-50 | -0.034492 | 3.17 | -26.48 | -0.1198 |
| chip_effective_area | 0.1234 | 6.78e-38 | 0.174048 | 2954.21 | 3117.31 | 0.9477 |
| lnl_b | 0.0768 | 1.37e-15 | 0.000280 | 1.19 | 1.47 | 0.8069 |
| lnl_r_squared | 0.0743 | 1.14e-14 | 0.000011 | 0.28 | 0.29 | 0.9659 |
| freq_sinefit_4hz_amplitude | 0.0638 | 3.29e-11 | 0.003348 | 40.25 | 42.45 | 0.9481 |
| freq_sinefit_2hz_amplitude | 0.0621 | 1.08e-10 | 0.002856 | 33.45 | 35.39 | 0.9452 |
| freq_sinefit_1hz_amplitude | 0.0614 | 1.68e-10 | 0.002729 | 27.83 | 29.79 | 0.9340 |
| lnl_nonlinearity_index | -0.0603 | 3.61e-10 | -0.000012 | 0.15 | 0.14 | 1.0919 |
| freq_sinefit_4hz_r_squared | 0.0585 | 1.16e-09 | 0.000021 | 0.38 | 0.41 | 0.9470 |
| lnl_a | 0.0575 | 2.29e-09 | 0.000000 | 0.00 | 0.00 | 0.9784 |
| lnl_n_spikes | 0.0570 | 3.11e-09 | 0.481972 | 6044.69 | 6389.45 | 0.9460 |
| freq_sinefit_2hz_r_squared | 0.0555 | 7.84e-09 | 0.000019 | 0.32 | 0.34 | 0.9397 |
| freq_sinefit_0p5hz_amplitude | 0.0547 | 1.28e-08 | 0.002243 | 21.10 | 22.89 | 0.9216 |
| freq_sinefit_0p5hz_r_squared | 0.0543 | 1.68e-08 | 0.000019 | 0.23 | 0.24 | 0.9269 |
| freq_sinefit_1hz_r_squared | 0.0527 | 4.16e-08 | 0.000019 | 0.27 | 0.29 | 0.9379 |

## Spatial Autocorrelation (Moran's I, all cells)

Positive Moran's I indicates clustering; values near 0 indicate random distribution.

| Feature | Moran's I | Hexbin CV | Gradient R^2 |
|---------|----------|----------|-------------|
| angle_correction_applied | 0.6864 | 4.9303 | 0.1786 |
| chip_effective_area | 0.3041 | 0.1444 | 0.0211 |
| freq_sinefit_10hz_r_squared | 0.2364 | 0.3982 | 0.0002 |
| dog_sigma_exc | 0.2352 | 0.2710 | 0.0167 |
| freq_sinefit_10hz_amplitude | 0.2164 | 0.5057 | 0.0016 |
| gaussian_r2 | 0.2123 | 0.0598 | 0.0293 |
| gaussian_sigma_x | 0.2078 | 0.3411 | 0.0235 |
| gaussian_sigma_y | 0.2061 | 0.3398 | 0.0226 |
| dog_r2 | 0.1886 | 0.0595 | 0.0228 |
| gaussian_amp | 0.1830 | 0.3433 | 0.0240 |
| dog_sigma_inh | 0.1733 | 0.3914 | 0.0108 |
| green_off_peak_extreme | 0.1492 | 0.5788 | 0.0213 |
| blue_off_peak_extreme | 0.1412 | 0.5850 | 0.0206 |
| off_peak_extreme | 0.1158 | 0.5861 | 0.0186 |
| freq_sinefit_1hz_r_squared | 0.1134 | 0.3100 | 0.0137 |

## Dominant Quadrant Summary (all cells)

Quadrant with the highest mean value for each feature.

| Quadrant | # Features |
|----------|-----------|
| SE | 22 |
| NE | 19 |
| NW | 18 |
| SW | 11 |

## Per-Subtype Highlights

### Subtypes with strongest average gradient

| Subtype | Avg Gradient Mag |
|---------|-----------------|
| OSGC_4 | 0.300339 |
| OSGC_6 | 0.227537 |
| Other_4 | 0.196324 |
| ipRGC_4 | 0.118359 |
| OSGC_1 | 0.087408 |

### Subtypes with strongest spatial clustering (avg Moran's I)

| Subtype | Avg Moran's I |
|---------|--------------|
| OSGC_3 | 0.0793 |
| ipRGC_7 | 0.0735 |
| Other_2 | 0.0656 |
| Other_3 | 0.0624 |
| DSGC_8 | 0.0601 |

## Output Files

- `hexbin_data_all_cells.parquet` -- hexbin bin centres, raw means, GAM predictions (all cells)
- `hexbin_data_per_subtype.parquet` -- hexbin bin centres, raw means (per subtype)
- `spatial_metrics.parquet` -- quantification metrics (all cells + per subtype)
- `figures_legacy/all_cells/` -- hexbin heatmaps (Raw + GAM) per feature
- `figures_legacy/per_subtype/` -- hexbin heatmaps per feature per subtype
