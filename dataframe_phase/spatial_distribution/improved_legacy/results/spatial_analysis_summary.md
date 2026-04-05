# Spatial Distribution Analysis (Improved Coordinates v6)

**Input**: `labeled_dataframe_improved_coords.parquet`  
**ONH method**: Robust (R^2>0.7, median+MAD), legacy transform  
**Coordinates**: improved_tx / improved_ty (electrode units x16 = microns)  
**Spatial filter**: |coord| < 100 electrode units  
**Cells after filter**: 15645  
**Features analysed**: 70  
**Valid subtypes**: 30  
**Color scale**: per-subplot (mean +/- 50% of |mean|)  

## Strongest Spatial Gradients (all cells)

| Feature | Gradient Mag | Direction (deg) | R^2 |
|---------|-------------|-----------------|-----|
| lnl_null_log_likelihood | 0.471300 | -90.2 | 0.0007 |
| lnl_log_likelihood | 0.443944 | -88.8 | 0.0006 |
| chip_effective_area | 0.170996 | 166.1 | 0.0156 |
| lnl_n_spikes | 0.127670 | -74.9 | 0.0003 |
| angle_correction_applied | 0.092256 | -103.1 | 0.1603 |
| freq_sinefit_1hz_phase_deg | 0.016128 | 112.1 | 0.0086 |
| off_peak_extreme | 0.015198 | -28.3 | 0.0175 |
| green_off_peak_extreme | 0.014981 | -27.6 | 0.0187 |
| freq_sinefit_2hz_phase_deg | 0.014817 | 13.3 | 0.0082 |
| blue_off_peak_extreme | 0.014128 | -25.2 | 0.0176 |
| freq_sinefit_4hz_phase_deg | 0.012383 | 17.4 | 0.0049 |
| dog_amp_inh | 0.010550 | -129.5 | 0.0004 |
| freq_sinefit_0p5hz_phase_deg | 0.009903 | 135.2 | 0.0033 |
| dog_amp_exc | 0.009408 | -131.2 | 0.0003 |
| on_peak_extreme | 0.006950 | -138.3 | 0.0021 |

## Most Spatially Uneven (all cells)

| Feature | Hexbin CV | Moran's I |
|---------|----------|-----------|
| freq_sinefit_1hz_phase_deg | 14.7033 | 0.0620 |
| off_sustained | 11.9911 | 0.0136 |
| on_off_sus_ratio | 4.2046 | 0.0515 |
| angle_correction_applied | 3.6849 | 0.7991 |
| dog_amp_inh | 2.7890 | 0.0223 |
| off_trans_sus_ratio | 2.7635 | 0.0556 |
| freq_sinefit_2hz_phase_deg | 2.5894 | 0.0449 |
| dog_amp_exc | 2.4747 | 0.0218 |
| freq_sinefit_0p5hz_phase_deg | 1.7939 | 0.0070 |
| on_sustained | 1.4283 | 0.0703 |
| freq_sinefit_10hz_phase_deg | 1.0904 | -0.0003 |
| iprgc_2hz_QI | 0.9135 | 0.1080 |
| freq_sinefit_4hz_phase_deg | 0.9015 | 0.1029 |
| lnl_bits_per_spike | 0.7932 | 0.0318 |
| on_trans_sus_ratio | 0.7852 | 0.1005 |

## Strongest Radial Trends (all cells)

| Feature | Radial r | p-value |
|---------|---------|---------|
| dog_sigma_exc | -0.0736 | 3.06e-20 |
| dsi | 0.0621 | 7.68e-15 |
| base_std | 0.0604 | 3.86e-14 |
| chip_effective_area | 0.0543 | 1.11e-11 |
| iprgc_2hz_QI | -0.0493 | 7.26e-10 |
| dog_sigma_inh | -0.0446 | 2.35e-08 |
| freq_sinefit_2hz_r_squared | 0.0438 | 4.39e-08 |
| lnl_b | 0.0414 | 2.29e-07 |
| iprgc_20hz_QI | -0.0402 | 5.33e-07 |
| ds_p_value | -0.0391 | 1.03e-06 |
| freq_sinefit_1hz_r_squared | 0.0387 | 1.29e-06 |
| freq_sinefit_0p5hz_r_squared | 0.0372 | 3.35e-06 |
| base_mean | 0.0367 | 4.33e-06 |
| freq_sinefit_4hz_r_squared | 0.0352 | 1.05e-05 |
| gb_base_std | 0.0345 | 1.60e-05 |

## Strongest Spatial Clustering (Moran's I, all cells)

| Feature | Moran's I | Hexbin CV | Gradient R^2 |
|---------|----------|----------|-------------|
| angle_correction_applied | 0.7991 | 3.6849 | 0.1603 |
| dog_sigma_exc | 0.3525 | 0.2044 | 0.0153 |
| chip_effective_area | 0.3288 | 0.1033 | 0.0156 |
| gaussian_sigma_y | 0.2904 | 0.2524 | 0.0231 |
| gaussian_sigma_x | 0.2834 | 0.2513 | 0.0226 |
| freq_sinefit_10hz_amplitude | 0.2640 | 0.4014 | 0.0087 |
| gaussian_r2 | 0.2239 | 0.0509 | 0.0204 |
| freq_sinefit_10hz_r_squared | 0.2164 | 0.3029 | 0.0048 |
| freq_sinefit_1hz_r_squared | 0.1931 | 0.2347 | 0.0129 |
| green_off_peak_extreme | 0.1893 | 0.4669 | 0.0187 |
| freq_sinefit_0p5hz_r_squared | 0.1840 | 0.2755 | 0.0123 |
| gaussian_amp | 0.1837 | 0.2927 | 0.0226 |
| freq_sinefit_2hz_r_squared | 0.1832 | 0.2079 | 0.0139 |
| off_peak_extreme | 0.1772 | 0.4713 | 0.0175 |
| dog_r2 | 0.1760 | 0.0532 | 0.0152 |

## Per-Subtype Highlights

### Subtypes with strongest average gradient

| Subtype | Avg Gradient Mag |
|---------|-----------------|
| OSGC_4 | 0.409196 |
| OSGC_6 | 0.292848 |
| Other_4 | 0.220996 |
| ipRGC_4 | 0.126285 |
| ipRGC_7 | 0.104364 |
| DSGC_4 | 0.096850 |
| Other_5 | 0.082529 |
| DSGC_6 | 0.071056 |
| OSGC_2 | 0.059833 |
| ipRGC_6 | 0.057894 |

### Subtypes with strongest spatial clustering

| Subtype | Avg Moran's I |
|---------|--------------|
| OSGC_3 | 0.1185 |
| DSGC_6 | 0.0995 |
| Other_4 | 0.0798 |
| OSGC_6 | 0.0770 |
| OSGC_0 | 0.0768 |
| Other_2 | 0.0741 |
| Other_3 | 0.0687 |
| Other_1 | 0.0595 |
| DSGC_0 | 0.0548 |
| ipRGC_7 | 0.0534 |

## All Features Summary (all cells)

| Feature | Mean | Std | n_bins | CV | Grad Mag | Grad Dir | Grad R^2 | Radial r | Moran I |
|---------|------|-----|--------|-----|----------|----------|----------|----------|---------|
| angle_correction_applied | -15.251 | 131.694 | 888 | 3.685 | 0.092256 | -103.1 | 0.1603 | -0.0157 | 0.7991 |
| base_mean | 15.806 | 20.859 | 888 | 0.507 | 0.000310 | -108.6 | 0.0001 | 0.0367 | 0.0740 |
| base_std | 4.638 | 3.965 | 888 | 0.342 | 0.000345 | -6.8 | 0.0024 | 0.0604 | 0.0919 |
| blue_off_peak_extreme | 49.923 | 58.987 | 888 | 0.475 | 0.014128 | -25.2 | 0.0176 | 0.0126 | 0.1642 |
| blue_on_peak_extreme | 87.786 | 82.374 | 888 | 0.351 | 0.006634 | -147.6 | 0.0021 | -0.0021 | 0.0176 |
| chip_effective_area | 3045.103 | 763.015 | 888 | 0.103 | 0.170996 | 166.1 | 0.0156 | 0.0543 | 0.3288 |
| dog_amp_exc | 59.016 | 301.627 | 888 | 2.475 | 0.009408 | -131.2 | 0.0003 | -0.0170 | 0.0218 |
| dog_amp_inh | 52.634 | 300.960 | 888 | 2.789 | 0.010550 | -129.5 | 0.0004 | -0.0174 | 0.0223 |
| dog_r2 | 0.763 | 0.092 | 888 | 0.053 | 0.000020 | 33.2 | 0.0152 | -0.0250 | 0.1760 |
| dog_sigma_exc | 2.645 | 1.439 | 888 | 0.204 | 0.000310 | -115.2 | 0.0153 | -0.0736 | 0.3525 |
| dog_sigma_inh | 2.431 | 1.971 | 888 | 0.301 | 0.000333 | -125.5 | 0.0094 | -0.0446 | 0.1691 |
| ds_p_value | 0.326 | 0.314 | 888 | 0.346 | 0.000016 | -160.0 | 0.0008 | -0.0391 | 0.0644 |
| dsi | 0.353 | 0.224 | 888 | 0.236 | 0.000011 | 137.6 | 0.0007 | 0.0621 | 0.1138 |
| freq_sinefit_0p5hz_amplitude | 21.881 | 21.607 | 888 | 0.321 | 0.002884 | 153.3 | 0.0055 | 0.0327 | 0.1037 |
| freq_sinefit_0p5hz_phase_deg | -20.451 | 94.666 | 854 | 1.794 | 0.009903 | 135.2 | 0.0033 | -0.0037 | 0.0070 |
| freq_sinefit_0p5hz_r_squared | 0.236 | 0.190 | 888 | 0.275 | 0.000038 | 152.8 | 0.0123 | 0.0372 | 0.1840 |
| freq_sinefit_10hz_amplitude | 21.675 | 21.675 | 888 | 0.401 | 0.003617 | -68.6 | 0.0087 | 0.0151 | 0.2640 |
| freq_sinefit_10hz_phase_deg | 26.402 | 67.644 | 850 | 1.090 | 0.005387 | -25.8 | 0.0020 | 0.0310 | -0.0003 |
| freq_sinefit_10hz_r_squared | 0.261 | 0.200 | 888 | 0.303 | 0.000025 | -67.8 | 0.0048 | 0.0293 | 0.2164 |
| freq_sinefit_1hz_amplitude | 28.715 | 23.471 | 888 | 0.262 | 0.002680 | 153.7 | 0.0040 | 0.0318 | 0.0833 |
| freq_sinefit_1hz_phase_deg | -0.668 | 97.203 | 874 | 14.703 | 0.016128 | 112.1 | 0.0086 | -0.0029 | 0.0620 |
| freq_sinefit_1hz_r_squared | 0.284 | 0.193 | 888 | 0.235 | 0.000040 | 151.5 | 0.0129 | 0.0387 | 0.1931 |
| freq_sinefit_2hz_amplitude | 34.307 | 24.414 | 888 | 0.238 | 0.001713 | 141.8 | 0.0015 | 0.0330 | 0.0651 |
| freq_sinefit_2hz_phase_deg | 12.673 | 92.548 | 879 | 2.589 | 0.014817 | 13.3 | 0.0082 | 0.0086 | 0.0449 |
| freq_sinefit_2hz_r_squared | 0.329 | 0.187 | 888 | 0.208 | 0.000040 | 148.1 | 0.0139 | 0.0438 | 0.1832 |
| freq_sinefit_4hz_amplitude | 41.236 | 28.004 | 888 | 0.225 | 0.001168 | 105.2 | 0.0005 | 0.0319 | 0.0691 |
| freq_sinefit_4hz_phase_deg | -50.198 | 100.740 | 880 | 0.902 | 0.012383 | 17.4 | 0.0049 | 0.0280 | 0.1029 |
| freq_sinefit_4hz_r_squared | 0.396 | 0.192 | 888 | 0.175 | 0.000035 | 136.3 | 0.0099 | 0.0352 | 0.1336 |
| gaussian_amp | 6.503 | 4.524 | 888 | 0.293 | 0.001183 | 63.4 | 0.0226 | 0.0208 | 0.1837 |
| gaussian_r2 | 0.760 | 0.089 | 888 | 0.051 | 0.000022 | 38.9 | 0.0204 | -0.0129 | 0.2239 |
| gaussian_sigma_x | 3.373 | 2.245 | 888 | 0.251 | 0.000587 | -114.4 | 0.0226 | -0.0300 | 0.2834 |
| gaussian_sigma_y | 3.370 | 2.248 | 888 | 0.252 | 0.000595 | -115.0 | 0.0231 | -0.0303 | 0.2904 |
| gb_base_mean | 15.997 | 21.910 | 888 | 0.512 | 0.000263 | 21.5 | 0.0000 | 0.0231 | 0.0615 |
| gb_base_std | 4.326 | 4.027 | 888 | 0.360 | 0.000481 | -21.1 | 0.0044 | 0.0345 | 0.0755 |
| green_blue_off_ratio | 0.527 | 0.503 | 888 | 0.332 | 0.000058 | -11.1 | 0.0042 | 0.0063 | 0.0891 |
| green_blue_on_ratio | 0.626 | 0.380 | 888 | 0.241 | 0.000037 | -103.8 | 0.0030 | -0.0288 | 0.0521 |
| green_off_peak_extreme | 51.388 | 60.569 | 888 | 0.467 | 0.014981 | -27.6 | 0.0187 | 0.0133 | 0.1893 |
| green_on_peak_extreme | 90.725 | 84.874 | 888 | 0.345 | 0.006676 | -141.8 | 0.0020 | -0.0035 | 0.0145 |
| iprgc_20hz_QI | 0.190 | 0.316 | 888 | 0.645 | 0.000038 | -33.0 | 0.0044 | -0.0402 | 0.0533 |
| iprgc_2hz_QI | 0.135 | 0.329 | 888 | 0.914 | 0.000050 | -54.8 | 0.0070 | -0.0493 | 0.1080 |
| lnl_a | 0.000 | 0.000 | 888 | 0.074 | 0.000000 | -50.7 | 0.0011 | 0.0242 | 0.0806 |
| lnl_a_norm | 0.756 | 0.284 | 888 | 0.106 | 0.000010 | 156.1 | 0.0004 | 0.0249 | 0.0137 |
| lnl_b | 1.343 | 1.867 | 888 | 0.396 | 0.000045 | 47.0 | 0.0002 | 0.0414 | 0.1141 |
| lnl_bits_per_spike | 0.548 | 1.781 | 888 | 0.793 | 0.000030 | -131.4 | 0.0001 | 0.0323 | 0.0318 |
| lnl_log_likelihood | 10302.199 | 10650.783 | 888 | 0.335 | 0.443944 | -88.8 | 0.0006 | 0.0142 | 0.0878 |
| lnl_n_frames | 10740.000 | 0.000 | 888 | 0.000 | 0.000000 | -85.4 | 0.0000 | nan | nan |
| lnl_n_spikes | 6129.091 | 4399.951 | 888 | 0.234 | 0.127670 | -74.9 | 0.0003 | 0.0176 | 0.0882 |
| lnl_nonlinearity_index | 0.142 | 0.103 | 888 | 0.245 | 0.000010 | -63.1 | 0.0029 | -0.0284 | 0.1204 |
| lnl_null_log_likelihood | 8566.764 | 10357.303 | 888 | 0.392 | 0.471300 | -90.2 | 0.0007 | 0.0130 | 0.0820 |
| lnl_r_squared | 0.286 | 0.083 | 888 | 0.103 | 0.000009 | 103.6 | 0.0037 | 0.0280 | 0.0925 |
| lnl_rectification_index | 0.811 | 0.095 | 888 | 0.039 | 0.000004 | 157.6 | 0.0006 | -0.0036 | 0.0421 |
| lnl_threshold_g | 0.188 | 0.119 | 888 | 0.187 | 0.000007 | -56.3 | 0.0010 | -0.0122 | 0.0397 |
| off_peak_extreme | 53.251 | 63.641 | 888 | 0.471 | 0.015198 | -28.3 | 0.0175 | 0.0050 | 0.1772 |
| off_sustained | 0.125 | 3.506 | 888 | 11.991 | 0.000187 | 18.8 | 0.0009 | -0.0063 | 0.0136 |
| off_trans_sus_ratio | 0.121 | 0.893 | 888 | 2.764 | 0.000050 | 21.7 | 0.0010 | -0.0078 | 0.0556 |
| on_off_ratio | 0.530 | 0.628 | 888 | 0.414 | 0.000015 | -49.2 | 0.0002 | -0.0151 | -0.0262 |
| on_off_sus_ratio | -0.095 | 0.965 | 888 | 4.205 | 0.000026 | -23.9 | 0.0002 | -0.0079 | 0.0515 |
| on_peak_extreme | 96.677 | 88.026 | 888 | 0.329 | 0.006950 | -138.3 | 0.0021 | 0.0013 | 0.0101 |
| on_sustained | 10.962 | 37.177 | 888 | 1.428 | 0.005481 | -161.9 | 0.0070 | 0.0052 | 0.0703 |
| on_trans_sus_ratio | 0.424 | 0.808 | 888 | 0.785 | 0.000147 | 172.0 | 0.0103 | 0.0011 | 0.1005 |
| os_p_value | 0.423 | 0.325 | 888 | 0.278 | 0.000038 | 65.3 | 0.0045 | 0.0072 | 0.0801 |
| osi | 0.057 | 0.079 | 888 | 0.434 | 0.000009 | -130.8 | 0.0042 | -0.0054 | 0.1429 |
| preferred_direction | 183.656 | 104.741 | 888 | 0.203 | 0.003094 | 150.6 | 0.0003 | -0.0070 | 0.0275 |
| step_up_QI | 0.811 | 0.129 | 888 | 0.055 | 0.000006 | 128.9 | 0.0007 | 0.0021 | 0.0586 |
| time_to_blue_off_peak | 12.854 | 11.610 | 872 | 0.323 | 0.001460 | 135.8 | 0.0048 | 0.0055 | 0.0904 |
| time_to_blue_on_peak | 11.628 | 9.113 | 864 | 0.328 | 0.001143 | -57.9 | 0.0048 | 0.0142 | 0.0703 |
| time_to_green_off_peak | 12.525 | 11.267 | 875 | 0.315 | 0.001329 | 118.9 | 0.0043 | 0.0017 | 0.1156 |
| time_to_green_on_peak | 11.253 | 9.194 | 865 | 0.339 | 0.001100 | -41.5 | 0.0043 | 0.0083 | 0.0589 |
| time_to_off_peak_extreme | 12.808 | 11.488 | 870 | 0.330 | 0.001266 | 124.6 | 0.0037 | 0.0018 | 0.0866 |
| time_to_on_peak_extreme | 10.159 | 8.598 | 863 | 0.337 | 0.001029 | -54.9 | 0.0044 | -0.0014 | 0.0598 |

## Output Files

- `hexbin_data_all_cells.parquet` (61938 rows) -- hexbin data, all cells
- `hexbin_data_per_subtype.parquet` (210556 rows) -- hexbin data, per subtype
- `spatial_metrics.parquet` (2170 rows) -- spatial metrics
- `figures_v2/all_cells/` -- 70 all-cells heatmaps (Raw + GAM)
- `figures_v2/per_subtype/` -- 70 per-subtype heatmaps (individual color scales)