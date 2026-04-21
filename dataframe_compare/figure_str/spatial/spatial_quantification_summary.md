# Spatial Quantification: Blocker Comparison


## Before -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.011995 | -8.4 | 0.0767 | 0.1129 | -0.0295 |
| on_peak_extreme | 0.009540 | 113.9 | 0.0304 | 0.0698 | -0.0070 |
| on_sustained | 0.004115 | 170.7 | 0.0576 | 0.0574 | 0.0544 |
| gb_base_mean | 0.001354 | 77.1 | 0.0073 | 0.0079 | 0.0026 |
| gb_base_mean_high | 0.001323 | 56.4 | 0.0024 | 0.0003 | 0.0079 |
| off_sustained | 0.000325 | -3.4 | 0.0244 | 0.0684 | 0.0874 |
| on_trans_sus_ratio | 0.000166 | 170.0 | 0.0789 | 0.0873 | 0.1269 |
| green_blue_off_ratio_high | 0.000084 | 17.4 | 0.0816 | 0.0571 | -0.1068 |
| green_blue_off_ratio | 0.000079 | 2.0 | 0.0720 | 0.0755 | -0.1299 |
| on_off_ratio | 0.000052 | 25.8 | 0.0175 | 0.0364 | -0.0779 |


## After -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.010884 | -34.2 | 0.0851 | 0.1550 | -0.0591 |
| gb_base_mean | 0.010059 | -146.8 | 0.0484 | 0.1655 | 0.0391 |
| on_peak_extreme | 0.007564 | 4.2 | 0.0193 | 0.0817 | -0.0669 |
| gb_base_mean_high | 0.003837 | -149.8 | 0.0394 | 0.1055 | 0.0261 |
| on_sustained | 0.002011 | -136.6 | 0.0152 | 0.0272 | 0.0294 |
| off_sustained | 0.000952 | -178.7 | 0.0406 | 0.1217 | 0.0252 |
| on_off_ratio | 0.000068 | -4.3 | 0.0237 | 0.0566 | -0.0884 |
| on_trans_sus_ratio | 0.000053 | 157.4 | 0.0087 | 0.0557 | 0.0233 |
| green_blue_off_ratio | 0.000050 | 27.5 | 0.0227 | 0.0152 | -0.0624 |
| off_trans_sus_ratio | 0.000050 | 130.1 | 0.0061 | 0.0308 | -0.0066 |


## Delta -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| on_peak_extreme | 0.014026 | -35.6 | 0.0551 | 0.0706 | -0.0550 |
| gb_base_mean | 0.011076 | -141.9 | 0.0515 | 0.1648 | 0.0338 |
| off_peak_extreme | 0.005206 | -123.3 | 0.0180 | 0.0766 | -0.0242 |
| gb_base_mean_high | 0.005057 | -143.2 | 0.0233 | 0.0356 | 0.0045 |
| on_sustained | 0.003308 | -38.3 | 0.0286 | 0.0272 | -0.0212 |
| off_sustained | 0.001276 | -179.9 | 0.0615 | 0.1281 | -0.0190 |
| on_trans_sus_ratio | 0.000115 | -4.2 | 0.0228 | 0.0357 | -0.0809 |
| green_blue_off_ratio_high | 0.000046 | -158.0 | 0.0128 | 0.0276 | 0.0882 |
| off_trans_sus_ratio | 0.000044 | 123.3 | 0.0027 | 0.0206 | -0.0590 |
| green_blue_off_ratio | 0.000040 | 149.6 | 0.0093 | -0.0161 | 0.0422 |


## Before vs After Comparison

### Features with largest change in spatial structure

| Feature | Before R2 | After R2 | Delta R2 | Before Moran | After Moran | Delta Moran |
|---------|-----------|----------|----------|--------------|-------------|-------------|
| on_trans_sus_ratio | 0.0789 | 0.0087 | -0.0702 | 0.0873 | 0.0557 | -0.0317 |
| green_blue_off_ratio_high | 0.0816 | 0.0123 | -0.0693 | 0.0571 | -0.0023 | -0.0594 |
| step_up_QI | 0.0602 | 0.0001 | -0.0601 | 0.1050 | 0.0913 | -0.0137 |
| green_blue_off_ratio | 0.0720 | 0.0227 | -0.0493 | 0.0755 | 0.0152 | -0.0603 |
| osi | 0.0860 | 0.0422 | -0.0438 | 0.1484 | 0.0456 | -0.1029 |
| on_sustained | 0.0576 | 0.0152 | -0.0424 | 0.0574 | 0.0272 | -0.0302 |
| gb_base_mean | 0.0073 | 0.0484 | 0.0412 | 0.0079 | 0.1655 | 0.1576 |
| gb_base_mean_high | 0.0024 | 0.0394 | 0.0370 | 0.0003 | 0.1055 | 0.1053 |
| green_blue_on_ratio_high | 0.0003 | 0.0171 | 0.0168 | 0.0069 | 0.0207 | 0.0139 |
| off_sustained | 0.0244 | 0.0406 | 0.0162 | 0.0684 | 0.1217 | 0.0532 |
