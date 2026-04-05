# Spatial Quantification: Blocker Comparison


## Before -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.013102 | -14.1 | 0.0694 | 0.1095 | 0.0506 |
| on_peak_extreme | 0.006689 | 137.0 | 0.0121 | 0.0578 | 0.0109 |
| on_sustained | 0.005023 | 174.6 | 0.0674 | 0.1011 | 0.0617 |
| gb_base_mean_high | 0.001610 | 42.6 | 0.0024 | -0.0004 | 0.0217 |
| gb_base_mean | 0.001407 | 65.5 | 0.0055 | 0.0091 | 0.0271 |
| off_sustained | 0.000408 | -6.5 | 0.0279 | 0.0670 | 0.0885 |
| on_trans_sus_ratio | 0.000209 | 177.1 | 0.0957 | 0.1286 | 0.0997 |
| green_blue_off_ratio_high | 0.000077 | 10.5 | 0.0559 | 0.0500 | -0.0680 |
| green_blue_off_ratio | 0.000072 | -0.2 | 0.0516 | 0.0348 | -0.0790 |
| on_off_sus_ratio | 0.000044 | -138.0 | 0.0046 | -0.0054 | -0.0139 |


## After -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.011869 | -48.3 | 0.0858 | 0.1583 | -0.0158 |
| on_peak_extreme | 0.007243 | -43.0 | 0.0152 | 0.0477 | -0.0599 |
| on_sustained | 0.003791 | -143.4 | 0.0385 | 0.0338 | 0.0441 |
| off_sustained | 0.000784 | 105.0 | 0.0595 | 0.1138 | 0.0880 |
| gb_base_mean_high | 0.000515 | -93.8 | 0.0016 | -0.0061 | -0.0165 |
| gb_base_mean | 0.000347 | -113.8 | 0.0007 | -0.0036 | 0.0069 |
| off_trans_sus_ratio | 0.000067 | 115.5 | 0.0087 | 0.0381 | 0.0454 |
| on_trans_sus_ratio | 0.000058 | 160.6 | 0.0086 | 0.0370 | 0.0399 |
| on_off_ratio | 0.000050 | -36.2 | 0.0104 | 0.0546 | -0.0481 |
| green_blue_off_ratio | 0.000045 | 27.1 | 0.0132 | 0.0140 | -0.0357 |


## Delta -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| on_peak_extreme | 0.013932 | -43.0 | 0.0482 | 0.1000 | -0.0647 |
| off_peak_extreme | 0.007424 | -130.3 | 0.0255 | 0.0763 | -0.0684 |
| on_sustained | 0.003359 | -54.4 | 0.0245 | 0.0357 | -0.0150 |
| gb_base_mean_high | 0.002014 | -127.2 | 0.0034 | 0.0049 | -0.0264 |
| gb_base_mean | 0.001753 | -114.4 | 0.0070 | 0.0265 | -0.0208 |
| off_sustained | 0.001008 | 127.1 | 0.0699 | 0.1016 | 0.0154 |
| on_trans_sus_ratio | 0.000154 | 3.3 | 0.0305 | 0.0604 | -0.0478 |
| off_trans_sus_ratio | 0.000071 | 105.6 | 0.0054 | 0.0193 | -0.0229 |
| green_blue_off_ratio_high | 0.000058 | -163.5 | 0.0144 | 0.0261 | 0.0512 |
| on_off_sus_ratio | 0.000055 | 60.5 | 0.0038 | 0.0084 | 0.0048 |


## Before vs After Comparison

### Features with largest change in spatial structure

| Feature | Before R2 | After R2 | Delta R2 | Before Moran | After Moran | Delta Moran |
|---------|-----------|----------|----------|--------------|-------------|-------------|
| on_trans_sus_ratio | 0.0957 | 0.0086 | -0.0871 | 0.1286 | 0.0370 | -0.0916 |
| osi | 0.0941 | 0.0229 | -0.0712 | 0.1152 | 0.0303 | -0.0849 |
| green_blue_off_ratio_high | 0.0559 | 0.0028 | -0.0531 | 0.0500 | -0.0162 | -0.0662 |
| green_blue_off_ratio | 0.0516 | 0.0132 | -0.0385 | 0.0348 | 0.0140 | -0.0208 |
| off_sustained | 0.0279 | 0.0595 | 0.0316 | 0.0670 | 0.1138 | 0.0468 |
| on_sustained | 0.0674 | 0.0385 | -0.0288 | 0.1011 | 0.0338 | -0.0673 |
| step_up_QI | 0.0232 | 0.0033 | -0.0198 | 0.0655 | 0.0936 | 0.0281 |
| off_peak_extreme | 0.0694 | 0.0858 | 0.0164 | 0.1095 | 0.1583 | 0.0488 |
| on_off_ratio | 0.0016 | 0.0104 | 0.0088 | 0.0179 | 0.0546 | 0.0367 |
| off_trans_sus_ratio | 0.0004 | 0.0087 | 0.0083 | 0.0192 | 0.0381 | 0.0189 |
