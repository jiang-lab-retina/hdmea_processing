# Spatial Quantification: Blocker Comparison


## Before -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.010235 | -22.6 | 0.1118 | 0.1937 | -0.1036 |
| on_peak_extreme | 0.008665 | 24.9 | 0.0515 | 0.0489 | 0.0590 |
| gb_base_mean_high | 0.002556 | 129.9 | 0.0277 | 0.0983 | 0.0026 |
| on_sustained | 0.002140 | -129.7 | 0.0347 | 0.0658 | 0.1984 |
| gb_base_mean | 0.001579 | 118.9 | 0.0217 | 0.0406 | 0.0391 |
| on_trans_sus_ratio | 0.000120 | -178.2 | 0.0736 | 0.2200 | 0.2578 |
| off_sustained | 0.000088 | 96.1 | 0.0031 | 0.0553 | 0.0073 |
| green_blue_off_ratio | 0.000078 | -0.5 | 0.0940 | 0.1226 | -0.1564 |
| green_blue_off_ratio_high | 0.000058 | 2.7 | 0.0628 | 0.1261 | -0.1152 |
| off_trans_sus_ratio | 0.000051 | -151.0 | 0.0161 | 0.0454 | 0.0370 |


## After -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.013977 | -30.1 | 0.1485 | 0.2407 | -0.2764 |
| on_peak_extreme | 0.008861 | -18.9 | 0.0424 | 0.1225 | -0.2628 |
| on_sustained | 0.002378 | -157.8 | 0.0491 | 0.1480 | 0.0334 |
| gb_base_mean | 0.001803 | 63.9 | 0.0452 | 0.0872 | -0.0090 |
| gb_base_mean_high | 0.001592 | 60.6 | 0.0334 | 0.0913 | -0.0106 |
| off_sustained | 0.000155 | -55.0 | 0.0056 | 0.0867 | -0.1023 |
| on_off_ratio | 0.000062 | 4.2 | 0.0361 | 0.1129 | -0.1282 |
| green_blue_off_ratio_high | 0.000046 | 1.2 | 0.0336 | 0.0343 | -0.0666 |
| green_blue_off_ratio | 0.000035 | -0.2 | 0.0166 | 0.0047 | -0.1181 |
| on_off_sus_ratio | 0.000034 | -45.4 | 0.0053 | 0.0295 | -0.0892 |


## Delta -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| on_peak_extreme | 0.006536 | -85.4 | 0.0199 | 0.1273 | -0.3026 |
| off_peak_extreme | 0.004054 | -49.3 | 0.0126 | 0.0545 | -0.1967 |
| gb_base_mean_high | 0.002488 | -13.3 | 0.0242 | 0.0938 | -0.0098 |
| gb_base_mean | 0.001573 | 8.6 | 0.0183 | 0.0559 | -0.0409 |
| on_sustained | 0.001120 | 138.1 | 0.0067 | 0.0450 | -0.1446 |
| off_sustained | 0.000236 | -65.3 | 0.0083 | 0.0505 | -0.0895 |
| on_trans_sus_ratio | 0.000119 | -0.3 | 0.0355 | 0.1482 | -0.2109 |
| off_trans_sus_ratio | 0.000051 | 4.3 | 0.0066 | 0.0401 | -0.0525 |
| green_blue_off_ratio | 0.000043 | 179.3 | 0.0164 | -0.0047 | 0.0244 |
| step_up_QI | 0.000042 | -119.9 | 0.1129 | 0.1706 | -0.1826 |


## Before vs After Comparison

### Features with largest change in spatial structure

| Feature | Before R2 | After R2 | Delta R2 | Before Moran | After Moran | Delta Moran |
|---------|-----------|----------|----------|--------------|-------------|-------------|
| green_blue_off_ratio | 0.0940 | 0.0166 | -0.0774 | 0.1226 | 0.0047 | -0.1179 |
| on_trans_sus_ratio | 0.0736 | 0.0001 | -0.0735 | 0.2200 | 0.0572 | -0.1628 |
| off_peak_extreme | 0.1118 | 0.1485 | 0.0368 | 0.1937 | 0.2407 | 0.0470 |
| step_up_QI | 0.0369 | 0.0727 | 0.0358 | 0.1056 | 0.1611 | 0.0555 |
| green_blue_off_ratio_high | 0.0628 | 0.0336 | -0.0292 | 0.1261 | 0.0343 | -0.0918 |
| gb_base_mean | 0.0217 | 0.0452 | 0.0234 | 0.0406 | 0.0872 | 0.0466 |
| on_off_ratio | 0.0193 | 0.0361 | 0.0169 | 0.0189 | 0.1129 | 0.0940 |
| on_sustained | 0.0347 | 0.0491 | 0.0144 | 0.0658 | 0.1480 | 0.0822 |
| off_trans_sus_ratio | 0.0161 | 0.0020 | -0.0141 | 0.0454 | 0.0402 | -0.0053 |
| osi | 0.0250 | 0.0129 | -0.0121 | 0.1068 | 0.0143 | -0.0924 |
