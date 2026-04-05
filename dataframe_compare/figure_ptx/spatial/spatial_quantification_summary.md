# Spatial Quantification: Blocker Comparison


## Before -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.006032 | -14.8 | 0.0176 | 0.0641 | -0.0528 |
| gb_base_mean | 0.004428 | 104.6 | 0.1347 | 0.0980 | -0.0468 |
| gb_base_mean_high | 0.003992 | 102.1 | 0.0951 | 0.0733 | -0.0763 |
| on_sustained | 0.003821 | -116.4 | 0.0517 | 0.0140 | 0.0602 |
| on_peak_extreme | 0.001827 | 179.8 | 0.0012 | 0.0736 | 0.0007 |
| off_sustained | 0.000163 | -103.9 | 0.0054 | 0.0345 | -0.0183 |
| on_trans_sus_ratio | 0.000137 | -178.4 | 0.0496 | 0.0775 | 0.1134 |
| on_off_sus_ratio | 0.000075 | -121.1 | 0.0170 | -0.0053 | 0.0095 |
| on_off_ratio | 0.000074 | -78.9 | 0.0344 | 0.0520 | 0.0092 |
| off_trans_sus_ratio | 0.000051 | -165.2 | 0.0090 | 0.0173 | 0.0523 |


## After -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.021699 | -19.4 | 0.1136 | 0.2219 | -0.1453 |
| on_peak_extreme | 0.019012 | -73.1 | 0.0949 | 0.1255 | -0.0436 |
| on_sustained | 0.007330 | -121.8 | 0.1311 | 0.1419 | 0.1454 |
| gb_base_mean | 0.001073 | 94.8 | 0.0090 | 0.0633 | -0.0784 |
| gb_base_mean_high | 0.000938 | 78.2 | 0.0059 | 0.0490 | -0.0727 |
| off_sustained | 0.000201 | 30.9 | 0.0053 | 0.0635 | -0.0867 |
| on_trans_sus_ratio | 0.000162 | -167.6 | 0.0562 | 0.1068 | 0.2108 |
| on_off_ratio | 0.000116 | -55.3 | 0.0797 | 0.0578 | -0.1122 |
| off_trans_sus_ratio | 0.000056 | -55.4 | 0.0087 | 0.0551 | -0.0637 |
| on_off_sus_ratio | 0.000056 | 152.5 | 0.0081 | 0.0079 | 0.0127 |


## Delta -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| on_peak_extreme | 0.019628 | -68.0 | 0.1071 | 0.1104 | -0.0472 |
| off_peak_extreme | 0.015694 | -21.1 | 0.0752 | 0.1335 | -0.1211 |
| on_sustained | 0.003545 | -127.7 | 0.0369 | 0.0653 | 0.1028 |
| gb_base_mean | 0.003375 | -72.3 | 0.0634 | 0.0769 | -0.0242 |
| gb_base_mean_high | 0.003157 | -71.0 | 0.0450 | 0.0574 | 0.0097 |
| off_sustained | 0.000336 | 51.0 | 0.0093 | 0.0442 | -0.0572 |
| on_off_sus_ratio | 0.000090 | 96.9 | 0.0135 | 0.0093 | 0.0032 |
| off_trans_sus_ratio | 0.000088 | -22.3 | 0.0120 | 0.0549 | -0.0834 |
| on_off_ratio | 0.000057 | -23.9 | 0.0123 | 0.0478 | -0.0990 |
| on_trans_sus_ratio | 0.000038 | -125.0 | 0.0025 | 0.0005 | 0.0943 |


## Before vs After Comparison

### Features with largest change in spatial structure

| Feature | Before R2 | After R2 | Delta R2 | Before Moran | After Moran | Delta Moran |
|---------|-----------|----------|----------|--------------|-------------|-------------|
| gb_base_mean | 0.1347 | 0.0090 | -0.1257 | 0.0980 | 0.0633 | -0.0347 |
| step_up_QI | 0.0063 | 0.1058 | 0.0996 | 0.0711 | 0.1835 | 0.1125 |
| off_peak_extreme | 0.0176 | 0.1136 | 0.0960 | 0.0641 | 0.2219 | 0.1578 |
| on_peak_extreme | 0.0012 | 0.0949 | 0.0937 | 0.0736 | 0.1255 | 0.0519 |
| gb_base_mean_high | 0.0951 | 0.0059 | -0.0892 | 0.0733 | 0.0490 | -0.0242 |
| on_sustained | 0.0517 | 0.1311 | 0.0793 | 0.0140 | 0.1419 | 0.1279 |
| on_off_ratio | 0.0344 | 0.0797 | 0.0453 | 0.0520 | 0.0578 | 0.0058 |
| green_blue_on_ratio | 0.0439 | 0.0041 | -0.0398 | 0.0492 | -0.0281 | -0.0773 |
| green_blue_off_ratio_high | 0.0015 | 0.0181 | 0.0166 | -0.0184 | 0.0381 | 0.0565 |
| green_blue_on_ratio_high | 0.0199 | 0.0062 | -0.0137 | 0.0296 | -0.0029 | -0.0325 |
