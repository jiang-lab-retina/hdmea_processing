# Spatial Quantification: Blocker Comparison


## Before -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.011803 | -11.4 | 0.0714 | 0.1578 | 0.0658 |
| on_peak_extreme | 0.008274 | -6.3 | 0.0219 | 0.0469 | 0.0665 |
| on_sustained | 0.004809 | -113.9 | 0.0931 | 0.1116 | 0.0730 |
| gb_base_mean | 0.002132 | 133.2 | 0.0326 | 0.0436 | 0.0960 |
| gb_base_mean_high | 0.001990 | 131.2 | 0.0286 | 0.0483 | 0.0840 |
| on_trans_sus_ratio | 0.000178 | -152.0 | 0.0804 | 0.1380 | 0.0961 |
| on_off_sus_ratio | 0.000124 | -60.8 | 0.0426 | 0.0543 | 0.0210 |
| green_blue_off_ratio | 0.000115 | 8.5 | 0.1073 | 0.1260 | -0.0713 |
| green_blue_off_ratio_high | 0.000088 | 18.3 | 0.0791 | 0.1227 | -0.0258 |
| off_trans_sus_ratio | 0.000072 | -121.2 | 0.0152 | 0.0583 | 0.0426 |


## After -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.017285 | -23.6 | 0.1161 | 0.1887 | -0.0418 |
| on_peak_extreme | 0.011979 | -22.5 | 0.0401 | 0.0744 | -0.0560 |
| on_sustained | 0.004249 | -131.3 | 0.0689 | 0.1334 | 0.0678 |
| gb_base_mean_high | 0.002889 | 77.6 | 0.0544 | 0.0871 | 0.0096 |
| gb_base_mean | 0.002849 | 78.9 | 0.0577 | 0.0706 | 0.0056 |
| on_off_ratio | 0.000105 | -15.8 | 0.0445 | 0.1108 | -0.0819 |
| off_sustained | 0.000096 | -76.7 | 0.0010 | 0.0862 | -0.0438 |
| on_trans_sus_ratio | 0.000076 | -90.3 | 0.0168 | 0.0457 | 0.0320 |
| green_blue_off_ratio_high | 0.000076 | -13.4 | 0.0370 | 0.0467 | -0.0659 |
| off_trans_sus_ratio | 0.000071 | -174.3 | 0.0098 | 0.0294 | 0.0051 |


## Delta -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.006260 | -46.9 | 0.0178 | 0.0303 | -0.1066 |
| on_peak_extreme | 0.004644 | -52.2 | 0.0062 | 0.0704 | -0.1225 |
| gb_base_mean | 0.002362 | 31.7 | 0.0368 | 0.0528 | -0.0780 |
| gb_base_mean_high | 0.002344 | 34.4 | 0.0344 | 0.0488 | -0.0603 |
| on_sustained | 0.001475 | 125.3 | 0.0064 | 0.0587 | -0.0036 |
| on_trans_sus_ratio | 0.000157 | 2.6 | 0.0334 | 0.1089 | -0.0503 |
| off_sustained | 0.000142 | -70.9 | 0.0015 | 0.0889 | -0.0050 |
| on_off_sus_ratio | 0.000122 | 129.8 | 0.0172 | 0.0574 | -0.0543 |
| off_trans_sus_ratio | 0.000064 | 121.5 | 0.0047 | 0.0645 | -0.0251 |
| green_blue_off_ratio | 0.000057 | -154.2 | 0.0146 | 0.0073 | 0.0029 |


## Before vs After Comparison

### Features with largest change in spatial structure

| Feature | Before R2 | After R2 | Delta R2 | Before Moran | After Moran | Delta Moran |
|---------|-----------|----------|----------|--------------|-------------|-------------|
| green_blue_off_ratio | 0.1073 | 0.0221 | -0.0852 | 0.1260 | 0.0368 | -0.0892 |
| on_trans_sus_ratio | 0.0804 | 0.0168 | -0.0636 | 0.1380 | 0.0457 | -0.0923 |
| off_peak_extreme | 0.0714 | 0.1161 | 0.0447 | 0.1578 | 0.1887 | 0.0310 |
| green_blue_off_ratio_high | 0.0791 | 0.0370 | -0.0421 | 0.1227 | 0.0467 | -0.0760 |
| on_off_sus_ratio | 0.0426 | 0.0012 | -0.0414 | 0.0543 | 0.0373 | -0.0170 |
| step_up_QI | 0.0092 | 0.0428 | 0.0336 | 0.0720 | 0.1275 | 0.0555 |
| gb_base_mean_high | 0.0286 | 0.0544 | 0.0257 | 0.0483 | 0.0871 | 0.0388 |
| gb_base_mean | 0.0326 | 0.0577 | 0.0251 | 0.0436 | 0.0706 | 0.0270 |
| green_blue_on_ratio_high | 0.0290 | 0.0041 | -0.0250 | 0.0491 | 0.0153 | -0.0338 |
| on_off_ratio | 0.0201 | 0.0445 | 0.0244 | 0.0140 | 0.1108 | 0.0969 |
