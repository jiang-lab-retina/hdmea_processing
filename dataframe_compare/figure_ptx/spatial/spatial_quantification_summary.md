# Spatial Quantification: Blocker Comparison


## Before -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.009204 | -21.3 | 0.0468 | 0.1144 | -0.0421 |
| on_peak_extreme | 0.004357 | 97.6 | 0.0080 | 0.0708 | 0.0346 |
| gb_base_mean | 0.003881 | 99.0 | 0.1304 | 0.1000 | -0.0462 |
| gb_base_mean_high | 0.003552 | 95.7 | 0.0943 | 0.0753 | -0.0785 |
| on_sustained | 0.003235 | -125.8 | 0.0406 | 0.0163 | 0.0779 |
| off_sustained | 0.000152 | -89.1 | 0.0058 | 0.0258 | -0.0194 |
| on_trans_sus_ratio | 0.000152 | 171.7 | 0.0682 | 0.1281 | 0.1126 |
| on_off_sus_ratio | 0.000076 | -137.7 | 0.0192 | 0.0118 | -0.0188 |
| green_blue_off_ratio | 0.000050 | -2.2 | 0.0328 | 0.0477 | -0.0081 |
| on_off_ratio | 0.000048 | -67.4 | 0.0168 | 0.0381 | 0.0169 |


## After -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| off_peak_extreme | 0.023137 | -31.3 | 0.1438 | 0.2298 | -0.1739 |
| on_peak_extreme | 0.018005 | -79.9 | 0.0966 | 0.1141 | -0.0666 |
| on_sustained | 0.006497 | -126.4 | 0.1257 | 0.1536 | 0.1672 |
| gb_base_mean | 0.000941 | 61.6 | 0.0074 | 0.0573 | -0.0822 |
| gb_base_mean_high | 0.000885 | 41.0 | 0.0057 | 0.0447 | -0.0860 |
| on_trans_sus_ratio | 0.000158 | -177.1 | 0.0622 | 0.1410 | 0.2257 |
| off_sustained | 0.000146 | -5.2 | 0.0035 | 0.0529 | -0.0721 |
| on_off_ratio | 0.000099 | -63.0 | 0.0636 | 0.0657 | -0.1193 |
| on_off_sus_ratio | 0.000070 | 153.4 | 0.0151 | 0.0237 | 0.0316 |
| off_trans_sus_ratio | 0.000063 | -68.4 | 0.0120 | 0.0371 | -0.0492 |


## Delta -- All Cells

### Strongest Spatial Gradients

| Feature | Grad Mag | Direction | Plane R2 | Moran I | Radial r |
|---------|----------|-----------|----------|---------|----------|
| on_peak_extreme | 0.022359 | -80.4 | 0.1413 | 0.1497 | -0.0968 |
| off_peak_extreme | 0.014164 | -37.8 | 0.0705 | 0.1533 | -0.1640 |
| on_sustained | 0.003263 | -127.0 | 0.0367 | 0.0649 | 0.1043 |
| gb_base_mean | 0.003185 | -70.7 | 0.0719 | 0.0919 | -0.0320 |
| gb_base_mean_high | 0.003125 | -70.9 | 0.0558 | 0.0693 | -0.0014 |
| off_sustained | 0.000199 | 44.3 | 0.0040 | 0.0323 | -0.0459 |
| on_off_sus_ratio | 0.000082 | 94.6 | 0.0128 | 0.0115 | 0.0377 |
| off_trans_sus_ratio | 0.000071 | -33.1 | 0.0090 | 0.0503 | -0.0856 |
| on_off_ratio | 0.000051 | -58.9 | 0.0114 | 0.0571 | -0.1108 |
| step_up_QI | 0.000045 | -104.0 | 0.1004 | 0.2099 | -0.0908 |


## Before vs After Comparison

### Features with largest change in spatial structure

| Feature | Before R2 | After R2 | Delta R2 | Before Moran | After Moran | Delta Moran |
|---------|-----------|----------|----------|--------------|-------------|-------------|
| gb_base_mean | 0.1304 | 0.0074 | -0.1230 | 0.1000 | 0.0573 | -0.0427 |
| step_up_QI | 0.0069 | 0.1234 | 0.1165 | 0.0751 | 0.2013 | 0.1262 |
| off_peak_extreme | 0.0468 | 0.1438 | 0.0969 | 0.1144 | 0.2298 | 0.1154 |
| gb_base_mean_high | 0.0943 | 0.0057 | -0.0887 | 0.0753 | 0.0447 | -0.0307 |
| on_peak_extreme | 0.0080 | 0.0966 | 0.0886 | 0.0708 | 0.1141 | 0.0433 |
| on_sustained | 0.0406 | 0.1257 | 0.0851 | 0.0163 | 0.1536 | 0.1373 |
| on_off_ratio | 0.0168 | 0.0636 | 0.0467 | 0.0381 | 0.0657 | 0.0276 |
| green_blue_on_ratio | 0.0354 | 0.0047 | -0.0307 | 0.0371 | -0.0395 | -0.0766 |
| green_blue_off_ratio | 0.0328 | 0.0120 | -0.0208 | 0.0477 | 0.0794 | 0.0318 |
| dsi | 0.0141 | 0.0252 | 0.0111 | 0.0604 | 0.0803 | 0.0199 |
