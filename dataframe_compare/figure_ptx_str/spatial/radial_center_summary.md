# Radial Center Analysis: Blocker Comparison


## Before -- All Cells (Raw Mean)

### Strongest radial trends (by |best_r|)

| Feature | Best Cx | Best Cy | best_r | best_p | origin_r | Improvement |
|---------|---------|---------|--------|--------|----------|-------------|
| green_blue_off_ratio | 1800 | -0 | -0.3349 | 6.73e-22 | -0.0713 | 0.2636 |
| off_peak_extreme | -1800 | 231 | 0.2924 | 7.62e-17 | 0.0658 | 0.2266 |
| on_trans_sus_ratio | 1800 | 576 | 0.2907 | 1.17e-16 | 0.0961 | 0.1946 |
| green_blue_off_ratio_high | -1800 | -593 | 0.2735 | 7.50e-15 | -0.0258 | 0.2477 |
| on_sustained | -1093 | -1797 | -0.2734 | 7.76e-15 | 0.0730 | 0.2004 |
| gb_base_mean | 480 | -377 | 0.1983 | 2.35e-08 | 0.0960 | 0.1023 |
| gb_base_mean_high | 536 | -404 | 0.1952 | 3.88e-08 | 0.0840 | 0.1112 |
| on_off_sus_ratio | -692 | 812 | 0.1922 | 6.33e-08 | 0.0210 | 0.1712 |
| green_blue_on_ratio | 52 | -691 | -0.1824 | 2.90e-07 | -0.0996 | 0.0829 |
| on_peak_extreme | -656 | 47 | 0.1641 | 4.10e-06 | 0.0665 | 0.0976 |


## After -- All Cells (Raw Mean)

### Strongest radial trends (by |best_r|)

| Feature | Best Cx | Best Cy | best_r | best_p | origin_r | Improvement |
|---------|---------|---------|--------|--------|----------|-------------|
| off_peak_extreme | 1145 | -563 | -0.3245 | 1.37e-20 | -0.0418 | 0.2828 |
| on_off_ratio | 709 | -278 | -0.2307 | 7.04e-11 | -0.0819 | 0.1488 |
| gb_base_mean | 420 | 1800 | -0.2264 | 1.59e-10 | 0.0056 | 0.2208 |
| step_up_QI | 43 | -627 | -0.2242 | 2.70e-10 | -0.1015 | 0.1226 |
| gb_base_mean_high | 499 | 1800 | -0.2238 | 2.61e-10 | 0.0096 | 0.2142 |
| on_sustained | -839 | -973 | -0.2192 | 6.11e-10 | 0.0678 | 0.1514 |
| on_peak_extreme | -1618 | 713 | 0.1685 | 2.21e-06 | -0.0560 | 0.1125 |
| green_blue_off_ratio_high | 585 | -193 | -0.1651 | 3.56e-06 | -0.0659 | 0.0992 |
| green_blue_off_ratio | 1121 | -478 | -0.1558 | 1.24e-05 | -0.0581 | 0.0977 |
| dsi | 266 | -32 | 0.1343 | 1.81e-04 | 0.1174 | 0.0169 |


## Delta -- All Cells (Raw Mean)

### Strongest radial trends (by |best_r|)

| Feature | Best Cx | Best Cy | best_r | best_p | origin_r | Improvement |
|---------|---------|---------|--------|--------|----------|-------------|
| step_up_QI | -172 | -522 | -0.2706 | 1.72e-14 | -0.1340 | 0.1366 |
| on_trans_sus_ratio | 1799 | 282 | -0.2450 | 4.01e-12 | -0.0503 | 0.1947 |
| gb_base_mean | 1799 | 1467 | -0.1984 | 2.28e-08 | -0.0780 | 0.1204 |
| gb_base_mean_high | 904 | 996 | -0.1921 | 6.45e-08 | -0.0603 | 0.1318 |
| on_off_sus_ratio | 1599 | -1650 | 0.1718 | 1.39e-06 | -0.0543 | 0.1175 |
| off_peak_extreme | 196 | -226 | -0.1407 | 8.10e-05 | -0.1066 | 0.0341 |
| on_peak_extreme | 58 | -190 | -0.1382 | 1.08e-04 | -0.1225 | 0.0157 |
| green_blue_off_ratio | 1800 | 728 | 0.1368 | 1.27e-04 | 0.0029 | 0.1338 |
| dsi | 257 | 136 | 0.1349 | 1.68e-04 | 0.1014 | 0.0335 |
| green_blue_off_ratio_high | -737 | -1055 | -0.1138 | 1.45e-03 | -0.0394 | 0.0744 |


## Before vs After Radial Center Shifts

| Feature | Before Cx | Before Cy | After Cx | After Cy | Shift (um) | Before |r| | After |r| |
|---------|-----------|-----------|----------|----------|------------|-----------|----------|
| dsi | -371 | -803 | 266 | -32 | 1000 | 0.1039 | 0.1343 |
| gb_base_mean | 480 | -377 | 420 | 1800 | 2177 | 0.1983 | 0.2264 |
| gb_base_mean_high | 536 | -404 | 499 | 1800 | 2204 | 0.1952 | 0.2238 |
| green_blue_off_ratio | 1800 | -0 | 1121 | -478 | 830 | 0.3349 | 0.1558 |
| green_blue_off_ratio_high | -1800 | -593 | 585 | -193 | 2418 | 0.2735 | 0.1651 |
| green_blue_on_ratio | 52 | -691 | -0 | 1794 | 2486 | 0.1824 | 0.0903 |
| green_blue_on_ratio_high | 125 | -754 | -34 | 723 | 1485 | 0.1556 | 0.0717 |
| off_peak_extreme | -1800 | 231 | 1145 | -563 | 3050 | 0.2924 | 0.3245 |
| off_sustained | -439 | 136 | 197 | -183 | 711 | 0.0649 | 0.0550 |
| off_trans_sus_ratio | -1800 | -1771 | 1182 | -530 | 3230 | 0.1143 | 0.1092 |
| on_off_ratio | 617 | 37 | 709 | -278 | 328 | 0.1295 | 0.2307 |
| on_off_sus_ratio | -692 | 812 | 1203 | -808 | 2493 | 0.1922 | 0.0780 |
| on_peak_extreme | -656 | 47 | -1618 | 713 | 1170 | 0.1641 | 0.1685 |
| on_sustained | -1093 | -1797 | -839 | -973 | 862 | 0.2734 | 0.2192 |
| on_trans_sus_ratio | 1800 | 576 | -478 | 626 | 2278 | 0.2907 | 0.1042 |
| osi | 934 | 352 | -353 | -508 | 1548 | 0.1377 | 0.1190 |
| step_up_QI | -711 | -210 | 43 | -627 | 862 | 0.1079 | 0.2242 |
