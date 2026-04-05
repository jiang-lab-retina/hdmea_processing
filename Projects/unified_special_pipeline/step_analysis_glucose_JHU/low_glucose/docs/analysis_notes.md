# Low Glucose Analysis Notes

## Experiment Protocol

3-phase glucose manipulation during full-field step-light stimulation:

1. **Normal glucose** (ionic AMES): 0 - 2.5 min
2. **High glucose** (25 mM): 2.5 - 15 min
3. **Low glucose** (2 mM): 15 min - end of recording

Light stimulus: 3 s step ON, 7 s interval (10 s per trial, 6 trials/min).

## Data Sources

| Date       | Folder                       | Transition recordings | Steady-state recordings |
|------------|------------------------------|-----------------------|-------------------------|
| 2026.03.04 | `S:\20260304_low_glucose`    | 2                     | 2                       |
| 2026.03.05 | `S:\20260305_low_glucose`    | 3                     | 4                       |

Metadata from **MEA dashboard** Google Sheet (`Condition` column), cached at
`Projects/load_gsheet/gsheet_table.csv`.

## Transition Recordings (processed)

Only transition recordings have CMTR (sorted spike data) files and can be processed.

| File                          | Date       | High glucose onset | Low glucose onset | Duration |
|-------------------------------|------------|--------------------|--------------------|----------|
| 2026.03.04-13.18.16-Rec.cmcr | 2026.03.04 | 2.5 min            | 15 min             | 25 min   |
| 2026.03.04-14.24.10-Rec.cmcr | 2026.03.04 | 2.5 min            | 15 min             | 25 min   |
| 2026.03.05-10.13.03-Rec.cmcr | 2026.03.05 | 2.5 min            | 15 min             | 25 min   |
| 2026.03.05-11.18.30-Rec.cmcr | 2026.03.05 | 2.5 min            | 15 min             | 25 min   |
| 2026.03.05-12.24.16-Rec.cmcr | 2026.03.05 | 2.5 min            | 15 min             | 25 min   |

## Steady-State Recordings (not processed)

These recordings are entirely in low glucose with no CMTR files:

- 2026.03.04-13.43.22-Rec.cmcr (20 min)
- 2026.03.04-14.49.23-Rec.cmcr (20 min)
- 2026.03.05-10.38.10-Rec.cmcr (20 min)
- 2026.03.05-11.43.36-Rec.cmcr (20 min)
- 2026.03.05-12.49.28-Rec.cmcr (14 min, error noted)
- 2026.03.05-13.03.47-Rec.cmcr (10 min)
- 2026.03.05-13.13.54-Rec.cmcr (10 min)

## Analysis Features

- **on_peak**: Peak firing rate during light ON (bins 10-20) minus baseline (bins 0-5)
- **off_peak**: Peak firing rate during light OFF (bins 40-50) minus baseline
- **on_sustained**: Mean sustained response during ON (bins 30-40) minus baseline
- **max_min_range**: Maximum minus minimum firing rate across the full trial

## Output Structure

```
low_glucose/
  data/       -> HDF5 files (one per transition recording)
  figures/    -> All generated plots
  docs/       -> This file
```

## Differences from High Glucose Pipeline

- **3-phase protocol**: normal -> high -> low (vs. normal -> high -> normal)
- **Glucose shading**: Red for high glucose region, blue for low glucose region
- **Combined transition plots**: Left = normal-to-high, Right = high-to-low
- **Data source**: Google Sheet CSV instead of local xlsx file
- **Multiple data folders**: Data spans two drives/dates
