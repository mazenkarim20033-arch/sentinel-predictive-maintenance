# Failure warnings, FD001 (generated)

Model: LightGBM, window 50, cap 125 (fixed). Truth: true RUL <= tau. Alert: predicted RUL <= tau for k=3 consecutive cycles unless stated. Thresholds are experimental operating points, not maintenance standards; predicted RUL is not a calibrated failure probability.

## Development: out-of-fold on 100 training assets, 20631 cycles

| tau | k | precision | recall | F1 | false-alarm rate | assets with a useful alert | missed | early alarms | median lead | lead p10-p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1 | 0.93 | 0.88 | 0.90 | 0.00 | 1.00 | 0.00 | 0.00 | 10.0 | 7-13 |
| 10 | 3 | 0.98 | 0.71 | 0.83 | 0.00 | 1.00 | 0.00 | 0.00 | 7.0 | 4-10 |
| 10 | 5 | 1.00 | 0.54 | 0.70 | 0.00 | 0.97 | 0.03 | 0.00 | 5.0 | 2-8 |
| 15 | 1 | 0.93 | 0.89 | 0.91 | 0.01 | 1.00 | 0.00 | 0.00 | 15.0 | 11-19 |
| 15 | 3 | 0.97 | 0.79 | 0.87 | 0.00 | 1.00 | 0.00 | 0.00 | 12.0 | 9-16 |
| 15 | 5 | 0.99 | 0.67 | 0.80 | 0.00 | 0.99 | 0.01 | 0.00 | 10.0 | 7-14 |
| 20 | 1 | 0.95 | 0.90 | 0.92 | 0.01 | 1.00 | 0.00 | 0.00 | 19.0 | 15-25 |
| 20 | 3 | 0.97 | 0.81 | 0.88 | 0.00 | 1.00 | 0.00 | 0.00 | 17.0 | 12-22 |
| 20 | 5 | 0.99 | 0.72 | 0.84 | 0.00 | 1.00 | 0.00 | 0.00 | 15.0 | 10-20 |
| 30 | 1 | 0.95 | 0.89 | 0.92 | 0.01 | 1.00 | 0.00 | 0.00 | 29.0 | 24-38 |
| 30 | 3 | 0.97 | 0.83 | 0.89 | 0.00 | 1.00 | 0.00 | 0.00 | 25.0 | 20-34 |
| 30 | 5 | 0.99 | 0.77 | 0.86 | 0.00 | 1.00 | 0.00 | 0.00 | 23.0 | 17-31 |
| 40 | 1 | 0.95 | 0.91 | 0.93 | 0.01 | 1.00 | 0.00 | 0.00 | 39.0 | 29-48 |
| 40 | 3 | 0.97 | 0.86 | 0.91 | 0.01 | 1.00 | 0.00 | 0.00 | 36.0 | 26-45 |
| 40 | 5 | 0.98 | 0.82 | 0.89 | 0.00 | 1.00 | 0.00 | 0.00 | 33.0 | 24-42 |
| 50 | 1 | 0.96 | 0.90 | 0.93 | 0.01 | 1.00 | 0.00 | 0.00 | 48.0 | 40-58 |
| 50 | 3 | 0.98 | 0.86 | 0.91 | 0.01 | 1.00 | 0.00 | 0.00 | 44.0 | 35-54 |
| 50 | 5 | 0.99 | 0.82 | 0.90 | 0.00 | 1.00 | 0.00 | 0.00 | 41.0 | 33-52 |

Early alarm = first alert with true RUL above 2 x tau. Lead = true RUL (cycles before failure) at the first alert.

### Three-class confusion at the default thresholds (normal > 30, warning 15-30, critical < 15), out-of-fold, no persistence

| true \ predicted | normal | warning | critical |
|---|---|---|---|
| normal | 17376 | 155 | 0 |
| warning | 334 | 1110 | 156 |
| critical | 2 | 119 | 1379 |

## Official test, fixed default operating points (k=3), cycle level only

Test assets stop before failure, so lead time is not measurable. Positives are few, and metrics pool many correlated cycles from 100-250 assets.

| tau | precision | recall | F1 | false-alarm rate | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 30 | 0.99 | 0.64 | 0.78 | 0.00 | 213 | 2 | 119 | 12762 |
| 15 | 0.82 | 0.53 | 0.65 | 0.00 | 32 | 7 | 28 | 13029 |

Test three-class confusion (no persistence):

| true \ predicted | normal | warning | critical |
|---|---|---|---|
| normal | 12748 | 16 | 0 |
| warning | 77 | 180 | 25 |
| critical | 0 | 9 | 41 |
