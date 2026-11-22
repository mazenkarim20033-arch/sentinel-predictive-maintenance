# Failure warnings, FD004 (generated)

Model: LightGBM, window 50, cap 125 (fixed). Truth: true RUL <= tau. Alert: predicted RUL <= tau for k=3 consecutive cycles unless stated. Thresholds are experimental operating points, not maintenance standards; predicted RUL is not a calibrated failure probability.

## Development: out-of-fold on 249 training assets, 61249 cycles

| tau | k | precision | recall | F1 | false-alarm rate | assets with a useful alert | missed | early alarms | median lead | lead p10-p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1 | 0.87 | 0.72 | 0.79 | 0.00 | 0.93 | 0.04 | 0.02 | 9.0 | 5-14 |
| 10 | 3 | 0.94 | 0.55 | 0.70 | 0.00 | 0.86 | 0.12 | 0.02 | 7.0 | 2-11 |
| 10 | 5 | 0.97 | 0.40 | 0.57 | 0.00 | 0.74 | 0.24 | 0.02 | 5.0 | 1-9 |
| 15 | 1 | 0.89 | 0.79 | 0.83 | 0.01 | 0.96 | 0.02 | 0.02 | 14.0 | 8-20 |
| 15 | 3 | 0.94 | 0.66 | 0.77 | 0.00 | 0.93 | 0.04 | 0.02 | 11.0 | 5-17 |
| 15 | 5 | 0.96 | 0.54 | 0.69 | 0.00 | 0.90 | 0.08 | 0.02 | 9.0 | 4-15 |
| 20 | 1 | 0.91 | 0.82 | 0.86 | 0.01 | 0.97 | 0.00 | 0.02 | 19.0 | 13-26 |
| 20 | 3 | 0.95 | 0.71 | 0.81 | 0.00 | 0.96 | 0.02 | 0.02 | 16.0 | 9-22 |
| 20 | 5 | 0.96 | 0.61 | 0.75 | 0.00 | 0.95 | 0.02 | 0.02 | 13.0 | 6-19 |
| 30 | 1 | 0.93 | 0.86 | 0.89 | 0.01 | 0.98 | 0.00 | 0.02 | 28.0 | 22-38 |
| 30 | 3 | 0.95 | 0.78 | 0.86 | 0.01 | 0.97 | 0.00 | 0.02 | 24.5 | 18-33 |
| 30 | 5 | 0.96 | 0.71 | 0.82 | 0.00 | 0.97 | 0.00 | 0.02 | 22.0 | 16-30 |
| 40 | 1 | 0.92 | 0.87 | 0.89 | 0.01 | 0.98 | 0.00 | 0.02 | 38.0 | 29-51 |
| 40 | 3 | 0.95 | 0.80 | 0.87 | 0.01 | 0.98 | 0.00 | 0.02 | 33.0 | 26-46 |
| 40 | 5 | 0.96 | 0.75 | 0.84 | 0.01 | 0.98 | 0.00 | 0.02 | 31.0 | 23-41 |
| 50 | 1 | 0.93 | 0.88 | 0.90 | 0.02 | 0.98 | 0.00 | 0.02 | 48.0 | 37-65 |
| 50 | 3 | 0.95 | 0.82 | 0.88 | 0.01 | 0.98 | 0.00 | 0.02 | 43.0 | 32-57 |
| 50 | 5 | 0.96 | 0.78 | 0.86 | 0.01 | 0.98 | 0.00 | 0.02 | 40.0 | 29-54 |

Early alarm = first alert with true RUL above 2 x tau. Lead = true RUL (cycles before failure) at the first alert.

### Three-class confusion at the default thresholds (normal > 30, warning 15-30, critical < 15), out-of-fold, no persistence

| true \ predicted | normal | warning | critical |
|---|---|---|---|
| normal | 53009 | 440 | 81 |
| warning | 1033 | 2534 | 417 |
| critical | 69 | 633 | 3033 |

## Official test, fixed default operating points (k=3), cycle level only

Test assets stop before failure, so lead time is not measurable. Positives are few, and metrics pool many correlated cycles from 100-250 assets.

| tau | precision | recall | F1 | false-alarm rate | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 30 | 0.85 | 0.53 | 0.65 | 0.00 | 455 | 79 | 409 | 40271 |
| 15 | 0.85 | 0.29 | 0.43 | 0.00 | 52 | 9 | 127 | 41026 |

Test three-class confusion (no persistence):

| true \ predicted | normal | warning | critical |
|---|---|---|---|
| normal | 40223 | 115 | 12 |
| warning | 281 | 412 | 28 |
| critical | 5 | 64 | 74 |
