# Prediction intervals, FD004 (generated)

LightGBM window 50 (fixed). Intervals are for RUL capped at 125 (values above are reported as 125+). Calibration on 20% held-out training assets; conformal methods give *marginal* coverage under exchangeability, not coverage at every RUL value. Intervals are not failure probabilities.

## Development: out-of-fold on 61249 cycles of 249 training assets (5-fold grouped CV)

| nominal | method | coverage | fold coverage (mean ± std) | mean width | median width |
|---|---|---|---|---|---|
| 80% | quantile | 0.62 | 0.62 ± 0.02 | 20.3 | 17.9 |
| 80% | cqr | 0.80 | 0.80 ± 0.01 | 22.4 | 20.1 |
| 80% | conformal | 0.80 | 0.80 ± 0.02 | 21.4 | 23.2 |
| 90% | quantile | 0.74 | 0.74 ± 0.02 | 29.5 | 28.7 |
| 90% | cqr | 0.89 | 0.89 ± 0.01 | 32.5 | 31.6 |
| 90% | conformal | 0.89 | 0.89 ± 0.02 | 31.1 | 30.3 |

### Coverage by true RUL (capped) bin, development

| nominal | method | 0-25 | 25-50 | 50-75 | 75-100 | 100-125 |
|---|---|---|---|---|---|---|
| 80% | quantile | 0.63 (w 17) | 0.63 (w 28) | 0.59 (w 36) | 0.58 (w 33) | 0.62 (w 15) |
| 80% | cqr | 0.80 (w 20) | 0.70 (w 31) | 0.65 (w 38) | 0.62 (w 36) | 0.87 (w 16) |
| 80% | conformal | 0.95 (w 24) | 0.81 (w 27) | 0.59 (w 27) | 0.47 (w 26) | 0.86 (w 18) |
| 90% | quantile | 0.76 (w 32) | 0.74 (w 45) | 0.73 (w 49) | 0.76 (w 43) | 0.74 (w 21) |
| 90% | cqr | 0.92 (w 36) | 0.82 (w 49) | 0.79 (w 53) | 0.80 (w 46) | 0.93 (w 23) |
| 90% | conformal | 0.98 (w 34) | 0.93 (w 41) | 0.75 (w 41) | 0.67 (w 38) | 0.93 (w 26) |

## Official test, evaluated once (248 assets, one point each)

| nominal | method | coverage vs capped truth | 95% CI | coverage vs uncapped truth | mean width |
|---|---|---|---|---|---|
| 80% | quantile | 0.63 | 0.57-0.69 | 0.46 | 24.0 |
| 80% | cqr | 0.76 | 0.71-0.81 | 0.51 | 26.2 |
| 80% | conformal | 0.76 | 0.70-0.81 | 0.53 | 23.4 |
| 90% | quantile | 0.76 | 0.70-0.81 | 0.57 | 35.1 |
| 90% | cqr | 0.88 | 0.84-0.92 | 0.63 | 37.9 |
| 90% | conformal | 0.89 | 0.84-0.92 | 0.63 | 34.4 |
