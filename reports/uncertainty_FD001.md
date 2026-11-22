# Prediction intervals, FD001 (generated)

LightGBM window 50 (fixed). Intervals are for RUL capped at 125 (values above are reported as 125+). Calibration on 20% held-out training assets; conformal methods give *marginal* coverage under exchangeability, not coverage at every RUL value. Intervals are not failure probabilities.

## Development: out-of-fold on 20631 cycles of 100 training assets (5-fold grouped CV)

| nominal | method | coverage | fold coverage (mean ± std) | mean width | median width |
|---|---|---|---|---|---|
| 80% | quantile | 0.50 | 0.50 ± 0.03 | 17.8 | 16.2 |
| 80% | cqr | 0.81 | 0.81 ± 0.03 | 23.8 | 22.3 |
| 80% | conformal | 0.80 | 0.80 ± 0.04 | 21.6 | 22.0 |
| 90% | quantile | 0.64 | 0.64 ± 0.03 | 28.4 | 26.9 |
| 90% | cqr | 0.91 | 0.91 ± 0.02 | 35.9 | 34.2 |
| 90% | conformal | 0.90 | 0.90 ± 0.03 | 30.6 | 32.5 |

### Coverage by true RUL (capped) bin, development

| nominal | method | 0-25 | 25-50 | 50-75 | 75-100 | 100-125 |
|---|---|---|---|---|---|---|
| 80% | quantile | 0.55 (w 12) | 0.55 (w 20) | 0.46 (w 28) | 0.51 (w 27) | 0.48 (w 14) |
| 80% | cqr | 0.93 (w 19) | 0.77 (w 27) | 0.62 (w 35) | 0.65 (w 34) | 0.87 (w 19) |
| 80% | conformal | 0.99 (w 22) | 0.90 (w 26) | 0.68 (w 26) | 0.51 (w 25) | 0.83 (w 19) |
| 90% | quantile | 0.70 (w 32) | 0.66 (w 39) | 0.58 (w 40) | 0.69 (w 35) | 0.62 (w 21) |
| 90% | cqr | 0.97 (w 41) | 0.89 (w 48) | 0.78 (w 50) | 0.81 (w 44) | 0.95 (w 27) |
| 90% | conformal | 1.00 (w 31) | 0.98 (w 38) | 0.80 (w 38) | 0.72 (w 36) | 0.93 (w 26) |

## Official test, evaluated once (100 assets, one point each)

| nominal | method | coverage vs capped truth | 95% CI | coverage vs uncapped truth | mean width |
|---|---|---|---|---|---|
| 80% | quantile | 0.61 | 0.51-0.70 | 0.57 | 24.0 |
| 80% | cqr | 0.78 | 0.69-0.85 | 0.68 | 29.9 |
| 80% | conformal | 0.71 | 0.61-0.79 | 0.64 | 22.7 |
| 90% | quantile | 0.72 | 0.63-0.80 | 0.66 | 31.9 |
| 90% | cqr | 0.85 | 0.77-0.91 | 0.74 | 36.9 |
| 90% | conformal | 0.79 | 0.70-0.86 | 0.72 | 29.8 |
