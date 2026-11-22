# Test-versus-CV gap diagnostics (generated)

Diagnostics only. Nothing here was used to choose a model, a threshold or a hyperparameter.

## FD001

- Official test (LightGBM w50, seed 0, one point per asset, n=100): RMSE 13.41, MAE 10.08, mean error (pred - true) +0.61.
- Out-of-fold RMSE over every cycle (uncapped truth): 46.25; over cycles with true RUL <= 150: 14.17.
- Out-of-fold RMSE re-sampled to the test RUL distribution (400 replicates, one cycle per training asset): mean 12.90, 5th-95th percentile 11.33-14.58. Share of replicates at or above the test RMSE: 0.282.
- Total asset life (observed + remaining): test median 199 (p10-p90 153-263) vs train median 199 (155-275).
- History length at the test cut: median 134 (min 31); share of test assets with at most 50 observed cycles: 0.08.
- Test final RUL: median 86, share above 125 (the training cap): 0.11.

| true RUL bin | test n | test RMSE | test bias | OOF re-sampled RMSE | OOF re-sampled bias |
|---|---|---|---|---|---|
| 0-25 | 19 | 5.1 | +2.0 | 4.0 | +0.9 |
| 25-50 | 11 | 5.0 | +1.7 | 7.5 | +2.6 |
| 50-75 | 13 | 9.9 | +3.1 | 13.2 | +6.1 |
| 75-100 | 23 | 19.2 | +8.1 | 16.1 | +6.7 |
| 100-inf | 34 | 14.8 | -6.5 | 15.1 | -7.6 |

## FD004

- Official test (LightGBM w50, seed 0, one point per asset, n=248): RMSE 25.62, MAE 17.84, mean error (pred - true) -7.50.
- Out-of-fold RMSE over every cycle (uncapped truth): 76.53; over cycles with true RUL <= 150: 16.80.
- Out-of-fold RMSE re-sampled to the test RUL distribution (400 replicates, one cycle per training asset): mean 24.51, 5th-95th percentile 22.14-26.92. Share of replicates at or above the test RMSE: 0.250.
- Total asset life (observed + remaining): test median 235 (p10-p90 160-368) vs train median 234 (166-348).
- History length at the test cut: median 154 (min 19); share of test assets with at most 50 observed cycles: 0.08.
- Test final RUL: median 88, share above 125 (the training cap): 0.27.

| true RUL bin | test n | test RMSE | test bias | OOF re-sampled RMSE | OOF re-sampled bias |
|---|---|---|---|---|---|
| 0-25 | 49 | 7.5 | +3.9 | 6.6 | +2.5 |
| 25-50 | 30 | 7.6 | +2.7 | 10.6 | +4.5 |
| 50-75 | 26 | 20.8 | +12.7 | 19.0 | +7.9 |
| 75-100 | 41 | 19.0 | +6.2 | 19.3 | +7.8 |
| 100-inf | 102 | 36.0 | -26.6 | 34.8 | -24.7 |

