# Leaderboard FD001 (generated from experiments/runs)

## Final configurations: dev CV and official test (each evaluated once on test)

| Model | CV cut RMSE | CV all-cycle RMSE | CV late-life RMSE | Test RMSE (5 seeds) | Test RMSE 95% CI | Test MAE | Test NASA score | Fit s | Latency ms | Size KB |
|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm_window50 | 15.12 ± 0.66 | 10.68 ± 0.17 | 6.25 ± 0.66 | 13.75 ± 0.18 | 11.6–15.6 | 10.27 ± 0.10 | 296.44 ± 8.21 | 2.8 | 1.42 | 1132 |
| gru_w50 | 15.78 ± 1.36 | 11.07 ± 0.53 | 6.49 ± 0.71 | 15.23 ± 0.33 | 12.3–17.0 | 11.15 ± 0.31 | 478.86 ± 122.36 | 194.6 | 5.82 | 171 |
| lstm_w50 | 16.52 ± 1.38 | 12.22 ± 0.67 | 7.14 ± 0.97 | 16.08 ± 0.41 | 13.0–17.4 | 12.16 ± 0.28 | 451.59 ± 55.63 | 142.5 | 0.98 | 224 |
| lightgbm | 16.87 ± 1.61 | 12.83 ± 0.73 | 8.13 ± 1.22 | 14.49 ± 0.28 | 12.4–16.3 | 11.24 ± 0.21 | 296.58 ± 16.22 | 2.9 | 1.40 | 1126 |
| gru_w30 | 17.30 ± 1.86 | 12.99 ± 0.85 | 8.31 ± 0.73 | 15.34 ± 0.11 | 12.4–17.4 | 11.19 ± 0.16 | 444.41 ± 49.66 | 120.1 | 3.31 | 171 |
| random_forest | 17.70 ± 2.04 | 13.66 ± 0.95 | 9.03 ± 1.94 | 14.76 ± 0.10 | 12.8–16.6 | 11.68 ± 0.11 | 326.64 ± 6.10 | 53.8 | 55.79 | 45115 |
| lstm_w30 | 18.08 ± 1.76 | 13.75 ± 0.73 | 8.31 ± 0.50 | 15.35 ± 0.40 | 12.4–16.7 | 11.45 ± 0.37 | 413.57 ± 87.13 | 77.6 | 0.85 | 224 |
| ridge_last | 24.41 ± 1.94 | 20.96 ± 1.10 | 21.49 ± 3.07 | 21.89 ± 0.00 | 18.9–24.7 | 17.62 ± 0.00 | 1316.89 ± 0.00 | 0.0 | 1.41 | 2 |
| constant | 38.99 ± 1.38 | 41.68 ± 0.09 | 63.56 ± 0.22 | 43.07 ± 0.00 | 38.6–47.4 | 35.90 ± 0.00 | 33629.23 ± 0.00 | 0.0 | 0.00 | 0 |

## Ablations (dev CV only; test set not touched)

| Variant | Config changes | CV cut RMSE | CV cut MAE | CV cut score/pt | CV all-cycle RMSE | CV late-life RMSE |
|---|---|---|---|---|---|---|
| lightgbm_with_cycle | cycle=True | 16.89 ± 1.19 | 12.53 ± 1.06 | 5.65 ± 1.37 | 12.18 ± 0.29 | 7.30 ± 0.84 |
| lightgbm_cap150 | cap=150 | 19.29 ± 2.02 | 14.64 ± 1.88 | 11.71 ± 3.91 | 13.42 ± 1.06 | 8.57 ± 1.24 |
| lightgbm_cap100 | cap=100 | 19.91 ± 1.16 | 14.78 ± 0.96 | 6.08 ± 1.33 | 19.92 ± 0.35 | 7.64 ± 1.08 |
| lightgbm_window15 | window=15 | 21.07 ± 2.29 | 16.44 ± 1.90 | 13.11 ± 7.74 | 16.62 ± 1.46 | 13.38 ± 2.94 |
| ridge_all_features | none | 22.64 ± 1.62 | 18.64 ± 1.51 | 10.94 ± 3.18 | 19.10 ± 0.89 | 18.99 ± 1.63 |
| lightgbm_uncapped | cap=None | 25.90 ± 4.56 | 19.06 ± 3.30 | 397.74 ± 525.16 | 14.32 ± 1.47 | 9.20 ± 1.13 |

Metric definitions: `cut` = up to 10 random truncation points per validation engine with true RUL in [10,150] and cycle >= 31, scored against uncapped true RUL; `all` = every cycle vs RUL capped at 125; `late` = cycles with true RUL <= 50. ± is std over 5 CV folds (model seed 0) or over 5 model seeds (test).
