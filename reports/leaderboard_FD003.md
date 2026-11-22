# Leaderboard FD003 (generated from experiments/runs)

## Final configurations: dev CV and official test (each evaluated once on test)

| Model | CV cut RMSE | CV all-step RMSE | CV late-life RMSE | Test RMSE (5 seeds) | Test RMSE 95% CI | Test MAE | Test NASA score | Fit s | Latency ms | Size KB |
|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm_window50 | 14.72 ± 1.26 | 9.83 ± 0.66 | 6.11 ± 0.54 | 13.88 ± 0.21 | 11.1–16.4 | 9.75 ± 0.13 | 333.45 ± 18.24 | 3.1 | 1.48 | 1132 |
| lightgbm | 15.87 ± 1.28 | 11.16 ± 0.50 | 7.39 ± 1.08 | 14.98 ± 0.24 | 12.3–17.3 | 11.00 ± 0.21 | 483.84 ± 42.58 | 3.2 | 1.46 | 1127 |
| random_forest | 16.79 ± 1.35 | 11.92 ± 0.62 | 8.27 ± 1.01 | 15.52 ± 0.06 | 12.9–18.1 | 11.33 ± 0.06 | 584.76 ± 28.12 | 80.9 | 48.48 | 44852 |
| ridge_last | 23.84 ± 1.40 | 19.02 ± 0.93 | 21.57 ± 2.47 | 22.31 ± 0.00 | 19.5–24.9 | 18.03 ± 0.00 | 1567.67 ± 0.00 | 0.0 | 1.46 | 2 |
| constant | 40.97 ± 1.37 | 40.64 ± 0.28 | 69.71 ± 0.27 | 45.07 ± 0.00 | 40.3–49.3 | 38.00 ± 0.00 | 57949.64 ± 0.00 | 0.0 | 0.00 | 0 |

## Ablations (dev CV only; test set not touched)

| Variant | Config changes | CV cut RMSE | CV cut MAE | CV cut score/pt | CV all-time RMSE | CV late-life RMSE |
|---|---|---|---|---|---|---|

Metric definitions: `cut` = up to 10 random truncation points per validation asset with true RUL in [10,150] and time >= 31, scored against uncapped true RUL; `all` = every time vs RUL capped at 125; `late` = steps with true RUL <= 50. ± is std over 5 CV folds (model seed 0) or over 5 model seeds (test).
