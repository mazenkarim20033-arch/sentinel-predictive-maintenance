# Leaderboard FD002 (generated from experiments/runs)

## Final configurations: dev CV and official test (each evaluated once on test)

| Model | CV cut RMSE | CV all-step RMSE | CV late-life RMSE | Test RMSE (5 seeds) | Test RMSE 95% CI | Test MAE | Test NASA score | Fit s | Latency ms | Size KB |
|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm_window50 | 17.22 ± 0.37 | 12.68 ± 0.30 | 7.96 ± 0.50 | 24.50 ± 0.06 | 21.3–27.3 | 16.30 ± 0.05 | 4860.90 ± 65.32 | 5.2 | 1.50 | 1142 |
| lightgbm | 19.42 ± 0.70 | 15.00 ± 0.25 | 10.79 ± 0.28 | 26.47 ± 0.11 | 23.0–29.5 | 17.23 ± 0.07 | 8523.59 ± 295.04 | 6.0 | 1.49 | 1137 |
| random_forest | 20.05 ± 0.73 | 15.66 ± 0.30 | 11.82 ± 0.44 | 26.45 ± 0.07 | 23.1–29.5 | 17.24 ± 0.06 | 7874.31 ± 176.90 | 216.1 | 60.38 | 119503 |
| ridge_last | 25.30 ± 0.95 | 21.23 ± 0.60 | 22.39 ± 0.73 | 32.41 ± 0.00 | 28.9–35.9 | 23.87 ± 0.00 | 19070.50 ± 0.00 | 0.0 | 1.46 | 2 |
| constant | 40.62 ± 0.63 | 41.67 ± 0.03 | 63.64 ± 0.08 | 54.08 ± 0.00 | 51.1–57.2 | 45.64 ± 0.00 | 153229.25 ± 0.00 | 0.0 | 0.00 | 0 |

## Ablations (dev CV only; test set not touched)

| Variant | Config changes | CV cut RMSE | CV cut MAE | CV cut score/pt | CV all-time RMSE | CV late-life RMSE |
|---|---|---|---|---|---|---|
| lightgbm_global_norm | window=50, regimes=1 (global normalisation) | 23.30 ± 0.89 | 18.43 ± 0.70 | 16.58 ± 3.62 | 18.59 ± 0.44 | 18.53 ± 0.77 |

Metric definitions: `cut` = up to 10 random truncation points per validation asset with true RUL in [10,150] and time >= 31, scored against uncapped true RUL; `all` = every time vs RUL capped at 125; `late` = steps with true RUL <= 50. ± is std over 5 CV folds (model seed 0) or over 5 model seeds (test).
