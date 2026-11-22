# Leaderboard FD004 (generated from experiments/runs)

## Final configurations: dev CV and official test (each evaluated once on test)

| Model | CV cut RMSE | CV all-step RMSE | CV late-life RMSE | Test RMSE (5 seeds) | Test RMSE 95% CI | Test MAE | Test NASA score | Fit s | Latency ms | Size KB |
|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm_window50 | 17.38 ± 1.01 | 12.79 ± 1.20 | 9.12 ± 0.97 | 25.65 ± 0.16 | 22.8–28.2 | 17.92 ± 0.11 | 4194.02 ± 121.48 | 5.2 | 1.48 | 1137 |
| lightgbm | 19.22 ± 1.18 | 14.15 ± 1.10 | 10.98 ± 1.09 | 26.45 ± 0.11 | 23.5–29.2 | 18.52 ± 0.06 | 4682.68 ± 98.78 | 5.2 | 1.45 | 1133 |
| random_forest | 19.48 ± 1.13 | 14.37 ± 1.08 | 11.53 ± 1.18 | 26.97 ± 0.04 | 24.0–29.7 | 19.24 ± 0.02 | 5086.17 ± 26.00 | 263.2 | 58.15 | 118831 |
| ridge_last | 25.69 ± 0.72 | 20.48 ± 0.63 | 24.67 ± 1.43 | 34.15 ± 0.00 | 31.3–36.9 | 27.10 ± 0.00 | 11757.80 ± 0.00 | 0.0 | 1.50 | 2 |
| constant | 42.62 ± 0.87 | 40.67 ± 0.06 | 69.56 ± 0.06 | 54.90 ± 0.00 | 51.8–58.0 | 46.75 ± 0.00 | 189682.81 ± 0.00 | 0.0 | 0.00 | 0 |

## Ablations (dev CV only; test set not touched)

| Variant | Config changes | CV cut RMSE | CV cut MAE | CV cut score/pt | CV all-time RMSE | CV late-life RMSE |
|---|---|---|---|---|---|---|
| lightgbm_global_norm | window=50, regimes=1 (global normalisation) | 23.66 ± 0.97 | 18.66 ± 0.80 | 17.88 ± 3.65 | 17.76 ± 0.92 | 18.67 ± 0.93 |

Metric definitions: `cut` = up to 10 random truncation points per validation asset with true RUL in [10,150] and time >= 31, scored against uncapped true RUL; `all` = every time vs RUL capped at 125; `late` = steps with true RUL <= 50. ± is std over 5 CV folds (model seed 0) or over 5 model seeds (test).
