# Battery demonstration (generated from experiments/runs)

Leave-one-cell-out over 4 NASA cells (B0005, B0006, B0007, B0018). No held-out test set; 4 trajectories only.
RUL is in discharge cycles. End of life = capacity at or below 1.5 Ah (project-defined, 25% fade).

## battery_nasa_sensors

| Model | RMSE (mean ± std over cells) | MAE | Late-life RMSE (RUL <= 20) | RMSE B0005 | RMSE B0006 | RMSE B0007 | RMSE B0018 |
|---|---|---|---|---|---|---|---|
| ridge_last | 13.7 ± 3.6 | 11.0 | 6.6 | 8.0 | 18.1 | 14.4 | 14.5 |
| lightgbm | 15.7 ± 6.2 | 13.4 | 4.8 | 12.8 | 19.6 | 23.1 | 7.1 |
| random_forest | 16.0 ± 5.3 | 14.0 | 5.8 | 15.8 | 18.1 | 22.5 | 7.8 |
| age_only | 23.6 ± 11.1 | 23.1 | 16.9 | 6.8 | 23.7 | 37.7 | 26.3 |
| constant | 30.9 ± 6.5 | 25.9 | 39.4 | 28.9 | 26.3 | 42.1 | 26.5 |

## battery_nasa_with_capacity

| Model | RMSE (mean ± std over cells) | MAE | Late-life RMSE (RUL <= 20) | RMSE B0005 | RMSE B0006 | RMSE B0007 | RMSE B0018 |
|---|---|---|---|---|---|---|---|
| ridge_last | 13.5 ± 5.6 | 11.2 | 6.6 | 9.6 | 20.3 | 17.4 | 6.7 |
| lightgbm | 16.8 ± 8.1 | 14.1 | 4.5 | 11.6 | 27.4 | 21.4 | 6.7 |
| random_forest | 17.0 ± 7.0 | 14.5 | 5.3 | 14.7 | 25.3 | 21.3 | 6.9 |
| age_only | 23.6 ± 11.1 | 23.1 | 16.9 | 6.8 | 23.7 | 37.7 | 26.3 |
| constant | 30.9 ± 6.5 | 25.9 | 39.4 | 28.9 | 26.3 | 42.1 | 26.5 |
