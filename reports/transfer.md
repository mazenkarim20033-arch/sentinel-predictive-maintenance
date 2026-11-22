# Cross-subset transfer (generated)

LightGBM, window 50, cap 125, seed 0, trained on all training assets of the row (source) subset and evaluated once on the official test set of the column (target) subset. Diagonal = in-subset. `source_norm` uses preprocessing fitted on the source; `target_norm` re-fits only the normalisation and regimes on the target's unlabelled training sensors (no target labels, no target test data). Single seed, no confidence intervals; differences of 1-2 RMSE points on 100-250 test assets are within noise.

## Test RMSE (cycles), source_norm

| source \ target | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| FD001 | **13.4** | 58.4 | 22.3 | 61.3 |
| FD002 | 14.7 | **24.6** | 34.9 | 36.6 |
| FD003 | 14.1 | 56.1 | **13.6** | 61.0 |
| FD004 | 17.8 | 25.4 | 20.9 | **25.6** |

## Mean error, prediction minus truth (cycles), source_norm

| source \ target | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| FD001 | **0.6** | -22.8 | 8.8 | -27.7 |
| FD002 | 3.3 | **-7.3** | 19.2 | 4.2 |
| FD003 | -0.4 | -3.9 | **0.8** | -7.3 |
| FD004 | 9.7 | -6.0 | 11.5 | **-7.5** |

## NASA score, source_norm

| source \ target | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| FD001 | **281** | 383,163 | 1,436 | 409,405 |
| FD002 | 384 | **4,771** | 18,716 | 38,876 |
| FD003 | 308 | 664,213 | **306** | 623,955 |
| FD004 | 654 | 5,306 | 1,616 | **4,198** |

## Test RMSE (cycles), target_norm

| source \ target | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| FD001 | **13.4** | 35.2 | 27.9 | 38.2 |
| FD002 | 15.4 | **24.6** | 34.9 | 36.9 |
| FD003 | 26.4 | 30.9 | **13.6** | 29.6 |
| FD004 | 20.3 | 26.3 | 19.9 | **25.6** |

## Mean error, prediction minus truth (cycles), target_norm

| source \ target | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| FD001 | **0.6** | -8.5 | 16.0 | -2.5 |
| FD002 | 5.1 | **-7.3** | 19.8 | 4.0 |
| FD003 | 14.9 | -10.9 | **0.8** | -12.2 |
| FD004 | 9.9 | -0.8 | 10.9 | **-7.5** |

## NASA score, target_norm

| source \ target | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| FD001 | **281** | 19,595 | 2,810 | 14,635 |
| FD002 | 450 | **4,771** | 18,133 | 28,600 |
| FD003 | 5,425 | 14,715 | **306** | 8,489 |
| FD004 | 1,790 | 6,716 | 1,032 | **4,198** |
