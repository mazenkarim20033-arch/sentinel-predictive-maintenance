# Explainability, FD004 (generated)

LightGBM window 50 (fixed). Attributions are exact TreeSHAP values computed on out-of-fold (held-out) assets and describe what the model does with its inputs; they are associations, not physical causes, and correlated signals share credit unpredictably. Signal importance sums a signal's last/mean/std/slope features.

Checks: additivity (sum of attributions + base value = prediction) max error 6.0e-13; max difference to the `shap` library TreeExplainer on 500 rows 0.0e+00.

## Global importance (mean |SHAP|, RUL units: cycles of the capped target)

| rank | signal | mean abs SHAP |
|---|---|---|
| 1 | s11 | 14.48 |
| 2 | s14 | 4.34 |
| 3 | s4 | 3.88 |
| 4 | s9 | 3.85 |
| 5 | s17 | 3.31 |
| 6 | s3 | 1.70 |
| 7 | s2 | 1.61 |
| 8 | s15 | 1.55 |
| 9 | s12 | 1.12 |
| 10 | s13 | 1.04 |

By feature type (fold 0): last 7.41, mean 5.97, std 6.60, slope 16.99

## How importance differs between models (fold 0; top 8 signals, value in parentheses)

- **LightGBM SHAP (all folds)** (mean |SHAP|): s11 (14.48), s14 (4.34), s4 (3.88), s9 (3.85), s17 (3.31), s3 (1.70), s2 (1.61), s15 (1.55)
- **LightGBM permutation** (RMSE increase when permuted): s11 (9.92), s4 (3.62), s14 (2.28), s9 (1.48), s17 (1.36), s15 (1.19), s3 (1.18), s2 (0.83)
- **Random Forest permutation** (RMSE increase when permuted): s11 (10.82), s4 (3.58), s14 (2.30), s9 (1.46), s3 (1.23), s15 (1.22), s17 (1.10), s2 (0.41)
- **Ridge abs coefficient** (sum |standardised coef|): s15 (16.22), s11 (16.04), s9 (14.17), s14 (13.70), s13 (11.72), s4 (11.68), s6 (11.36), s12 (11.25)

Spearman rank correlation of signal importance between methods:

| | LightGBM SHAP (all folds) | LightGBM permutation | Random Forest permutation | Ridge abs coefficient |
|---|---|---|---|---|
| LightGBM SHAP (all folds) | 1.00 | 0.94 | 0.91 | 0.68 |
| LightGBM permutation | 0.94 | 1.00 | 0.94 | 0.65 |
| Random Forest permutation | 0.91 | 0.94 | 1.00 | 0.64 |
| Ridge abs coefficient | 0.68 | 0.65 | 0.64 | 1.00 |

## Mean signed SHAP by true RUL (capped), top 6 signals

Positive pushes the predicted RUL up, negative pushes it down, relative to the base prediction.

| RUL bin | s11 | s14 | s4 | s9 | s17 | s3 |
|---|---|---|---|---|---|---|
| 0-25 | -36.8 | -7.2 | -7.0 | -6.2 | -7.4 | -3.2 |
| 25-50 | -22.0 | -6.9 | -3.5 | -6.4 | -3.5 | -1.6 |
| 50-75 | -8.0 | -4.1 | -0.8 | -4.0 | -1.2 | -0.7 |
| 75-100 | +2.7 | -0.3 | +0.8 | -0.8 | +0.3 | -0.1 |
| 100-125 | +9.8 | +2.5 | +3.2 | +1.9 | +2.1 | +1.2 |
