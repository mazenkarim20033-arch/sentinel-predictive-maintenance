# Explainability, FD001 (generated)

LightGBM window 50 (fixed). Attributions are exact TreeSHAP values computed on out-of-fold (held-out) assets and describe what the model does with its inputs; they are associations, not physical causes, and correlated signals share credit unpredictably. Signal importance sums a signal's last/mean/std/slope features.

Checks: additivity (sum of attributions + base value = prediction) max error 4.5e-13; max difference to the `shap` library TreeExplainer on 500 rows 0.0e+00.

## Global importance (mean |SHAP|, RUL units: cycles of the capped target)

| rank | signal | mean abs SHAP |
|---|---|---|
| 1 | s11 | 7.07 |
| 2 | s4 | 5.49 |
| 3 | s14 | 5.25 |
| 4 | s12 | 3.59 |
| 5 | s3 | 3.19 |
| 6 | s9 | 2.57 |
| 7 | s7 | 2.51 |
| 8 | s2 | 2.21 |
| 9 | s17 | 2.02 |
| 10 | s6 | 1.99 |

By feature type (fold 0): last 7.50, mean 9.57, std 7.63, slope 14.91

## How importance differs between models (fold 0; top 8 signals, value in parentheses)

- **LightGBM SHAP (all folds)** (mean |SHAP|): s11 (7.07), s4 (5.49), s14 (5.25), s12 (3.59), s3 (3.19), s9 (2.57), s7 (2.51), s2 (2.21)
- **LightGBM permutation** (RMSE increase when permuted): s14 (3.21), s11 (3.16), s6 (2.45), s12 (2.44), s4 (2.28), s9 (1.59), s3 (1.48), s2 (1.27)
- **Random Forest permutation** (RMSE increase when permuted): s11 (6.10), s14 (3.08), s4 (2.96), s12 (2.49), s9 (1.39), s2 (1.35), s3 (1.24), s7 (0.94)
- **Ridge abs coefficient** (sum |standardised coef|): s8 (24.28), s13 (17.75), s9 (17.42), s12 (13.97), s11 (12.45), s4 (12.04), s7 (11.94), s14 (7.41)

Spearman rank correlation of signal importance between methods:

| | LightGBM SHAP (all folds) | LightGBM permutation | Random Forest permutation | Ridge abs coefficient |
|---|---|---|---|---|
| LightGBM SHAP (all folds) | 1.00 | 0.82 | 0.89 | 0.11 |
| LightGBM permutation | 0.82 | 1.00 | 0.67 | 0.04 |
| Random Forest permutation | 0.89 | 0.67 | 1.00 | 0.34 |
| Ridge abs coefficient | 0.11 | 0.04 | 0.34 | 1.00 |

## Mean signed SHAP by true RUL (capped), top 6 signals

Positive pushes the predicted RUL up, negative pushes it down, relative to the base prediction.

| RUL bin | s11 | s4 | s14 | s12 | s3 | s9 |
|---|---|---|---|---|---|---|
| 0-25 | -15.3 | -11.3 | -7.4 | -6.4 | -6.2 | -2.4 |
| 25-50 | -8.3 | -6.0 | -6.9 | -3.6 | -2.4 | -2.4 |
| 50-75 | -1.6 | -0.7 | -5.0 | -0.1 | -0.3 | -1.8 |
| 75-100 | +2.2 | +1.7 | -0.9 | +2.0 | +1.0 | -0.5 |
| 100-125 | +5.3 | +4.3 | +2.7 | +2.8 | +2.2 | +1.2 |
