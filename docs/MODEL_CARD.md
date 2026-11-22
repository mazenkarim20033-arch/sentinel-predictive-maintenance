# Model Card: SENTINEL default RUL model

## Summary
A gradient-boosted regression model (LightGBM) that estimates remaining useful life (RUL, in operating cycles) of a machine from the causal history of its sensor signals. It is the default model of SENTINEL; the same pipeline also produces warning levels, anomaly scores, prediction intervals and SHAP explanations. Research and demonstration only; **not production-ready**.

## Model details
- **Inputs:** the last 50 time steps of each asset's signals, causally windowed (a prediction at step t uses steps up to t only; short histories are left-padded with the first observation and a mask). Signals are z-scored per operating regime (regimes found by k-means on the operating conditions, fitted on training assets only; one regime for single-condition data).
- **Features:** per signal, the last value, mean, standard deviation and slope over the window. The cycle number is not used (an ablation showed no gain).
- **Target:** RUL capped at 125 cycles; predictions are therefore at most 125.
- **Algorithm:** LightGBM, 400 trees, learning rate 0.05, 31 leaves, subsample 0.8, column subsample 0.8 (defaults fixed in `src/sentinel/models/baselines.py`; no tuning). About 1.1 MB; single-prediction latency about 1.5 ms on one CPU laptop; training about 3-5 s.
- **Extras built on it:** warning levels from thresholds on predicted RUL (normal > 30, warning 15-30, critical < 15; experimental), 80% / 90% prediction intervals by conformalised quantile regression, TreeSHAP explanations, a separate statistical anomaly detector.

## Training and evaluation data
NASA C-MAPSS FD001-FD004 (simulated turbofan data; see DATASET_CARD.md). Training on the official training engines of each subset; engine-level splits for development; official test sets used once per configuration.

## Performance (official test sets, one point per engine, 5 seeds; see `reports/REPORT.md` for everything)

| Subset | Test RMSE (cycles) | 95% CI | Constant-guess baseline |
|---|---|---|---|
| FD001 | 13.75 | 11.6-15.6 | 43.1 |
| FD002 | 24.50 | 21.3-27.3 | 54.1 |
| FD003 | 13.88 | 11.1-16.4 | 45.1 |
| FD004 | 25.65 | 22.8-28.2 | 54.9 |

- Models with overlapping confidence intervals (Random Forest, GRU, LSTM, other LightGBM settings) are not separable from this model on these test sets; LightGBM was chosen as default for equal-or-better accuracy at much lower training cost and size, not because other models were shown to be worse.
- Error is dominated by assets with true RUL above the 125 cap, where the model cannot predict beyond 125 (see `reports/test_predictions.png`).
- Difficulty tracks the number of operating regimes (FD002/FD004 about 25 RMSE versus about 14 for FD001/FD003), not the number of fault modes, on these results.
- Warnings (default thresholds, 3-cycle persistence), FD001 out-of-fold: precision 0.97, recall 0.83 at 30 cycles, median lead 25 cycles; on the official test set recall is lower (0.64) at cycle level.
- Intervals: CQR coverage was consistent with nominal on the official tests (for example FD004, 80% nominal: 0.76, 95% CI 0.71-0.81), but coverage is uneven across RUL values and applies to the capped target.

## Intended use
Comparing and demonstrating predictive-maintenance methods on run-to-failure data; as a template to be retrained on other machines' data.

## Out of scope / do not use for
Safety-critical or real maintenance decisions; machines whose data differ from the training data without retraining (transfer experiments show large failures, for example models trained on single-condition subsets reaching RMSE of 56-61 on six-condition subsets, worse than a constant guess); predicting RUL beyond the cap; treating the alert level or interval as a failure probability.

## Caveats
Simulated data; small test sets; one hardware configuration; explanations are associations, not causes. See LIMITATIONS.md.

## Reproducibility
Configs, seeds, splits, metrics, hardware and timings for every run are in `experiments/runs/`; commands in the README; `scripts/make_report.py` regenerates the report from those records.
