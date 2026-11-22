# Evaluation Plan

## Splits

| Level | Data | Use |
|---|---|---|
| Development | Training engines split **by engine id** into train / validation (e.g. 80/20, stratified by lifetime; repeated with grouped K-fold for the headline FD001 comparison) | Model selection, early stopping, calibration |
| Final | Official test files + official RUL files | Reported once per final model; never used for tuning, scaling, clustering, or threshold choice |

Split seeds and engine ids are written to the registry. After selection, models may be refit on all training engines with the settings fixed on validation.

## Metrics

**RUL**: MAE, RMSE, and the NASA asymmetric score (Saxena et al. 2008): with d = predicted − true, `s = Σ exp(−d/13) − 1` for d < 0 and `Σ exp(d/10) − 1` for d ≥ 0 (late predictions penalised more). I wrote this from memory; the unit test will pin it against hand-computed values and I will cross-check it against the paper. Report on the official test (one point per engine) with bootstrap 95% CIs, plus actual-vs-predicted and error histograms. When models are trained on capped targets, report error against both capped and true RUL, and say which.

**Rolling evaluation (validation engines)**: all cycles, with prediction at cycle t from rows ≤ t. Report error as a function of RUL-remaining, since early-life error matters less than late-life error.

**Failure warning**: ground truth class from true RUL thresholds; precision, recall, F1, false-alarm rate, confusion matrix, missed warnings, lead time (cycles between first critical alert and failure) per engine. Sweep thresholds and show the trade-off curve. Thresholds chosen on validation.

**Anomaly**: injected-fault precision/recall, detection delay, false-positive rate on clean held-out sequences. Never merged with dataset results.

**Uncertainty**: empirical coverage vs nominal (80/90%), mean width, coverage by RUL range. Calibration of any risk "probability" via reliability curve and Brier score; otherwise it is called a score.

**Systems**: training time, p50/p95 inference latency, artifact size, hardware string, library versions.

## Model selection

Not by a single metric. A model is preferred if it is within noise of the best on validation RMSE and score, stable across seeds, cheaper, and not worse on late-life error. Differences smaller than seed std or bootstrap CI are reported as ties.

## Leakage controls (each has a test)

1. No engine appears in both train and validation.
2. Scaler, regime clusterer, feature selection fitted on training rows only.
3. Windows use rows ≤ t; perturbing rows > t does not change the output at t.
4. Test RUL never enters training or any threshold/hyperparameter choice.
5. Cycle count/engine age as a feature is allowed only if available at inference (it is), but is ablated since it can act as a proxy for RUL in training data.
6. Replay uses held-out or test engines, never training engines.

## Generalisation

Within-subset (FD001–FD004) and cross-subset transfer matrix. Report the drop honestly; note FD002/FD004 transfer from single-condition subsets is expected to be poor.

## Reporting

`make_report.py` builds the leaderboard, plots and tables from registry records only. Failed and negative runs stay in the registry and are listed in EXPERIMENTS.md.
