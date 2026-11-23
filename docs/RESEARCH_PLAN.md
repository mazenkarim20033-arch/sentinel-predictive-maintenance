# Research Plan

> **Status at project end (plan versus what was done).** Done as planned: baselines, LightGBM, GRU and LSTM on FD001, per-regime normalisation, capped versus uncapped targets, window and feature ablations, cross-subset transfer, anomaly detectors with injected faults, quantile/CQR/conformal intervals, SHAP. **Not done:** the optional Transformer (nothing justified its cost), an autoencoder anomaly detector (the simple statistical detector already matched or beat Isolation Forest), GRU/LSTM on FD002-FD004 (stopped as a cost decision), hyperparameter search (all models use fixed defaults), MLflow and SQLite (a file-based registry was enough). Results are in `docs/EXPERIMENTS.md`; the generated `reports/REPORT.md` is the source of truth.

## Questions

1. How much does each model family add over a constant and linear baseline on FD001, and does the ranking hold on FD002–FD004?
2. Does RUL capping (125) help, and does the answer depend on the metric (capped vs true-RUL error)?
3. What window length and feature set matter?
4. How well do models generalise across operating conditions and fault modes (train FD001 -> test FD003, etc.)?
5. Are prediction intervals honest (empirical coverage vs nominal)?
6. Can warning thresholds give useful lead time at an acceptable false-alarm rate?

## Data pipeline

- Loader validates column count (26), dtypes, no NaN/inf, contiguous cycles from 1, one test RUL per test engine. Fails loudly on missing files.
- Train RUL label: `max_cycle - cycle`. Variants: raw and capped at 125 (cap configurable).
- Sensor analysis per subset: constant/near-constant sensors, monotonic trend strength, per-condition behaviour. Drop decisions come from training data only.
- Operating conditions (FD002/FD004): cluster the 3 settings into the discrete regimes on training data (KMeans, k chosen from data, expected 6) and **normalise each sensor per regime**. Global min-max on FD002/4 is a known failure mode.
- Windows: causal, length w (try 20/30/50), left-padded with first observed row plus a mask for short histories (FD004 has test engines with very few cycles). Scaler and clusterer fitted on training engines only; stored in the artifact and reused in inference.
- Tabular features for GBM/RF: last-window statistics (last value, mean, slope, std) over selected sensors, causal only.

## Models (each justified)

| Model | Reason it exists |
|---|---|
| Constant (train mean RUL) | Floor; any model must beat it |
| Linear / Ridge on last-cycle sensors | Shows how much is simple trend |
| Random Forest | Strong nonlinear baseline, minimal tuning |
| LightGBM | Usually best tabular; quantile objective for intervals |
| LSTM, GRU | Sequence models standard for C-MAPSS; compare on identical windows |
| Small Transformer encoder (optional, time-boxed) | Test whether attention helps; expected not to dominate at this data size. Reported regardless |

Tuning: small, budgeted random search on validation engines, same trial budget per model family. No test-set tuning. Seeds: 5 seeds for NN and GBM, report mean ± std. Compute: CPU first; Colab if NN ensembles become slow.

## Experiment ladder

1. FD001 baselines -> GBM -> GRU/LSTM, capped vs uncapped.
2. Window and feature ablations on FD001.
3. FD004 (and FD002/FD003) with per-regime normalisation vs without.
4. Cross-subset transfer (FD001->FD003, FD002->FD004, etc.); expect degradation and report it.
5. Risk thresholds sweep, then anomaly, uncertainty, explainability.

## Anomaly design

Unsupervised detectors (Isolation Forest, per-sensor z-score/CUSUM on regime-normalised data, optional autoencoder only if it beats the first two) fitted on early-life cycles of training engines. Two separate evaluations: (a) does anomaly score rise as engines approach failure (descriptive, no accuracy claim); (b) injected faults on held-out engines — spike, step drift, ramp drift, stuck sensor, noise burst — with documented magnitude, location and seed. Report precision/recall, detection delay, false-positive rate per magnitude.

## Uncertainty design

LightGBM quantile models (10/90), NN ensembles, then split-conformal calibration on validation engines. Check coverage on held-out engines and on the official test set, plus mean interval width. Caveat: engine-level exchangeability is assumed; coverage may fail under subset shift, which will be tested.
