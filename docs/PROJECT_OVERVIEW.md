# Project Overview

## Problem

Given the sensor history of an engine up to operating cycle *t*, estimate how many cycles remain before failure, decide whether that is cause for a warning, flag abnormal behaviour, and say how uncertain the estimate is.

## Dataset facts (to be verified against the downloaded files)

C-MAPSS: four subsets, each with run-to-failure training engines and truncated test engines plus one true RUL per test engine. Each row: engine id, cycle, 3 operating settings, 21 sensors (26 columns, space separated, no header).

| Subset | Operating conditions | Fault modes | Train / test engines |
|---|---|---|---|
| FD001 | 1 | 1 (HPC degradation) | 100 / 100 |
| FD002 | 6 | 1 | 260 / 259 |
| FD003 | 1 | 2 (HPC + fan) | 100 / 100 |
| FD004 | 6 | 2 | 249 / 248 |

**Verified on the downloaded files (2026-11-22, `CMAPSSData.zip` from data.nasa.gov, 26 columns, no NaN in any subset):**

| Subset | Train rows | Test rows | Train life min/median/max | Shortest test history | Constant sensors (train) |
|---|---|---|---|---|---|
| FD001 | 20,631 | 13,096 | 128 / 199 / 362 | 31 | 1, 5, 10, 16, 18, 19 |
| FD002 | 53,759 | 33,991 | 128 / 199 / 378 | 21 | none |
| FD003 | 24,720 | 16,596 | 145 / 220 / 525 | 38 | 1, 5, 16, 18, 19 |
| FD004 | 61,249 | 41,214 | 128 / 234 / 543 | 19 | none |

Engine counts match the table above, and each RUL file has one value per test engine. Sensor 6 is not exactly constant in FD001 in this check, so the commonly cited list from papers differs slightly; sensor decisions must come from the data per subset (Milestone 2). Rounding the operating settings to 2 decimals yields 3 distinct combinations in FD001/FD003 and 11 in FD002/FD004, so regimes have to be recovered by clustering, not by counting unique values.

Important structural facts that shape the design:

1. The data are simulated. Claims must say "on simulated turbofan data".
2. The official test set gives **one RUL per test engine** (at its last observed cycle). Final benchmark metrics are therefore computed on ~100 points for FD001. That is noisy; report bootstrap confidence intervals.
3. Because the test truncation point and its RUL are known, true RUL at every earlier test cycle is `RUL_T + (T - t)`. This enables honest replay on test engines.
4. Test engines never reach failure, training engines do, and training engines are in the training data. A replay on a training engine only shows memorisation. Replay must use validation-held-out or official test engines.

**Sensor and regime analysis (training data only, `scripts/analyze_sensors.py`):** k-means silhouette on standardised settings is 0.997 at k=6 for FD002 and FD004 (next best 0.94), so six regimes are clearly recoverable. FD001/FD003 have no cluster structure (silhouette about 0.36), so one regime is used. After per-regime normalisation the informative sensors are 15 (FD001), 17 (FD002), 16 (FD003), 17 (FD004). The strongest Spearman correlations with RUL are around 0.7 in magnitude (s11, s4, s12, s7, s15 in FD001). These are associations measured on training engines, not causal claims.

## ML formulation

- **RUL regression** (primary): `f(x_{t-w+1..t}) -> RUL_t`, causal windows only.
- **Failure warning**: derived from predicted RUL via thresholds (default >30 normal, 15–30 warning, <15 critical). Operational definition for experiments, not a maintenance standard. Also evaluate a direct classifier only if the derived approach is clearly weak.
- **Anomaly detection**: unsupervised on healthy-phase data; evaluated only on (a) degradation-trajectory analysis and (b) injected synthetic faults. No accuracy claim on "real" anomalies because none are labelled.
- **Uncertainty**: quantile regression and ensembles, calibrated and checked with conformal prediction on held-out engines.
- **Explainability**: SHAP for tree models; gradient-based or permutation methods for sequence models. Associations, not causes.

## Scope decisions (challenging the brief)

- **SQLite and a database**: the fleet is a static dataset. Start with Parquet/JSON artifacts read by FastAPI. Add SQLite only if a concrete need appears.
- **Transformer**: on ~20k–60k rows and 2 CPU cores it is unlikely to beat a GRU and costs time. Plan it as an optional, time-boxed experiment after LSTM/GRU, reported even if it loses.
- **MLflow**: adds disk/memory overhead on a 10 GB-free, 8 GB-RAM machine. Start with a simple file-based registry with the same fields; MLflow-compatible logging can be added later if wanted.
- **"Calibrated failure probability"**: not claimed unless reliability is measured.
