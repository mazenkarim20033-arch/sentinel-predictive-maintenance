# Experiments

Source of truth: `experiments/runs/*.json` and the generated `reports/leaderboard_<subset>.md`. Numbers below are copied from that generated file; if they differ, the generated file wins.

## M3: FD001 baselines (hardware: i7-6820HQ, 2 cores, CPU only)

Protocol: grouped 5-fold CV by engine (preprocessing refit inside each fold, model seed 0), causal windows of 30 cycles, per-sensor last/mean/std/slope features, RUL target capped at 125. Final configurations were fixed before the run and evaluated once on the official FD001 test set (100 engines) after refit on all training engines with 5 model seeds. Ablations were never scored on the test set. No hyperparameter tuning was done; defaults are in `models/baselines.py`.

| Model | CV cut RMSE | Test RMSE | Test 95% CI | Test NASA score |
|---|---|---|---|---|
| Constant (train mean) | 38.99 | 43.07 | 38.6–47.4 | 33,629 |
| Ridge (last-cycle sensors) | 24.41 | 21.89 | 18.9–24.7 | 1,317 |
| Random Forest | 17.70 | 14.76 | 12.8–16.6 | 327 |
| LightGBM | 16.87 | 14.49 | 12.4–16.3 | 297 |

### What the results support

- Sensor history matters: every learned model beats the constant by a wide margin.
- LightGBM and Random Forest are statistically indistinguishable on this test set (CIs overlap almost completely; CV gap is within fold std). LightGBM is far cheaper: about 3 s vs 54 s to fit, 1.1 MB vs 45 MB, 1.4 ms vs 56 ms single-row latency.
- Capping matters under the test-like metric. Uncapped LightGBM has cut RMSE 25.9 and a very unstable score (397 ± 525 per point) because overestimates are penalised exponentially. Cap 125 (16.9) beat 100 (19.9) and 150 (19.3).
- The cycle number as a feature did not help (16.89 vs 16.87), so it is not acting as a useful shortcut here.
- A longer window helped in CV: 50 cycles gave 15.12 vs 16.87 for 30, and 15 cycles was worse (21.07).

### What they do not support

- Test RMSE is lower than CV cut RMSE, but the two use different protocols and the test set is only 100 points; do not read it as the model generalising better.
- The window=50 gain is a single-seed CV result and has not been checked on the test set; it should be adopted as a new fixed configuration and evaluated once, not retro-fitted onto the table above.
- Test numbers apply to simulated data, to FD001 only, with default hyperparameters.
- The NASA score constants (13 and 10) are the commonly cited ones; they were not found in the primary paper shipped with the data and remain unverified.

## M4: GRU / LSTM and window 50 (FD001, same protocol)

Declared before running: GRU and LSTM (hidden 64, 2 layers, dropout 0.2, Adam 1e-3, early stopping on 15% held-out training engines), windows 30 and 50, plus LightGBM with window 50. Each was evaluated once on the official test set with 5 seeds. Training curves: `reports/training_curves_FD001.png`.

| Model | CV cut RMSE | Test RMSE | Test 95% CI | Test NASA score | Fit s | Latency ms |
|---|---|---|---|---|---|---|
| LightGBM w50 | 15.12 | 13.75 | 11.6–15.6 | 296 | 2.8 | 1.4 |
| GRU w50 | 15.78 | 15.23 | 12.3–17.0 | 479 | 195 | 5.8 |
| LSTM w50 | 16.52 | 16.08 | 13.0–17.4 | 452 | 143 | 1.0 |
| LightGBM w30 | 16.87 | 14.49 | 12.4–16.3 | 297 | 2.9 | 1.4 |
| GRU w30 | 17.30 | 15.34 | 12.4–17.4 | 444 | 120 | 3.3 |
| LSTM w30 | 18.08 | 15.35 | 12.4–16.7 | 414 | 78 | 0.9 |

What this supports:
- With default settings and no tuning, the recurrent models did not beat gradient boosting on FD001. LightGBM is as good or better on every metric and about 50-70x cheaper to train.
- The 50-cycle window helped LightGBM and GRU in CV (about 1.7 and 1.5 RMSE points), but not LSTM on test.
- The NASA score is clearly worse for the recurrent models (about 410-480 vs 296), which suggests more over-estimates; this was not investigated.
- GRU and LSTM are not distinguishable from each other given the seed spread.

What it does not support:
- That deep learning is worse in general: hyperparameters were not tuned, only one architecture size was tried, and FD001 is small and low-noise. FD004 (six regimes, two fault modes) may favour sequence models; that is untested.
- Any ranking among models whose test CIs overlap (all of the above do).
- No Transformer was run; nothing here justifies the cost yet.

## M4b: FD004 (six operating regimes, two fault modes)

Same protocol as FD001, with the operating settings clustered into 6 regimes (fit on training engines only) and sensors normalised per regime. Configurations were declared before the run. No GRU or LSTM results exist for FD004: a GRU run was started and then stopped by decision (cost vs expected value), so nothing is reported for it. The deep-learning comparison therefore rests on FD001 only.

| Model | CV cut RMSE | Test RMSE | Test 95% CI | Test NASA score | Fit s |
|---|---|---|---|---|---|
| LightGBM w50 | 17.38 | 25.65 | 22.8–28.2 | 4,194 | 5.2 |
| LightGBM w30 | 19.22 | 26.45 | 23.5–29.2 | 4,683 | 5.2 |
| Random Forest | 19.48 | 26.97 | 24.0–29.7 | 5,086 | 263 |
| Ridge (last cycle) | 25.69 | 34.15 | 31.3–36.9 | 11,758 | 0 |
| Constant | 42.62 | 54.90 | 51.8–58.0 | 189,683 | 0 |
| *Ablation:* LightGBM w50, global normalisation | 23.66 | not evaluated | | | |

What this supports:
- Per-regime normalisation matters: removing it raised CV cut RMSE from 17.38 to 23.66.
- FD004 is much harder than FD001 for every model.
- LightGBM and Random Forest are statistically indistinguishable again; LightGBM is about 50x cheaper to fit and about 100x smaller.

What it does not support or explain:
- Test RMSE (25.65) is far above the development CV cut RMSE (17.38). Investigated in M6 (`reports/gap_diagnostics.md`): this is a difference in evaluation distribution, not a train/test shift. Re-sampling out-of-fold predictions to the test set's RUL distribution gives RMSE 24.5 (5th-95th percentile 22.1-26.9), which contains the test value; the asset-life distributions of train and test are the same (median 234 vs 235). 27% of FD004 test assets have a final RUL above the 125-cycle training cap, and that bin dominates the error (test RMSE 36, bias -27 cycles). The dev "cut" metric samples fewer such points, so it understates the official-test error.
- Whether sequence models do better on FD004 is untested.

Decision: LightGBM (window 50) is the default model going forward, chosen for equal-or-better accuracy at a fraction of the cost on the two subsets tested, not because deep learning was shown to be worse.

## M4.5: Second machine type, NASA Li-ion battery aging (generality demonstration)

Purpose: show the same pipeline (spec, loader adapter, per-asset splits, causal windows, tabular features, models, registry) runs on a non-engine machine with different signals. This is a **demonstration, not a benchmark**.

Data: `5. Battery Data Set.zip` from NASA PCoE (S3 mirror, 209,708,670 bytes; the catalog page lists no files and the dashlink landing page was unreachable). The archive holds 34 distinct cells, but capacity series are often unusable as run-to-failure: several cells start far below rated capacity (e.g. B0033 at 0.07 Ah), some contain zeros or spikes, and many stop before end of life. By NASA's own 1.4 Ah criterion only B0005, B0006 and B0018 reach end of life cleanly; B0007 bottoms out at about 1.40 Ah. I therefore used the four original cells (B0005, B0006, B0007, B0018; 24 C, 2 A discharge) with a **project-defined end of life of 1.5 Ah (25% fade)**, chosen after seeing the data so that all four qualify. This is a deviation from NASA's criterion. End of life is the first cycle where the trailing 3-cycle median capacity is at or below 1.5 Ah (so the second low reading marks it). Resulting lifetimes: 100, 77, 127 and 75 discharge cycles (379 labelled rows).

Features per discharge cycle: duration, mean voltage, time to 3.5 V, peak and rise in temperature, plus duration and peak temperature of the preceding charge. Capacity is excluded from the main variant (it defines the label) and added in a second variant. No operating-condition columns and one regime.

Protocol: leave-one-cell-out over 4 cells; windows of 10 cycles; fixed configurations, no tuning; random forest and LightGBM averaged over 5 seeds per fold. **No held-out test set exists**, so every number is a development estimate from 4 trajectories.

| Model (sensor-only variant) | RMSE, cycles (mean ± std over 4 cells) | Late-life RMSE (RUL <= 20) |
|---|---|---|
| Constant | 30.9 ± 6.5 | 39.4 |
| Age only (mean lifetime minus elapsed cycles) | 23.6 ± 11.1 | 16.9 |
| Ridge, last-cycle signals | 13.7 ± 3.6 | 6.6 |
| LightGBM | 15.7 ± 6.2 | 4.8 |
| Random Forest | 16.0 ± 5.3 | 5.8 |

What this supports:
- The pipeline ran unchanged on unrelated signals, with threshold-defined failure, per-cell splits and censored cells excluded and reported.
- Cycle-level signals carry information beyond elapsed time: all sensor models beat the constant and age-only baselines on average.
- Adding capacity as an input did not help (Ridge 13.5, LightGBM 16.8).

What it does not support:
- Any ranking among Ridge, LightGBM and Random Forest: the spread across cells (std 3.6–6.2) is as large as the gaps, and with 4 folds nothing is statistically separable. LightGBM is best in late life, Ridge best overall, and the age-only baseline is best on one cell (B0005, 6.8).
- Any claim about batteries in general: one chemistry, one temperature and load, four cells, a project-chosen threshold, and the cutoff voltage differs by cell, which the models could partly exploit.
- Generalisation across conditions: untested (other cell groups were not usable).

## M5: Failure warnings and anomaly detection (FD001 and FD004)

Generated reports: `reports/warning_FD00X.md`, `reports/warning_tradeoff_FD00X.png`, `reports/anomaly_FD00X.md`, `reports/anomaly_synthetic_FD00X.png`; run records under `experiments/runs/*-warning-*` and `*-anomaly-*`. Thresholds (normal > 30, warning 15-30, critical < 15) are the brief's experimental defaults, not maintenance standards. Predicted RUL is not a calibrated failure probability and nothing here evaluates calibration.

### Failure warnings (LightGBM window 50, fixed)

Sweep and lead time use out-of-fold predictions on full run-to-failure training trajectories (5-fold grouped CV). The official test set is used once, for the fixed default operating points at cycle level only, because test engines stop before failure and lead time cannot be measured there. Alert = predicted RUL at or below tau for k consecutive cycles.

| | tau = 30, k = 3 | tau = 15, k = 3 |
|---|---|---|
| FD001 out-of-fold: precision / recall | 0.97 / 0.83 | 0.97 / 0.79 |
| FD001: engines with a useful alert, median lead (cycles) | 100%, 25 | 100%, 12 |
| FD004 out-of-fold: precision / recall | 0.95 / 0.78 | 0.94 / 0.66 |
| FD004: engines with a useful alert, median lead | 97%, 24.5 | 93%, 11 (4% missed) |
| FD001 official test: precision / recall | 0.99 / 0.64 | 0.82 / 0.53 |
| FD004 official test: precision / recall | 0.85 / 0.53 | 0.85 / 0.29 |

What this supports:
- With the default thresholds the system gives useful advance warning on almost every held-out engine in development, with median lead roughly equal to tau and few early alarms (0-2%).
- Persistence is a clear trade-off: raising k from 1 to 5 raises precision by a few points but costs recall and lead time (FD004, tau = 10: recall 0.72 to 0.40, missed engines 4% to 24%).
- Errors are mostly between adjacent classes; on FD001 out-of-fold only 2 of 1,500 critical cycles were classed normal.

What it does not support:
- The cycle-level false-alarm rate (about 0.00-0.02) looks excellent mainly because most cycles are far from failure. The engine-level early-alarm share and the missed share are the more informative figures.
- Official-test recall is clearly lower than out-of-fold recall (FD004 tau = 30: 0.53 vs 0.78; FD004 tau = 15: 0.29 vs 0.66). The RMSE gap between test and development has been explained (an evaluation-distribution effect, see M4b notes and `reports/gap_diagnostics.md`), but the recall gap has not been separately verified. A plausible, untested reason is that test engines contain only the first part of their warning window, so the delay before an alert weighs more on those short segments. Test positives are few (for example 60 cycles at tau = 15 on FD001).
- Cycle-level recall also counts the lag before an alert fires, so lead time is the better operational measure.
- Which operating point to choose is a cost decision (missed failure versus false alarm) that this project cannot make.

### Anomaly detection

Two detectors: a statistical one (mean squared z-score against the healthy profile) and Isolation Forest, both fit on the first 30% of life of training engines; the alert threshold is the 99th percentile on a separate healthy calibration slice of training engines. An autoencoder was not tried because the simpler detector already matches or beats Isolation Forest (see below), so it was not justified.

A. Degradation analysis on real trajectories (descriptive only; C-MAPSS has no anomaly labels, so no accuracy is claimed). Scores rise as engines degrade: on FD001 the statistical score is 0.3x the threshold for RUL above 100 and 4.2x for RUL under 25 (mean Spearman with time 0.74). Every engine eventually produces a sustained alert, with median lead of 50 cycles (FD001, statistical), 51 (Isolation Forest), and 43 / 53 on FD004. A sustained alert already fired inside the healthy phase for 2-3% of engines, and the clean-phase false-positive rate is 1.5-2.5%, above the nominal 1% (the threshold was calibrated on different engines).

B. Controlled synthetic faults, kept separate (not real anomalies). One signal modified inside the healthy phase of held-out engines (10 injections per engine and setting; magnitudes 1, 2, 3, 5 healthy standard deviations; 15 cycles, spike 1 cycle). Event detection rate at 5 standard deviations, FD001 statistical detector: noise burst 0.99, step 0.93, ramp 0.65, spike 0.42, stuck sensor 0.13. On FD004 the same detector gives 0.91, 0.44, 0.28, 0.16 and 0.09. Isolation Forest is worse for large faults (FD001: noise burst 0.33, step 0.35).

What this supports:
- Large noise bursts and step shifts are detectable; small faults (2 standard deviations or less) and single-cycle spikes mostly are not.
- A stuck sensor is not detected at all: its rate (0.09-0.13) is the same as the false-alarm hit rate without any fault. A frozen but normal-looking value is invisible to a per-step detector; catching it needs a temporal feature such as a rolling variance.
- The simple statistical detector beats Isolation Forest on these faults, so complexity did not help.
- Detection is harder on FD004 (six regimes) than FD001.

What it does not support:
- Any claim about real faults: injected faults are single-signal, applied in normalised space and scaled by healthy variance.
- Detection rates include chance hits from the 1.5-2.5% per-step false-positive rate, so compare them to the stuck-sensor rows.
- Detection delay and precision are reported per setting in the generated report; detectors have no temporal model.

## M6: Uncertainty and explainability (FD001 and FD004, LightGBM window 50)

Reports: `reports/uncertainty_FD00X.md/.png`, `reports/explain_FD00X.md/.png`, `reports/gap_diagnostics.md`.

### Prediction intervals

Intervals are for the capped target (RUL above 125 is reported as 125+). Three methods: raw quantile regression (LightGBM, 10/90 and 5/95), conformalised quantile regression (CQR) and split conformal on the point model. Calibration uses 20% of training assets held out from fitting; development coverage is out-of-fold over 5 grouped folds; the official test set was evaluated once.

| nominal | method | FD001 dev | FD001 test (n=100, 95% CI) | FD004 dev | FD004 test (n=248, 95% CI) |
|---|---|---|---|---|---|
| 80% | raw quantile | 0.50 | 0.61 (0.51-0.70) | 0.62 | 0.63 (0.57-0.69) |
| 80% | CQR | 0.81 | 0.78 (0.69-0.85) | 0.80 | 0.76 (0.71-0.81) |
| 80% | split conformal | 0.80 | 0.71 (0.61-0.79) | 0.80 | 0.76 (0.70-0.81) |
| 90% | raw quantile | 0.64 | 0.72 (0.63-0.80) | 0.74 | 0.76 (0.70-0.81) |
| 90% | CQR | 0.91 | 0.85 (0.77-0.91) | 0.89 | 0.88 (0.84-0.92) |
| 90% | split conformal | 0.90 | 0.79 (0.70-0.86) | 0.89 | 0.89 (0.84-0.92) |

Mean width of the 80% / 90% intervals in development: FD001 CQR 23.8 / 35.9 cycles, conformal 21.6 / 30.6; FD004 CQR 22 / 32, conformal 21 / 31.

What this supports:
- Raw quantile regression is badly overconfident: its nominal 80% interval covered 50-62% of held-out points. Reporting those intervals as-is would mislead.
- Conformal calibration restores the nominal coverage on average in development (0.80-0.91 for both methods) and CQR was statistically consistent with nominal on all four official-test checks.
- Split conformal under-covered on the FD001 test set (80%: 0.71, CI upper bound 0.79; 90%: 0.79, upper bound 0.86).

What it does not support:
- Coverage at every RUL value. Conformal coverage is marginal. In development, mid-life bins are under-covered while the ends are over-covered: FD001 80% conformal covers 0.99 of points in RUL 0-25 but 0.51 in 75-100 (CQR 0.93 and 0.65); FD004 conformal 0.47 in 75-100. Any interface showing these intervals must say so.
- Coverage of the true (uncapped) RUL. Because the interval cannot exceed 125, coverage of the official uncapped RUL is much lower (FD004 CQR 80%: 0.51; FD001: 0.68).
- Coverage under distribution shift: assets from another machine type or subset were not tested, and the exchangeability assumption is only partly checked by the test results above. One seed and one fold assignment.

### Explainability (TreeSHAP on out-of-fold assets)

Exact TreeSHAP from LightGBM, aggregated from the last, mean, std and slope features to one value per signal. Additivity error is 4.5e-13 and the values equal the `shap` library's TreeExplainer exactly (max difference 0.0 on 500 rows).

- Top signals on FD001: s11, s4, s14, s12, s3, s9. On FD004: s11 (dominant, mean |SHAP| 14.5), s14, s4, s9, s17, s3.
- Slope features carry the most attribution (FD001 14.9 vs 7.5-9.6 for last/mean/std; FD004 17.0 vs 6.0-7.4): the model relies on recent trends more than on current levels.
- Mean signed attribution by true RUL: for the top signals it is positive (+2 to +5 cycles) early in life and strongly negative (-6 to -15) in the last 25 cycles on FD001, consistent with steady degradation trends.
- Models agree on which signals matter but not exactly: rank correlation of signal importance between LightGBM SHAP, LightGBM permutation and Random Forest permutation is 0.67-0.89 on FD001 and 0.91-0.94 on FD004, while ridge coefficients agree poorly (0.04-0.34 on FD001, 0.64-0.68 on FD004). Importance is a property of the model, not of the machine.
- s14 is 3rd on FD001 by SHAP although it was not in the top 5 by simple correlation with RUL (M2 analysis): attribution is conditional on the other signals and on non-linear effects.

What it does not support:
- Causal statements. These are associations learned by one model; strongly correlated signals (for example s11, s4, s12) share credit unpredictably, so individual rankings among them should not be over-read.
- Permutation and cross-model comparison used one fold, 4,000 rows and 3 repeats; SHAP used all folds.

### Test-versus-development gap (resolved)

See the M4b notes: the larger official-test RMSE is an evaluation-distribution effect (27% of FD004 test assets have final RUL above the 125 cap), not a train/test shift. The lower recall of warnings on the test set has not been separately verified.

### Notes

- NASA's own `readme.txt` lists FD004 as 248 train / 249 test engines; the files contain 249 train / 248 test. This project uses the files.
- Failed or unhelpful variants (uncapped, cap 100, window 15, cycle feature) are kept in the registry and the leaderboard.
