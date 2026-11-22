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
- Test RMSE (25.65) is far above CV cut RMSE (17.38), and Ridge shows the same gap. Cause not investigated (candidates: how test trajectories are truncated, train/test shift, the CV cut protocol being easier). Until explained, treat absolute FD004 numbers with caution.
- Whether sequence models do better on FD004 is untested.

Decision: LightGBM (window 50) is the default model going forward, chosen for equal-or-better accuracy at a fraction of the cost on the two subsets tested, not because deep learning was shown to be worse.

### Notes

- NASA's own `readme.txt` lists FD004 as 248 train / 249 test engines; the files contain 249 train / 248 test. This project uses the files.
- Failed or unhelpful variants (uncapped, cap 100, window 15, cycle feature) are kept in the registry and the leaderboard.
