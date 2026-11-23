# Interview guide: defending SENTINEL

Short, honest answers to the questions a reviewer is likely to ask, with where the evidence is. Rule of thumb: state the claim, the evidence, and the limit.

**Why split by engine instead of by row?** Rows from one engine are strongly correlated; a random row split puts neighbouring cycles of the same engine on both sides and inflates scores. Every split here is by engine id (tests assert no engine appears on both sides). Alternative: time-based splits, which do not apply to run-to-failure fleets.

**How do you know there is no data leakage?** Preprocessing (regime clustering, normalisation, sensor selection) is fitted on training engines only and applied row by row; windows are causal; tests alter later rows and check earlier predictions do not change; the official test set was evaluated once per final configuration, ablations only on development folds. Limit: hyperparameters were not tuned at all, which also removes the main route for test leakage.

**Why cap the RUL at 125?** Early in life degradation is invisible in the sensors, so predicting 300 versus 200 cycles is noise. Capping is standard and it measurably helped (uncapped LightGBM: development error 25.9 versus 16.9). Cost: predictions cannot exceed 125, which dominates the error on test engines with more life left (27% of FD004) and means intervals are for the capped target. Evidence: docs/EXPERIMENTS.md, reports/gap_diagnostics.md.

**Why LightGBM and not deep learning?** On every subset tested it matched or beat GRU/LSTM within the confidence intervals while training in seconds instead of minutes and being easy to explain with SHAP. I did not claim deep learning is worse in general: no tuning, one network size, FD001 only. I stopped GRU/LSTM on FD004 as a cost decision and said so.

**What does "indistinguishable within the confidence interval" mean?** The official test sets have 100 to 259 engines, so a bootstrap interval around RMSE spans several cycles; two models whose intervals overlap heavily cannot be ranked from this data. I report ties as ties and choose among tied models on cost and explainability.

**Why does the test error exceed the cross-validation error?** It looked like shift, but resampling held-out predictions to the test set's RUL distribution reproduces the test error (FD004: 24.5 with a 22.1-26.9 range versus 25.6), and train and test lifetimes match. The test set has more high-RUL engines, where a capped model is forced to be wrong. My development metric had understated this.

**What is conformal prediction, and did it work?** It adjusts an interval using errors on held-out calibration engines so that, on average, the stated share of true values falls inside. It fixed the badly over-confident raw quantile intervals (80% nominal covered 50-62% before and 71-81% after, depending on method and dataset; split conformal was the weakest on the FD001 test set). It does not give coverage at every RUL: mid-life coverage was only 47-65% in places. I show that, and the interface warns about it.

**Is the risk level a failure probability?** No. It is a threshold on predicted RUL (30 and 15 cycles are working levels). Calibration was not evaluated, so I do not call it a probability.

**How do you know the replay is honest?** The server cuts the history before the model runs; tests change later rows and check the prediction is identical; a test spies on the model input to confirm the truth column never reaches it; blind mode returns no truth at all; replay serves only held-out engines. The public demo plays back recordings, which equals live computation because predictions depend only on past cycles (checked against the live API).

**What does SHAP show and not show?** How the model's output moves with each signal's recent level, trend and variability. It does not show physical causes, correlated signals share credit, and rankings differ between models (ridge disagreed with the tree models).

**How general is it really?** The pipeline is config-driven and ran on a battery dataset with different signals, a threshold failure rule and censored units. Trained models are not general: single-condition models failed on six-condition data (RMSE 56-61 versus a 54-55 constant guess). The battery result is four cells with no test set, so it demonstrates the plumbing, not battery performance.

**What was the hardest bug?** An extreme sensor value (1e308) overflowed the anomaly score to infinity and crashed the API response. Found by hostile-input testing; fixed by rejecting out-of-range inputs and saturating scores.

**What would you change?** Tune and test the deep models properly on the hard subsets; use a real, larger run-to-failure dataset; add temporal anomaly features; model censoring; put real costs on false alarms and misses before fixing a warning threshold.

**What is not production-ready?** Everything operational: no authentication, monitoring, drift detection, retraining pipeline or security review; one laptop CPU benchmark; simulated data.
