# SENTINEL case study

A predictive-maintenance platform built to show machine-learning engineering practice, not just a model. It estimates how many operating cycles a machine has left, raises warnings, flags unusual sensor behaviour, says how uncertain it is, and explains itself, on NASA's simulated C-MAPSS turbofan data. Everything below is backed by experiment records in the repository; where a result is weak or untested, it says so.

## The problem
A maintenance team wants to service a machine neither too early (wasted life) nor too late (failure). That needs a remaining-useful-life (RUL) estimate that is accurate when it matters, honest about its uncertainty, and checkable. The most common way such projects go wrong is evaluation: leakage between train and test, tuning on the test set, single lucky numbers, claims about data the model never saw. This project treats evaluation as the main deliverable.

## What I built
- **A data and modelling pipeline** with engine-level splits, per-operating-regime normalisation fitted on training data only, causal windows, and a file-based experiment registry that records config, seeds, splits, metrics, hardware and timings for every run.
- **Models:** constant, Ridge, Random Forest, LightGBM, GRU and LSTM, compared on the same splits; the report is generated from the records.
- **Failure intelligence:** warning levels with lead-time analysis, a statistical anomaly detector benchmarked on injected faults, prediction intervals (quantile, conformalised quantile, split conformal) with measured coverage, and exact SHAP explanations.
- **A serving layer and web app:** a FastAPI service and a React interface with a replay lab in which the model sees only the cycles up to now, a blind mode that hides the truth, and a label on every figure saying whether it is a prediction, ground truth, dataset signal or recorded benchmark.
- **A machine-agnostic design:** machines are described by a config; the same pipeline ran on a second, non-engine dataset (lithium-ion batteries) as a small demonstration.

## What the results say (official test sets, 100 to 259 engines each)
| Dataset | LightGBM test RMSE (cycles) | 95% CI | Constant guess |
|---|---|---|---|
| FD001 (1 condition, 1 fault) | 13.75 | 11.6-15.6 | 43.1 |
| FD002 (6 conditions, 1 fault) | 24.50 | 21.3-27.3 | 54.1 |
| FD003 (1 condition, 2 faults) | 13.88 | 11.1-16.4 | 45.1 |
| FD004 (6 conditions, 2 faults) | 25.65 | 22.8-28.2 | 54.9 |

- Operating conditions, not fault modes, make the problem hard. Per-regime normalisation is what makes the six-condition sets workable (FD002: 23.3 without it, 17.2 with it, in development CV).
- LightGBM, Random Forest and the recurrent networks are statistically indistinguishable on these test sets. LightGBM is the default because it is as good and about 40 to 70 times cheaper to train than the recurrent networks (about 20 times cheaper than Random Forest on FD001), not because the others were shown to be worse.
- Warnings are useful in development (FD001, 30-cycle level: precision 0.97, recall 0.83, median 25 cycles of notice), with lower recall on the test set.
- Raw quantile intervals badly under-cover (nominal 80% covered 50-62%). Conformal calibration restores the average coverage, but coverage is uneven: weakest in mid-life.
- A stuck sensor is invisible to the anomaly detector; large noise bursts and level shifts are caught.
- Models trained on single-condition data fail on multi-condition data (RMSE 56-61, worse than a constant guess): a model does not transfer to different operating conditions without retraining.

## What went wrong, and what I learned
- **A big test-versus-development gap looked like leakage or shift.** Diagnosing it showed neither: the official test set simply has more high-RUL points, which a model trained on a 125-cycle cap cannot reach. My development metric had understated the test error. Lesson: build the development metric to look like the test distribution, and diagnose gaps before explaining them.
- **I designed a "universal" ambition and then scoped it honestly.** The pipeline generalises; trained models do not. The battery dataset turned out to be mostly unusable as run-to-failure data (glitched capacities, early stops), leaving four cells and a threshold I had to choose; I reported it as a demonstration.
- **Testing the whole stack found bugs unit tests missed:** an extreme sensor value crashed the API, the first dashboard load took 25 s, and a real-browser audit found contrast and naming failures. All were fixed and are recorded in docs/TESTING.md.
- **Several of my own tests were wrong** (a hand-computed count, an assertion that could never fail, a benchmark that measured the wrong process). Each was caught by checking and fixed in the open.

## What this does not show
Simulated data only; no real machine. No tuning of models; deep learning tried on one subset. Intervals are calibrated on average, not everywhere. Anomaly accuracy is known only for injected faults. SHAP explains the model, not the machine. No security, monitoring or drift handling. Details: docs/LIMITATIONS.md.

## What I would do next
Run the same pipeline on a real run-to-failure dataset with more units (for example a bearing data set, which needs a signal-feature step); tune and test sequence models on the harder subsets; add survival-style handling for machines that have not failed; evaluate warning thresholds with an explicit cost of a miss versus a false alarm; add temporal anomaly features so stuck sensors are detectable.

## Where to look
Results and caveats: docs/EXPERIMENTS.md and the generated reports/REPORT.md. Model and data: docs/MODEL_CARD.md, docs/DATASET_CARD.md. Honest gaps: docs/LIMITATIONS.md and docs/TESTING.md. How to defend each decision: docs/INTERVIEW_GUIDE.md.
