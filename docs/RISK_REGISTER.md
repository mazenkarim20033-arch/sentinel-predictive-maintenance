# Risk Register

| # | Risk / mistake | Impact | Mitigation |
|---|---|---|---|
| 1 | Row-level splitting or fitting scalers on all data | Inflated results | Engine-level splits, fit-on-train pipeline objects, leakage tests |
| 2 | Tuning on the official test set | Optimistic benchmark | Test touched only for final reports; run log records when |
| 3 | Test set has one point per engine (~100) | Noisy ranking | Bootstrap CIs, plus rolling validation metrics; call ties ties |
| 4 | Capped target distorts metrics | Misleading comparison | Report capped and true-RUL error; label clearly |
| 5 | Global normalisation on FD002/FD004 | Poor results that look like "model weakness" | Per-regime normalisation, ablated |
| 6 | Replay on training engines | Memorisation shown as skill | Replay only on held-out/test engines, enforced in API |
| 7 | Future rows reaching the model (centered smoothing, bidirectional features) | Invalid replay | Causal-only features; perturbation test |
| 8 | Cycle number acting as a shortcut | Model learns age, not health | Ablation with and without |
| 9 | Treating SHAP as causal | Wrong claims | Wording rule in docs and UI |
| 10 | Anomaly "accuracy" on unlabelled data | Fabricated claim | Only injected-fault metrics; degradation analysis is descriptive |
| 11 | Intervals under-cover under subset shift | Overconfident UI | Test coverage across subsets; show widths |
| 12 | Simulated data presented as real | Credibility | Provenance labels everywhere; "simulated turbofan data" wording |
| 13 | Weak hardware (i7-6820HQ, 2 cores, 7.9 GB RAM, ~10 GB disk, no NVIDIA GPU) | Slow NN tuning, disk pressure | Small models, capped trial budgets, Colab for ensembles, keep artifacts small, avoid heavyweight tooling |
| 14 | Scope creep (SQLite, MLflow, Transformer, autoencoder, Docker) | Unfinished flagship | Optional items time-boxed; each must justify itself |
| 15 | Deep nets not beating GBM | Disappointing story | Report it; that is a legitimate finding |
| 16 | Dataset URL or format changes | Blocked pipeline | Loader accepts the official files from any path; docs give source |
| 17 | Package gaps: lightgbm, shap, fastapi, pytest not installed | Setup friction | Project venv with pinned `pyproject.toml` |

| 18 | "Works on any machine" overclaim | Misleading, not defensible in an interview | Claim a configurable pipeline, not a universal model; per-machine training; prove on a second dataset |
| 19 | Target machines lack failure data | No supervised RUL possible | Anomaly/health-index mode, no accuracy claim; survival methods only if censoring is handled correctly |
| 20 | Abstraction built before a second dataset exists | Wrong abstraction | Design `DatasetSpec` from C-MAPSS plus one concrete second dataset, then refactor; avoid speculative options |
| 21 | Hard-coded C-MAPSS constants (column names, 21 sensors, 1..N ids, 125 cap) in current code | Blocks generality | Refactor in Milestone 4.5, with regression tests that old results reproduce |

## Assumptions

- The NASA files are unchanged from the public release; counts in PROJECT_OVERVIEW are checked on load.
- The NASA score formula as written in EVALUATION_PLAN matches Saxena et al. 2008 (to verify).
- Warning thresholds 30/15 are experimental defaults.

## Open decisions

- Window length, RUL cap value, hold-out fraction (settled by experiment).
- Whether to use public hosting beyond a static snapshot.
- Whether to add MLflow later.
