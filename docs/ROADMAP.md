# Roadmap

Each milestone ends with: results shown, docs updated, tests passing, your approval before the next. Time estimates are rough part-time guesses.

| # | Milestone | Exit criteria | Est. |
|---|---|---|---|
| 1 | Research and architecture | These docs reviewed and agreed; repo created; dataset obtained | 2–3 d |
| 2 | Data engineering | Loader + schema validation, sensor analysis, RUL labels, engine-level splits, causal windows, preprocessing tests green | 4–5 d |
| 3 | Baselines | Constant, Ridge, RF, LightGBM on FD001 with registry and 5 seeds; first honest leaderboard | 4 d |
| 4 | Deep learning | GRU/LSTM on identical splits; capped vs uncapped; Transformer experiment time-boxed | 5–7 d |
| 4.5 | Generalisation (new; done: spec, loader, preprocessing, exact reproduction of FD001/FD004 results, and a small NASA battery demonstration with 4 cells and no test set) | `DatasetSpec` + adapter interface; C-MAPSS loader/preprocessing/runners refactored to read only the spec; all existing results reproduce unchanged; a second, non-engine dataset (e.g. battery aging or bearings) runs end to end with its own honest results | 4–6 d |
| 5 | Failure intelligence (done: warning thresholds, lead time, two anomaly detectors, synthetic-fault benchmark; no autoencoder) | Threshold sweep, lead-time evaluation, anomaly detectors, injected-fault benchmark | 4–5 d |
| 6 | Explainability and uncertainty (done: TreeSHAP, cross-model comparison, quantile/CQR/conformal intervals with coverage, test-gap diagnosis) | SHAP global/local, quantile + conformal intervals, coverage evaluation | 4–5 d |
| 7 | Evaluation framework (done: FD002/FD003, cross-subset transfer, generated REPORT.md, figures, dataset/model cards, limitations) | FD002–FD004 and transfer runs, generated report, EXPERIMENTS/MODEL_CARD/LIMITATIONS drafts | 5 d |
| 8 | Backend (done: serving bundle, FastAPI endpoints, replay with blind/eval modes, provenance labels, 40 API/inference tests; see docs/API.md) | FastAPI with replay (blind/eval), provenance field, API tests | 4 d |
| 9 | Frontend (done: six pages, provenance labels, replay lab, 19 frontend tests, checked in a browser at desktop and phone width; see web/README.md) | Dashboard, fleet, engine detail, simulation lab, model lab | 7–10 d |
| 10 | Integration and quality (done: hostile-input hardening, live HTTP and full-stack tests, benchmark and optimisation, browser accessibility audit; see docs/TESTING.md and reports/performance.md) | End-to-end test, latency measurement, error handling | 3 d |
| 11 | Deployment and write-up (done locally: setup verified from a clean checkout, static recorded-playback demo with a GitHub Pages workflow, screenshots, README, case study and interview guide; published: public repository and GitHub Pages demo, verified live) | Static snapshot demo, screenshots, case study, repo cleanup | 3–4 d |

Frontend work did not begin before Milestone 7 was approved. Decisions taken at the first gate: file-based registry instead of MLflow, LightGBM instead of XGBoost, SQLite deferred, Transformer optional.
