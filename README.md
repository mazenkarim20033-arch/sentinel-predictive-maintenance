# SENTINEL — Predictive Maintenance & Failure Intelligence Platform

**Status: Milestone 1 (research and planning). No models have been trained and no results exist yet.**

SENTINEL predicts the Remaining Useful Life (RUL) of turbofan engines from sensor histories, flags failure risk, detects abnormal sensor behaviour, and explains and bounds its predictions. The research pipeline comes first; the application is a thin layer on top of it.

## Data

NASA C-MAPSS turbofan degradation dataset (FD001–FD004), Saxena et al., 2008. It is **simulated** data from a physics-based engine model, not measurements from real engines. Every number shown anywhere in this project is labelled as one of: measured/simulated dataset value, model prediction, ground truth, synthetic injected anomaly, or illustrative.

Obtain it from the NASA open data catalog: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data (file `CMAPSSData.zip`). Place the extracted `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt` files in `data/raw/`. The loader will refuse to run on anything else and will never substitute generated data silently.

## Documents

| Doc | Purpose |
|---|---|
| [PROJECT_OVERVIEW](docs/PROJECT_OVERVIEW.md) | Problem, formulation, scope |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Components, folder structure, tooling decisions |
| [RESEARCH_PLAN](docs/RESEARCH_PLAN.md) | Models, experiments, data pipeline |
| [EVALUATION_PLAN](docs/EVALUATION_PLAN.md) | Splits, metrics, leakage controls |
| [ROADMAP](docs/ROADMAP.md) | Milestones with exit criteria |
| [RISK_REGISTER](docs/RISK_REGISTER.md) | Risks, assumptions, open decisions |

## Results

Only FD001 baselines exist (Milestone 3): see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and the generated `reports/leaderboard_FD001.md`. On the official FD001 test set (100 simulated engines), LightGBM reaches RMSE 14.5 (95% CI 12.4–16.3) against 43.1 for the constant baseline; Random Forest is statistically indistinguishable. On the harder FD004 subset (6 operating regimes), LightGBM reaches test RMSE 25.7 (95% CI 22.8–28.2); GRU/LSTM were run on FD001 only and did not beat LightGBM there. LightGBM (window 50) is the default model. Failure warnings (normal/warning/critical from predicted RUL) and anomaly detection are evaluated in the same document, with synthetic-fault results kept separate from real-trajectory analysis. Uncertainty intervals and explainability are not implemented yet.

## Setup

Python 3.12, CPU on Windows (Colab for any GPU work).

```bash
python -m venv .venv --system-site-packages
.venv\Scripts\python.exe -m pip install pytest lightgbm pyyaml pandas scikit-learn pydantic matplotlib
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\analyze_sensors.py
```

Extract the NASA zip to `data/raw/extracted/`. Tests marked `realdata` skip automatically if the files are missing.

Machines are described by a `DatasetSpec` in `configs/` (asset id, time, signal and condition columns, regimes, label rule); the shared pipeline reads columns only from that spec. Adapters exist for C-MAPSS and for NASA Li-ion battery aging. The battery result is a small demonstration (4 cells, leave-one-cell-out, no held-out test, project-defined end of life), not a benchmark; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Implemented so far (Milestones 2–4.5 part 1)

Loader with schema validation, RUL labels, engine-level splits, per-regime preprocessing fitted on training data only, causal windows, sensor and regime analysis (`reports/data_analysis/`), tabular features, metrics with bootstrap CIs, a file-based experiment registry, FD001 baselines, and 36 automated tests.

```bash
set PYTHONPATH=src
.venv\Scripts\python.exe scripts\run_baselines.py
.venv\Scripts\python.exe scripts\make_leaderboard.py
```
