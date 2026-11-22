# Architecture

Principle: the research pipeline is a Python package that produces versioned artifacts and a results registry. The API only loads artifacts and calls the same inference code. The frontend only calls the API.

```
data/raw (NASA files)
   -> sentinel.data   (load, validate, RUL labels, engine-level splits)
   -> sentinel.features (causal windows, scaler fit on train only)
   -> sentinel.models   (baselines, GBM, LSTM/GRU, optional Transformer)
   -> sentinel.eval     (metrics, risk, anomaly, uncertainty, SHAP)
   -> experiments/registry + artifacts/
                         \
                          -> sentinel.inference (one predict path, streaming-safe)
                                -> FastAPI -> React
```

## Components

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Already installed (3.12.9); torch 2.13, sklearn, pandas, pydantic present |
| Classical ML | scikit-learn, LightGBM (preferred over XGBoost) | Faster on 2 cores, native quantile objective |
| Deep learning | PyTorch, CPU by default, `--device` flag | No local NVIDIA GPU; Colab for ensembles |
| Explainability | SHAP (TreeExplainer; permutation/gradient for NN) | |
| Experiments | File registry (JSON per run + `leaderboard.parquet`), seeds and configs recorded | Light; MLflow optional later |
| Config | Pydantic models, YAML files | Validated, reproducible |
| API | FastAPI + Pydantic | |
| Frontend | React + TypeScript + Vite, Recharts | Only after Milestone 7 |
| Hosting | Static snapshot on GitHub Pages (as with Procura AI) with precomputed replay; live API only locally | Free; Render-style free tiers sleep and are fragile |

No Docker. Windows-local first.

## Machine-agnostic design (requirement added after Milestone 3)

SENTINEL is a template that can be pointed at any machine with time-series condition data, not an engine-specific tool. C-MAPSS is one *adapter*; the rest of the pipeline must not mention engines, cycles of a turbofan, or the 21 NASA sensors.

A `DatasetSpec` (Pydantic, YAML) describes any dataset:

| Field | Meaning | C-MAPSS example | Electric motor / battery example |
|---|---|---|---|
| `asset_id` | column identifying one machine/unit | `engine` | `motor_id`, `cell_id` |
| `time` | ordering column and its unit | `cycle` | timestamp, charge cycle |
| `signals` | measured columns | `s1..s21` | current, vibration, winding temperature, capacity |
| `conditions` | operating context columns (optional) | `setting_1..3` | load, speed, ambient temp |
| `label` | how the target is obtained | run-to-failure -> RUL = last - t | failure timestamps, or capacity threshold (e.g. 80%) |
| `censoring` | assets that did not fail | test engines | machines still running |
| `horizon_cap`, `window` | modelling defaults | 125, 30 | set per dataset by validation |

Pipeline stages read only the spec: loader/validator -> label builder -> regime detection (skipped if no conditions) -> per-regime normalisation -> windows -> models -> evaluation. Models never see column names, only arrays.

What generality does and does not mean (to keep claims honest):

1. The **pipeline and methodology** transfer. A **trained model does not**: each machine type needs its own training on its own data. There is no zero-shot "works on any machine".
2. Supervised RUL needs run-to-failure (or degradation-to-threshold) histories. Machines with few or no failures get the anomaly/health-index path (unsupervised, no accuracy claim) or survival-style methods with censoring, not a fabricated RUL.
3. Sampling rate, units and signal quality differ; the spec carries units and the validator checks ranges, gaps and irregular sampling instead of assuming C-MAPSS-style clean cycles.
4. Raw high-frequency signals (vibration waveforms) need a feature-extraction adapter (RMS, kurtosis, spectral bands) before the shared pipeline. That adapter is per signal type.
5. Generality is claimed only after it is demonstrated on at least one non-engine dataset, with its own reported results.

## Inference contract

`predict(history: engine history up to cycle t) -> {rul, interval, risk_class, anomaly_score}`. The function receives rows with cycle <= t only; ground-truth RUL is not an argument anywhere. The replay endpoint enforces this by slicing server-side, and a test asserts that outputs at cycle t are bit-identical when rows after t are altered.

## API surface (planned)

`GET /fleet`, `GET /engines/{id}`, `POST /predict/rul`, `POST /predict/risk`, `POST /predict/anomaly`, `GET /replay/{id}?cycle=t&mode=blind|eval`, `GET /models`, `GET /evaluation`. Every response carries a `provenance` field: `dataset | prediction | ground_truth | synthetic | illustrative`.

## Folder structure

```
sentinel/
  README.md
  pyproject.toml
  docs/                 (plan docs now; DATASET_CARD, EXPERIMENTS, MODEL_CARD, LIMITATIONS later)
  data/raw/             (gitignored; NASA files)
  data/processed/       (gitignored)
  configs/              (YAML experiment configs)
  src/sentinel/
    data/ features/ models/ eval/ anomaly/ uncertainty/ explain/ inference/ registry/
    api/
  scripts/              (train.py, evaluate.py, make_report.py, export_snapshot.py)
  notebooks/            (EDA, Colab training)
  experiments/          (run records, committed; small)
  artifacts/            (model files, gitignored or Git LFS-free small subset)
  reports/              (generated leaderboard, plots)
  tests/                (data, ml, api, integration)
  web/                  (React app, Milestone 9)
```
