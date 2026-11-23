# API

FastAPI service in `src/sentinel/api/`; inference in `src/sentinel/inference/bundle.py`. Research demonstration, not production.

```bash
python scripts/build_bundle.py                      # builds artifacts/bundles/FD00X.joblib (git-ignored)
uvicorn sentinel.api.app:app --port 8000            # needs PYTHONPATH=src or an editable install
```

Interactive docs: `http://localhost:8000/docs`. CORS allows `http://localhost:5173` (the planned Vite dev server) only.

## Rules the API enforces
1. **Causality.** A prediction at cycle t uses only rows with time <= t. Replay slices the history on the server before calling the model; tests alter later rows and check the prediction does not change.
2. **Ground truth is never a model input.** The model receives only the dataset's declared signal and condition columns. Truth appears only in `eval` replay and `include_truth=true` engine detail, and is labelled `ground_truth`. `blind` mode (the default) returns `ground_truth: null`.
3. **Held-out assets only.** The fleet and replay use the official test engines (truncated before failure). Training engines are never served, because the model has seen them.
4. **Provenance on every payload:** `dataset` (raw signals), `prediction`, `ground_truth`, `measured_evaluation` (recorded benchmark results); `synthetic` and `illustrative` are reserved for values that are not real model output.
5. **Honest wording in responses:** RUL is capped (a value at the cap means "at least"); risk levels are experimental thresholds, not failure probabilities; intervals are calibrated on average, not at every RUL.

## Endpoints
| Method and path | Purpose |
|---|---|
| `GET /health`, `GET /subsets` | status; per subset: model bundle and data files available |
| `GET /fleet/{subset}` | state of every held-out engine at its last cycle; summary counts; filters `risk`, `anomalous`, `search`, `sort`, `limit`, `offset` |
| `GET /engines/{subset}/{id}` | signals, predictions over time, optional `include_truth` |
| `POST /predict/rul`, `/predict/risk`, `/predict/anomaly` | body `{subset, history: [{time, <conditions>, <signals>}...]}`; stateless, works for any history that matches the subset's spec |
| `GET /replay/{subset}/{id}?cycle=t&mode=blind\|eval` | prediction state at cycle t, history up to t, sensor values at t |
| `GET /explain/{subset}/{id}?cycle=t` | TreeSHAP contribution of each signal to the prediction made at cycle t (default: last observed cycle); contributions sum to the raw model output; history is cut server-side like replay |
| `GET /models` | bundle metadata per subset |
| `GET /evaluation/{subset}` | leaderboard from the experiment records (test RMSE with confidence interval, NASA score, cost) |

## Errors
`404` unknown subset or engine; `422` invalid input (missing, unknown or non-finite values, non-integer or duplicate time, cycle beyond the observed history, bad query parameters); `503` model bundle or dataset files missing or incompatible, with the command to fix it in the message.

## Generality
The service is driven by `DatasetSpec` files in `configs/`; the prediction endpoints take whichever signals the chosen subset's spec lists. Only the C-MAPSS adapter is wired into the fleet loader so far; a new machine type needs a spec, an adapter and a built bundle.
