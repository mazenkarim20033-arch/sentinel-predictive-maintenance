# Testing and quality

## How to run everything

```bash
# Python: unit, integration and live-HTTP tests (needs the built bundles and data for the integration/live ones, which otherwise skip)
python -m pytest -q

# Frontend: component, client, accessibility tests (API mocked)
cd web && npm test

# Full stack: starts the real API, runs the frontend's live tests against it
python scripts/run_e2e.py      # run with PYTHONPATH=src;scripts

# Performance
python scripts/benchmark_api.py    # writes reports/performance.md
```

## What is tested

| Area | Where | What it checks |
|---|---|---|
| Data pipeline | `tests/test_data.py`, `test_features.py` | file parsing, engine grouping, RUL labels, NaN/inf and bad cycles rejected, preprocessing fitted on training data only and reproducible, causal windows |
| Dataset specs and generality | `test_data.py`, `test_generality.py`, `test_battery.py` | specs validate, a pipeline with unrelated column names runs, battery adapter labels and censoring |
| ML and metrics | `test_metrics_and_tabular.py`, `test_sequence.py`, `test_warning.py`, `test_anomaly.py`, `test_uncertainty.py`, `test_explain.py` | metric values by hand, bootstrap, no future leakage, reproducibility, alert logic, injected-fault properties, conformal coverage on exchangeable data, SHAP additivity and equality with the `shap` library |
| Inference bundle | `test_inference.py` | output schema and ranges, prediction at t unchanged when later rows change, truncated history equals full history, save/load and incompatible files, input validation, saturation of extreme scores, batched and tail computations identical to the full computation |
| Backend API | `test_api.py` | schemas, filters, 404/405/422/503 handling, malformed and hostile requests (bad JSON, NaN/Infinity, extreme magnitudes, oversized histories), ground truth never passed to the model, blind versus evaluation replay, causal replay, CORS |
| Real server | `test_live_http.py` | separate uvicorn process over real HTTP: OpenAPI, CORS preflight, 8-way concurrent requests equal sequential results, simultaneous cold fleet requests agree, HTTP prediction equals direct model call, server survives abuse with no traceback in its log |
| Dataset to backend | `test_integration.py` | real FD001 files, built bundle, fleet error sanity bound, blind/eval replay against dataset truth, evaluation endpoint reflects run records |
| Frontend | `web/src/test/app.test.tsx` | dashboard rendering, loading, error and retry, API-unreachable message, empty states, engine selection, filters, comparison limit, replay step/play/pause/restart without requesting beyond the history, blind vs evaluation truth visibility, model lab and research pages, API client error handling |
| Accessibility | `web/src/test/a11y.test.tsx`, browser run | axe-core on every page in jsdom (with a negative control proving it can fail); colour contrast checked separately in a real browser in light and dark themes |
| Full stack | `web/src/test/live.test.tsx` via `scripts/run_e2e.py` | the frontend rendering real API data end to end, replay truth equals the dataset's, server log clean |
| Public demo | `web/src/test/snapshot.test.ts` | the recorded-playback client follows the server's filtering, sorting, paging, cycle slicing and truth rules; with the live API and an exported snapshot present, fleet state for all 100 FD001 engines and replay/explanation at several cycles equal the live API's values |
| Clean checkout | manual, before release | fresh clone, fresh virtual environment, `pip install -e ".[all]"`, `fetch_data.py` (checksum verified), `build_bundle.py`, `pytest` and `npm ci && npm test && npm run build`; found and fixed a test that assumed all four bundles were built |

## Problems the quality work found and fixed
- Extreme sensor values (about 1e308) overflowed the anomaly score to infinity and produced a 500; inputs beyond 1e9 are now rejected with a 422 and scores saturate at 1e12.
- The first dashboard load took 8.6 s on FD001 and 25 s on FD004 because every cycle of every engine was predicted; fleet state now needs only the last cycles, batched into one model call (now 1.4 s and 3.2 s, about 6-8x faster). `/explain` dropped from 643 ms to 38 ms and a 3000-row prediction from 881 ms to 158 ms, with tests proving the results are identical to the full computation.
- The API-unreachable message hid the underlying cause; the original error is now attached to the exception.
- In a real browser, axe-core found low colour contrast (amber, green, teal and grey tokens), scale ticks whose accessible names did not match their text, a banner outside any landmark, and a skipped heading level. All fixed; every page now passes in light and dark themes.
- My own benchmark first reported 4 MB of memory because it measured the Windows launcher process, not the server; it now sums the process tree.

## Not covered
- No automated end-to-end browser test of clicks and keyboard paths (the frontend tests use jsdom, and the live test uses jsdom against the real API). Browser checks of layout at desktop and phone width and of replay playback were done by hand.
- Colour-contrast checks are a manual browser run, not part of `npm test`.
- No authentication, rate limiting or security testing: the API is for local use.
- Load testing is limited to one machine and 8 concurrent clients.
- Models were not retested for drift or on data outside the simulated engines.
