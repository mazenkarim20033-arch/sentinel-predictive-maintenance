# SENTINEL web app

React, TypeScript and Vite, with Recharts. It only talks to the SENTINEL API; no model logic lives here.

```bash
cd web
npm install
npm run dev          # http://localhost:5173, proxies /api to http://127.0.0.1:8000
npm test             # vitest, 19 component and client tests (the API is mocked)
npm run build        # type-check and production build
```

Start the API first (`python scripts/build_bundle.py`, then `uvicorn sentinel.api.app:app --port 8000` from the repository root with `PYTHONPATH=src`). On Windows PowerShell use `npm.cmd` / `npx.cmd` if script execution is disabled.

## Pages
- **Dashboard:** fleet summary and every engine on one remaining-life scale; engines needing attention.
- **Fleet:** filter, search, sort, page, and compare up to four engines.
- **Engine:** predicted remaining life with intervals, warning timeline, anomaly score, signals, per-signal explanation at any cycle, optional ground truth.
- **Simulation lab:** cycle-by-cycle replay with play, pause, step and seek; blind and evaluation modes.
- **Model lab:** recorded test results with confidence intervals and cost.
- **Research:** data, method and what the project does not show.

## Labelling
Every figure carries a chip: model prediction, ground truth (evaluation only), dataset signal, measured benchmark or illustrative. Ground truth is requested from the API only in evaluation mode or when the user asks for it, and is never sent to a model.

## Design notes
Light and dark themes follow the system setting. Status is encoded by text and shape as well as colour. Charts have text alternatives; controls are keyboard reachable.
