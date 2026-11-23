"""SENTINEL API. Inference lives in `sentinel.inference`; this layer validates input, slices history so a prediction at
cycle t can only see cycles <= t, labels every value with its provenance, and never passes ground truth to a model."""
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sentinel.api import schemas as S
from sentinel.api.services import DiskServices, FleetData, Services, ServiceError
from sentinel.inference.bundle import RISK_NAMES, BundleError, ModelBundle
from sentinel.registry.records import load_records

DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
REPLAY_NOTE = "Replay uses held-out (official test) assets only; training assets are never replayed because the model has seen them."


def _interval(row, tag: int) -> S.Interval:
    return S.Interval(level=tag / 100, lower=float(row[f"lo{tag}"]), upper=float(row[f"hi{tag}"]))


def _point(row) -> S.PredictionPoint:
    return S.PredictionPoint(
        time=int(row["time"]), rul=float(row["rul"]), interval_80=_interval(row, 80), interval_90=_interval(row, 90),
        risk=RISK_NAMES[int(row["risk"])], risk_instant=RISK_NAMES[int(row["risk_instant"])],
        anomaly_score=float(row["anomaly_score"]), anomaly_alert=bool(row["anomaly_alert"]),
    )


def _series(df: pd.DataFrame) -> S.SeriesOut:
    return S.SeriesOut(
        time=df["time"].astype(int).tolist(), rul=df["rul"].astype(float).tolist(),
        lower_80=df["lo80"].tolist(), upper_80=df["hi80"].tolist(), lower_90=df["lo90"].tolist(), upper_90=df["hi90"].tolist(),
        risk=[RISK_NAMES[int(i)] for i in df["risk"]], anomaly_score=df["anomaly_score"].astype(float).tolist(),
        anomaly_alert=[bool(x) for x in df["anomaly_alert"]],
    )


def _prediction(subset: str, bundle: ModelBundle, row) -> S.Prediction:
    return S.Prediction(subset=subset, point=_point(row), rul_cap=bundle.cap, anomaly_threshold=bundle.anomaly_threshold,
                        notes=[S.CAP_NOTE, S.RISK_NOTE])


def create_app(services: Services | None = None, origins: list[str] | None = None) -> FastAPI:
    svc: Services = services or DiskServices()
    app = FastAPI(title="SENTINEL", version="0.1.0", description="Predictive maintenance inference API (research demonstration, not production).")
    app.add_middleware(CORSMiddleware, allow_origins=origins or DEFAULT_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
    fleet_cache: dict[str, pd.DataFrame] = {}

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    def model_input(fleet: FleetData, engine_id: int) -> pd.DataFrame:
        if engine_id not in fleet.index:
            raise HTTPException(404, f"engine {engine_id} not found")
        return fleet.index[engine_id][fleet.spec.columns]  # ground truth is not a model input

    def predict_or_422(bundle: ModelBundle, frame: pd.DataFrame, tail: int | None = None) -> pd.DataFrame:
        try:
            return bundle.predict_history(frame, tail=tail)
        except BundleError as exc:
            raise HTTPException(422, str(exc)) from exc

    def fleet_last(subset: str) -> pd.DataFrame:
        if subset not in fleet_cache:
            bundle, fleet = svc.bundle(subset), svc.fleet(subset)
            try:
                last = bundle.predict_last_many({eid: g[fleet.spec.columns] for eid, g in fleet.index.items()})
            except BundleError as exc:
                raise HTTPException(422, str(exc)) from exc
            last.insert(0, "engine_id", last.pop("asset").astype(int))
            last.insert(1, "observed_cycles", last["time"].astype(int))
            fleet_cache[subset] = last
        return fleet_cache[subset]

    def history_frame(req: S.PredictRequest) -> tuple[ModelBundle, pd.DataFrame]:
        bundle = svc.bundle(req.subset)
        allowed = set(bundle.spec.columns) - {"asset"}
        for i, row in enumerate(req.history):
            unknown, missing = set(row) - allowed, allowed - set(row)
            if unknown or missing:
                raise HTTPException(422, f"row {i}: unknown columns {sorted(unknown)}, missing columns {sorted(missing)}")
        df = pd.DataFrame(req.history)
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise HTTPException(422, "history contains missing or non-finite values")
        if not (df["time"] % 1 == 0).all() or (df["time"] < 1).any():
            raise HTTPException(422, "time must be integer steps starting at 1 or later")
        df["time"] = df["time"].astype(int)
        return bundle, df

    @app.get("/health", response_model=S.Health)
    def health():
        subsets = [S.SubsetInfo(**s) for s in svc.subsets()]
        ok = any(s.model_available and s.data_available for s in subsets)
        return S.Health(status="ok" if ok else "degraded", subsets=subsets)

    @app.get("/subsets", response_model=list[S.SubsetInfo])
    def subsets():
        return [S.SubsetInfo(**s) for s in svc.subsets()]

    @app.get("/fleet/{subset}", response_model=S.FleetResponse)
    def fleet(subset: str, risk: S.Risk | None = None, anomalous: bool | None = None, search: str | None = Query(None, max_length=10),
              sort: Literal["rul", "engine_id"] = "rul", limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        df = fleet_last(subset)
        counts = {n: int((df["risk"] == i).sum()) for i, n in enumerate(RISK_NAMES)}
        summary = S.FleetSummary(total=len(df), anomalous=int(df["anomaly_alert"].sum()), **counts)
        view = df
        if risk is not None:
            view = view[view["risk"] == RISK_NAMES.index(risk)]
        if anomalous is not None:
            view = view[view["anomaly_alert"] == anomalous]
        if search:
            view = view[view["engine_id"].astype(str).str.contains(search, regex=False)]
        view = view.sort_values(sort if sort == "engine_id" else "rul", kind="stable")
        engines = [S.FleetEngine(engine_id=int(r["engine_id"]), observed_cycles=int(r["observed_cycles"]), rul=float(r["rul"]),
                                 interval_80=_interval(r, 80), risk=RISK_NAMES[int(r["risk"])], anomaly_alert=bool(r["anomaly_alert"]))
                   for _, r in view.iloc[offset : offset + limit].iterrows()]
        return S.FleetResponse(subset=subset, note="State of each held-out engine at its last observed cycle. " + S.RISK_NOTE,
                               summary=summary, total_matching=len(view), engines=engines)

    @app.get("/engines/{subset}/{engine_id}", response_model=S.EngineDetail)
    def engine(subset: str, engine_id: int, include_truth: bool = False):
        bundle, fleet_data = svc.bundle(subset), svc.fleet(subset)
        pred = predict_or_422(bundle, model_input(fleet_data, engine_id))
        g = fleet_data.index[engine_id]
        truth = S.GroundTruthSeries(time=g["time"].astype(int).tolist(), rul=g["rul"].astype(float).tolist()) if include_truth else None
        return S.EngineDetail(subset=subset, engine_id=engine_id, observed_cycles=int(g["time"].max()),
                              signals={c: g[c].astype(float).tolist() for c in fleet_data.spec.signals}, predictions=_series(pred),
                              anomaly_threshold=bundle.anomaly_threshold, ground_truth=truth, notes=[S.CAP_NOTE, S.RISK_NOTE, REPLAY_NOTE])

    @app.post("/predict/rul", response_model=S.RulResponse)
    def predict_rul(req: S.PredictRequest):
        bundle, df = history_frame(req)
        row = predict_or_422(bundle, df, tail=1).iloc[-1]
        return S.RulResponse(subset=req.subset, time=int(row["time"]), rul=float(row["rul"]), interval_80=_interval(row, 80),
                             interval_90=_interval(row, 90), rul_cap=bundle.cap, notes=[S.CAP_NOTE])

    @app.post("/predict/risk", response_model=S.RiskResponse)
    def predict_risk(req: S.PredictRequest):
        bundle, df = history_frame(req)
        row = predict_or_422(bundle, df, tail=1).iloc[-1]
        return S.RiskResponse(subset=req.subset, time=int(row["time"]), risk=RISK_NAMES[int(row["risk"])],
                              risk_instant=RISK_NAMES[int(row["risk_instant"])], rul=float(row["rul"]), notes=[S.RISK_NOTE])

    @app.post("/predict/anomaly", response_model=S.AnomalyResponse)
    def predict_anomaly(req: S.PredictRequest):
        bundle, df = history_frame(req)
        row = predict_or_422(bundle, df, tail=1).iloc[-1]
        return S.AnomalyResponse(subset=req.subset, time=int(row["time"]), anomaly_score=float(row["anomaly_score"]),
                                 anomaly_threshold=bundle.anomaly_threshold, anomaly_alert=bool(row["anomaly_alert"]),
                                 notes=["Statistical detector fitted on healthy-phase data; no accuracy is claimed on real anomalies."])

    @app.get("/replay/{subset}/{engine_id}", response_model=S.ReplayState)
    def replay(subset: str, engine_id: int, cycle: int = Query(..., ge=1), mode: Literal["blind", "eval"] = "blind"):
        bundle, fleet_data = svc.bundle(subset), svc.fleet(subset)
        inputs = model_input(fleet_data, engine_id)
        observed = int(inputs["time"].max())
        if cycle > observed:
            raise HTTPException(422, f"cycle {cycle} is beyond the {observed} observed cycles of engine {engine_id}")
        visible = inputs[inputs["time"] <= cycle]  # server-side cut: nothing after `cycle` reaches the model
        pred = predict_or_422(bundle, visible)
        now = visible.iloc[-1]
        truth = None
        if mode == "eval":
            g = fleet_data.index[engine_id]
            truth = S.GroundTruthPoint(rul=float(g.loc[g["time"] == cycle, "rul"].iloc[0]))
        return S.ReplayState(subset=subset, engine_id=engine_id, cycle=cycle, observed_cycles=observed, mode=mode,
                             sensors={c: float(now[c]) for c in fleet_data.spec.signals}, prediction=_prediction(subset, bundle, pred.iloc[-1]),
                             history=_series(pred), ground_truth=truth,
                             notes=[REPLAY_NOTE, "Blind mode withholds ground truth; eval mode shows it for comparison only, and it is never a model input.",
                                    S.CAP_NOTE])

    @app.get("/explain/{subset}/{engine_id}", response_model=S.ExplainResponse)
    def explain(subset: str, engine_id: int, cycle: int | None = Query(None, ge=1)):
        bundle, fleet_data = svc.bundle(subset), svc.fleet(subset)
        inputs = model_input(fleet_data, engine_id)
        observed = int(inputs["time"].max())
        t = observed if cycle is None else cycle
        if t > observed:
            raise HTTPException(422, f"cycle {t} is beyond the {observed} observed cycles of engine {engine_id}")
        visible = inputs[inputs["time"] <= t]
        try:
            contrib, raw, base = bundle.explain(visible, tail=1)
        except BundleError as exc:
            raise HTTPException(422, str(exc)) from exc
        row = contrib.iloc[-1]
        now = visible.iloc[-1]
        signals = [c for c in contrib.columns if c != "time"]
        items = sorted((S.Contribution(signal=s, contribution=float(row[s]), value=float(now[s])) for s in signals), key=lambda c: -abs(c.contribution))
        return S.ExplainResponse(subset=subset, engine_id=engine_id, cycle=t, base_value=base, raw_prediction=float(raw[-1]),
                                 rul=float(np.clip(raw[-1], 0, bundle.cap)), contributions=items,
                                 notes=["Attributions are TreeSHAP values of this model: they describe how it uses its inputs, not physical causes.",
                                        "Each signal's contribution sums the effect of its recent level, trend and variability; correlated signals share credit.",
                                        S.CAP_NOTE])

    @app.get("/models", response_model=list[S.ModelInfo])
    def models():
        out = []
        for s in svc.subsets():
            if s["model_available"]:
                try:
                    out.append(S.ModelInfo(subset=s["subset"], available=True, meta=svc.bundle(s["subset"]).meta,
                                           notes=["Default model: LightGBM, window 50, per-regime normalisation, conformalised intervals."]))
                    continue
                except ServiceError as exc:
                    out.append(S.ModelInfo(subset=s["subset"], available=False, meta={}, notes=[exc.detail]))
                    continue
            out.append(S.ModelInfo(subset=s["subset"], available=False, meta={}, notes=["model bundle not built"]))
        return out

    @app.get("/evaluation/{subset}", response_model=S.EvaluationResponse)
    def evaluation(subset: str):
        svc.spec(subset)
        latest = {}
        for r in load_records(svc.runs_dir) if svc.runs_dir.exists() else []:
            if r.subset == subset and r.test_metrics and "rmse" in r.test_metrics:
                latest[r.name] = r
        if not latest:
            raise HTTPException(404, f"no evaluation records for {subset}")
        rows = [S.EvaluationRow(
            model=r.name, cv_cut_rmse=r.cv_metrics.get("cut_rmse", {}).get("mean"), test_rmse=r.test_metrics["rmse"]["mean"],
            test_rmse_ci95=list(r.test_metrics.get("rmse_ci95_seed_mean_prediction", [])) or None,
            test_nasa_score=r.test_metrics.get("score", {}).get("mean"), fit_seconds=r.train_seconds, latency_ms=r.latency_single_ms,
            size_kb=(r.model_size_bytes or 0) / 1024 or None) for r in sorted(latest.values(), key=lambda r: r.test_metrics["rmse"]["mean"])]
        return S.EvaluationResponse(subset=subset, leaderboard=rows,
                                    notes=["Measured on the official test set, one point per engine; models whose confidence intervals overlap are not distinguishable.",
                                           "Simulated C-MAPSS data."])

    return app


app = create_app()
