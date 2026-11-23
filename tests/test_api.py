"""API tests on synthetic fixtures (clearly synthetic; used only to exercise code paths)."""
import copy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sentinel.api.app import create_app
from sentinel.api.services import FleetData, MemoryServices
from sentinel.data.labels import add_train_rul
from sentinel.inference.bundle import ModelBundle
from sentinel.registry.records import RunRecord, save_record
from conftest import fixture_spec, make_fixture_frame

SPEC = fixture_spec()
SIGNALS = SPEC.signals


@pytest.fixture(scope="module")
def bundle():
    train = add_train_rul(make_fixture_frame([60, 70, 80, 90, 100, 110, 120, 130, 140, 150], seed=3))
    return ModelBundle.build(SPEC, train, window=10)


@pytest.fixture(scope="module")
def fleet_frame():
    full = add_train_rul(make_fixture_frame([90, 100, 110, 120, 130, 140], seed=11))
    cut = {a: int(0.6 * n) for a, n in full.groupby("asset")["time"].max().items()}
    return full[full.apply(lambda r: r["time"] <= cut[int(r["asset"])], axis=1)].reset_index(drop=True)


def make_client(bundle, fleet_frame, tmp_path, with_bundle=True, with_fleet=True, origins=None):
    svc = MemoryServices({"FDX": SPEC}, {"FDX": bundle} if with_bundle else {}, {"FDX": FleetData(SPEC, fleet_frame)} if with_fleet else {}, tmp_path)
    return TestClient(create_app(svc, origins=origins)), svc


@pytest.fixture
def client(bundle, fleet_frame, tmp_path):
    return make_client(bundle, fleet_frame, tmp_path)[0]


def history(n=30, seed=5):
    return make_fixture_frame([n], seed=seed).drop(columns=["asset"]).to_dict(orient="records")


def test_health_and_subsets(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["subsets"][0]["subset"] == "FDX" and h["subsets"][0]["model_available"]
    assert client.get("/subsets").json()[0]["signals"] == 21


def test_fleet_schema_summary_and_filters(client):
    body = client.get("/fleet/FDX").json()
    assert body["provenance"] == "prediction" and body["summary"]["total"] == 6 == len(body["engines"])
    s = body["summary"]
    assert s["normal"] + s["warning"] + s["critical"] == s["total"]
    ruls = [e["rul"] for e in body["engines"]]
    assert ruls == sorted(ruls)
    e = body["engines"][0]
    assert e["interval_80"]["lower"] <= e["interval_80"]["upper"] and e["risk"] in ("normal", "warning", "critical")
    only = client.get("/fleet/FDX", params={"risk": "normal"}).json()
    assert all(x["risk"] == "normal" for x in only["engines"]) and only["total_matching"] == s["normal"]
    assert len(client.get("/fleet/FDX", params={"limit": 2}).json()["engines"]) == 2
    assert client.get("/fleet/FDX", params={"limit": 2, "offset": 5}).json()["engines"].__len__() == 1
    assert [x["engine_id"] for x in client.get("/fleet/FDX", params={"sort": "engine_id"}).json()["engines"]] == [1, 2, 3, 4, 5, 6]
    assert client.get("/fleet/FDX", params={"search": "3"}).json()["total_matching"] == 1


@pytest.mark.parametrize("params", [{"risk": "doomed"}, {"limit": 0}, {"limit": 501}, {"offset": -1}, {"sort": "price"}])
def test_fleet_rejects_invalid_query_parameters(client, params):
    assert client.get("/fleet/FDX", params=params).status_code == 422


def test_engine_detail_and_truth_visibility(client):
    d = client.get("/engines/FDX/2").json()
    n = d["observed_cycles"]
    assert len(d["predictions"]["rul"]) == n == len(d["signals"][SIGNALS[0]]) and d["ground_truth"] is None
    assert d["signals_provenance"] == "dataset" and d["predictions_provenance"] == "prediction"
    t = client.get("/engines/FDX/2", params={"include_truth": True}).json()["ground_truth"]
    assert t["provenance"] == "ground_truth" and len(t["rul"]) == n and t["rul"][-1] > 0


def test_invalid_engine_and_subset_ids(client):
    assert client.get("/engines/FDX/999").status_code == 404
    assert client.get("/engines/FDX/abc").status_code == 422
    assert client.get("/engines/NOPE/1").status_code == 404
    assert client.get("/fleet/NOPE").status_code == 404
    assert client.get("/replay/NOPE/1", params={"cycle": 1}).status_code == 404
    assert client.get("/evaluation/NOPE").status_code == 404
    assert client.get("/fleet/bad$name").status_code == 404
    assert client.post("/predict/rul", json={"subset": "bad name!", "history": history()}).status_code == 422
    assert client.post("/predict/rul", json={"subset": "NOPE", "history": history()}).status_code == 404


def test_predict_endpoints_return_consistent_values(client):
    req = {"subset": "FDX", "history": history(40)}
    rul, risk, anom = (client.post(f"/predict/{k}", json=req).json() for k in ("rul", "risk", "anomaly"))
    assert rul["time"] == risk["time"] == anom["time"] == 40 and rul["provenance"] == "prediction"
    assert 0 <= rul["rul"] <= rul["rul_cap"]
    assert 0 <= rul["interval_80"]["lower"] <= rul["interval_80"]["upper"] <= rul["rul_cap"]
    assert rul["interval_80"]["level"] == 0.8 and rul["interval_90"]["level"] == 0.9
    assert risk["rul"] == rul["rul"] and risk["risk"] in ("normal", "warning", "critical")
    assert isinstance(anom["anomaly_alert"], bool) and anom["anomaly_threshold"] > 0


def test_predict_validation_errors(client):
    good = history(20)
    cases = {
        "empty": {"subset": "FDX", "history": []},
        "missing column": {"subset": "FDX", "history": [{k: v for k, v in r.items() if k != "s4"} for r in good]},
        "unknown column": {"subset": "FDX", "history": [{**r, "mystery": 1.0} for r in good]},
        "non-numeric": {"subset": "FDX", "history": [{**good[0], "s4": "abc"}] + good[1:]},
        "fractional time": {"subset": "FDX", "history": [{**good[0], "time": 1.5}] + good[1:]},
        "time below 1": {"subset": "FDX", "history": [{**good[0], "time": 0}] + good[1:]},
        "too long": {"subset": "FDX", "history": good * 160},
        "duplicate time": {"subset": "FDX", "history": good + [good[0]]},
        "no subset": {"history": good},
    }
    for name, payload in cases.items():
        for ep in ("rul", "risk", "anomaly"):
            r = client.post(f"/predict/{ep}", json=payload)
            assert r.status_code == 422, (name, ep, r.status_code, r.text[:200])


@pytest.mark.parametrize("scale", [1e6, 1e12, 1e30, 1e200, 1e300])
def test_extreme_magnitudes_never_cause_a_server_error(client, scale):
    rows = [{**r, "s4": r["s4"] * scale} for r in history(25)]
    for ep in ("rul", "risk", "anomaly"):
        r = client.post(f"/predict/{ep}", json={"subset": "FDX", "history": rows})
        assert r.status_code == (200 if scale <= 1e6 else 422), (scale, ep, r.status_code, r.text[:200])
        if r.status_code == 200:
            assert all(np.isfinite(v) for v in _numbers(r.json()))
        else:
            assert "supported range" in r.json()["detail"]


def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield obj


@pytest.mark.parametrize("content,headers", [
    ('{"subset": "FDX", "history": [', {"Content-Type": "application/json"}),
    ("", {"Content-Type": "application/json"}),
    ('{"subset": "FDX", "history": [{"time": NaN}]}', {"Content-Type": "application/json"}),
    ('{"subset": "FDX", "history": [{"time": Infinity, "s4": 1}]}', {"Content-Type": "application/json"}),
    ("subset=FDX", {"Content-Type": "application/x-www-form-urlencoded"}),
    ("[1, 2, 3]", {"Content-Type": "application/json"}),
    ('{"subset": ["FDX"], "history": "nope"}', {"Content-Type": "application/json"}),
])
def test_malformed_requests_get_422_not_500(client, content, headers):
    for ep in ("rul", "risk", "anomaly"):
        assert client.post(f"/predict/{ep}", content=content, headers=headers).status_code == 422


def test_unsupported_methods_and_unknown_paths(client):
    assert client.get("/predict/rul").status_code == 405
    assert client.post("/fleet/FDX").status_code == 405
    assert client.delete("/engines/FDX/1").status_code == 405
    assert client.get("/nope").status_code == 404


def test_replay_blind_hides_truth_and_eval_shows_it(client, fleet_frame):
    blind = client.get("/replay/FDX/3", params={"cycle": 20}).json()
    assert blind["mode"] == "blind" and blind["ground_truth"] is None
    ev = client.get("/replay/FDX/3", params={"cycle": 20, "mode": "eval"}).json()
    row = fleet_frame[(fleet_frame["asset"] == 3) & (fleet_frame["time"] == 20)].iloc[0]
    assert ev["ground_truth"]["provenance"] == "ground_truth" and ev["ground_truth"]["rul"] == row["rul"]
    assert ev["prediction"] == blind["prediction"] and ev["history"] == blind["history"]
    assert len(blind["history"]["time"]) == 20 == blind["cycle"] and blind["prediction"]["point"]["time"] == 20
    assert set(blind["sensors"]) == set(SIGNALS)


@pytest.mark.parametrize("params", [{"cycle": 0}, {"cycle": 10_000}, {"cycle": 5, "mode": "cheat"}, {}])
def test_replay_rejects_bad_requests(client, params):
    assert client.get("/replay/FDX/1", params=params).status_code == 422
    assert client.get("/replay/FDX/999", params={"cycle": 1}).status_code == 404


def test_replay_prediction_ignores_future_rows(bundle, fleet_frame, tmp_path):
    t = 25
    tampered = fleet_frame.copy()
    mask = (tampered["asset"] == 3) & (tampered["time"] > t)
    tampered.loc[mask, SIGNALS] += 500.0
    a = make_client(bundle, fleet_frame, tmp_path)[0].get("/replay/FDX/3", params={"cycle": t}).json()
    b = make_client(bundle, tampered, tmp_path)[0].get("/replay/FDX/3", params={"cycle": t}).json()
    assert a["prediction"] == b["prediction"] and a["history"] == b["history"] and a["sensors"] == b["sensors"]
    assert a["observed_cycles"] == b["observed_cycles"]


def test_replay_matches_stateless_prediction_from_the_same_history(client, fleet_frame):
    t = 30
    rows = fleet_frame[(fleet_frame["asset"] == 2) & (fleet_frame["time"] <= t)].drop(columns=["asset", "rul"]).to_dict(orient="records")
    direct = client.post("/predict/rul", json={"subset": "FDX", "history": rows}).json()
    state = client.get("/replay/FDX/2", params={"cycle": t}).json()["prediction"]["point"]
    assert direct["rul"] == state["rul"] and direct["interval_90"] == state["interval_90"]


def test_ground_truth_is_never_passed_to_the_model(bundle, fleet_frame, tmp_path, monkeypatch):
    seen = []
    original = bundle.predict_history
    monkeypatch.setattr(bundle, "predict_history", lambda frame, **kw: (seen.append(list(frame.columns)), original(frame, **kw))[1])
    c, _ = make_client(bundle, fleet_frame, tmp_path)
    c.get("/fleet/FDX")
    c.get("/engines/FDX/1", params={"include_truth": True})
    c.get("/replay/FDX/1", params={"cycle": 10, "mode": "eval"})
    assert seen and all("rul" not in cols and "rul_capped" not in cols for cols in seen)


def test_explain_contributions_are_additive_sorted_and_causal(bundle, fleet_frame, tmp_path, client):
    r = client.get("/explain/FDX/2", params={"cycle": 30}).json()
    assert r["provenance"] == "prediction" and r["cycle"] == 30 and len(r["contributions"]) == len(bundle.pre.feature_cols) == 20
    total = r["base_value"] + sum(c["contribution"] for c in r["contributions"])
    assert total == pytest.approx(r["raw_prediction"], abs=1e-6)
    assert r["rul"] == pytest.approx(min(max(r["raw_prediction"], 0), 125), abs=1e-9)
    mags = [abs(c["contribution"]) for c in r["contributions"]]
    assert mags == sorted(mags, reverse=True) and any("causes" in n for n in r["notes"])
    tampered = fleet_frame.copy()
    tampered.loc[(tampered["asset"] == 2) & (tampered["time"] > 30), SIGNALS] += 500.0
    other = make_client(bundle, tampered, tmp_path)[0].get("/explain/FDX/2", params={"cycle": 30}).json()
    assert other["contributions"] == r["contributions"] and other["raw_prediction"] == r["raw_prediction"]
    default = client.get("/explain/FDX/2").json()
    assert default["cycle"] == client.get("/engines/FDX/2").json()["observed_cycles"]


@pytest.mark.parametrize("url,status", [("/explain/FDX/999", 404), ("/explain/NOPE/1", 404), ("/explain/FDX/1?cycle=0", 422),
                                        ("/explain/FDX/1?cycle=99999", 422), ("/explain/FDX/abc", 422)])
def test_explain_rejects_bad_requests(client, url, status):
    assert client.get(url).status_code == status


def test_missing_artifacts_give_clear_503s(bundle, fleet_frame, tmp_path):
    no_model, _ = make_client(bundle, fleet_frame, tmp_path, with_bundle=False)
    for url in ("/fleet/FDX", "/engines/FDX/1", "/replay/FDX/1?cycle=1"):
        r = no_model.get(url)
        assert r.status_code == 503 and "bundle" in r.json()["detail"]
    assert no_model.post("/predict/rul", json={"subset": "FDX", "history": history()}).status_code == 503
    assert no_model.get("/health").json()["status"] == "degraded"
    no_data, _ = make_client(bundle, fleet_frame, tmp_path, with_fleet=False)
    assert no_data.get("/fleet/FDX").status_code == 503
    assert no_data.post("/predict/rul", json={"subset": "FDX", "history": history()}).status_code == 200


def test_models_and_evaluation_endpoints(bundle, fleet_frame, tmp_path):
    c, svc = make_client(bundle, fleet_frame, tmp_path)
    m = c.get("/models").json()[0]
    assert m["available"] and m["meta"]["n_training_assets"] > 0
    assert c.get("/evaluation/FDX").status_code == 404
    rec = RunRecord(run_id="FDX-lgbm-1", subset="FDX", name="lgbm", model="lgbm", config={}, split={}, preprocessing={}, seeds=[0],
                    cv_metrics={"cut_rmse": {"mean": 10.0, "std": 1.0}},
                    test_metrics={"rmse": {"mean": 12.0, "std": 0.1}, "rmse_ci95_seed_mean_prediction": [10.0, 14.0], "score": {"mean": 300.0, "std": 5.0}},
                    train_seconds=2.0, latency_single_ms=1.5, model_size_bytes=2048)
    save_record(rec, tmp_path)
    ev = c.get("/evaluation/FDX").json()
    assert ev["provenance"] == "measured_evaluation" and ev["leaderboard"][0]["test_rmse"] == 12.0 and ev["leaderboard"][0]["test_rmse_ci95"] == [10.0, 14.0]
    no_model, _ = make_client(bundle, fleet_frame, tmp_path, with_bundle=False)
    assert no_model.get("/models").json()[0]["available"] is False


def test_cors_only_for_configured_origins(bundle, fleet_frame, tmp_path):
    c, _ = make_client(bundle, fleet_frame, tmp_path, origins=["http://localhost:5173"])
    ok = c.get("/health", headers={"Origin": "http://localhost:5173"})
    bad = c.get("/health", headers={"Origin": "http://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers
