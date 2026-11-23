"""The real server in a separate process, spoken to over real HTTP (skipped without the built bundle and data)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAVE = (ROOT / "artifacts" / "bundles" / "FD001.joblib").exists() and (ROOT / "data" / "raw" / "extracted" / "test_FD001.txt").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="real FD001 data and bundle not present")


@pytest.fixture(scope="module")
def server():
    from serve_helper import running_server

    with running_server() as (base, proc, log, _):
        yield base, log


def test_health_openapi_and_cors_over_real_http(server):
    base, _ = server
    h = httpx.get(f"{base}/health").json()
    assert h["status"] == "ok" and {s["subset"] for s in h["subsets"]} >= {"FD001"}
    paths = httpx.get(f"{base}/openapi.json").json()["paths"]
    for p in ("/fleet/{subset}", "/replay/{subset}/{engine_id}", "/explain/{subset}/{engine_id}", "/predict/rul", "/evaluation/{subset}"):
        assert p in paths
    ok = httpx.options(f"{base}/fleet/FD001", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    bad = httpx.options(f"{base}/fleet/FD001", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers


def test_concurrent_requests_match_sequential_results(server):
    base, _ = server
    jobs = [("replay", c) for c in (5, 40, 80, 120, 160)] * 4

    def call(job):
        _, cycle = job
        r = httpx.get(f"{base}/replay/FD001/12", params={"cycle": cycle, "mode": "blind"}, timeout=60)
        assert r.status_code == 200
        return cycle, r.json()["prediction"]["point"]["rul"]

    serial = {c: call(("replay", c))[1] for c in (5, 40, 80, 120, 160)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(call, jobs))
    assert all(rul == serial[c] for c, rul in results)


def test_simultaneous_cold_fleet_requests_agree(server):
    base, _ = server
    with ThreadPoolExecutor(max_workers=4) as pool:
        bodies = list(pool.map(lambda _: httpx.get(f"{base}/fleet/FD003", params={"limit": 500}, timeout=120).json(), range(4)))
    first = [(e["engine_id"], e["rul"], e["risk"]) for e in bodies[0]["engines"]]
    assert len(first) == 100 and all([(e["engine_id"], e["rul"], e["risk"]) for e in b["engines"]] == first for b in bodies)


def test_prediction_over_http_equals_direct_bundle_call(server):
    base, _ = server
    from sentinel.inference.bundle import ModelBundle
    from sentinel.api.services import DiskServices

    svc = DiskServices()
    g = svc.fleet("FD001").index[20]
    rows = g[svc.spec("FD001").columns].drop(columns=["asset"]).to_dict(orient="records")
    http = httpx.post(f"{base}/predict/rul", json={"subset": "FD001", "history": rows}, timeout=60).json()
    direct = ModelBundle.load(ROOT / "artifacts" / "bundles" / "FD001.joblib").predict_history(g[svc.spec("FD001").columns]).iloc[-1]
    assert http["rul"] == pytest.approx(float(direct["rul"]), abs=1e-9)
    assert http["interval_90"]["upper"] == pytest.approx(float(direct["hi90"]), abs=1e-9)


def test_server_survives_abuse_and_logs_no_tracebacks(server):
    base, log = server
    junk = [
        httpx.post(f"{base}/predict/rul", content=b"\x00\xff garbage", headers={"Content-Type": "application/json"}),
        httpx.post(f"{base}/predict/rul", json={"subset": "FD001", "history": [{"time": 1.0}] * 3500}),
        httpx.get(f"{base}/replay/FD001/1", params={"cycle": -5}),
        httpx.get(f"{base}/replay/FD001/99999", params={"cycle": 1}),
        httpx.get(f"{base}/fleet/..%2F..%2Fetc", params={"limit": 10}),
        httpx.get(f"{base}/explain/FD001/1", params={"cycle": "abc"}),
        httpx.post(f"{base}/predict/rul", content=b"x" * 5_000_000, headers={"Content-Type": "application/json"}),
    ]
    assert all(r.status_code in (404, 405, 413, 422) for r in junk), [r.status_code for r in junk]
    assert httpx.get(f"{base}/health").status_code == 200
    assert "Traceback" not in log.read_text()


def test_unsupported_subset_and_engine_ids_over_http(server):
    base, _ = server
    assert httpx.get(f"{base}/fleet/NOPE").status_code == 404
    assert httpx.get(f"{base}/engines/FD001/0").status_code == 404
    r = httpx.get(f"{base}/engines/FD001/5", params={"include_truth": "true"}).json()
    assert r["ground_truth"]["provenance"] == "ground_truth" and len(r["ground_truth"]["rul"]) == r["observed_cycles"]
    assert np.isfinite(r["predictions"]["rul"]).all()
