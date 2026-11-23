"""Dataset -> preprocessing -> model inference -> API, on the real FD001 files and the built bundle (skipped if absent)."""
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentinel.api.app import create_app
from sentinel.api.services import DiskServices

ROOT = Path(__file__).resolve().parents[1]
HAVE = (ROOT / "artifacts" / "bundles" / "FD001.joblib").exists() and (ROOT / "data" / "raw" / "extracted" / "test_FD001.txt").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="real FD001 data and bundle not present (run scripts/build_bundle.py)")


@pytest.fixture(scope="module")
def stack():
    svc = DiskServices()
    return TestClient(create_app(svc)), svc


def test_health_lists_real_subsets(stack):
    client, _ = stack
    h = client.get("/health").json()
    fd001 = next(s for s in h["subsets"] if s["subset"] == "FD001")
    assert fd001["model_available"] and fd001["data_available"] and h["status"] == "ok"


def test_fleet_predictions_match_known_final_ruls_to_a_sane_error(stack):
    client, svc = stack
    fleet = client.get("/fleet/FD001", params={"limit": 500}).json()
    assert fleet["summary"]["total"] == 100
    data = svc.fleet("FD001")
    truth = {e: float(g["rul"].iloc[-1]) for e, g in data.index.items()}
    err = np.array([e["rul"] - truth[e["engine_id"]] for e in fleet["engines"]])
    assert np.sqrt((err**2).mean()) < 20  # bundle is trained on 80% of the engines; test RMSE of the full model is about 13-14
    assert fleet["summary"]["critical"] > 0 and fleet["summary"]["normal"] > 0


def test_blind_replay_never_leaks_future_or_truth_and_eval_matches_dataset(stack):
    client, svc = stack
    data = svc.fleet("FD001")
    eid = 7
    cycle = 60
    blind = client.get(f"/replay/FD001/{eid}", params={"cycle": cycle}).json()
    ev = client.get(f"/replay/FD001/{eid}", params={"cycle": cycle, "mode": "eval"}).json()
    assert blind["ground_truth"] is None and blind["prediction"] == ev["prediction"]
    g = data.index[eid]
    assert ev["ground_truth"]["rul"] == float(g.loc[g["time"] == cycle, "rul"].iloc[0])
    assert blind["history"]["time"] == list(range(1, cycle + 1))


def test_replay_over_full_history_matches_engine_endpoint(stack):
    client, _ = stack
    detail = client.get("/engines/FD001/12").json()
    last = detail["observed_cycles"]
    replay = client.get("/replay/FD001/12", params={"cycle": last}).json()
    assert np.allclose(replay["history"]["rul"], detail["predictions"]["rul"])


def test_evaluation_endpoint_reflects_recorded_results(stack):
    client, _ = stack
    ev = client.get("/evaluation/FD001").json()
    names = [r["model"] for r in ev["leaderboard"]]
    assert "lightgbm_window50" in names and "constant" in names
    best = next(r for r in ev["leaderboard"] if r["model"] == "lightgbm_window50")
    assert 10 < best["test_rmse"] < 20
