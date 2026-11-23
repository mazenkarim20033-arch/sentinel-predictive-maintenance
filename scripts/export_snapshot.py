"""Exports the real model's outputs as static JSON for the public demo (web/public/snapshot).

The public demo cannot run a Python server for free, so it replays recorded outputs of the same API code: predictions,
intervals, risk, anomaly scores and explanations for every cycle of every held-out engine. Because a prediction at cycle t
depends only on cycles <= t (tested in tests/test_inference.py), a replay of the recording at cycle t equals what the live
API computes from the history cut at t; scripts/run_e2e.py checks this against the live API.

    python scripts/export_snapshot.py --subsets FD001 FD004
"""
import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from sentinel.api.app import create_app
from sentinel.api.services import DiskServices
from sentinel.inference.bundle import RISK_NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "snapshot"


def dump(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text)


def r(values, nd: int):
    return [round(float(v), nd) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", default=["FD001", "FD004"])
    args = parser.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    svc = DiskServices()
    client = TestClient(create_app(svc))
    total = 0

    subsets = [s for s in client.get("/subsets").json() if s["subset"] in args.subsets]
    total += dump(OUT / "subsets.json", subsets)
    total += dump(OUT / "models.json", [m for m in client.get("/models").json() if m["subset"] in args.subsets])

    meta = {}
    for subset in args.subsets:
        bundle, fleet = svc.bundle(subset), svc.fleet(subset)
        total += dump(OUT / "evaluation" / f"{subset}.json", client.get(f"/evaluation/{subset}").json())
        sample_id = next(iter(fleet.index))
        meta[subset] = {
            "rul_cap": bundle.cap, "anomaly_threshold": bundle.anomaly_threshold,
            "engine_notes": client.get(f"/engines/{subset}/{sample_id}").json()["notes"],
            "replay_notes": client.get(f"/replay/{subset}/{sample_id}", params={"cycle": 1}).json()["notes"],
            "prediction_notes": client.get(f"/replay/{subset}/{sample_id}", params={"cycle": 1}).json()["prediction"]["notes"],
            "explain_notes": client.get(f"/explain/{subset}/{sample_id}").json()["notes"],
            "fleet_note": client.get(f"/fleet/{subset}", params={"limit": 1}).json()["note"],
        }
        last = bundle.predict_last_many({eid: g[fleet.spec.columns] for eid, g in fleet.index.items()})
        engines = []
        for _, row in last.iterrows():
            engines.append({
                "engine_id": int(row["asset"]), "observed_cycles": int(row["time"]), "rul": round(float(row["rul"]), 3),
                "interval_80": {"level": 0.8, "lower": round(float(row["lo80"]), 3), "upper": round(float(row["hi80"]), 3)},
                "risk": RISK_NAMES[int(row["risk"])], "anomaly_alert": bool(row["anomaly_alert"]),
            })
        total += dump(OUT / "fleet" / f"{subset}.json", {"subset": subset, "engines": engines})
        for eid, g in fleet.index.items():
            inputs = g[fleet.spec.columns]
            pred = bundle.predict_history(inputs)
            contrib, raw, base = bundle.explain(inputs)
            signals = [c for c in contrib.columns if c != "time"]
            total += dump(OUT / "engines" / subset / f"{eid}.json", {
                "engine_id": eid, "observed_cycles": int(g["time"].max()),
                "signals": {s: r(g[s], 4) for s in fleet.spec.signals},
                "predictions": {
                    "time": [int(t) for t in pred["time"]], "rul": r(pred["rul"], 3), "lower_80": r(pred["lo80"], 3), "upper_80": r(pred["hi80"], 3),
                    "lower_90": r(pred["lo90"], 3), "upper_90": r(pred["hi90"], 3), "risk": [RISK_NAMES[int(i)] for i in pred["risk"]],
                    "risk_instant": [RISK_NAMES[int(i)] for i in pred["risk_instant"]], "anomaly_score": r(pred["anomaly_score"], 4),
                    "anomaly_alert": [bool(x) for x in pred["anomaly_alert"]],
                },
                "truth": {"time": [int(t) for t in g["time"]], "rul": r(g["rul"], 1)},
            })
            total += dump(OUT / "explain" / subset / f"{eid}.json", {"base_value": round(base, 4), "signals": signals, "raw": r(raw, 3), "contrib": [r(row, 3) for row in contrib[signals].to_numpy()]})
        print(subset, len(engines), "engines exported", flush=True)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    total += dump(OUT / "meta.json", meta)
    total += dump(OUT / "manifest.json", {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_version": commit, "subsets": args.subsets,
                                          "note": "Recorded outputs of the SENTINEL API on held-out C-MAPSS engines. Simulated data; research demonstration."})
    print(f"snapshot written to {OUT} ({total / 1e6:.1f} MB uncompressed)")


if __name__ == "__main__":
    main()
