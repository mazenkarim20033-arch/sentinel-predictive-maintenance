"""Latency, concurrency and memory measurements for the real API process on this machine. Writes reports/performance.md
and experiments/performance.json. Numbers describe one CPU laptop and one uvicorn worker, not production capacity."""
import json
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psutil

from serve_helper import ROOT, running_server

REPS = 30
CONC_WORKERS, CONC_REQUESTS = 8, 25


def ms(fn) -> float:
    t0 = time.perf_counter()
    r = fn()
    dt = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200, (r.status_code, r.text[:200])
    return dt


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "p50_ms": round(statistics.median(xs), 1), "p95_ms": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1),
            "max_ms": round(xs[-1], 1), "mean_ms": round(statistics.fmean(xs), 1)}


def history_rows(n: int) -> list[dict]:
    """Real FD001 test history of engine 34 (203 cycles), tiled with re-indexed time when more rows are needed (stress input only)."""
    from sentinel.api.services import DiskServices

    svc = DiskServices()
    g = svc.fleet("FD001").index[34]
    base = g[svc.spec("FD001").columns].drop(columns=["asset"])
    rows = []
    while len(rows) < n:
        rows += base.to_dict(orient="records")
    rows = rows[:n]
    for i, r in enumerate(rows, start=1):
        r["time"] = float(i)
    return rows


def main() -> None:
    out: dict = {"hardware": {"processor": platform.processor(), "logical_cpus": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
                              "python": platform.python_version(), "platform": platform.platform(), "server": "uvicorn, 1 worker, CPU only"}}
    with running_server() as (base, proc, _, startup_s):
        p = psutil.Process(proc.pid)

        def rss_mb() -> int:
            """Resident memory of the whole process tree (the Windows venv launcher starts the real interpreter as a child)."""
            procs = [p, *p.children(recursive=True)]
            return round(sum(q.memory_info().rss for q in procs) / 1e6)

        out["startup_seconds_to_healthy"] = round(startup_s, 2)
        out["server_rss_mb_idle"] = rss_mb()
        c = httpx.Client(base_url=base, timeout=120)

        cold = {}
        for subset in ("FD001", "FD004"):
            cold[f"GET /fleet/{subset} (first call, computes every engine)"] = round(ms(lambda s=subset: c.get(f"/fleet/{s}", params={"limit": 500})), 1)
        out["cold_ms"] = cold
        out["server_rss_mb_after_fleet"] = rss_mb()

        rows50, rows200, rows1000, rows3000 = (history_rows(n) for n in (50, 200, 1000, 3000))
        calls = {
            "GET /health": lambda: c.get("/health"),
            "GET /subsets": lambda: c.get("/subsets"),
            "GET /fleet/FD001 (cached, 25 rows)": lambda: c.get("/fleet/FD001", params={"limit": 25}),
            "GET /engines/FD001/34 (203 cycles)": lambda: c.get("/engines/FD001/34"),
            "GET /replay cycle 10": lambda: c.get("/replay/FD001/34", params={"cycle": 10}),
            "GET /replay cycle 100": lambda: c.get("/replay/FD001/34", params={"cycle": 100}),
            "GET /replay cycle 200": lambda: c.get("/replay/FD001/34", params={"cycle": 200, "mode": "eval"}),
            "GET /explain cycle 200": lambda: c.get("/explain/FD001/34", params={"cycle": 200}),
            "GET /evaluation/FD001": lambda: c.get("/evaluation/FD001"),
            "GET /models": lambda: c.get("/models"),
            "POST /predict/rul 50 rows": lambda: c.post("/predict/rul", json={"subset": "FD001", "history": rows50}),
            "POST /predict/rul 200 rows": lambda: c.post("/predict/rul", json={"subset": "FD001", "history": rows200}),
            "POST /predict/rul 1000 rows (tiled stress input)": lambda: c.post("/predict/rul", json={"subset": "FD001", "history": rows1000}),
            "POST /predict/rul 3000 rows (tiled stress input, API maximum)": lambda: c.post("/predict/rul", json={"subset": "FD001", "history": rows3000}),
        }
        for fn in calls.values():
            fn()  # warm-up
        out["warm_sequential"] = {name: stats([ms(fn) for _ in range(REPS)]) for name, fn in calls.items()}

        def worker(fn):
            return [ms(fn) for _ in range(CONC_REQUESTS)]

        out["concurrent"] = {}
        for name in ("GET /replay cycle 100", "POST /predict/rul 200 rows"):
            with httpx.Client(base_url=base, timeout=120) as _:
                t0 = time.perf_counter()
                with ThreadPoolExecutor(CONC_WORKERS) as pool:
                    lat = [x for xs in pool.map(lambda _: worker(calls[name]), range(CONC_WORKERS)) for x in xs]
                wall = time.perf_counter() - t0
            out["concurrent"][name] = {**stats(lat), "workers": CONC_WORKERS, "requests": len(lat), "throughput_rps": round(len(lat) / wall, 1)}
        out["server_rss_mb_after_load"] = rss_mb()

    from sentinel.inference.bundle import ModelBundle

    t0 = time.perf_counter()
    bundle = ModelBundle.load(ROOT / "artifacts" / "bundles" / "FD001.joblib")
    out["bundle_load_seconds"] = round(time.perf_counter() - t0, 2)
    import pandas as pd

    frame = pd.DataFrame(history_rows(1000)).assign(asset=1)
    timings = {}
    for n in (50, 200, 1000):
        sub = frame.iloc[:n]
        bundle.predict_history(sub)
        xs = []
        for _ in range(20):
            t0 = time.perf_counter()
            bundle.predict_history(sub)
            xs.append((time.perf_counter() - t0) * 1000)
        timings[f"predict_history, {n} rows"] = stats(xs)
    out["model_in_process"] = timings

    (ROOT / "experiments" / "performance.json").write_text(json.dumps(out, indent=2))
    hw = out["hardware"]
    L = ["# API performance (generated by scripts/benchmark_api.py)", "",
         f"Machine: {hw['processor']}, {hw['logical_cpus']} logical CPUs, {hw['ram_gb']} GB RAM; {hw['server']}; Python {hw['python']}. "
         "These are single-machine, single-worker figures measured on a quiet laptop, not production capacity. Requests are sequential unless stated; 30 repetitions after one warm-up.", "",
         f"Server start to healthy: {out['startup_seconds_to_healthy']} s. Memory (resident): {out['server_rss_mb_idle']} MB idle, {out['server_rss_mb_after_fleet']} MB after building two fleets, {out['server_rss_mb_after_load']} MB after the load test. Bundle load from disk (in process): {out['bundle_load_seconds']} s.", "",
         "## First calls (cold)", "", "| Request | ms |", "|---|---|"]
    L += [f"| {k} | {v:,.0f} |" for k, v in out["cold_ms"].items()]
    L += ["", "## Warm sequential requests", "", "| Request | p50 ms | p95 ms | max ms |", "|---|---|---|---|"]
    L += [f"| {k} | {v['p50_ms']} | {v['p95_ms']} | {v['max_ms']} |" for k, v in out["warm_sequential"].items()]
    L += ["", f"## Concurrent ({CONC_WORKERS} clients, {CONC_REQUESTS} requests each)", "", "| Request | throughput req/s | p50 ms | p95 ms | max ms |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['throughput_rps']} | {v['p50_ms']} | {v['p95_ms']} | {v['max_ms']} |" for k, v in out["concurrent"].items()]
    L += ["", "## Model alone (no HTTP), whole-history prediction for every cycle", "", "| Call | p50 ms | p95 ms |", "|---|---|---|"]
    L += [f"| {k} | {v['p50_ms']} | {v['p95_ms']} |" for k, v in out["model_in_process"].items()]
    L += ["", "Notes: replay recomputes the model on the visible history at every step (so the server provably never sees later cycles); its cost grows with the cycle number. "
          "The 1000 and 3000 row inputs are real engine history tiled with re-indexed time, used only to stress the endpoint."]
    (ROOT / "reports" / "performance.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
