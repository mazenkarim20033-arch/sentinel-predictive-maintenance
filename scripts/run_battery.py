"""NASA battery demonstration of the machine-agnostic pipeline.

Protocol: leave-one-cell-out validation over 4 cells. There is NO held-out test set, so every number here is
a development estimate from 4 trajectories, not a benchmark. Configurations are fixed in advance, nothing is tuned.
Stochastic models are averaged over 5 seeds inside each fold.
"""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_baselines import ROOT
from sentinel.data.labels import add_train_rul
from sentinel.data.loader import load_dataset
from sentinel.data.splits import leave_one_asset_out, save_split, select_assets
from sentinel.eval.metrics import mae, rmse
from sentinel.features.preprocess import Preprocessor, PreprocessConfig
from sentinel.features.tabular import tabular_features
from sentinel.features.windows import make_windows
from sentinel.models.baselines import build_model
from sentinel.registry.records import RUNS_DIR, RunRecord, hardware_info, save_record
from sentinel.spec import CONFIGS_DIR, load_spec

RAW = ROOT / "data" / "raw" / "battery" / "extracted"
SEEDS = [0, 1, 2, 3, 4]
LATE_RUL = 20
MODELS = [
    {"name": "constant", "model": "constant", "features": "last", "stochastic": False},
    {"name": "age_only", "model": "age", "features": "last", "stochastic": False},
    {"name": "ridge_last", "model": "ridge", "features": "last", "stochastic": False},
    {"name": "random_forest", "model": "rf", "features": "all", "stochastic": True},
    {"name": "lightgbm", "model": "lgbm", "features": "all", "stochastic": True},
]


def fold_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.maximum(pred, 0)
    late = true <= LATE_RUL
    return {"rmse": rmse(pred, true), "mae": mae(pred, true), "late_rmse": rmse(pred[late], true[late])}


def build(frame, pre, window):
    z = pre.transform(frame).assign(rul=frame["rul"].to_numpy())
    win = make_windows(z, pre.feature_cols, window, "rul")
    return tabular_features(win, pre.feature_cols), win


def predict_fold(cfg, spec, tr, va):
    pre = Preprocessor(PreprocessConfig.from_spec(spec)).fit(tr)
    Xtr, wtr = build(tr, pre, spec.window)
    Xva, wva = build(va, pre, spec.window)
    if cfg["features"] == "last":
        cols = [c for c in Xtr.columns if c.endswith("_last")]
        Xtr, Xva = Xtr[cols], Xva[cols]
    if cfg["model"] == "age":
        mean_life = tr.groupby("asset")["time"].max().mean()
        return [np.maximum(mean_life - wva.time, 0)], wva.y
    seeds = SEEDS if cfg["stochastic"] else [0]
    preds = [build_model(cfg["model"], seed=s).fit(Xtr, wtr.y).predict(Xva) for s in seeds]
    return preds, wva.y


def run(spec_name: str, runs_dir: Path) -> list[RunRecord]:
    spec = load_spec(CONFIGS_DIR / f"{spec_name}.yaml")
    data = load_dataset(spec, RAW)
    train = add_train_rul(data.train)
    cells = data.asset_info.set_index("asset")["cell"].to_dict()
    folds = leave_one_asset_out(train)
    save_split({"folds": folds, "cells": cells}, ROOT / "experiments" / "splits" / f"{spec_name}_loco.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    records = []
    for cfg in MODELS:
        per_cell, t0 = {}, time.perf_counter()
        for fold in folds:
            tr, va = select_assets(train, fold["train"]), select_assets(train, fold["val"])
            preds, true = predict_fold(cfg, spec, tr, va)
            seed_metrics = [fold_metrics(p, true) for p in preds]
            per_cell[cells[fold["val"][0]]] = {k: float(np.mean([m[k] for m in seed_metrics])) for k in seed_metrics[0]}
        elapsed = time.perf_counter() - t0
        summary = {k: {"mean": float(np.mean([c[k] for c in per_cell.values()])),
                       "std": float(np.std([c[k] for c in per_cell.values()]))} for k in ("rmse", "mae", "late_rmse")}
        rec = RunRecord(
            run_id=f"{spec_name}-{cfg['name']}-{stamp}", subset=spec_name, name=cfg["name"], model=cfg["model"],
            config={**cfg, "window": spec.window, "regimes": 1, "use_time": False},
            split={"type": "leave_one_asset_out", "n_folds": len(folds), "file": f"experiments/splits/{spec_name}_loco.json"},
            preprocessing={"n_regimes": 1, "per_regime_zscore": True, "fit_scope": "training cells only"},
            seeds=SEEDS if cfg["stochastic"] else [0], cv_metrics={**summary, "per_cell": per_cell},
            train_seconds=elapsed / len(folds), hardware=hardware_info(),
            notes="Leave-one-cell-out over 4 cells; no held-out test set; mean/std are over cells.",
        )
        save_record(rec, runs_dir)
        records.append(rec)
        print(f"{spec_name:<28} {cfg['name']:<14} rmse {summary['rmse']['mean']:.1f}±{summary['rmse']['std']:.1f}"
              f" mae {summary['mae']['mean']:.1f} late_rmse {summary['late_rmse']['mean']:.1f}", flush=True)
    return records


def write_report(all_records: dict[str, list[RunRecord]], spec_names: list[str]) -> Path:
    lines = ["# Battery demonstration (generated from experiments/runs)", "",
             "Leave-one-cell-out over 4 NASA cells (B0005, B0006, B0007, B0018). No held-out test set; 4 trajectories only.",
             "RUL is in discharge cycles. End of life = capacity at or below 1.5 Ah (project-defined, 25% fade).", ""]
    for name in spec_names:
        recs = all_records[name]
        cells = list(recs[0].cv_metrics["per_cell"])
        lines += [f"## {name}", "", "| Model | RMSE (mean ± std over cells) | MAE | Late-life RMSE (RUL <= 20) | " +
                  " | ".join(f"RMSE {c}" for c in cells) + " |", "|---|---|---|---|" + "---|" * len(cells)]
        for r in sorted(recs, key=lambda r: r.cv_metrics["rmse"]["mean"]):
            m = r.cv_metrics
            lines.append(f"| {r.name} | {m['rmse']['mean']:.1f} ± {m['rmse']['std']:.1f} | {m['mae']['mean']:.1f} | "
                         f"{m['late_rmse']['mean']:.1f} | " + " | ".join(f"{m['per_cell'][c]['rmse']:.1f}" for c in cells) + " |")
        lines.append("")
    out = ROOT / "reports" / "leaderboard_battery.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    names = ["battery_nasa_sensors", "battery_nasa_with_capacity"]
    all_records = {n: run(n, args.runs_dir) for n in names}
    print(write_report(all_records, names))


if __name__ == "__main__":
    main()
