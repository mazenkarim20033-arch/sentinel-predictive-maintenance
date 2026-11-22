"""Baseline experiments on one C-MAPSS subset.

Dev evaluation: grouped 5-fold CV over training engines (preprocessing refit inside every fold).
Official test: evaluated once per final configuration (test=True), never for ablations.
"""
import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import load_subset
from sentinel.data.splits import grouped_kfold_engines, save_split, select_engines
from sentinel.eval.metrics import bootstrap_ci, mae, nasa_score, rmse
from sentinel.features.preprocess import Preprocessor, PreprocessConfig
from sentinel.features.tabular import tabular_features
from sentinel.features.windows import make_windows
from sentinel.models.baselines import build_model
from sentinel.registry.records import RunRecord, hardware_info, save_record

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "extracted"
EVAL_CAP = 125
FINAL_SEEDS = [0, 1, 2, 3, 4]
REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

BASE = dict(features="all", cap=125, window=30, cycle=False, test=False)
CONFIGS = [
    {**BASE, "name": "constant", "model": "constant", "test": True},
    {**BASE, "name": "ridge_last", "model": "ridge", "features": "last", "test": True},
    {**BASE, "name": "random_forest", "model": "rf", "test": True},
    {**BASE, "name": "lightgbm", "model": "lgbm", "test": True},
    {**BASE, "name": "ridge_all_features", "model": "ridge"},
    {**BASE, "name": "lightgbm_uncapped", "model": "lgbm", "cap": None},
    {**BASE, "name": "lightgbm_cap100", "model": "lgbm", "cap": 100},
    {**BASE, "name": "lightgbm_cap150", "model": "lgbm", "cap": 150},
    {**BASE, "name": "lightgbm_window15", "model": "lgbm", "window": 15},
    {**BASE, "name": "lightgbm_window50", "model": "lgbm", "window": 50, "test": True},
    {**BASE, "name": "lightgbm_with_cycle", "model": "lgbm", "cycle": True},
    {**BASE, "name": "lightgbm_global_norm", "model": "lgbm", "window": 50, "regimes": 1},
]


def build_xy(frame: pd.DataFrame, pre: Preprocessor, cfg: dict, last_only: bool = False):
    z = pre.transform(frame).assign(rul=frame["rul"].to_numpy())
    win = make_windows(z, pre.feature_cols, cfg["window"], "rul", last_only=last_only)
    X = tabular_features(win, pre.feature_cols, include_cycle=cfg["cycle"])
    if cfg["features"] == "last":
        X = X[[c for c in X.columns if c.endswith("_last")]]
    return X, win


def train_target(y: np.ndarray, cap: int | None) -> np.ndarray:
    return np.minimum(y, cap) if cap else y


def dev_metrics(pred: np.ndarray, win, rng: np.random.Generator) -> dict[str, float]:
    true = win.y
    p_c, t_c = np.clip(pred, 0, EVAL_CAP), np.minimum(true, EVAL_CAP)
    out = {"all_mae": mae(p_c, t_c), "all_rmse": rmse(p_c, t_c)}
    late = true <= 50
    out["late_mae"], out["late_rmse"] = mae(p_c[late], t_c[late]), rmse(p_c[late], t_c[late])
    eligible = (true >= 10) & (true <= 150) & (win.cycle >= 31)
    chosen = []
    for engine in np.unique(win.engine[eligible]):
        idx = np.where(eligible & (win.engine == engine))[0]
        chosen.extend(rng.choice(idx, size=min(10, len(idx)), replace=False))
    chosen = np.array(chosen)
    p_u = np.maximum(pred[chosen], 0)
    out["cut_mae"], out["cut_rmse"] = mae(p_u, true[chosen]), rmse(p_u, true[chosen])
    out["cut_score_per_point"] = nasa_score(p_u, true[chosen]) / len(chosen)
    return out


def summarise(per_fold: list[dict]) -> dict:
    keys = per_fold[0].keys()
    return {k: {"mean": float(np.mean([f[k] for f in per_fold])), "std": float(np.std([f[k] for f in per_fold]))} for k in keys}


def run_cv(cfg: dict, train: pd.DataFrame, folds: list[dict]) -> tuple[dict, float]:
    per_fold, fit_times = [], []
    for i, fold in enumerate(folds):
        tr, va = select_engines(train, fold["train"]), select_engines(train, fold["val"])
        pre = Preprocessor(PreprocessConfig(n_regimes=cfg["regimes"])).fit(tr)
        Xtr, wtr = build_xy(tr, pre, cfg)
        Xva, wva = build_xy(va, pre, cfg)
        t0 = time.perf_counter()
        model = build_model(cfg["model"], seed=0).fit(Xtr, train_target(wtr.y, cfg["cap"]))
        fit_times.append(time.perf_counter() - t0)
        per_fold.append(dev_metrics(model.predict(Xva), wva, np.random.default_rng(1000 + i)))
    return summarise(per_fold), float(np.mean(fit_times))


def run_test(cfg: dict, train: pd.DataFrame, test: pd.DataFrame, subset: str):
    pre = Preprocessor(PreprocessConfig(n_regimes=cfg["regimes"])).fit(train)
    Xtr, wtr = build_xy(train, pre, cfg)
    Xte, wte = build_xy(test, pre, cfg, last_only=True)
    y_tr, truth = train_target(wtr.y, cfg["cap"]), wte.y
    preds, fit_times, first_model = [], [], None
    for seed in FINAL_SEEDS:
        t0 = time.perf_counter()
        model = build_model(cfg["model"], seed=seed).fit(Xtr, y_tr)
        fit_times.append(time.perf_counter() - t0)
        preds.append(np.maximum(model.predict(Xte), 0))
        first_model = first_model or model
    per_seed = [{"rmse": rmse(p, truth), "mae": mae(p, truth), "score": nasa_score(p, truth)} for p in preds]
    mean_pred = np.mean(preds, axis=0)
    metrics = {k: {"mean": float(np.mean([m[k] for m in per_seed])), "std": float(np.std([m[k] for m in per_seed]))} for k in per_seed[0]}
    metrics["rmse_ci95_seed_mean_prediction"] = bootstrap_ci(rmse, mean_pred, truth)
    metrics["score_ci95_seed_mean_prediction"] = bootstrap_ci(nasa_score, mean_pred, truth)
    metrics["n_test_engines"] = int(len(truth))
    row = Xte.iloc[[0]]
    lat = []
    for _ in range(30):
        t0 = time.perf_counter()
        first_model.predict(row)
        lat.append((time.perf_counter() - t0) * 1000)
    buf = io.BytesIO()
    joblib.dump(first_model, buf)
    artifact = ROOT / "artifacts" / subset / f"{cfg['name']}.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": first_model, "preprocessor": pre, "config": cfg, "columns": list(Xtr.columns)}, artifact)
    return metrics, float(np.mean(fit_times)), float(np.median(lat)), buf.getbuffer().nbytes, str(artifact.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    data = load_subset(RAW, args.subset)
    train = add_train_rul(data.train)
    test = add_test_rul(data.test, data.test_final_rul)
    folds = grouped_kfold_engines(train, n_splits=5, seed=0)
    save_split({"folds": folds}, ROOT / "experiments" / "splits" / f"{args.subset}_kfold5_seed0.json")
    hw = hardware_info()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    for cfg in CONFIGS:
        if args.only and cfg["name"] not in args.only:
            continue
        cfg = {**cfg, "regimes": cfg.get("regimes") or REGIMES[args.subset]}
        cv, fit_s = run_cv(cfg, train, folds)
        record = RunRecord(
            run_id=f"{args.subset}-{cfg['name']}-{stamp}", subset=args.subset, name=cfg["name"], model=cfg["model"],
            config=cfg, split={"type": "grouped_kfold", "n_splits": 5, "seed": 0, "file": f"experiments/splits/{args.subset}_kfold5_seed0.json"},
            preprocessing={"n_regimes": cfg["regimes"], "per_regime_zscore": True, "fit_scope": "fold train engines only"},
            seeds=FINAL_SEEDS if cfg["test"] else [0], cv_metrics=cv, train_seconds=fit_s, hardware=hw,
            notes="CV uses model seed 0; test metrics (if present) are mean over 5 seeds on all train engines.",
        )
        if cfg["test"]:
            m, fit_s, lat, size, art = run_test(cfg, train, test, args.subset)
            record.test_metrics, record.train_seconds = m, fit_s
            record.latency_single_ms, record.model_size_bytes, record.artifact_path = lat, size, art
        save_record(record)
        t = record.test_metrics
        print(f"{cfg['name']:<22} cv cut_rmse {cv['cut_rmse']['mean']:.2f}±{cv['cut_rmse']['std']:.2f}"
              f" all_rmse {cv['all_rmse']['mean']:.2f}" + (f" | TEST rmse {t['rmse']['mean']:.2f} score {t['score']['mean']:.0f}" if t else ""), flush=True)


if __name__ == "__main__":
    main()
