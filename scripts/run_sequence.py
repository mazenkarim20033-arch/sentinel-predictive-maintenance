"""GRU/LSTM experiments with the same splits, windows, preprocessing and metrics as run_baselines.py.

All four configurations are declared here before any result was seen and all are evaluated once on the
official test set. Early stopping uses held-out training engines; the CV validation fold and the test
set are never used for it.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from run_baselines import EVAL_CAP, FINAL_SEEDS, RAW, REGIMES, ROOT, dev_metrics, summarise
from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import load_subset
from sentinel.data.splits import grouped_kfold_engines, select_engines
from sentinel.eval.metrics import bootstrap_ci, mae, nasa_score, rmse
from sentinel.features.preprocess import Preprocessor, PreprocessConfig
from sentinel.features.windows import make_windows
from sentinel.models.sequence import SeqConfig, SequenceRegressor
from sentinel.registry.records import RunRecord, hardware_info, save_record

CONFIGS = [
    {"name": "gru_w30", "kind": "gru", "window": 30, "cap": 125},
    {"name": "lstm_w30", "kind": "lstm", "window": 30, "cap": 125},
    {"name": "gru_w50", "kind": "gru", "window": 50, "cap": 125},
    {"name": "lstm_w50", "kind": "lstm", "window": 50, "cap": 125},
]


def windows_for(frame, pre, window, last_only=False):
    z = pre.transform(frame).assign(rul=frame["rul"].to_numpy())
    return make_windows(z, pre.feature_cols, window, "rul", last_only=last_only)


def fit_model(cfg, win, seed):
    model = SequenceRegressor(SeqConfig(kind=cfg["kind"]))
    return model.fit(win.X, win.mask, np.minimum(win.y, cfg["cap"]), win.engine, seed=seed)


def run_cv(cfg, train, folds):
    per_fold, fit_times, epochs = [], [], []
    for i, fold in enumerate(folds):
        tr, va = select_engines(train, fold["train"]), select_engines(train, fold["val"])
        pre = Preprocessor(PreprocessConfig(n_regimes=cfg["regimes"])).fit(tr)
        wtr, wva = windows_for(tr, pre, cfg["window"]), windows_for(va, pre, cfg["window"])
        model = fit_model(cfg, wtr, seed=0)
        fit_times.append(model.fit_seconds_)
        epochs.append(model.best_epoch_)
        per_fold.append(dev_metrics(model.predict(wva.X, wva.mask), wva, np.random.default_rng(1000 + i)))
        print(f"  fold {i} cut_rmse {per_fold[-1]['cut_rmse']:.2f} best_epoch {model.best_epoch_} fit {fit_times[-1]:.0f}s", flush=True)
    return summarise(per_fold), float(np.mean(fit_times)), epochs


def run_test(cfg, train, test, subset):
    pre = Preprocessor(PreprocessConfig(n_regimes=cfg["regimes"])).fit(train)
    wtr, wte = windows_for(train, pre, cfg["window"]), windows_for(test, pre, cfg["window"], last_only=True)
    preds, fit_times, models = [], [], []
    for seed in FINAL_SEEDS:
        m = fit_model(cfg, wtr, seed)
        models.append(m)
        fit_times.append(m.fit_seconds_)
        preds.append(np.maximum(m.predict(wte.X, wte.mask), 0))
        print(f"  seed {seed} rmse {rmse(preds[-1], wte.y):.2f} best_epoch {m.best_epoch_} fit {fit_times[-1]:.0f}s", flush=True)
    truth = wte.y
    per_seed = [{"rmse": rmse(p, truth), "mae": mae(p, truth), "score": nasa_score(p, truth)} for p in preds]
    metrics = {k: {"mean": float(np.mean([m[k] for m in per_seed])), "std": float(np.std([m[k] for m in per_seed]))} for k in per_seed[0]}
    mean_pred = np.mean(preds, axis=0)
    metrics["rmse_ci95_seed_mean_prediction"] = bootstrap_ci(rmse, mean_pred, truth)
    metrics["score_ci95_seed_mean_prediction"] = bootstrap_ci(nasa_score, mean_pred, truth)
    metrics["n_test_engines"] = int(len(truth))
    metrics["best_epochs"] = [m.best_epoch_ for m in models]
    one = wte.X[:1], wte.mask[:1]
    lat = []
    for _ in range(30):
        t0 = time.perf_counter()
        models[0].predict(*one)
        lat.append((time.perf_counter() - t0) * 1000)
    artifact = ROOT / "artifacts" / subset / f"{cfg['name']}.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": models[0].net_.state_dict(), "config": models[0].config.model_dump(), "run": cfg,
                "feature_cols": pre.feature_cols}, artifact)
    curve = ROOT / "experiments" / "curves" / f"{subset}_{cfg['name']}_seed0.json"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text(json.dumps(models[0].history_))
    return metrics, float(np.mean(fit_times)), float(np.median(lat)), len(models[0].state_bytes()), artifact, curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    data = load_subset(RAW, args.subset)
    train, test = add_train_rul(data.train), add_test_rul(data.test, data.test_final_rul)
    folds = grouped_kfold_engines(train, n_splits=5, seed=0)
    hw, stamp = hardware_info(), datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    for cfg in CONFIGS:
        if args.only and cfg["name"] not in args.only:
            continue
        cfg = {**cfg, "regimes": cfg.get("regimes") or REGIMES[args.subset]}
        print(cfg["name"], flush=True)
        cv, cv_fit, cv_epochs = run_cv(cfg, train, folds)
        m, fit_s, lat, size, art, curve = run_test(cfg, train, test, args.subset)
        rec = RunRecord(
            run_id=f"{args.subset}-{cfg['name']}-{stamp}", subset=args.subset, name=cfg["name"], model=cfg["kind"],
            config={**cfg, **SeqConfig(kind=cfg["kind"]).model_dump(), "features": "raw sequence + mask channel", "test": True},
            split={"type": "grouped_kfold", "n_splits": 5, "seed": 0, "file": f"experiments/splits/{args.subset}_kfold5_seed0.json"},
            preprocessing={"n_regimes": cfg["regimes"], "per_regime_zscore": True, "fit_scope": "fold train engines only"},
            seeds=FINAL_SEEDS, cv_metrics={**cv, "cv_best_epochs": cv_epochs}, test_metrics=m, train_seconds=fit_s,
            latency_single_ms=lat, model_size_bytes=size, artifact_path=str(art.relative_to(ROOT)),
            training_curve_path=str(curve.relative_to(ROOT)), hardware=hw,
            notes="Early stopping on 15% held-out training engines; CV uses model seed 0; test = mean of 5 seeds trained on all training engines.",
        )
        save_record(rec)
        print(f"{cfg['name']}: cv cut_rmse {cv['cut_rmse']['mean']:.2f} | TEST rmse {m['rmse']['mean']:.2f}±{m['rmse']['std']:.2f} score {m['score']['mean']:.0f}", flush=True)


if __name__ == "__main__":
    main()
