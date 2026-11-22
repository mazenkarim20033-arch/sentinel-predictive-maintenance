"""Failure-warning evaluation for the default model (LightGBM, window 50).

Development: out-of-fold predictions on training assets (5-fold grouped CV, full run-to-failure trajectories),
used for the threshold/persistence sweep and for lead time. Official test: cycle-level metrics for the fixed
default operating points only; test assets are truncated before failure, so lead time is not measurable there.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_baselines import CONFIGS, RAW, ROOT, build_xy, make_pre, train_target
from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import load_dataset
from sentinel.data.splits import grouped_kfold_assets, select_assets
from sentinel.eval.warning import CLASS_NAMES, evaluate_warnings, three_class_confusion
from sentinel.models.baselines import build_model
from sentinel.registry.records import RUNS_DIR, RunRecord, hardware_info, save_record
from sentinel.spec import CONFIGS_DIR, load_spec

DEFAULT_TAUS = (30, 15)
DEFAULT_K = 3
SWEEP_TAUS = (10, 15, 20, 30, 40, 50)
SWEEP_KS = (1, 3, 5)
CYCLE_KEYS = ("precision", "recall", "f1", "false_alarm_rate")


def fit_predict(cfg, spec, train, evaluate_frame, last_only=False, seed=0):
    pre = make_pre(spec, cfg).fit(train)
    Xtr, wtr = build_xy(train, pre, cfg)
    Xev, wev = build_xy(evaluate_frame, pre, cfg, last_only=last_only)
    model = build_model(cfg["model"], seed=seed).fit(Xtr, train_target(wtr.y, cfg["cap"]))
    return pd.DataFrame({"asset": wev.asset, "time": wev.time, "true": wev.y, "pred": np.maximum(model.predict(Xev), 0)})


def oof_predictions(cfg, spec, train, folds):
    return pd.concat(
        [fit_predict(cfg, spec, select_assets(train, f["train"]), select_assets(train, f["val"])) for f in folds],
        ignore_index=True,
    )


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def run(subset: str, runs_dir: Path) -> None:
    spec = load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
    data = load_dataset(spec, RAW)
    train, test = add_train_rul(data.train), add_test_rul(data.test, data.test_final_rul)
    cfg = {**next(c for c in CONFIGS if c["name"] == "lightgbm_window50"), "regimes": spec.n_regimes}
    folds = grouped_kfold_assets(train, n_splits=5, seed=0)
    oof = oof_predictions(cfg, spec, train, folds)

    sweep = [evaluate_warnings(oof, tau, k) for tau in SWEEP_TAUS for k in SWEEP_KS]
    levels = np.arange(5, 91, 5)
    curves = {(tau, k): [evaluate_warnings(oof, tau, k, alert_level=lv) for lv in levels] for tau in DEFAULT_TAUS for k in (1, DEFAULT_K)}
    defaults = {tau: evaluate_warnings(oof, tau, DEFAULT_K) for tau in DEFAULT_TAUS}
    conf = three_class_confusion(oof["true"].to_numpy(), oof["pred"].to_numpy())

    test_pred = fit_predict(cfg, spec, train, test)
    test_default = {tau: evaluate_warnings(test_pred, tau, DEFAULT_K) for tau in DEFAULT_TAUS}
    test_conf = three_class_confusion(test_pred["true"].to_numpy(), test_pred["pred"].to_numpy())

    lines = [f"# Failure warnings, {subset} (generated)", "",
             f"Model: LightGBM, window 50, cap 125 (fixed). Truth: true RUL <= tau. Alert: predicted RUL <= tau for k={DEFAULT_K} consecutive cycles unless stated. "
             "Thresholds are experimental operating points, not maintenance standards; predicted RUL is not a calibrated failure probability.", "",
             f"## Development: out-of-fold on {oof['asset'].nunique()} training assets, {len(oof)} cycles", "",
             "| tau | k | precision | recall | F1 | false-alarm rate | assets with a useful alert | missed | early alarms | median lead | lead p10-p90 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sweep:
        lines.append(f"| {m['tau']} | {m['k']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['false_alarm_rate'])} | "
                     f"{fmt(m['useful_alert_frac'])} | {fmt(m['missed_frac'])} | {fmt(m['early_alarm_frac'])} | {fmt(m['median_lead'], 1)} | "
                     f"{fmt(m['lead_p10'], 0)}-{fmt(m['lead_p90'], 0)} |")
    lines += ["", "Early alarm = first alert with true RUL above 2 x tau. Lead = true RUL (cycles before failure) at the first alert.", "",
              "### Three-class confusion at the default thresholds (normal > 30, warning 15-30, critical < 15), out-of-fold, no persistence",
              "", "| true \\ predicted | " + " | ".join(CLASS_NAMES) + " |", "|---|---|---|---|"]
    lines += [f"| {CLASS_NAMES[i]} | " + " | ".join(str(v) for v in conf[i]) + " |" for i in range(3)]
    lines += ["", f"## Official test, fixed default operating points (k={DEFAULT_K}), cycle level only", "",
              "Test assets stop before failure, so lead time is not measurable. Positives are few, and metrics pool many correlated cycles from 100-250 assets.", "",
              "| tau | precision | recall | F1 | false-alarm rate | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|---|"]
    for tau, m in test_default.items():
        lines.append(f"| {tau} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['false_alarm_rate'])} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |")
    lines += ["", "Test three-class confusion (no persistence):", "", "| true \\ predicted | " + " | ".join(CLASS_NAMES) + " |", "|---|---|---|---|"]
    lines += [f"| {CLASS_NAMES[i]} | " + " | ".join(str(v) for v in test_conf[i]) + " |" for i in range(3)]
    out_md = ROOT / "reports" / f"warning_{subset}.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (tau, k), ms in curves.items():
        ax.plot([m["false_alarm_rate"] for m in ms], [m["recall"] for m in ms], marker=".", label=f"tau={tau}, k={k}")
    ax.set_xlabel("false-alarm rate (cycle level)")
    ax.set_ylabel("recall (cycle level)")
    ax.set_title(f"{subset}: recall vs false alarms as the alert level varies (out-of-fold)")
    ax.title.set_fontsize(8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"warning_tradeoff_{subset}.png", dpi=110)

    rec = RunRecord(
        run_id=f"{subset}-warning-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}", subset=f"{subset}-warning",
        name="warning_lightgbm_window50", model="lgbm", config={**cfg, "default_taus": DEFAULT_TAUS, "default_k": DEFAULT_K},
        split={"type": "grouped_kfold", "n_splits": 5, "seed": 0}, preprocessing={"n_regimes": cfg["regimes"]}, seeds=[0],
        cv_metrics={"default_oof": {str(t): m for t, m in defaults.items()}, "sweep": sweep, "confusion_oof": conf.tolist()},
        test_metrics={"default": {str(t): {k: m[k] for k in (*CYCLE_KEYS, "tp", "fp", "fn", "tn")} for t, m in test_default.items()},
                      "confusion": test_conf.tolist()},
        hardware=hardware_info(), notes="Sweep and lead time on out-of-fold training assets; test = fixed default points, cycle level only.",
    )
    save_record(rec, runs_dir)
    for tau, m in defaults.items():
        print(f"{subset} OOF tau={tau} k={DEFAULT_K}: precision {fmt(m['precision'])} recall {fmt(m['recall'])} F1 {fmt(m['f1'])} "
              f"FAR {fmt(m['false_alarm_rate'])} useful {fmt(m['useful_alert_frac'])} missed {fmt(m['missed_frac'])} "
              f"early {fmt(m['early_alarm_frac'])} median lead {fmt(m['median_lead'], 1)}")
    for tau, m in test_default.items():
        print(f"{subset} TEST tau={tau}: precision {fmt(m['precision'])} recall {fmt(m['recall'])} FAR {fmt(m['false_alarm_rate'])}")
    print(out_md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", default=["FD001", "FD004"])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    for subset in args.subsets:
        run(subset, args.runs_dir)


if __name__ == "__main__":
    main()
