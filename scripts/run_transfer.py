"""Cross-subset transfer: train LightGBM (window 50, cap 125, seed 0) on one C-MAPSS subset and evaluate it once on every
subset's official test set. Two variants:
  source_norm: preprocessing (regimes, per-regime z-scoring) fitted on the source training data only;
  target_norm: same signals, but normalisation statistics and regimes fitted on the *unlabelled training sensors of the
               target subset* (no target labels and no target test data).
The diagonal is the in-subset result. Nothing is tuned; this is a one-shot evaluation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_baselines import CONFIGS, RAW, ROOT, build_xy, make_pre, train_target
from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import SUBSETS, load_dataset
from sentinel.eval.metrics import mae, nasa_score, rmse
from sentinel.features.preprocess import Preprocessor, PreprocessConfig
from sentinel.models.baselines import build_model
from sentinel.registry.records import RUNS_DIR, RunRecord, hardware_info, save_record
from sentinel.spec import CONFIGS_DIR, load_spec

CAP = 125
VARIANTS = ("source_norm", "target_norm")


def evaluate(model, pre, cfg, test):
    X, w = build_xy(test, pre, cfg, last_only=True)
    pred = np.maximum(model.predict(X), 0)
    return {"rmse": rmse(pred, w.y), "mae": mae(pred, w.y), "score": nasa_score(pred, w.y), "bias": float(np.mean(pred - w.y)), "n": int(len(w.y))}


def run(runs_dir: Path) -> None:
    base = next(c for c in CONFIGS if c["name"] == "lightgbm_window50")
    specs = {s: load_spec(CONFIGS_DIR / f"cmapss_{s}.yaml") for s in SUBSETS}
    data = {s: load_dataset(specs[s], RAW) for s in SUBSETS}
    train = {s: add_train_rul(data[s].train) for s in SUBSETS}
    test = {s: add_test_rul(data[s].test, data[s].test_final_rul) for s in SUBSETS}
    results = {v: {} for v in VARIANTS}
    for src in SUBSETS:
        cfg = {**base, "regimes": specs[src].n_regimes}
        pre = make_pre(specs[src], cfg).fit(train[src])
        X, w = build_xy(train[src], pre, cfg)
        model = build_model("lgbm", 0).fit(X, train_target(w.y, CAP))
        for tgt in SUBSETS:
            results["source_norm"][f"{src}->{tgt}"] = evaluate(model, pre, cfg, test[tgt])
            tgt_pre = Preprocessor(PreprocessConfig(signals=pre.signals_, conditions=specs[tgt].conditions, n_regimes=specs[tgt].n_regimes, min_std=1e-30)).fit(train[tgt])
            tgt_pre.signals_ = list(pre.signals_)  # the source model needs exactly these columns; constant target signals become 0
            results["target_norm"][f"{src}->{tgt}"] = evaluate(model, tgt_pre, cfg, test[tgt])
        print(src, "done", flush=True)

    lines = ["# Cross-subset transfer (generated)", "",
             "LightGBM, window 50, cap 125, seed 0, trained on all training assets of the row (source) subset and evaluated once on the official test set of the column (target) subset. "
             "Diagonal = in-subset. `source_norm` uses preprocessing fitted on the source; `target_norm` re-fits only the normalisation and regimes on the target's unlabelled training sensors "
             "(no target labels, no target test data). Single seed, no confidence intervals; differences of 1-2 RMSE points on 100-250 test assets are within noise.", ""]
    for v in VARIANTS:
        for metric, label in (("rmse", "Test RMSE (cycles)"), ("bias", "Mean error, prediction minus truth (cycles)"), ("score", "NASA score")):
            lines += [f"## {label}, {v}", "", "| source \\ target | " + " | ".join(SUBSETS) + " |", "|---|" + "---|" * len(SUBSETS)]
            for src in SUBSETS:
                cells = []
                for tgt in SUBSETS:
                    val = results[v][f"{src}->{tgt}"][metric]
                    cells.append((f"**{val:.1f}**" if src == tgt else f"{val:.1f}") if metric != "score" else (f"**{val:,.0f}**" if src == tgt else f"{val:,.0f}"))
                lines.append(f"| {src} | " + " | ".join(cells) + " |")
            lines.append("")
    (ROOT / "reports" / "transfer.md").write_text("\n".join(lines), encoding="utf-8")
    save_record(RunRecord(
        run_id=f"transfer-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}", subset="transfer", name="transfer_lightgbm_window50", model="lgbm",
        config={**base, "variants": VARIANTS}, split={"type": "official_train_test_cross_subset"}, preprocessing={"variants": VARIANTS}, seeds=[0],
        cv_metrics={}, test_metrics=results, hardware=hardware_info(), notes="One-shot cross-subset evaluation; no tuning.",
    ), runs_dir)
    for v in VARIANTS:
        print(v)
        for src in SUBSETS:
            print("  ", src, " ".join(f"{results[v][f'{src}->{t}']['rmse']:6.1f}" for t in SUBSETS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    run(parser.parse_args().runs_dir)


if __name__ == "__main__":
    main()
