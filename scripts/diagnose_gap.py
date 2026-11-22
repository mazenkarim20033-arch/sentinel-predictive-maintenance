"""Why is official-test error much larger than cross-validation error? Diagnostics only; no model selection.

Compares the actual test RMSE (one point per test asset) with out-of-fold RMSE re-sampled to look like the test set:
for every held-out training asset, draw a target RUL from the empirical test final-RUL distribution and score the cycle with that true RUL.
"""
import argparse

import numpy as np
import pandas as pd

from run_baselines import CONFIGS, RAW, ROOT
from run_warning import fit_predict, oof_predictions
from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import load_dataset
from sentinel.data.splits import grouped_kfold_assets
from sentinel.eval.metrics import mae, rmse
from sentinel.spec import CONFIGS_DIR, load_spec

BINS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 400)]
N_REP = 400


def run(subset: str) -> list[str]:
    spec = load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
    data = load_dataset(spec, RAW)
    train, test = add_train_rul(data.train), add_test_rul(data.test, data.test_final_rul)
    cfg = {**next(c for c in CONFIGS if c["name"] == "lightgbm_window50"), "regimes": spec.n_regimes}
    oof = oof_predictions(cfg, spec, train, grouped_kfold_assets(train, n_splits=5, seed=0))
    tp = fit_predict(cfg, spec, train, test, last_only=True)
    finals = tp["true"].to_numpy()
    test_rmse, test_mae = rmse(tp["pred"], tp["true"]), mae(tp["pred"], tp["true"])

    by_asset = {a: g.sort_values("time") for a, g in oof.groupby("asset")}
    rng = np.random.default_rng(0)
    reps, bias_rows = [], []
    for _ in range(N_REP):
        picks = []
        for g in by_asset.values():
            target = rng.choice(finals)
            idx = np.where(g["true"].to_numpy() == target)[0]
            if idx.size:
                picks.append(g.iloc[idx[0]])
        pk = pd.DataFrame(picks)
        reps.append(rmse(pk["pred"], pk["true"]))
        bias_rows.append(pk)
    reps = np.array(reps)
    matched = pd.concat(bias_rows)

    test_life = data.test.groupby("asset")["time"].max().to_numpy() + data.test_final_rul.to_numpy()
    train_life = train.groupby("asset")["time"].max().to_numpy()
    hist_test = data.test.groupby("asset")["time"].max().to_numpy()
    lines = [f"## {subset}", "",
             f"- Official test (LightGBM w50, seed 0, one point per asset, n={len(tp)}): RMSE {test_rmse:.2f}, MAE {test_mae:.2f}, mean error (pred - true) {np.mean(tp['pred'] - tp['true']):+.2f}.",
             f"- Out-of-fold RMSE over every cycle (uncapped truth): {rmse(oof['pred'], oof['true']):.2f}; over cycles with true RUL <= 150: {rmse(oof[oof['true'] <= 150]['pred'], oof[oof['true'] <= 150]['true']):.2f}.",
             f"- Out-of-fold RMSE re-sampled to the test RUL distribution ({N_REP} replicates, one cycle per training asset): mean {reps.mean():.2f}, 5th-95th percentile {np.quantile(reps, 0.05):.2f}-{np.quantile(reps, 0.95):.2f}. "
             f"Share of replicates at or above the test RMSE: {(reps >= test_rmse).mean():.3f}.",
             f"- Total asset life (observed + remaining): test median {np.median(test_life):.0f} (p10-p90 {np.quantile(test_life, 0.1):.0f}-{np.quantile(test_life, 0.9):.0f}) vs train median {np.median(train_life):.0f} ({np.quantile(train_life, 0.1):.0f}-{np.quantile(train_life, 0.9):.0f}).",
             f"- History length at the test cut: median {np.median(hist_test):.0f} (min {hist_test.min()}); share of test assets with at most 50 observed cycles: {(hist_test <= 50).mean():.2f}.",
             f"- Test final RUL: median {np.median(finals):.0f}, share above 125 (the training cap): {(finals > 125).mean():.2f}.", "",
             "| true RUL bin | test n | test RMSE | test bias | OOF re-sampled RMSE | OOF re-sampled bias |", "|---|---|---|---|---|---|"]
    for lo, hi in BINS:
        t = tp[(tp["true"] >= lo) & (tp["true"] < hi)]
        m = matched[(matched["true"] >= lo) & (matched["true"] < hi)]
        if len(t) and len(m):
            lines.append(f"| {lo}-{hi if hi < 400 else 'inf'} | {len(t)} | {rmse(t['pred'], t['true']):.1f} | {np.mean(t['pred'] - t['true']):+.1f} | "
                         f"{rmse(m['pred'], m['true']):.1f} | {np.mean(m['pred'] - m['true']):+.1f} |")
    lines.append("")
    print("\n".join(lines[:9]))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", default=["FD001", "FD004"])
    args = parser.parse_args()
    out = ["# Test-versus-CV gap diagnostics (generated)", "",
           "Diagnostics only. Nothing here was used to choose a model, a threshold or a hyperparameter.", ""]
    for s in args.subsets:
        out += run(s)
    (ROOT / "reports" / "gap_diagnostics.md").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
