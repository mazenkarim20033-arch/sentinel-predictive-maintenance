"""Explainability for the default model (LightGBM, window 50): global and individual TreeSHAP attributions on out-of-fold
(held-out) assets, plus a cross-model comparison of signal importance using grouped permutation importance.
Attributions describe model behaviour, not physical causes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_baselines import CONFIGS, RAW, ROOT, build_xy, make_pre, train_target
from sentinel.data.labels import add_train_rul
from sentinel.data.loader import load_dataset
from sentinel.data.splits import grouped_kfold_assets, select_assets
from sentinel.eval.metrics import rmse
from sentinel.explain.shap_utils import aggregate_by_signal, importance, lgbm_shap, signal_of
from sentinel.models.baselines import build_model
from sentinel.registry.records import RUNS_DIR, RunRecord, hardware_info, save_record
from sentinel.spec import CONFIGS_DIR, load_spec

CAP = 125
RUL_BINS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]
PERM_ROWS, PERM_REPEATS = 4000, 3


def grouped_permutation(model, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[int]], rng) -> dict[str, float]:
    base = rmse(model.predict(X), y)
    out = {}
    for name, idx in groups.items():
        deltas = []
        for _ in range(PERM_REPEATS):
            Xp = X.copy()
            Xp.iloc[:, idx] = X.iloc[rng.permutation(len(X)), idx].to_numpy()
            deltas.append(rmse(model.predict(Xp), y) - base)
        out[name] = float(np.mean(deltas))
    return out


def ridge_importance(model, features: list[str], signals: list[str]) -> dict[str, float]:
    coef = np.abs(model[-1].coef_)
    return {s: float(sum(coef[i] for i, f in enumerate(features) if signal_of(f) == s)) for s in signals}


def run(subset: str, runs_dir: Path) -> None:
    spec = load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
    train = add_train_rul(load_dataset(spec, RAW).train)
    cfg = {**next(c for c in CONFIGS if c["name"] == "lightgbm_window50"), "regimes": spec.n_regimes}
    folds = grouped_kfold_assets(train, n_splits=5, seed=0)

    rows, shap_parts, max_additivity_err = [], [], 0.0
    fold0 = None
    for i, f in enumerate(folds):
        tr, va = select_assets(train, f["train"]), select_assets(train, f["val"])
        pre = make_pre(spec, cfg).fit(tr)
        Xtr, wtr = build_xy(tr, pre, cfg)
        Xva, wva = build_xy(va, pre, cfg)
        model = build_model("lgbm", 0).fit(Xtr, train_target(wtr.y, CAP))
        values, base = lgbm_shap(model, Xva)
        max_additivity_err = max(max_additivity_err, float(np.abs(values.sum(axis=1) + base - model.predict(Xva)).max()))
        sig_vals, signals = aggregate_by_signal(values, list(Xva.columns))
        shap_parts.append(sig_vals)
        rows.append(pd.DataFrame({"asset": wva.asset, "time": wva.time, "true": wva.y, "pred": np.clip(model.predict(Xva), 0, CAP)}))
        if i == 0:
            fold0 = (tr, va, pre, Xtr, wtr, Xva, wva, model, values)
    meta = pd.concat(rows, ignore_index=True)
    sv = np.concatenate(shap_parts)
    features = list(fold0[5].columns)
    shap_imp = dict(zip(signals, importance(sv)))

    type_imp = {}
    _, _, _, _, _, Xva0, _, _, values0 = fold0
    for kind in ("last", "mean", "std", "slope"):
        idx = [j for j, c in enumerate(features) if c.endswith(f"_{kind}")]
        type_imp[kind] = float(np.abs(values0[:, idx].sum(axis=1)).mean())

    tr0, va0, pre0, Xtr0, wtr0, Xva0, wva0, lgbm0, _ = fold0
    rng = np.random.default_rng(0)
    sample = rng.choice(len(Xva0), size=min(PERM_ROWS, len(Xva0)), replace=False)
    Xs, ys = Xva0.iloc[sample].reset_index(drop=True), np.minimum(wva0.y[sample], CAP)
    groups = {s: [j for j, c in enumerate(features) if signal_of(c) == s] for s in signals}
    rf0 = build_model("rf", 0).fit(Xtr0, train_target(wtr0.y, CAP))
    ridge0 = build_model("ridge").fit(Xtr0, train_target(wtr0.y, CAP))
    comparison = {
        "LightGBM SHAP (all folds)": shap_imp,
        "LightGBM permutation": grouped_permutation(lgbm0, Xs, ys, groups, rng),
        "Random Forest permutation": grouped_permutation(rf0, Xs, ys, groups, rng),
        "Ridge abs coefficient": ridge_importance(ridge0, features, signals),
    }
    names = list(comparison)
    corr = pd.DataFrame([[spearmanr([comparison[a][s] for s in signals], [comparison[b][s] for s in signals]).statistic for b in names] for a in names], index=names, columns=names)

    shap_check = None
    try:
        import shap

        sub = Xva0.iloc[:500]
        shap_check = float(np.abs(shap.TreeExplainer(lgbm0).shap_values(sub) - lgbm_shap(lgbm0, sub)[0]).max())
    except ImportError:
        pass

    mean_signed = {}
    true_c = np.minimum(meta["true"].to_numpy(), CAP)
    for lo, hi in RUL_BINS:
        sel = (true_c >= lo) & (true_c < hi)
        mean_signed[f"{lo}-{hi - 1 if hi == 126 else hi}"] = dict(zip(signals, sv[sel].mean(axis=0)))
    top = sorted(shap_imp, key=shap_imp.get, reverse=True)

    def top_row(d, n=8):
        return ", ".join(f"{s} ({d[s]:.2f})" for s in sorted(d, key=d.get, reverse=True)[:n])

    lines = [f"# Explainability, {subset} (generated)", "",
             "LightGBM window 50 (fixed). Attributions are exact TreeSHAP values computed on out-of-fold (held-out) assets and describe what the model does with its inputs; "
             "they are associations, not physical causes, and correlated signals share credit unpredictably. Signal importance sums a signal's last/mean/std/slope features.", "",
             f"Checks: additivity (sum of attributions + base value = prediction) max error {max_additivity_err:.1e}" +
             (f"; max difference to the `shap` library TreeExplainer on 500 rows {shap_check:.1e}." if shap_check is not None else "."), "",
             "## Global importance (mean |SHAP|, RUL units: cycles of the capped target)", "",
             "| rank | signal | mean abs SHAP |", "|---|---|---|"]
    lines += [f"| {r + 1} | {s} | {shap_imp[s]:.2f} |" for r, s in enumerate(top[:10])]
    lines += ["", "By feature type (fold 0): " + ", ".join(f"{k} {v:.2f}" for k, v in type_imp.items()), "",
              "## How importance differs between models (fold 0; top 8 signals, value in parentheses)", ""]
    for n in names:
        unit = "mean |SHAP|" if "SHAP" in n else ("sum |standardised coef|" if "Ridge" in n else "RMSE increase when permuted")
        lines.append(f"- **{n}** ({unit}): {top_row(comparison[n])}")
    lines += ["", "Spearman rank correlation of signal importance between methods:", "",
              "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {a} | " + " | ".join(f"{corr.loc[a, b]:.2f}" for b in names) + " |" for a in names]
    lines += ["", "## Mean signed SHAP by true RUL (capped), top 6 signals", "", "Positive pushes the predicted RUL up, negative pushes it down, relative to the base prediction.", "",
              "| RUL bin | " + " | ".join(top[:6]) + " |", "|---|" + "---|" * 6]
    lines += [f"| {k} | " + " | ".join(f"{v[s]:+.1f}" for s in top[:6]) + " |" for k, v in mean_signed.items()]
    (ROOT / "reports" / f"explain_{subset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pick = meta.groupby("asset")["time"].max().sort_values().index[len(meta["asset"].unique()) // 2]
    sel = (meta["asset"] == pick).to_numpy()
    order = np.argsort(meta["time"].to_numpy()[sel])
    t = meta["time"].to_numpy()[sel][order]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    ax = axes[0, 0]
    ax.barh([s for s in top[:10]][::-1], [shap_imp[s] for s in top[:10]][::-1])
    ax.set_title(f"{subset}: global importance (mean |SHAP|)", fontsize=9)
    ax = axes[0, 1]
    ax.plot(t, meta["pred"].to_numpy()[sel][order], label="prediction")
    ax.plot(t, np.minimum(meta["true"].to_numpy()[sel][order], CAP), "k--", label="true RUL (capped)")
    ax.set_title(f"held-out asset {pick}: prediction", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[0, 2]
    for s in top[:5]:
        ax.plot(t, sv[sel][order][:, signals.index(s)], label=s)
    ax.set_title(f"asset {pick}: signed SHAP of top signals over time", fontsize=9)
    ax.legend(fontsize=7)
    true_asset = np.minimum(meta["true"].to_numpy()[sel][order], CAP)
    for ax, target in zip(axes[1], (100, 50, 15)):
        k = int(np.argmin(np.abs(true_asset - target)))
        vals = sv[sel][order][k]
        o = np.argsort(np.abs(vals))[::-1][:6][::-1]
        ax.barh([signals[j] for j in o], vals[o], color=["tab:green" if vals[j] > 0 else "tab:red" for j in o])
        ax.set_title(f"asset {pick}, true RUL {true_asset[k]:.0f} (pred {meta['pred'].to_numpy()[sel][order][k]:.0f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"explain_{subset}.png", dpi=110)

    save_record(RunRecord(
        run_id=f"{subset}-explain-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}", subset=f"{subset}-explain", name="shap_lightgbm_window50",
        model="lgbm", config={**cfg, "perm_rows": PERM_ROWS, "perm_repeats": PERM_REPEATS},
        split={"type": "grouped_kfold", "n_splits": 5, "seed": 0}, preprocessing={"n_regimes": cfg["regimes"]}, seeds=[0],
        cv_metrics={"shap_importance": shap_imp, "comparison": comparison, "rank_corr": corr.to_dict(), "feature_type_importance_fold0": type_imp,
                    "additivity_max_error": max_additivity_err, "shap_library_max_diff": shap_check},
        hardware=hardware_info(), notes="TreeSHAP on out-of-fold assets; permutation/coefficients on fold 0.",
    ), runs_dir)
    print(subset, "top signals:", top_row(shap_imp, 6), "| feature types:", {k: round(v, 2) for k, v in type_imp.items()})
    print(corr.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", default=["FD001", "FD004"])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    for subset in args.subsets:
        run(subset, args.runs_dir)


if __name__ == "__main__":
    main()
