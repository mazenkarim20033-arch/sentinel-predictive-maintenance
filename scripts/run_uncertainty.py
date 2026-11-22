"""Prediction intervals for RUL: raw quantile regression, conformalised quantile regression (CQR) and split conformal.

Intervals are for the capped RUL target (cap 125): RUL beyond 125 is reported as 125+. Calibration uses held-out
calibration assets (20% of training assets), never the evaluation assets. Development coverage comes from 5-fold grouped CV;
the official test set is evaluated once. Coverage on the uncapped truth is reported as a diagnostic.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMRegressor

from run_baselines import CONFIGS, RAW, ROOT, build_xy, make_pre
from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import load_dataset
from sentinel.data.splits import grouped_kfold_assets, select_assets, split_assets
from sentinel.models.baselines import build_model
from sentinel.registry.records import RUNS_DIR, RunRecord, hardware_info, save_record
from sentinel.spec import CONFIGS_DIR, load_spec
from sentinel.uncertainty.intervals import coverage_by_bins, cqr, interval_metrics, split_conformal

LEVELS = (0.8, 0.9)
CAP = 125
BINS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]
METHODS = ("quantile", "cqr", "conformal")


def quantile_model(point: LGBMRegressor, alpha: float) -> LGBMRegressor:
    return LGBMRegressor(**{**point.get_params(), "objective": "quantile", "alpha": alpha})


def clip(lo, hi):
    lo, hi = np.maximum(lo, 0), np.minimum(hi, CAP)
    return lo, np.maximum(hi, lo)


def intervals_for(cfg, spec, tr, ev, last_only=False, seed=0):
    inner = split_assets(tr, 0.2, seed=seed)
    prop, cal = select_assets(tr, inner["train"]), select_assets(tr, inner["val"])
    pre = make_pre(spec, cfg).fit(prop)
    Xp, wp = build_xy(prop, pre, cfg)
    Xc, wc = build_xy(cal, pre, cfg)
    Xe, we = build_xy(ev, pre, cfg, last_only=last_only)
    yp, yc = np.minimum(wp.y, CAP), np.minimum(wc.y, CAP)
    point = build_model("lgbm", seed).fit(Xp, yp)
    pc, pe = point.predict(Xc), point.predict(Xe)
    out = {"point": np.clip(pe, 0, CAP), "true": we.y, "asset": we.asset, "time": we.time}
    for level in LEVELS:
        a = 1 - level
        qlo, qhi = quantile_model(point, a / 2).fit(Xp, yp), quantile_model(point, 1 - a / 2).fit(Xp, yp)
        lo_c, hi_c = np.minimum(qlo.predict(Xc), qhi.predict(Xc)), np.maximum(qlo.predict(Xc), qhi.predict(Xc))
        lo_e, hi_e = np.minimum(qlo.predict(Xe), qhi.predict(Xe)), np.maximum(qlo.predict(Xe), qhi.predict(Xe))
        out[(level, "quantile")] = clip(lo_e, hi_e)
        out[(level, "cqr")] = clip(*cqr(lo_c, hi_c, yc, lo_e, hi_e, a))
        out[(level, "conformal")] = clip(*split_conformal(pc, yc, pe, a))
    return out


def concat(parts):
    out = {}
    for key in parts[0]:
        if isinstance(parts[0][key], tuple):
            out[key] = (np.concatenate([p[key][0] for p in parts]), np.concatenate([p[key][1] for p in parts]))
        else:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


def run(subset: str, runs_dir: Path) -> None:
    spec = load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
    data = load_dataset(spec, RAW)
    train, test = add_train_rul(data.train), add_test_rul(data.test, data.test_final_rul)
    cfg = {**next(c for c in CONFIGS if c["name"] == "lightgbm_window50"), "regimes": spec.n_regimes}
    folds = grouped_kfold_assets(train, n_splits=5, seed=0)
    fold_parts = [intervals_for(cfg, spec, select_assets(train, f["train"]), select_assets(train, f["val"]), seed=i) for i, f in enumerate(folds)]
    dev = concat(fold_parts)
    dev_true = np.minimum(dev["true"], CAP)
    test_part = intervals_for(cfg, spec, train, test, last_only=True)
    test_capped, test_raw = np.minimum(test_part["true"], CAP), test_part["true"]

    results = {"dev": {}, "test": {}}
    for level in LEVELS:
        for m in METHODS:
            lo, hi = dev[(level, m)]
            per_fold = [interval_metrics(np.minimum(p["true"], CAP), *p[(level, m)])["coverage"] for p in fold_parts]
            r = interval_metrics(dev_true, lo, hi)
            r.update(by_rul_bin=coverage_by_bins(dev_true, lo, hi, BINS), fold_coverage_mean=float(np.mean(per_fold)), fold_coverage_std=float(np.std(per_fold)))
            results["dev"][f"{level}|{m}"] = r
            tlo, thi = test_part[(level, m)]
            t = interval_metrics(test_capped, tlo, thi)
            t["coverage_vs_uncapped_truth"] = float(((test_raw >= tlo) & (test_raw <= thi)).mean())
            results["test"][f"{level}|{m}"] = t

    f = lambda x: f"{x:.2f}"
    lines = [f"# Prediction intervals, {subset} (generated)", "",
             "LightGBM window 50 (fixed). Intervals are for RUL capped at 125 (values above are reported as 125+). Calibration on 20% held-out training assets; "
             "conformal methods give *marginal* coverage under exchangeability, not coverage at every RUL value. Intervals are not failure probabilities.", "",
             f"## Development: out-of-fold on {len(dev_true)} cycles of {len(np.unique(dev['asset']))} training assets (5-fold grouped CV)", "",
             "| nominal | method | coverage | fold coverage (mean ± std) | mean width | median width |", "|---|---|---|---|---|---|"]
    for level in LEVELS:
        for m in METHODS:
            r = results["dev"][f"{level}|{m}"]
            lines.append(f"| {level:.0%} | {m} | {f(r['coverage'])} | {f(r['fold_coverage_mean'])} ± {f(r['fold_coverage_std'])} | {r['mean_width']:.1f} | {r['median_width']:.1f} |")
    lines += ["", "### Coverage by true RUL (capped) bin, development", "",
              "| nominal | method | " + " | ".join(f"{a}-{b - 1 if b == 126 else b}" for a, b in BINS) + " |", "|---|---|" + "---|" * len(BINS)]
    for level in LEVELS:
        for m in METHODS:
            bins = results["dev"][f"{level}|{m}"]["by_rul_bin"]
            lines.append(f"| {level:.0%} | {m} | " + " | ".join(f"{f(bins[k]['coverage'])} (w {bins[k]['mean_width']:.0f})" if k in bins else "n/a" for k in (f"{a}-{b}" for a, b in BINS)) + " |")
    lines += ["", f"## Official test, evaluated once ({len(test_capped)} assets, one point each)", "",
              "| nominal | method | coverage vs capped truth | 95% CI | coverage vs uncapped truth | mean width |", "|---|---|---|---|---|---|"]
    for level in LEVELS:
        for m in METHODS:
            t = results["test"][f"{level}|{m}"]
            lines.append(f"| {level:.0%} | {m} | {f(t['coverage'])} | {t['coverage_ci95'][0]:.2f}-{t['coverage_ci95'][1]:.2f} | {f(t['coverage_vs_uncapped_truth'])} | {t['mean_width']:.1f} |")
    (ROOT / "reports" / f"uncertainty_{subset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, level in zip(axes[:2], LEVELS):
        for m in METHODS:
            bins = results["dev"][f"{level}|{m}"]["by_rul_bin"]
            ax.plot([k for k in bins], [bins[k]["coverage"] for k in bins], marker="o", label=m)
        ax.axhline(level, color="k", ls="--", lw=1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("true RUL bin (capped)")
        ax.set_title(f"{subset}: coverage by RUL, nominal {level:.0%}", fontsize=9)
    axes[0].set_ylabel("coverage")
    axes[0].legend(fontsize=7)
    part = fold_parts[0]
    lives = {a: (part["asset"] == a).sum() for a in np.unique(part["asset"])}
    pick = sorted(lives, key=lives.get)[len(lives) // 2]
    sel = part["asset"] == pick
    order = np.argsort(part["time"][sel])
    lo, hi = part[(0.9, "cqr")]
    ax = axes[2]
    ax.fill_between(part["time"][sel][order], lo[sel][order], hi[sel][order], alpha=0.3, label="90% CQR interval")
    ax.plot(part["time"][sel][order], part["point"][sel][order], label="prediction")
    ax.plot(part["time"][sel][order], np.minimum(part["true"][sel][order], CAP), "k--", label="true RUL (capped)")
    ax.set_title(f"{subset}: held-out asset {pick}", fontsize=9)
    ax.set_xlabel("time (cycles)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"uncertainty_{subset}.png", dpi=110)

    save_record(RunRecord(
        run_id=f"{subset}-uncertainty-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}", subset=f"{subset}-uncertainty",
        name="intervals_lightgbm_window50", model="lgbm", config={**cfg, "levels": LEVELS, "methods": METHODS, "cap": CAP, "calibration_fraction": 0.2},
        split={"type": "grouped_kfold", "n_splits": 5, "seed": 0}, preprocessing={"n_regimes": cfg["regimes"]}, seeds=[0],
        cv_metrics=results["dev"], test_metrics=results["test"], hardware=hardware_info(),
        notes="Intervals for capped RUL; calibration on held-out training assets; test evaluated once.",
    ), runs_dir)
    for level in LEVELS:
        print(subset, f"nominal {level:.0%}:", " | ".join(
            f"{m} dev {results['dev'][f'{level}|{m}']['coverage']:.2f} (w {results['dev'][f'{level}|{m}']['mean_width']:.0f}) test {results['test'][f'{level}|{m}']['coverage']:.2f}" for m in METHODS))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", default=["FD001", "FD004"])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    for subset in args.subsets:
        run(subset, args.runs_dir)


if __name__ == "__main__":
    main()
