"""Builds reports/leaderboard_<subset>.md from experiments/runs/*.json only."""
import argparse
from pathlib import Path

from sentinel.registry.records import load_records

ROOT = Path(__file__).resolve().parents[1]


def pm(m: dict) -> str:
    return f"{m['mean']:.2f} ± {m['std']:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001")
    args = parser.parse_args()
    latest = {}
    for r in load_records():
        if r.subset == args.subset:
            latest[r.name] = r
    runs = sorted(latest.values(), key=lambda r: r.cv_metrics["cut_rmse"]["mean"])
    final = [r for r in runs if r.test_metrics]
    abl = [r for r in runs if not r.test_metrics]
    lines = [f"# Leaderboard {args.subset} (generated from experiments/runs)", ""]
    lines += ["## Final configurations: dev CV and official test (each evaluated once on test)", "",
              "| Model | CV cut RMSE | CV all-step RMSE | CV late-life RMSE | Test RMSE (5 seeds) | Test RMSE 95% CI | Test MAE | Test NASA score | Fit s | Latency ms | Size KB |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in final:
        t, c = r.test_metrics, r.cv_metrics
        lo, hi = t["rmse_ci95_seed_mean_prediction"]
        lines.append(f"| {r.name} | {pm(c['cut_rmse'])} | {pm(c['all_rmse'])} | {pm(c['late_rmse'])} | {pm(t['rmse'])} | {lo:.1f}–{hi:.1f} | {pm(t['mae'])} | {pm(t['score'])} | {r.train_seconds:.1f} | {r.latency_single_ms:.2f} | {r.model_size_bytes/1024:.0f} |")
    lines += ["", "## Ablations (dev CV only; test set not touched)", "",
              "| Variant | Config changes | CV cut RMSE | CV cut MAE | CV cut score/pt | CV all-time RMSE | CV late-life RMSE |", "|---|---|---|---|---|---|---|"]
    base = {"features": "all", "cap": 125, "window": 30, "use_time": False}
    for r in abl:
        cfg = {**r.config, "use_time": r.config.get("use_time", r.config.get("cycle", False))}
        diff = [f"{k}={cfg[k]}" for k in base if cfg[k] != base[k]]
        if r.config.get("regimes") == 1 and args.subset in ("FD002", "FD004"):
            diff.append("regimes=1 (global normalisation)")
        diff = ", ".join(diff) or "none"
        c = r.cv_metrics
        lines.append(f"| {r.name} | {diff} | {pm(c['cut_rmse'])} | {pm(c['cut_mae'])} | {pm(c['cut_score_per_point'])} | {pm(c['all_rmse'])} | {pm(c['late_rmse'])} |")
    lines += ["", "Metric definitions: `cut` = up to 10 random truncation points per validation asset with true RUL in [10,150] and time >= 31, scored against uncapped true RUL; `all` = every time vs RUL capped at 125; `late` = steps with true RUL <= 50. ± is std over 5 CV folds (model seed 0) or over 5 model seeds (test).", ""]
    out = ROOT / "reports" / f"leaderboard_{args.subset}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
