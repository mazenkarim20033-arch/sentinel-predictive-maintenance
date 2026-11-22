"""Anomaly detection, two strictly separate evaluations.

(a) Degradation analysis on real training trajectories (descriptive; there are no anomaly labels, so no accuracy is claimed).
(b) Controlled synthetic faults injected into the healthy phase of held-out assets (scored with precision, recall and delay).
Detectors are fit on the healthy phase (first 30% of life) of training assets only; the alert threshold is the 99th percentile of
scores on a separate healthy calibration slice of training assets. Held-out assets never influence fitting or thresholds.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from run_baselines import RAW, ROOT
from sentinel.anomaly.detectors import IsoDetector, StatDetector, healthy_rows, quantile_threshold
from sentinel.anomaly.inject import KINDS, inject
from sentinel.data.labels import add_train_rul
from sentinel.data.loader import load_dataset
from sentinel.data.splits import grouped_kfold_assets, select_assets, split_assets
from sentinel.features.preprocess import Preprocessor, PreprocessConfig
from sentinel.registry.records import RUNS_DIR, RunRecord, hardware_info, save_record
from sentinel.spec import CONFIGS_DIR, load_spec

HEALTHY_FRAC, DURATION, MAGNITUDES, REPS, QUANTILE, K = 0.3, 15, (1, 2, 3, 5), 10, 0.99, 3
RUL_BINS = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 10_000)]


def sustained(mask: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(mask.astype(int), np.ones(k, dtype=int))[: len(mask)] >= k


def by_asset(z, rul, cols):
    out = {}
    for asset, idx in z.groupby("asset").indices.items():
        out[asset] = (z[cols].to_numpy()[idx], rul[idx])
    return out


def run(subset: str, runs_dir: Path) -> None:
    spec = load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
    train = add_train_rul(load_dataset(spec, RAW).train)
    folds = grouped_kfold_assets(train, n_splits=5, seed=0)
    inj = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "events": 0, "detected": 0, "delays": []})
    clean = defaultdict(lambda: [0, 0])
    deg = defaultdict(lambda: {"leads": [], "frac": [], "healthy_alert": 0, "assets": 0, "rho": [], "bins": defaultdict(list)})
    for fi, fold in enumerate(folds):
        tr, va = select_assets(train, fold["train"]), select_assets(train, fold["val"])
        pre = Preprocessor(PreprocessConfig.from_spec(spec)).fit(tr)
        cols = pre.feature_cols
        ztr, zva = pre.transform(tr), pre.transform(va)
        inner = split_assets(tr, 0.2, seed=fi)
        fit_rows = healthy_rows(select_assets(ztr, inner["train"]), cols)
        cal_rows = healthy_rows(select_assets(ztr, inner["val"]), cols)
        scale = fit_rows.std(axis=0)
        val_assets = by_asset(zva, va["rul"].to_numpy(), cols)
        rng = np.random.default_rng(1000 + fi)
        events = {
            (kind, mag): [(a, int(rng.integers(len(cols))), int(rng.integers(5, max(6, int(len(Z) * HEALTHY_FRAC) - DURATION - 1))),
                           int(rng.integers(1 << 31))) for a, (Z, _) in val_assets.items() for _ in range(REPS)]
            for kind in KINDS for mag in MAGNITUDES
        }
        for det in (StatDetector().fit(fit_rows), IsoDetector(seed=0).fit(fit_rows)):
            thr = quantile_threshold(det.score(cal_rows), QUANTILE)
            for a, (Z, rul) in val_assets.items():
                s = det.score(Z)
                h = int(len(Z) * HEALTHY_FRAC)
                clean[det.name][0] += int((s[:h] > thr).sum())
                clean[det.name][1] += h
                d = deg[det.name]
                d["assets"] += 1
                d["rho"].append(spearmanr(s, np.arange(len(s))).statistic)
                alert = sustained(s > thr, K)
                if alert.any():
                    first = int(np.argmax(alert))
                    d["leads"].append(float(rul[first]))
                    d["frac"].append(first / len(Z))
                    d["healthy_alert"] += int(first < h)
                for lo, hi in RUL_BINS:
                    sel = (rul >= lo) & (rul < hi)
                    if sel.any():
                        d["bins"][(lo, hi)].extend((s[sel] / thr).tolist())
            for (kind, mag), evs in events.items():
                seqs, masks, lens = [], [], []
                for a, sensor, start, seed in evs:
                    Z = val_assets[a][0]
                    zi, m = inject(Z[: int(len(Z) * HEALTHY_FRAC)], kind, mag, sensor, start, DURATION, scale, np.random.default_rng(seed))
                    seqs.append(zi)
                    masks.append(m)
                    lens.append(len(zi))
                scores = det.score(np.concatenate(seqs))
                a_ = inj[(det.name, kind, mag)]
                pos = 0
                for (a, sensor, start, seed), m, n in zip(evs, masks, lens):
                    al = scores[pos : pos + n] > thr
                    pos += n
                    a_["tp"] += int((al & m).sum())
                    a_["fp"] += int((al & ~m).sum())
                    a_["fn"] += int((~al & m).sum())
                    a_["events"] += 1
                    if (al & m).any():
                        a_["detected"] += 1
                        a_["delays"].append(int(np.argmax(al & m)) - start)

    def ratio(a, b):
        return a / b if b else float("nan")

    results = {f"{d}|{k}|{m}": {
        "precision": ratio(v["tp"], v["tp"] + v["fp"]), "step_recall": ratio(v["tp"], v["tp"] + v["fn"]),
        "event_detection_rate": ratio(v["detected"], v["events"]), "median_delay": float(np.median(v["delays"])) if v["delays"] else float("nan"),
        "n_events": v["events"]} for (d, k, m), v in inj.items()}
    clean_fpr = {d: ratio(a, b) for d, (a, b) in clean.items()}
    degradation = {d: {
        "assets": v["assets"], "sustained_alert_frac": ratio(len(v["leads"]), v["assets"]),
        "alert_in_healthy_phase_frac": ratio(v["healthy_alert"], v["assets"]),
        "median_lead": float(np.median(v["leads"])) if v["leads"] else float("nan"),
        "lead_p10": float(np.quantile(v["leads"], 0.1)) if v["leads"] else float("nan"),
        "lead_p90": float(np.quantile(v["leads"], 0.9)) if v["leads"] else float("nan"),
        "mean_spearman_score_vs_time": float(np.nanmean(v["rho"])),
        "median_score_over_threshold_by_rul": {f"{lo}-{hi if hi < 10_000 else 'inf'}": float(np.median(x)) for (lo, hi), x in sorted(v["bins"].items())},
    } for d, v in deg.items()}

    lines = [f"# Anomaly detection, {subset} (generated)", "",
             f"Detectors: statistical (mean squared z against healthy profile) and Isolation Forest, fit on the first {int(HEALTHY_FRAC*100)}% of life of training assets; "
             f"threshold = {int(QUANTILE*100)}th percentile of scores on a separate healthy calibration slice. 5-fold grouped CV; held-out assets never touch fitting or thresholds.", "",
             "## A. Degradation analysis on real trajectories (descriptive, no accuracy claim)", "",
             f"Alert = score above threshold for {K} consecutive cycles. There are no ground-truth anomaly labels in C-MAPSS, so this only shows how the scores behave as engines degrade.", "",
             "| detector | assets | with a sustained alert | alert inside healthy phase | median lead (cycles before failure) | lead p10-p90 | mean Spearman(score, time) | clean-phase false-positive rate |",
             "|---|---|---|---|---|---|---|---|"]
    for d, v in degradation.items():
        lines.append(f"| {d} | {v['assets']} | {v['sustained_alert_frac']:.2f} | {v['alert_in_healthy_phase_frac']:.2f} | {v['median_lead']:.0f} | "
                     f"{v['lead_p10']:.0f}-{v['lead_p90']:.0f} | {v['mean_spearman_score_vs_time']:.2f} | {clean_fpr[d]:.3f} |")
    lines += ["", "Median score divided by threshold, by true RUL bin (above 1 means the score exceeds the alert level):", "",
              "| detector | " + " | ".join(f"RUL {lo}-{hi if hi < 10_000 else 'inf'}" for lo, hi in RUL_BINS) + " |", "|---|" + "---|" * len(RUL_BINS)]
    for d, v in degradation.items():
        lines.append(f"| {d} | " + " | ".join(f"{v['median_score_over_threshold_by_rul'].get(f'{lo}-{hi if hi < 10_000 else chr(105)+chr(110)+chr(102)}', float('nan')):.2f}" for lo, hi in RUL_BINS) + " |")
    lines += ["", "## B. Controlled synthetic faults (SYNTHETIC, not real anomalies)", "",
              f"One signal is modified inside the healthy phase of a held-out engine, {REPS} random injections per engine and setting. Magnitude is in healthy standard deviations; "
              f"duration {DURATION} cycles (spike: 1 cycle). Per-step alerts, no persistence. Precision counts alerts outside the injected interval as false.", ""]
    for d in clean_fpr:
        lines += [f"### {d} (clean healthy-phase false-positive rate {clean_fpr[d]:.3f})", "",
                  "| fault | magnitude | event detection rate | step recall | precision | median delay (cycles) |", "|---|---|---|---|---|---|"]
        for kind in KINDS:
            for mag in MAGNITUDES:
                r = results[f"{d}|{kind}|{mag}"]
                lines.append(f"| {kind} | {mag} | {r['event_detection_rate']:.2f} | {r['step_recall']:.2f} | {r['precision']:.2f} | {r['median_delay']:.1f} |")
        lines.append("")
    (ROOT / "reports" / f"anomaly_{subset}.md").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, len(clean_fpr), figsize=(4.2 * len(clean_fpr), 3.4), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], clean_fpr):
        for kind in KINDS:
            ax.plot(MAGNITUDES, [results[f"{d}|{kind}|{m}"]["event_detection_rate"] for m in MAGNITUDES], marker="o", label=kind)
        ax.set_title(f"{subset} {d} (synthetic faults)", fontsize=9)
        ax.set_xlabel("magnitude (healthy std)")
    axes[0][0].set_ylabel("event detection rate")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"anomaly_synthetic_{subset}.png", dpi=110)

    save_record(RunRecord(
        run_id=f"{subset}-anomaly-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}", subset=f"{subset}-anomaly", name="anomaly_detectors",
        model="statistical+isolation_forest", config={"healthy_frac": HEALTHY_FRAC, "duration": DURATION, "magnitudes": MAGNITUDES, "reps": REPS,
                                                      "quantile": QUANTILE, "persistence_k": K, "kinds": KINDS},
        split={"type": "grouped_kfold", "n_splits": 5, "seed": 0}, preprocessing={"n_regimes": spec.n_regimes}, seeds=[0],
        cv_metrics={"synthetic_injection": results, "clean_fpr": clean_fpr, "degradation": degradation}, hardware=hardware_info(),
        notes="Synthetic injection results are separate from degradation analysis; no accuracy is claimed on real anomalies.",
    ), runs_dir)
    for d, v in degradation.items():
        print(f"{subset} {d}: sustained alert {v['sustained_alert_frac']:.2f}, in healthy phase {v['alert_in_healthy_phase_frac']:.2f}, "
              f"median lead {v['median_lead']:.0f}, clean FPR {clean_fpr[d]:.3f}")
        for kind in KINDS:
            print("   ", kind, " ".join(f"{results[f'{d}|{kind}|{m}']['event_detection_rate']:.2f}" for m in MAGNITUDES))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", default=["FD001", "FD004"])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()
    for subset in args.subsets:
        run(subset, args.runs_dir)


if __name__ == "__main__":
    main()
