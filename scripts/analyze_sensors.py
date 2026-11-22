"""Per-subset sensor and regime analysis on training data only. Writes reports/data_analysis/."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel.data.labels import add_train_rul  # noqa: E402
from sentinel.data.loader import SENSOR_COLS, SETTING_COLS, SUBSETS, load_subset  # noqa: E402
from sentinel.features.preprocess import Preprocessor, PreprocessConfig  # noqa: E402

RAW = ROOT / "data" / "raw" / "extracted"
OUT = ROOT / "reports" / "data_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def regime_scan(train: pd.DataFrame) -> dict[int, float]:
    x = StandardScaler().fit_transform(train[SETTING_COLS])
    sample = x[np.random.default_rng(0).choice(len(x), size=min(8000, len(x)), replace=False)]
    scores = {}
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(sample)
        scores[k] = float(silhouette_score(sample, labels))
    return scores


def main() -> None:
    summary = []
    for name in SUBSETS:
        data = load_subset(RAW, name)
        train = add_train_rul(data.train, cap=125)
        scan = regime_scan(train)
        best_k = max(scan, key=scan.get)
        k = best_k if scan[best_k] > 0.6 else 1
        pre = Preprocessor(PreprocessConfig(n_regimes=k)).fit(train)
        z = pre.transform(train)
        z["rul"] = train["rul"].to_numpy()
        rows = []
        for s in SENSOR_COLS:
            raw_std = float(train[s].std(ddof=0))
            if s in pre.sensors_:
                rho = spearmanr(z[s], z["rul"]).statistic
                late = z[train["rul"].to_numpy() <= 50][s].mean() - z[train["rul"].to_numpy() >= 150][s].mean()
            else:
                rho, late = np.nan, np.nan
            rows.append(
                {"sensor": s, "raw_std": raw_std, "kept": s in pre.sensors_,
                 "spearman_vs_rul": rho, "late_minus_early_z": late}
            )
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f"sensors_{name}.csv", index=False)
        top = table[table.kept].reindex(table[table.kept].spearman_vs_rul.abs().sort_values(ascending=False).index)
        summary.append(
            {"subset": name, "silhouette_by_k": {k_: round(v, 3) for k_, v in scan.items()},
             "chosen_regimes": k, "kept_sensors": len(pre.sensors_),
             "dropped": [s for s in SENSOR_COLS if s not in pre.sensors_],
             "top5_abs_spearman": top.head(5)[["sensor", "spearman_vs_rul"]].round(2).values.tolist()}
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for s in top.head(4).sensor:
            mean_by_rul = z.assign(r=train["rul"].clip(upper=250)).groupby("r")[s].mean()
            ax.plot(mean_by_rul.index[::-1], mean_by_rul.values[::-1], label=s)
        ax.set_xlabel("cycles to failure (training engines, regime-normalised)")
        ax.set_ylabel("mean z-score")
        ax.set_title(f"{name}: strongest RUL-correlated sensors (training data)")
        ax.invert_xaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"trends_{name}.png", dpi=110)
        plt.close(fig)
    for item in summary:
        print(item)
    pd.Series(summary).to_json(OUT / "summary.json", indent=2)


if __name__ == "__main__":
    main()
