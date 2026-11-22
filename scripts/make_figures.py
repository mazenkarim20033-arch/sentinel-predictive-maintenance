"""Actual-versus-predicted and error-distribution figures for the default model on each official test set.
Saves the per-asset predictions to experiments/predictions so the figures can be regenerated without refitting."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_baselines import CONFIGS, RAW, ROOT, build_xy, make_pre, train_target
from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import SUBSETS, load_dataset
from sentinel.models.baselines import build_model
from sentinel.spec import CONFIGS_DIR, load_spec

PRED_DIR = ROOT / "experiments" / "predictions"


def predictions(subset: str) -> pd.DataFrame:
    path = PRED_DIR / f"{subset}_lightgbm_window50_test.csv"
    if path.exists():
        return pd.read_csv(path)
    spec = load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
    data = load_dataset(spec, RAW)
    train, test = add_train_rul(data.train), add_test_rul(data.test, data.test_final_rul)
    cfg = {**next(c for c in CONFIGS if c["name"] == "lightgbm_window50"), "regimes": spec.n_regimes}
    pre = make_pre(spec, cfg).fit(train)
    X, w = build_xy(train, pre, cfg)
    model = build_model("lgbm", 0).fit(X, train_target(w.y, 125))
    Xt, wt = build_xy(test, pre, cfg, last_only=True)
    df = pd.DataFrame({"asset": wt.asset, "true_rul": wt.y, "pred_rul": np.maximum(model.predict(Xt), 0), "observed_cycles": wt.time})
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def main() -> None:
    argparse.ArgumentParser().parse_args()
    fig, axes = plt.subplots(2, len(SUBSETS), figsize=(3.6 * len(SUBSETS), 6.4))
    for j, s in enumerate(SUBSETS):
        d = predictions(s)
        err = d["pred_rul"] - d["true_rul"]
        ax = axes[0, j]
        ax.scatter(d["true_rul"], d["pred_rul"], s=9, alpha=0.7)
        lim = max(d["true_rul"].max(), d["pred_rul"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.axhline(125, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{s}: RMSE {np.sqrt((err ** 2).mean()):.1f}, bias {err.mean():+.1f}", fontsize=9)
        ax.set_xlabel("true RUL")
        ax.set_ylabel("predicted RUL")
        ax = axes[1, j]
        ax.hist(err, bins=25, color="tab:orange")
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("prediction - truth (cycles)")
        ax.set_ylabel("assets")
    fig.suptitle("LightGBM (window 50), official test sets, one point per asset; dotted line = training cap of 125", fontsize=10)
    fig.tight_layout()
    out = ROOT / "reports" / "test_predictions.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
