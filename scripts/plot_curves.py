"""Training/validation loss curves (seed 0, final refit) for sequence models, from experiments/curves."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="FD001")
    args = parser.parse_args()
    files = sorted((ROOT / "experiments" / "curves").glob(f"{args.subset}_*_seed0.json"))
    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 3.4), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], files):
        h = json.loads(f.read_text())
        ax.plot([e["epoch"] for e in h], [e["train_loss"] for e in h], label="train")
        ax.plot([e["epoch"] for e in h], [e["val_loss"] for e in h], label="held-out engines")
        ax.set_title(f.stem.replace(f"{args.subset}_", "").replace("_seed0", ""))
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
    axes[0][0].set_ylabel("MSE (target / 125)")
    axes[0][0].legend()
    fig.tight_layout()
    out = ROOT / "reports" / f"training_curves_{args.subset}.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
