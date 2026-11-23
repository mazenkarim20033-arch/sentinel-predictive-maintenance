"""Builds the serving artifact for a subset: artifacts/bundles/<subset>.joblib (git-ignored; rebuild with this script)."""
import argparse
import json
import time

from run_baselines import RAW, ROOT
from sentinel.data.labels import add_train_rul
from sentinel.data.loader import load_dataset
from sentinel.inference.bundle import ModelBundle
from sentinel.spec import CONFIGS_DIR, load_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="*", default=["FD001", "FD002", "FD003", "FD004"])
    args = parser.parse_args()
    for subset in args.subsets:
        spec = load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
        train = add_train_rul(load_dataset(spec, RAW).train)
        t0 = time.perf_counter()
        bundle = ModelBundle.build(spec, train)
        path = ROOT / "artifacts" / "bundles" / f"{subset}.joblib"
        bundle.save(path)
        bundle.meta["build_seconds"] = round(time.perf_counter() - t0, 1)
        print(subset, f"{path.stat().st_size / 1e6:.1f} MB", json.dumps({k: bundle.meta[k] for k in ("n_training_assets", "n_calibration_assets", "calibration_mae_cycles", "build_seconds")}), flush=True)


if __name__ == "__main__":
    main()
