"""The shared pipeline must work with column names and counts that have nothing to do with C-MAPSS.
The frame below is a synthetic fixture for unit testing only; it is not a dataset and carries no results."""
import numpy as np
import pandas as pd

from sentinel.data.labels import add_train_rul
from sentinel.data.loader import validate_frame
from sentinel.data.splits import grouped_kfold_assets
from sentinel.features.preprocess import PreprocessConfig, Preprocessor
from sentinel.features.tabular import tabular_features
from sentinel.features.windows import make_windows
from sentinel.spec import DatasetSpec, SourceSpec

SIGNALS = ["current_a", "vibration_rms", "winding_temp"]
CONDITIONS = ["load_pct", "speed_rpm"]


def motor_frame(n_assets: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for asset in range(7, 7 + n_assets):
        life = int(rng.integers(60, 120))
        for t in range(1, life + 1):
            regime = t % 2
            wear = (t / life) ** 2
            rows.append([asset, t, 40.0 + 40 * regime, 1500.0 + 1500 * regime,
                         10 + 5 * regime + wear * 3 + rng.normal(0, 0.1),
                         0.2 + wear + rng.normal(0, 0.02),
                         60 + 10 * regime + rng.normal(0, 0.2)])
    return pd.DataFrame(rows, columns=["asset", "time", *CONDITIONS, *SIGNALS])


def test_pipeline_runs_on_unrelated_column_names_and_counts():
    spec = DatasetSpec(name="motor_fixture", signals=SIGNALS, conditions=CONDITIONS, n_regimes=2,
                       contiguous_time=True, source=SourceSpec(adapter="cmapss"))
    frame = add_train_rul(validate_frame(motor_frame(), spec, "motor"))
    pre = Preprocessor(PreprocessConfig.from_spec(spec)).fit(frame)
    assert pre.feature_cols == SIGNALS
    z = pre.transform(frame).assign(rul=frame["rul"].to_numpy())
    win = make_windows(z, pre.feature_cols, window=12, target_col="rul")
    feats = tabular_features(win, pre.feature_cols)
    assert feats.shape == (len(frame), 4 * len(SIGNALS))
    assert np.isfinite(feats.to_numpy()).all()
    assert set(np.unique(win.asset)) == set(range(7, 19))
    assert z["regime"].nunique() == 2


def test_non_contiguous_asset_ids_work_in_splits():
    frame = add_train_rul(motor_frame())
    folds = grouped_kfold_assets(frame, n_splits=3)
    ids = sorted(e for f in folds for e in f["val"])
    assert ids == sorted(frame["asset"].unique().tolist())
