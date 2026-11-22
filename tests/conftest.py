from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.features.preprocess import PreprocessConfig
from sentinel.spec import DatasetSpec, SourceSpec

REAL_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "extracted"
CONDITIONS = ["setting_1", "setting_2", "setting_3"]
SIGNALS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["asset", "time", *CONDITIONS, *SIGNALS]


def fixture_spec(**kw) -> DatasetSpec:
    return DatasetSpec(name="fixture", signals=SIGNALS, conditions=CONDITIONS, source=SourceSpec(adapter="cmapss"), **kw)


def pcfg(**kw) -> PreprocessConfig:
    return PreprocessConfig(signals=SIGNALS, conditions=CONDITIONS, **kw)


def make_fixture_frame(lifetimes: list[int], n_regimes: int = 1, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic frame used only to unit-test code paths. Never used for results."""
    rng = np.random.default_rng(seed)
    rows = []
    for asset, life in enumerate(lifetimes, start=1):
        for time in range(1, life + 1):
            regime = rng.integers(0, n_regimes)
            settings = [regime * 10.0, regime * 0.2, 100.0 if regime else 60.0]
            drift = time / life
            sensors = [5.0 if i == 0 else 10 * (regime + 1) + drift * (i % 4) + rng.normal(0, 0.1)
                       for i in range(len(SIGNALS))]
            rows.append([asset, time, *settings, *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)


def write_raw(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep=" ", header=False, index=False)


@pytest.fixture
def small_frame() -> pd.DataFrame:
    return make_fixture_frame([30, 40, 50, 60, 70, 80, 90, 100, 110, 120])


@pytest.fixture
def raw_dir(tmp_path: Path, small_frame: pd.DataFrame) -> Path:
    train = small_frame
    test = train[train["time"] <= 25].copy()
    write_raw(train, tmp_path / "train_FD001.txt")
    write_raw(test, tmp_path / "test_FD001.txt")
    pd.Series([10 + i for i in range(train["asset"].nunique())]).to_csv(
        tmp_path / "RUL_FD001.txt", header=False, index=False
    )
    return tmp_path


real_data = pytest.mark.skipif(not REAL_RAW.exists(), reason="official NASA files not present")
