from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.loader import COLUMNS, SENSOR_COLS, SETTING_COLS

REAL_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "extracted"


def make_fixture_frame(lifetimes: list[int], n_regimes: int = 1, seed: int = 0) -> pd.DataFrame:
    """Tiny synthetic frame used only to unit-test code paths. Never used for results."""
    rng = np.random.default_rng(seed)
    rows = []
    for engine, life in enumerate(lifetimes, start=1):
        for cycle in range(1, life + 1):
            regime = rng.integers(0, n_regimes)
            settings = [regime * 10.0, regime * 0.2, 100.0 if regime else 60.0]
            drift = cycle / life
            sensors = [5.0 if i == 0 else 10 * (regime + 1) + drift * (i % 4) + rng.normal(0, 0.1)
                       for i in range(len(SENSOR_COLS))]
            rows.append([engine, cycle, *settings, *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)


def write_raw(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep=" ", header=False, index=False)


@pytest.fixture
def small_frame() -> pd.DataFrame:
    return make_fixture_frame([30, 40, 50, 60, 70, 80, 90, 100, 110, 120])


@pytest.fixture
def raw_dir(tmp_path: Path, small_frame: pd.DataFrame) -> Path:
    train = small_frame
    test = train[train["cycle"] <= 25].copy()
    write_raw(train, tmp_path / "train_FD001.txt")
    write_raw(test, tmp_path / "test_FD001.txt")
    pd.Series([10 + i for i in range(train["engine"].nunique())]).to_csv(
        tmp_path / "RUL_FD001.txt", header=False, index=False
    )
    return tmp_path


real_data = pytest.mark.skipif(not REAL_RAW.exists(), reason="official NASA files not present")
__all__ = ["SETTING_COLS", "make_fixture_frame", "real_data", "REAL_RAW"]
