from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["engine", "cycle", *SETTING_COLS, *SENSOR_COLS]


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SubsetData:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_final_rul: pd.Series


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download CMAPSSData.zip from "
            "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data and extract it there."
        )
    return pd.read_csv(path, sep=r"\s+", header=None)


def validate_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.shape[1] != len(COLUMNS):
        raise DataValidationError(f"{label}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df = df.copy()
    df.columns = COLUMNS
    values = df.to_numpy(dtype=float, na_value=np.nan)
    if np.isnan(values).any():
        raise DataValidationError(f"{label}: contains missing values")
    if not np.isfinite(values).all():
        raise DataValidationError(f"{label}: contains infinite values")
    df["engine"] = df["engine"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    for engine, group in df.groupby("engine", sort=True):
        cycles = group["cycle"].to_numpy()
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1)):
            raise DataValidationError(f"{label}: engine {engine} cycles are not 1..N in order")
    ids = np.sort(df["engine"].unique())
    if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise DataValidationError(f"{label}: engine ids are not 1..N")
    return df.reset_index(drop=True)


def load_subset(raw_dir: str | Path, name: str) -> SubsetData:
    if name not in SUBSETS:
        raise ValueError(f"unknown subset {name!r}; expected one of {SUBSETS}")
    raw = Path(raw_dir)
    train = validate_frame(_read_table(raw / f"train_{name}.txt"), f"train_{name}")
    test = validate_frame(_read_table(raw / f"test_{name}.txt"), f"test_{name}")
    rul = _read_table(raw / f"RUL_{name}.txt")
    if rul.shape[1] != 1:
        raise DataValidationError(f"RUL_{name}: expected a single column")
    n_test = test["engine"].nunique()
    if len(rul) != n_test:
        raise DataValidationError(f"RUL_{name}: {len(rul)} values for {n_test} test engines")
    if (rul[0] < 0).any():
        raise DataValidationError(f"RUL_{name}: negative RUL")
    final_rul = pd.Series(rul[0].to_numpy(dtype=float), index=np.arange(1, n_test + 1), name="final_rul")
    return SubsetData(name=name, train=train, test=test, test_final_rul=final_rul)
