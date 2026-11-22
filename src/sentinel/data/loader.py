from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.spec import DatasetSpec, cmapss_spec

SUBSETS = ("FD001", "FD002", "FD003", "FD004")


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetData:
    spec: DatasetSpec
    train: pd.DataFrame
    test: pd.DataFrame
    test_final_rul: pd.Series
    asset_info: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return self.spec.name


def validate_frame(df: pd.DataFrame, spec: DatasetSpec, label: str) -> pd.DataFrame:
    """Checks a frame already in canonical form: `asset`, `time`, then the spec's columns."""
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"{label}: missing columns {missing}")
    df = df[spec.columns].copy()
    values = df.to_numpy(dtype=float, na_value=np.nan)
    if np.isnan(values).any():
        raise DataValidationError(f"{label}: contains missing values")
    if not np.isfinite(values).all():
        raise DataValidationError(f"{label}: contains infinite values")
    df["asset"] = df["asset"].astype(int)
    df["time"] = df["time"].astype(int)
    for asset, group in df.groupby("asset", sort=True):
        steps = group["time"].to_numpy()
        if spec.contiguous_time:
            ok = np.array_equal(steps, np.arange(1, len(steps) + 1))
        else:
            ok = bool((np.diff(steps) > 0).all())
        if not ok:
            raise DataValidationError(f"{label}: asset {asset} time steps are not ordered as the spec requires")
    return df.reset_index(drop=True)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download CMAPSSData.zip from "
            "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data and extract it there."
        )
    return pd.read_csv(path, sep=r"\s+", header=None)


def _load_cmapss(spec: DatasetSpec, raw_dir: Path) -> DatasetData:
    name = spec.source.params["subset"]
    raw_cols = ["asset", "time", *spec.conditions, *spec.signals]

    def read(prefix: str) -> pd.DataFrame:
        df = _read_table(raw_dir / f"{prefix}_{name}.txt")
        if df.shape[1] != len(raw_cols):
            raise DataValidationError(f"{prefix}_{name}: expected {len(raw_cols)} columns, got {df.shape[1]}")
        df.columns = raw_cols
        frame = validate_frame(df, spec, f"{prefix}_{name}")
        ids = np.sort(frame["asset"].unique())
        if not np.array_equal(ids, np.arange(1, len(ids) + 1)):
            raise DataValidationError(f"{prefix}_{name}: asset ids are not 1..N")
        return frame

    train, test = read("train"), read("test")
    rul = _read_table(raw_dir / f"RUL_{name}.txt")
    if rul.shape[1] != 1:
        raise DataValidationError(f"RUL_{name}: expected a single column")
    n_test = test["asset"].nunique()
    if len(rul) != n_test:
        raise DataValidationError(f"RUL_{name}: {len(rul)} values for {n_test} test assets")
    if (rul[0] < 0).any():
        raise DataValidationError(f"RUL_{name}: negative RUL")
    final_rul = pd.Series(rul[0].to_numpy(dtype=float), index=np.arange(1, n_test + 1), name="final_rul")
    return DatasetData(spec=spec, train=train, test=test, test_final_rul=final_rul)


def _load_nasa_battery(spec: DatasetSpec, raw_dir: Path) -> DatasetData:
    """Discharge-cycle features per cell. End of life is the first cycle whose smoothed capacity is at or
    below `eol_capacity_ah`; cells that never reach it are censored and left out of the labelled data."""
    from sentinel.data.battery import cell_cycles, find_cell_file, first_crossing

    params = spec.source.params
    frames, info = [], []
    for asset, cell in enumerate(params["cells"], start=1):
        cycles = cell_cycles(find_cell_file(raw_dir, cell))
        eol = first_crossing(cycles["capacity_ah"], params["eol_capacity_ah"])
        info.append({"asset": asset, "cell": cell, "n_cycles": len(cycles), "eol_time": eol, "reached_eol": eol is not None})
        if eol is None:
            continue
        kept = cycles[cycles["time"] <= eol].copy()
        kept.insert(0, "asset", asset)
        frames.append(kept)
    if not frames:
        raise DataValidationError("no cell reached the end-of-life threshold")
    train = validate_frame(pd.concat(frames, ignore_index=True), spec, spec.name)
    return DatasetData(
        spec=spec, train=train, test=train.iloc[0:0], test_final_rul=pd.Series(dtype=float, name="final_rul"),
        asset_info=pd.DataFrame(info),
    )


ADAPTERS = {"cmapss": _load_cmapss, "nasa_battery": _load_nasa_battery}


def load_dataset(spec: DatasetSpec, raw_dir: str | Path) -> DatasetData:
    return ADAPTERS[spec.source.adapter](spec, Path(raw_dir))


def load_subset(raw_dir: str | Path, name: str) -> DatasetData:
    if name not in SUBSETS:
        raise ValueError(f"unknown subset {name!r}; expected one of {SUBSETS}")
    return load_dataset(cmapss_spec(name), raw_dir)
