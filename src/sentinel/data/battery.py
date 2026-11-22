from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

CYCLE_COLUMNS = [
    "time", "capacity_ah", "dur_s", "v_mean", "t_to_3p5v_s", "t_max", "t_rise", "chg_dur_s", "chg_t_max",
]
SOURCE_HINT = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"


def _scalar(x) -> float:
    a = np.asarray(x).ravel()
    return float(a[0]) if a.size else float("nan")


def _discharge_row(data: dict) -> dict:
    v = np.asarray(data["Voltage_measured"], dtype=float)
    t = np.asarray(data["Time"], dtype=float)
    temp = np.asarray(data["Temperature_measured"], dtype=float)
    below = np.where(v <= 3.5)[0]
    return {
        "capacity_ah": _scalar(data["Capacity"]),
        "dur_s": float(t[-1]),
        "v_mean": float(v.mean()),
        "t_to_3p5v_s": float(t[below[0]]) if below.size else float(t[-1]),
        "t_max": float(temp.max()),
        "t_rise": float(temp.max() - temp[0]),
    }


def _charge_row(data: dict) -> dict:
    t = np.asarray(data["Time"], dtype=float)
    return {"chg_dur_s": float(t[-1]), "chg_t_max": float(np.max(data["Temperature_measured"]))}


def cell_cycles(mat_path: str | Path) -> pd.DataFrame:
    raw = loadmat(str(mat_path), simplify_cells=True)
    key = next(k for k in raw if not k.startswith("__"))
    return parse_cycles(raw[key]["cycle"])


def parse_cycles(cycles: list[dict]) -> pd.DataFrame:
    """One row per discharge cycle, paired with the charge cycle that preceded it."""
    rows, last_charge = [], {}
    for cyc in cycles:
        if cyc["type"] == "charge":
            last_charge = _charge_row(cyc["data"])
        elif cyc["type"] == "discharge":
            rows.append({**_discharge_row(cyc["data"]), **last_charge})
    df = pd.DataFrame(rows)
    for col in ("chg_dur_s", "chg_t_max"):
        if col not in df:
            df[col] = np.nan
    df = df.ffill().bfill()
    df.insert(0, "time", np.arange(1, len(df) + 1))
    return df[CYCLE_COLUMNS]


def first_crossing(capacity: pd.Series, threshold: float, smooth: int = 3) -> int | None:
    """1-based index of the first cycle whose rolling-median capacity is at or below the threshold."""
    rolled = capacity.rolling(smooth, min_periods=1).median().to_numpy()
    hit = np.where(rolled <= threshold)[0]
    return int(hit[0]) + 1 if hit.size else None


def find_cell_file(raw_dir: Path, cell: str) -> Path:
    path = next(Path(raw_dir).rglob(f"{cell}.mat"), None)
    if path is None:
        raise FileNotFoundError(f"{cell}.mat not found under {raw_dir}. Source: {SOURCE_HINT}")
    return path
