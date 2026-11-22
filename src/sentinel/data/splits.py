import json
from pathlib import Path

import numpy as np
import pandas as pd


def _lifetime_strata(train: pd.DataFrame, n_bins: int = 4) -> pd.Series:
    life = train.groupby("asset")["time"].max()
    return pd.qcut(life.rank(method="first"), q=n_bins, labels=False)


def split_assets(train: pd.DataFrame, val_fraction: float = 0.2, seed: int = 0) -> dict[str, list[int]]:
    """Engine-level split stratified by lifetime quartile; no asset appears in both sides."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    strata = _lifetime_strata(train)
    val: list[int] = []
    for _, ids in strata.groupby(strata):
        ids = np.array(ids.index)
        rng.shuffle(ids)
        val.extend(ids[: max(1, round(len(ids) * val_fraction))].tolist())
    val_set = set(val)
    tr = [int(e) for e in strata.index if e not in val_set]
    return {"train": sorted(tr), "val": sorted(int(e) for e in val), "seed": seed}


def grouped_kfold_assets(train: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> list[dict[str, list[int]]]:
    rng = np.random.default_rng(seed)
    strata = _lifetime_strata(train)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for _, ids in strata.groupby(strata):
        ids = np.array(ids.index)
        rng.shuffle(ids)
        for i, e in enumerate(ids):
            folds[i % n_splits].append(int(e))
    all_ids = {int(e) for e in strata.index}
    return [
        {"train": sorted(all_ids - set(f)), "val": sorted(f), "seed": seed}
        for f in folds
    ]


def select_assets(df: pd.DataFrame, assets: list[int]) -> pd.DataFrame:
    return df[df["asset"].isin(assets)].reset_index(drop=True)


def save_split(split: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=2))
