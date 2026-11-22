from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Windows:
    X: np.ndarray  # (n, window, n_features), left-padded with the asset's first row
    mask: np.ndarray  # (n, window) True where the step is a real observation
    asset: np.ndarray
    time: np.ndarray
    y: np.ndarray | None


def make_windows(
    frame: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    target_col: str | None = None,
    last_only: bool = False,
) -> Windows:
    """Causal windows: the window ending at time t uses rows with time <= t only."""
    if window < 1:
        raise ValueError("window must be >= 1")
    xs, masks, assets, steps, ys = [], [], [], [], []
    for asset, g in frame.sort_values(["asset", "time"]).groupby("asset", sort=True):
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        n = len(g)
        ends = np.array([n - 1]) if last_only else np.arange(n)
        idx = ends[:, None] - (window - 1) + np.arange(window)[None, :]
        masks.append(idx >= 0)
        xs.append(feats[np.clip(idx, 0, None)])
        assets.append(np.full(len(ends), asset))
        steps.append(g["time"].to_numpy()[ends])
        if target_col is not None:
            ys.append(g[target_col].to_numpy(dtype=np.float32)[ends])
    return Windows(
        X=np.concatenate(xs),
        mask=np.concatenate(masks),
        asset=np.concatenate(assets),
        time=np.concatenate(steps),
        y=np.concatenate(ys) if target_col is not None else None,
    )
