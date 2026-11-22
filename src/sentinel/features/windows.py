from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Windows:
    X: np.ndarray  # (n, window, n_features), left-padded with the engine's first row
    mask: np.ndarray  # (n, window) True where the step is a real observation
    engine: np.ndarray
    cycle: np.ndarray
    y: np.ndarray | None


def make_windows(
    frame: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    target_col: str | None = None,
    last_only: bool = False,
) -> Windows:
    """Causal windows: the window ending at cycle t uses rows with cycle <= t only."""
    if window < 1:
        raise ValueError("window must be >= 1")
    xs, masks, engines, cycles, ys = [], [], [], [], []
    for engine, g in frame.sort_values(["engine", "cycle"]).groupby("engine", sort=True):
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        n = len(g)
        ends = np.array([n - 1]) if last_only else np.arange(n)
        idx = ends[:, None] - (window - 1) + np.arange(window)[None, :]
        masks.append(idx >= 0)
        xs.append(feats[np.clip(idx, 0, None)])
        engines.append(np.full(len(ends), engine))
        cycles.append(g["cycle"].to_numpy()[ends])
        if target_col is not None:
            ys.append(g[target_col].to_numpy(dtype=np.float32)[ends])
    return Windows(
        X=np.concatenate(xs),
        mask=np.concatenate(masks),
        engine=np.concatenate(engines),
        cycle=np.concatenate(cycles),
        y=np.concatenate(ys) if target_col is not None else None,
    )
