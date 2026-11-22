import numpy as np
import pandas as pd

from sentinel.features.windows import Windows


def tabular_features(win: Windows, feature_cols: list[str], include_time: bool = False) -> pd.DataFrame:
    """Per-sensor last value, mean, std and slope over the real (unpadded) steps of a causal window."""
    X, mask = win.X, win.mask
    w = X.shape[1]
    mf = mask[:, :, None].astype(np.float32)
    cnt = mask.sum(axis=1)[:, None].astype(np.float32)
    last = X[:, -1, :]
    mean = (X * mf).sum(axis=1) / cnt
    dev = X - mean[:, None, :]
    std = np.sqrt((dev**2 * mf).sum(axis=1) / cnt)
    t = np.arange(w, dtype=np.float32)[None, :, None]
    t_c = t - ((t * mf).sum(axis=1) / cnt)[:, None, :]
    slope = (t_c * dev * mf).sum(axis=1) / np.maximum((t_c**2 * mf).sum(axis=1), 1e-8)
    parts = {"last": last, "mean": mean, "std": std, "slope": slope}
    cols = {f"{s}_{k}": v[:, i] for k, v in parts.items() for i, s in enumerate(feature_cols)}
    out = pd.DataFrame(cols)
    if include_time:
        out["time"] = win.time.astype(np.float32)
    return out
