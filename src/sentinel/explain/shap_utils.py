"""TreeSHAP attributions for the tabular models. These describe what the *model* does with its inputs;
they are associations, not evidence of physical causes, and correlated signals share credit unpredictably."""
import numpy as np
import pandas as pd


def signal_of(feature: str) -> str:
    """'s11_slope' -> 's11'."""
    return feature.rsplit("_", 1)[0]


def lgbm_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP values from LightGBM (per feature) and the expected value. Rows sum to the prediction."""
    contrib = model.predict(X, pred_contrib=True)
    return contrib[:, :-1], float(contrib[0, -1])


def aggregate_by_signal(values: np.ndarray, features: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum per-feature attributions into one column per underlying signal."""
    signals = list(dict.fromkeys(signal_of(f) for f in features))
    cols = [[i for i, f in enumerate(features) if signal_of(f) == s] for s in signals]
    return np.stack([values[:, idx].sum(axis=1) for idx in cols], axis=1), signals


def importance(values: np.ndarray) -> np.ndarray:
    return np.abs(values).mean(axis=0)
