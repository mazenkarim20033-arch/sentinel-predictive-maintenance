from collections.abc import Callable

import numpy as np


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """PHM08 asymmetric score as commonly cited: d = pred - true; late (d >= 0) is penalised
    more steeply. Lower is better. Constants not yet checked against the primary paper."""
    d = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    pred: np.ndarray,
    true: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    pred, true = np.asarray(pred), np.asarray(true)
    rng = np.random.default_rng(seed)
    n = len(pred)
    stats = [fn(pred[i], true[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
