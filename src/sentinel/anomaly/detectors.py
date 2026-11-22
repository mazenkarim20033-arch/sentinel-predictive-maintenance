import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def healthy_rows(z: pd.DataFrame, feature_cols: list[str], healthy_frac: float = 0.3) -> np.ndarray:
    """Rows from the first `healthy_frac` of each asset's recorded life. Needs the full life, so training data only."""
    parts = []
    for _, g in z.sort_values(["asset", "time"]).groupby("asset"):
        parts.append(g[feature_cols].to_numpy()[: max(1, int(len(g) * healthy_frac))])
    return np.concatenate(parts)


class StatDetector:
    """Mean squared z-score against the healthy profile (a diagonal Mahalanobis distance)."""

    name = "statistical"

    def fit(self, healthy: np.ndarray) -> "StatDetector":
        self.mu_ = healthy.mean(axis=0)
        self.sd_ = np.maximum(healthy.std(axis=0), 1e-6)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        return (((z - self.mu_) / self.sd_) ** 2).mean(axis=1)


class IsoDetector:
    name = "isolation_forest"

    def __init__(self, seed: int = 0, n_estimators: int = 200):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=2)

    def fit(self, healthy: np.ndarray) -> "IsoDetector":
        self.model.fit(healthy)
        self.sd_ = np.maximum(healthy.std(axis=0), 1e-6)
        return self

    def score(self, z: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(z)


def quantile_threshold(scores: np.ndarray, q: float = 0.99) -> float:
    return float(np.quantile(scores, q))
