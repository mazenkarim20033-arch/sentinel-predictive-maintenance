from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sentinel.data.loader import SENSOR_COLS, SETTING_COLS


class PreprocessConfig(BaseModel):
    n_regimes: int = Field(1, ge=1)
    seed: int = 0
    min_std: float = Field(1e-8, gt=0)


class Preprocessor:
    """Per-regime z-scoring of sensors. Every statistic comes from `fit` data only and
    `transform` works row by row, so it cannot leak information from later cycles."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self.sensors_: list[str] | None = None

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        cfg = self.config
        if cfg.n_regimes > 1:
            self.setting_scaler_ = StandardScaler().fit(train[SETTING_COLS])
            self.kmeans_ = KMeans(n_clusters=cfg.n_regimes, n_init=10, random_state=cfg.seed).fit(
                self.setting_scaler_.transform(train[SETTING_COLS])
            )
        regimes = self._regimes(train)
        raw = train[SENSOR_COLS]
        self.means_ = raw.groupby(regimes).mean()
        self.stds_ = raw.groupby(regimes).std(ddof=0)
        informative = (self.stds_ > cfg.min_std).any(axis=0)
        self.sensors_ = [s for s in SENSOR_COLS if informative[s]]
        self.stds_ = self.stds_.where(self.stds_ > cfg.min_std, 1.0)
        return self

    def _regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.config.n_regimes == 1:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(self.setting_scaler_.transform(df[SETTING_COLS]))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sensors_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        regimes = self._regimes(df)
        mean = self.means_.loc[regimes, self.sensors_].to_numpy()
        std = self.stds_.loc[regimes, self.sensors_].to_numpy()
        z = (df[self.sensors_].to_numpy() - mean) / std
        out = df[["engine", "cycle"]].copy()
        out[self.sensors_] = z
        out["regime"] = regimes
        return out.reset_index(drop=True)

    @property
    def feature_cols(self) -> list[str]:
        if self.sensors_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        return list(self.sensors_)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)
