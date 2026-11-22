from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sentinel.spec import DatasetSpec


class PreprocessConfig(BaseModel):
    signals: list[str] = Field(min_length=1)
    conditions: list[str] = []
    n_regimes: int = Field(1, ge=1)
    seed: int = 0
    min_std: float = Field(1e-8, gt=0)

    @model_validator(mode="after")
    def _regimes_need_conditions(self) -> "PreprocessConfig":
        if self.n_regimes > 1 and not self.conditions:
            raise ValueError("n_regimes > 1 needs condition columns")
        return self

    @classmethod
    def from_spec(cls, spec: DatasetSpec, n_regimes: int | None = None) -> "PreprocessConfig":
        return cls(
            signals=spec.signals,
            conditions=spec.conditions,
            n_regimes=spec.n_regimes if n_regimes is None else n_regimes,
        )


class Preprocessor:
    """Per-regime z-scoring of signals. Every statistic comes from `fit` data only and
    `transform` works row by row, so it cannot leak information from later steps."""

    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.signals_: list[str] | None = None

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        cfg = self.config
        if cfg.n_regimes > 1:
            self.condition_scaler_ = StandardScaler().fit(train[cfg.conditions])
            self.kmeans_ = KMeans(n_clusters=cfg.n_regimes, n_init=10, random_state=cfg.seed).fit(
                self.condition_scaler_.transform(train[cfg.conditions])
            )
        regimes = self._regimes(train)
        raw = train[cfg.signals]
        self.means_ = raw.groupby(regimes).mean()
        self.stds_ = raw.groupby(regimes).std(ddof=0)
        informative = (self.stds_ > cfg.min_std).any(axis=0)
        self.signals_ = [s for s in cfg.signals if informative[s]]
        self.stds_ = self.stds_.where(self.stds_ > cfg.min_std, 1.0)
        return self

    def _regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.config.n_regimes == 1:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(self.condition_scaler_.transform(df[self.config.conditions]))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.signals_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        regimes = self._regimes(df)
        mean = self.means_.loc[regimes, self.signals_].to_numpy()
        std = self.stds_.loc[regimes, self.signals_].to_numpy()
        z = (df[self.signals_].to_numpy() - mean) / std
        out = df[["asset", "time"]].copy()
        out[self.signals_] = z
        out["regime"] = regimes
        return out.reset_index(drop=True)

    @property
    def feature_cols(self) -> list[str]:
        if self.signals_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        return list(self.signals_)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)
