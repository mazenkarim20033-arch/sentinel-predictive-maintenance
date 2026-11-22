import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ConstantModel:
    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


def build_model(kind: str, seed: int = 0):
    if kind == "constant":
        return ConstantModel()
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=2, random_state=seed
        )
    if kind == "lgbm":
        return LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=20, n_jobs=2, random_state=seed, verbose=-1,
        )
    raise ValueError(f"unknown model {kind!r}")
