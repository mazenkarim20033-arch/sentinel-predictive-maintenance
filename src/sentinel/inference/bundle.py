"""Everything needed to turn raw signal history into a prediction, packaged as one versioned artifact.

The same code path is used for training-time evaluation and for serving. A prediction for time step t depends only on
rows with time <= t: preprocessing is row-wise, windows are causal and the anomaly score is per step."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from sentinel.anomaly.detectors import StatDetector, healthy_rows, quantile_threshold
from sentinel.data.splits import select_assets, split_assets
from sentinel.eval.warning import alert_state, classify
from sentinel.features.preprocess import Preprocessor, PreprocessConfig
from sentinel.features.tabular import tabular_features
from sentinel.features.windows import make_windows
from sentinel.models.baselines import build_model
from sentinel.spec import DatasetSpec
from sentinel.uncertainty.intervals import conformal_quantile

FORMAT_VERSION = 1
LEVELS = (0.8, 0.9)
RISK_NAMES = ("normal", "warning", "critical")
PERSISTENCE = 3


class BundleError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    spec: DatasetSpec
    pre: Preprocessor
    point: LGBMRegressor
    quantile_models: dict[float, tuple[LGBMRegressor, LGBMRegressor, float]]
    detector: StatDetector
    anomaly_threshold: float
    window: int
    cap: int
    warn: float = 30.0
    crit: float = 15.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def build(cls, spec: DatasetSpec, train: pd.DataFrame, window: int = 50, cap: int = 125, seed: int = 0,
              calibration_fraction: float = 0.2, quantile: float = 0.99) -> "ModelBundle":
        """`train` is a labelled run-to-failure frame (asset, time, spec columns, rul). Calibration assets are held out
        from every fitted component so intervals and the anomaly threshold are calibrated on unseen assets."""
        inner = split_assets(train, calibration_fraction, seed=seed)
        order = ["asset", "time"]
        prop = select_assets(train, inner["train"]).sort_values(order).reset_index(drop=True)
        cal = select_assets(train, inner["val"]).sort_values(order).reset_index(drop=True)
        pre = Preprocessor(PreprocessConfig.from_spec(spec)).fit(prop)
        Xp, yp = _features(prop, pre, window), np.minimum(prop["rul"].to_numpy(), cap)
        Xc, yc = _features(cal, pre, window), np.minimum(cal["rul"].to_numpy(), cap)
        point = build_model("lgbm", seed).fit(Xp, yp)
        pc = point.predict(Xc)
        qmodels = {}
        for level in LEVELS:
            a = 1 - level
            lo = LGBMRegressor(**{**point.get_params(), "objective": "quantile", "alpha": a / 2}).fit(Xp, yp)
            hi = LGBMRegressor(**{**point.get_params(), "objective": "quantile", "alpha": 1 - a / 2}).fit(Xp, yp)
            lo_c, hi_c = np.minimum(lo.predict(Xc), hi.predict(Xc)), np.maximum(lo.predict(Xc), hi.predict(Xc))
            q = conformal_quantile(np.maximum(lo_c - yc, yc - hi_c), a)
            qmodels[level] = (lo, hi, float(q))
        cols = pre.feature_cols
        detector = StatDetector().fit(healthy_rows(pre.transform(prop), cols))
        thr = quantile_threshold(detector.score(healthy_rows(pre.transform(cal), cols)), quantile)
        meta = {"subset": spec.name, "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_training_assets": int(prop["asset"].nunique()), "n_calibration_assets": int(cal["asset"].nunique()),
                "seed": seed, "signals": pre.feature_cols, "format_version": FORMAT_VERSION,
                "calibration_mae_cycles": float(np.mean(np.abs(pc - yc)))}
        return cls(spec=spec, pre=pre, point=point, quantile_models=qmodels, detector=detector, anomaly_threshold=float(thr),
                   window=window, cap=cap, meta=meta)

    def predict_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predictions for every time step of ONE asset. Row i uses only rows <= i."""
        frame = self._validate(frame)
        z = self.pre.transform(frame)
        cols = self.pre.feature_cols
        X = tabular_features(make_windows(z, cols, self.window), cols)
        pred = np.clip(self.point.predict(X), 0, self.cap)
        out = pd.DataFrame({"time": z["time"].to_numpy(), "rul": pred})
        for level in LEVELS:
            lo_m, hi_m, q = self.quantile_models[level]
            lo = np.minimum(lo_m.predict(X), hi_m.predict(X)) - q
            hi = np.maximum(lo_m.predict(X), hi_m.predict(X)) + q
            lo, hi = np.maximum(lo, 0), np.minimum(hi, self.cap)
            tag = int(level * 100)
            out[f"lo{tag}"], out[f"hi{tag}"] = lo, np.maximum(hi, lo)
        crit = alert_state(pred, self.crit, PERSISTENCE)
        warn = alert_state(pred, self.warn, PERSISTENCE)
        out["risk"] = np.where(crit, 2, np.where(warn, 1, 0))
        out["risk_instant"] = classify(pred, self.warn, self.crit)
        scores = self.detector.score(z[cols].to_numpy())
        out["anomaly_score"] = scores
        out["anomaly_alert"] = alert_state(-scores, -self.anomaly_threshold, PERSISTENCE)
        return out

    def explain(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
        """TreeSHAP attributions per signal for every step of ONE asset (row i uses rows <= i only).
        Returns (signal contributions, raw model output before clipping, base value). Associations, not causes."""
        from sentinel.explain.shap_utils import aggregate_by_signal, lgbm_shap

        frame = self._validate(frame)
        z = self.pre.transform(frame)
        cols = self.pre.feature_cols
        X = tabular_features(make_windows(z, cols, self.window), cols)
        values, base = lgbm_shap(self.point, X)
        agg, signals = aggregate_by_signal(values, list(X.columns))
        contrib = pd.DataFrame(agg, columns=signals)
        contrib.insert(0, "time", z["time"].to_numpy())
        return contrib, self.point.predict(X), base

    def _validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.spec.columns if c != "asset" and c not in frame.columns]
        if missing:
            raise BundleError(f"missing columns: {missing}")
        frame = frame.copy()
        if "asset" not in frame.columns:
            frame["asset"] = 1
        if frame["asset"].nunique() != 1:
            raise BundleError("predict_history takes one asset at a time")
        frame = frame.sort_values("time").reset_index(drop=True)
        values = frame[[c for c in self.spec.columns if c != "asset"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise BundleError("history contains missing or non-finite values")
        if frame["time"].duplicated().any():
            raise BundleError("duplicate time steps")
        return frame

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format_version": FORMAT_VERSION, "bundle": self}, path)

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        try:
            obj = joblib.load(path)
        except Exception as exc:  # corrupt or incompatible artifact
            raise BundleError(f"could not load bundle {path}: {exc}") from exc
        if not isinstance(obj, dict) or obj.get("format_version") != FORMAT_VERSION or not isinstance(obj.get("bundle"), ModelBundle):
            raise BundleError(f"{path} is not a compatible SENTINEL bundle (expected format version {FORMAT_VERSION})")
        return obj["bundle"]


def _features(frame: pd.DataFrame, pre: Preprocessor, window: int) -> pd.DataFrame:
    z = pre.transform(frame)
    return tabular_features(make_windows(z, pre.feature_cols, window), pre.feature_cols)
