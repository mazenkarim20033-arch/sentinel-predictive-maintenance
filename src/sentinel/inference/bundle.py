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
MAX_SCORE = 1e12  # anomaly scores saturate here so extreme inputs cannot overflow to infinity
MAX_ABS_INPUT = 1e9  # sensor values beyond this are rejected as not physically meaningful


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

    def predict_history(self, frame: pd.DataFrame, tail: int | None = None) -> pd.DataFrame:
        """Predictions for every time step of ONE asset. Row i uses only rows <= i.
        With `tail`, only the last `tail` rows are computed (identical values, much cheaper for long histories)."""
        frame = self._validate(frame)
        z = self.pre.transform(frame)
        cols = self.pre.feature_cols
        context = None if tail is None else max(1, tail) + PERSISTENCE - 1  # extra rows so persistent alerts see their full context
        X = tabular_features(make_windows(z, cols, self.window, tail=context), cols)
        z_used = z.iloc[len(z) - len(X):]
        pred = np.clip(self.point.predict(X), 0, self.cap)
        out = pd.DataFrame({"time": z_used["time"].to_numpy(), "rul": pred})
        for level in LEVELS:
            lo_m, hi_m, q = self.quantile_models[level]
            a, b = lo_m.predict(X), hi_m.predict(X)
            lo = np.minimum(a, b) - q
            hi = np.maximum(a, b) + q
            lo, hi = np.maximum(lo, 0), np.minimum(hi, self.cap)
            tag = int(level * 100)
            out[f"lo{tag}"], out[f"hi{tag}"] = lo, np.maximum(hi, lo)
        crit = alert_state(pred, self.crit, PERSISTENCE)
        warn = alert_state(pred, self.warn, PERSISTENCE)
        out["risk"] = np.where(crit, 2, np.where(warn, 1, 0))
        out["risk_instant"] = classify(pred, self.warn, self.crit)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = self.detector.score(z_used[cols].to_numpy())
        scores = np.nan_to_num(scores, nan=MAX_SCORE, posinf=MAX_SCORE, neginf=0.0).clip(0, MAX_SCORE)
        out["anomaly_score"] = scores
        out["anomaly_alert"] = alert_state(-scores, -self.anomaly_threshold, PERSISTENCE)
        return out.iloc[-tail:].reset_index(drop=True) if tail else out

    def predict_last_many(self, frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Last-cycle prediction for many assets in one batched model call (used for fleet views).
        Equivalent to `predict_history(frame, tail=1)` per asset, but with one preprocessing and one model pass."""
        parts = [self._validate(f).assign(asset=aid) for aid, f in frames.items()]
        z = self.pre.transform(pd.concat(parts, ignore_index=True))
        cols = self.pre.feature_cols
        win = make_windows(z, cols, self.window, tail=PERSISTENCE)
        X = tabular_features(win, cols)
        pred = np.clip(self.point.predict(X), 0, self.cap)
        out = pd.DataFrame({"asset": win.asset, "time": win.time, "rul": pred})
        for level in LEVELS:
            lo_m, hi_m, q = self.quantile_models[level]
            a, b = lo_m.predict(X), hi_m.predict(X)
            lo = np.maximum(np.minimum(a, b) - q, 0)
            hi = np.minimum(np.maximum(a, b) + q, self.cap)
            tag = int(level * 100)
            out[f"lo{tag}"], out[f"hi{tag}"] = lo, np.maximum(hi, lo)
        z_tail = z.sort_values(["asset", "time"]).groupby("asset", sort=True).tail(PERSISTENCE)  # same rows and order as the windows
        with np.errstate(over="ignore", invalid="ignore"):
            scores = self.detector.score(z_tail[cols].to_numpy())
        scores = np.nan_to_num(scores, nan=MAX_SCORE, posinf=MAX_SCORE, neginf=0.0).clip(0, MAX_SCORE)
        out["anomaly_score"] = scores
        risk, instant, alert = np.zeros(len(out), int), np.zeros(len(out), int), np.zeros(len(out), bool)
        for a in np.unique(win.asset):
            idx = np.where(win.asset == a)[0]
            crit, warn = alert_state(pred[idx], self.crit, PERSISTENCE), alert_state(pred[idx], self.warn, PERSISTENCE)
            risk[idx] = np.where(crit, 2, np.where(warn, 1, 0))
            instant[idx] = classify(pred[idx], self.warn, self.crit)
            alert[idx] = alert_state(-scores[idx], -self.anomaly_threshold, PERSISTENCE)
        out["risk"], out["risk_instant"], out["anomaly_alert"] = risk, instant, alert
        return out.groupby("asset", sort=True).tail(1).reset_index(drop=True)

    def explain(self, frame: pd.DataFrame, tail: int | None = None) -> tuple[pd.DataFrame, np.ndarray, float]:
        """TreeSHAP attributions per signal for every step of ONE asset (row i uses rows <= i only); with `tail`, only the
        last `tail` rows. Returns (signal contributions, raw model output before clipping, base value). Associations, not causes."""
        from sentinel.explain.shap_utils import aggregate_by_signal, lgbm_shap

        frame = self._validate(frame)
        z = self.pre.transform(frame)
        cols = self.pre.feature_cols
        X = tabular_features(make_windows(z, cols, self.window, tail=tail), cols)
        values, base = lgbm_shap(self.point, X)
        agg, signals = aggregate_by_signal(values, list(X.columns))
        contrib = pd.DataFrame(agg, columns=signals)
        contrib.insert(0, "time", z["time"].to_numpy()[len(z) - len(X):])
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
        if (np.abs(values) > MAX_ABS_INPUT).any():
            raise BundleError(f"history contains values beyond the supported range of +/-{MAX_ABS_INPUT:.0e}")
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
