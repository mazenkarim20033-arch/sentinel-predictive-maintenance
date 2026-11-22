"""Failure-warning evaluation. Alert thresholds are experimental operating points, not maintenance standards,
and a predicted RUL is never treated as a calibrated failure probability."""
import numpy as np
import pandas as pd

CLASS_NAMES = ["normal", "warning", "critical"]


def alert_state(pred: np.ndarray, level: float, k: int = 1) -> np.ndarray:
    """Causal alert: predicted RUL has been at or below `level` for the last k consecutive steps."""
    below = np.asarray(pred) <= level
    if k == 1:
        return below
    trailing = np.convolve(below.astype(int), np.ones(k, dtype=int))[: len(below)]
    return trailing >= k


def classify(rul: np.ndarray, warn: float = 30, crit: float = 15) -> np.ndarray:
    """0 normal (> warn), 1 warning (crit <= RUL <= warn), 2 critical (< crit)."""
    rul = np.asarray(rul)
    return np.where(rul < crit, 2, np.where(rul <= warn, 1, 0))


def three_class_confusion(true: np.ndarray, pred: np.ndarray, warn: float = 30, crit: float = 15) -> np.ndarray:
    m = np.zeros((3, 3), dtype=int)
    for t, p in zip(classify(true, warn, crit), classify(pred, warn, crit)):
        m[t, p] += 1
    return m


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def evaluate_warnings(
    frame: pd.DataFrame, tau: float, k: int = 1, alert_level: float | None = None, early_factor: float = 2.0
) -> dict:
    """`frame` has columns asset, time, true, pred. Ground truth positive means true RUL <= tau.
    An alert fires when predicted RUL <= alert_level (default tau) for k consecutive steps.
    Engine-level lead time is the true RUL at the first alert; an alert with lead > early_factor * tau counts as early."""
    level = tau if alert_level is None else alert_level
    tp = fp = fn = tn = 0
    leads, n_assets, detected, early = [], 0, 0, 0
    for _, g in frame.sort_values(["asset", "time"]).groupby("asset"):
        n_assets += 1
        true, pred = g["true"].to_numpy(), g["pred"].to_numpy()
        state = alert_state(pred, level, k)
        pos = true <= tau
        tp += int((state & pos).sum())
        fp += int((state & ~pos).sum())
        fn += int((~state & pos).sum())
        tn += int((~state & ~pos).sum())
        valid = np.where(state & (true >= 1))[0]
        if valid.size:
            detected += 1
            lead = float(true[valid[0]])
            if lead > early_factor * tau:
                early += 1
            else:
                leads.append(lead)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall) if tp else (0.0 if (tp + fp + fn) else float("nan"))
    return {
        "tau": tau, "k": k, "alert_level": level, "precision": precision, "recall": recall, "f1": f1,
        "false_alarm_rate": _ratio(fp, fp + tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_assets": n_assets, "detected_frac": _ratio(detected, n_assets), "missed_frac": _ratio(n_assets - detected, n_assets),
        "early_alarm_frac": _ratio(early, n_assets),
        "useful_alert_frac": _ratio(len(leads), n_assets),
        "median_lead": float(np.median(leads)) if leads else float("nan"),
        "lead_p10": float(np.quantile(leads, 0.1)) if leads else float("nan"),
        "lead_p90": float(np.quantile(leads, 0.9)) if leads else float("nan"),
    }
