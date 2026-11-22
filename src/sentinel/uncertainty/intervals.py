"""Prediction intervals. Conformal guarantees assume calibration and evaluation assets are exchangeable;
they are marginal (averaged over the evaluation distribution), not conditional on the RUL value."""
import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (inf if that exceeds n)."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(s[k - 1])


def split_conformal(pred_cal, y_cal, pred_new, alpha):
    q = conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(pred_cal)), alpha)
    return np.asarray(pred_new) - q, np.asarray(pred_new) + q


def cqr(lo_cal, hi_cal, y_cal, lo_new, hi_new, alpha):
    """Conformalised quantile regression (Romano et al., 2019)."""
    y_cal = np.asarray(y_cal)
    q = conformal_quantile(np.maximum(np.asarray(lo_cal) - y_cal, y_cal - np.asarray(hi_cal)), alpha)
    return np.asarray(lo_new) - q, np.asarray(hi_new) + q


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


def interval_metrics(y, lo, hi) -> dict:
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    covered = (y >= lo) & (y <= hi)
    wlo, whi = wilson(int(covered.sum()), len(y))
    return {"coverage": float(covered.mean()), "coverage_ci95": (wlo, whi), "mean_width": float((hi - lo).mean()),
            "median_width": float(np.median(hi - lo)), "n": int(len(y))}


def coverage_by_bins(y, lo, hi, bins) -> dict:
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    out = {}
    for a, b in bins:
        sel = (y >= a) & (y < b)
        if sel.any():
            out[f"{a}-{b}"] = {"coverage": float(((y[sel] >= lo[sel]) & (y[sel] <= hi[sel])).mean()),
                               "mean_width": float((hi[sel] - lo[sel]).mean()), "n": int(sel.sum())}
    return out
