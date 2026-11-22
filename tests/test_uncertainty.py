import numpy as np
import pytest

from sentinel.uncertainty.intervals import (
    conformal_quantile, coverage_by_bins, cqr, interval_metrics, split_conformal, wilson,
)


def test_conformal_quantile_hand_computed():
    scores = np.arange(1, 10, dtype=float)  # n = 9
    assert conformal_quantile(scores, 0.2) == 8.0  # ceil(10 * 0.8) = 8th smallest
    assert conformal_quantile(scores, 0.1) == 9.0  # ceil(10 * 0.9) = 9th smallest
    assert conformal_quantile(scores, 0.05) == float("inf")  # ceil(10 * 0.95) = 10 > n


def test_split_conformal_interval_is_symmetric_around_prediction():
    lo, hi = split_conformal(np.zeros(9), np.arange(1, 10), np.array([5.0, 7.0]), alpha=0.2)
    assert lo.tolist() == [-3.0, -1.0] and hi.tolist() == [13.0, 15.0]


def test_cqr_shrinks_overcovering_intervals_and_widens_undercovering_ones():
    y = np.arange(10, dtype=float)
    shrunk = cqr(y - 5, y + 5, y, np.array([0.0]), np.array([10.0]), alpha=0.1)
    assert shrunk[0][0] == 5.0 and shrunk[1][0] == 5.0
    narrow = cqr(y - 0.1, y + 0.1, y + 2, np.array([0.0]), np.array([1.0]), alpha=0.1)
    assert narrow[0][0] < 0.0 and narrow[1][0] > 1.0


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_split_conformal_achieves_nominal_coverage_on_exchangeable_data(alpha):
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(300):
        y_cal, y_new = rng.normal(0, 1, 200), rng.normal(0, 1, 50)
        lo, hi = split_conformal(np.zeros(200), y_cal, np.zeros(50), alpha)
        covered.append(((y_new >= lo) & (y_new <= hi)).mean())
    assert np.mean(covered) >= 1 - alpha - 0.01
    assert np.mean(covered) <= 1 - alpha + 0.03


def test_interval_metrics_and_bins():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = interval_metrics(y, y - 1, y + np.array([1, 1, -1.5, 1]))
    assert m["coverage"] == 0.75 and m["mean_width"] == pytest.approx(1.375)
    b = coverage_by_bins(y, y - 1, y + 1, [(0, 2.5), (2.5, 10)])
    assert b["0-2.5"]["n"] == 2 and b["2.5-10"]["coverage"] == 1.0


def test_wilson_interval_contains_proportion_and_handles_edges():
    lo, hi = wilson(80, 100)
    assert lo < 0.8 < hi and 0.7 < lo and hi < 0.88
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan
    assert wilson(10, 10)[1] == pytest.approx(1.0)
