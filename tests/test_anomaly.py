import numpy as np
import pandas as pd
import pytest

from sentinel.anomaly.detectors import IsoDetector, StatDetector, healthy_rows, quantile_threshold
from sentinel.anomaly.inject import KINDS, inject

SCALE = np.array([1.0, 2.0, 0.5])


def base(n=40):
    return np.random.default_rng(0).normal(0, 1, (n, 3)) * SCALE


def test_every_kind_marks_only_its_interval_and_only_the_chosen_signal():
    z = base()
    for kind in KINDS:
        out, mask = inject(z, kind, 4.0, signal=1, start=10, duration=8, scale=SCALE, rng=np.random.default_rng(1))
        assert mask.sum() == (1 if kind == "spike" else 8) and mask[10]
        assert np.array_equal(out[:, [0, 2]], z[:, [0, 2]])
        assert np.array_equal(out[~mask], z[~mask])
        assert not np.array_equal(out[mask, 1], z[mask, 1]) or kind == "stuck"


def test_step_ramp_stuck_and_spike_values():
    z = np.zeros((20, 3))
    step, _ = inject(z, "step", 3.0, 1, 5, 4, SCALE, np.random.default_rng(0))
    assert np.allclose(step[5:9, 1], 6.0)
    ramp, _ = inject(z, "ramp", 3.0, 1, 5, 4, SCALE, np.random.default_rng(0))
    assert ramp[5, 1] == 0 and ramp[8, 1] == pytest.approx(6.0)
    z2 = np.arange(60, dtype=float).reshape(20, 3)
    stuck, _ = inject(z2, "stuck", 0.0, 0, 5, 6, SCALE, np.random.default_rng(0))
    assert np.all(stuck[5:11, 0] == z2[5, 0])
    spike, _ = inject(z, "spike", 5.0, 2, 3, 9, SCALE, np.random.default_rng(0))
    assert abs(spike[3, 2]) == pytest.approx(2.5) and (spike != 0).sum() == 1


def test_injection_is_reproducible_and_does_not_mutate_input():
    z = base()
    before = z.copy()
    a, _ = inject(z, "noise_burst", 3.0, 0, 5, 10, SCALE, np.random.default_rng(7))
    b, _ = inject(z, "noise_burst", 3.0, 0, 5, 10, SCALE, np.random.default_rng(7))
    assert np.array_equal(a, b) and np.array_equal(z, before)


def test_invalid_injection_arguments_rejected():
    with pytest.raises(ValueError):
        inject(base(), "meltdown", 1, 0, 0, 5, SCALE, np.random.default_rng(0))
    with pytest.raises(ValueError):
        inject(base(10), "step", 1, 0, 10, 5, SCALE, np.random.default_rng(0))


def test_healthy_rows_take_only_the_early_fraction_of_each_asset():
    z = pd.DataFrame({"asset": [1] * 10 + [2] * 20, "time": list(range(1, 11)) + list(range(1, 21)),
                      "s": list(range(10)) + list(range(100, 120))})
    rows = healthy_rows(z, ["s"], 0.3)
    assert rows.ravel().tolist() == [0, 1, 2, 100, 101, 102, 103, 104, 105]


@pytest.mark.parametrize("det", [StatDetector(), IsoDetector(seed=0, n_estimators=50)])
def test_detectors_score_shifted_points_higher_than_healthy(det):
    healthy = np.random.default_rng(0).normal(0, 1, (400, 4))
    det.fit(healthy)
    normal = np.random.default_rng(1).normal(0, 1, (200, 4))
    shifted = normal + np.array([6, 0, 0, 0])
    assert det.score(shifted).mean() > det.score(normal).mean() + 0.5 * det.score(normal).std()


def test_threshold_gives_roughly_nominal_false_positive_rate_on_fresh_healthy_data():
    rng = np.random.default_rng(0)
    det = StatDetector().fit(rng.normal(0, 1, (3000, 5)))
    thr = quantile_threshold(det.score(rng.normal(0, 1, (3000, 5))), 0.99)
    fpr = (det.score(rng.normal(0, 1, (20000, 5))) > thr).mean()
    assert 0.003 < fpr < 0.03


def test_stuck_sensor_is_invisible_to_a_marginal_detector():
    """A frozen healthy-looking value is within the healthy range, so a per-step detector cannot see it."""
    rng = np.random.default_rng(0)
    det = StatDetector().fit(rng.normal(0, 1, (2000, 3)))
    thr = quantile_threshold(det.score(rng.normal(0, 1, (2000, 3))), 0.99)
    z = rng.normal(0, 1, (40, 3))
    out, mask = inject(z, "stuck", 0.0, 0, 10, 15, np.ones(3), rng)
    assert (det.score(out)[mask] > thr).mean() < 0.2
