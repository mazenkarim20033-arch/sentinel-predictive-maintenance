import numpy as np
import pandas as pd
import pytest

from sentinel.eval.warning import alert_state, classify, evaluate_warnings, three_class_confusion


def test_alert_state_requires_k_consecutive_low_predictions():
    pred = np.array([50, 20, 50, 20, 20, 20, 50])
    assert alert_state(pred, 30, 1).tolist() == [False, True, False, True, True, True, False]
    assert alert_state(pred, 30, 3).tolist() == [False, False, False, False, False, True, False]


def test_alert_state_is_causal():
    pred = np.array([50, 50, 20, 20, 20, 50.0])
    changed = pred.copy()
    changed[4:] = 999
    assert alert_state(pred, 30, 2)[:4].tolist() == alert_state(changed, 30, 2)[:4].tolist()


def test_classify_boundaries():
    assert classify(np.array([31, 30, 15, 14.9, 0])).tolist() == [0, 1, 1, 2, 2]


def test_three_class_confusion_counts():
    m = three_class_confusion(np.array([40, 20, 5, 5]), np.array([40, 40, 5, 20]))
    assert m.tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 1]]


def frame(true, pred, asset=1):
    return pd.DataFrame({"asset": asset, "time": np.arange(1, len(true) + 1), "true": true, "pred": pred})


def test_perfect_predictor_has_perfect_cycle_metrics_and_full_lead():
    true = np.arange(60, -1, -1)
    m = evaluate_warnings(frame(true, true), tau=30)
    assert (m["precision"], m["recall"], m["f1"], m["false_alarm_rate"]) == (1.0, 1.0, 1.0, 0.0)
    assert m["detected_frac"] == 1.0 and m["median_lead"] == 30.0 and m["early_alarm_frac"] == 0.0


def test_hand_computed_confusion_counts():
    true = np.array([40, 35, 30, 25, 20, 10, 0])
    pred = np.array([40, 28, 35, 25, 35, 10, 0])
    m = evaluate_warnings(frame(true, pred), tau=30)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 2, 1)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.6)
    assert m["false_alarm_rate"] == pytest.approx(0.5)


def test_late_alert_counts_as_late_lead_and_never_alerting_is_missed():
    true = np.arange(40, -1, -1)
    never = evaluate_warnings(frame(true, np.full(41, 100.0)), tau=30)
    assert never["missed_frac"] == 1.0 and np.isnan(never["median_lead"]) and never["recall"] == 0.0
    over = evaluate_warnings(frame(true, true + 20), tau=30)
    assert over["median_lead"] == 10.0 and over["recall"] < 1.0


def test_alert_far_too_early_is_flagged_as_early_alarm():
    true = np.arange(100, -1, -1)
    pred = np.full(101, 10.0)
    m = evaluate_warnings(frame(true, pred), tau=30)
    assert m["early_alarm_frac"] == 1.0 and m["useful_alert_frac"] == 0.0


def test_alert_at_failure_cycle_is_not_a_detection():
    true = np.array([5, 4, 3, 2, 1, 0])
    pred = np.array([100, 100, 100, 100, 100, 0])
    assert evaluate_warnings(frame(true, pred), tau=30)["detected_frac"] == 0.0


def test_persistence_trades_recall_for_fewer_false_alarms():
    rng = np.random.default_rng(0)
    true = np.arange(120, -1, -1).astype(float)
    pred = true + rng.normal(0, 15, true.size)
    k1 = evaluate_warnings(frame(true, pred), tau=30, k=1)
    k5 = evaluate_warnings(frame(true, pred), tau=30, k=5)
    assert k5["false_alarm_rate"] <= k1["false_alarm_rate"]
    assert k5["recall"] <= k1["recall"]
