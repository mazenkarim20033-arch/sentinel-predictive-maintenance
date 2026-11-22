import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from sentinel.explain.shap_utils import aggregate_by_signal, importance, lgbm_shap, signal_of


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 6)), columns=["a_last", "a_slope", "b_last", "b_slope", "c_last", "c_slope"])
    y = 3 * X["a_last"] + X["b_slope"] ** 2 + rng.normal(0, 0.1, 400)
    return LGBMRegressor(n_estimators=60, verbose=-1, random_state=0).fit(X, y), X


def test_signal_of():
    assert signal_of("s11_slope") == "s11" and signal_of("setting_1_last") == "setting_1"


def test_shap_values_are_additive_and_match_the_prediction(fitted):
    model, X = fitted
    values, base = lgbm_shap(model, X)
    assert values.shape == X.shape
    assert np.allclose(values.sum(axis=1) + base, model.predict(X), atol=1e-6)


def test_signal_aggregation_preserves_total_and_groups_features(fitted):
    model, X = fitted
    values, base = lgbm_shap(model, X)
    agg, signals = aggregate_by_signal(values, list(X.columns))
    assert signals == ["a", "b", "c"] and agg.shape == (400, 3)
    assert np.allclose(agg.sum(axis=1), values.sum(axis=1))
    assert np.allclose(agg[:, 0], values[:, 0] + values[:, 1])


def test_informative_signals_rank_above_noise_signal(fitted):
    model, X = fitted
    agg, signals = aggregate_by_signal(lgbm_shap(model, X)[0], list(X.columns))
    imp = dict(zip(signals, importance(agg)))
    assert imp["a"] > imp["c"] and imp["b"] > imp["c"]


def test_matches_the_shap_library_treeexplainer(fitted):
    shap = pytest.importorskip("shap")
    model, X = fitted
    ours, _ = lgbm_shap(model, X)
    theirs = shap.TreeExplainer(model).shap_values(X)
    assert np.allclose(ours, theirs, atol=1e-5)
