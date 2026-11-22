import numpy as np
import pytest

from sentinel.eval.metrics import bootstrap_ci, mae, nasa_score, rmse
from sentinel.features.preprocess import Preprocessor
from sentinel.features.tabular import tabular_features
from sentinel.features.windows import make_windows
from sentinel.models.baselines import ConstantModel, build_model
from sentinel.registry.records import RunRecord, load_records, save_record


def test_rmse_mae_known_values():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)


def test_nasa_score_hand_computed_and_asymmetric():
    assert nasa_score([10], [10]) == 0
    assert nasa_score([20], [10]) == pytest.approx(np.exp(1.0) - 1)
    assert nasa_score([0], [13]) == pytest.approx(np.exp(1.0) - 1)
    assert nasa_score([20], [10]) > nasa_score([0], [10])


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 100, 200)
    pred = true + rng.normal(0, 5, 200)
    lo, hi = bootstrap_ci(rmse, pred, true, n_boot=300, seed=1)
    assert lo < rmse(pred, true) < hi
    assert (lo, hi) == bootstrap_ci(rmse, pred, true, n_boot=300, seed=1)


def test_tabular_features_match_manual_stats(small_frame):
    pre = Preprocessor().fit(small_frame)
    z = pre.transform(small_frame)
    win = make_windows(z, pre.feature_cols, window=10)
    feats = tabular_features(win, pre.feature_cols)
    i = np.where((win.engine == 3) & (win.cycle == 20))[0][0]
    s = pre.feature_cols[0]
    seg = z[(z["engine"] == 3) & (z["cycle"].between(11, 20))][s].to_numpy()
    assert feats.loc[i, f"{s}_last"] == pytest.approx(seg[-1], abs=1e-5)
    assert feats.loc[i, f"{s}_mean"] == pytest.approx(seg.mean(), abs=1e-5)
    assert feats.loc[i, f"{s}_std"] == pytest.approx(seg.std(), abs=1e-5)
    assert feats.loc[i, f"{s}_slope"] == pytest.approx(np.polyfit(np.arange(10), seg, 1)[0], abs=1e-4)


def test_tabular_features_ignore_padding_and_future(small_frame):
    pre = Preprocessor().fit(small_frame)
    z = pre.transform(small_frame)
    base = tabular_features(make_windows(z, pre.feature_cols, 10), pre.feature_cols)
    tampered = z.copy()
    tampered.loc[tampered["cycle"] > 15, pre.feature_cols] += 500
    after = tabular_features(make_windows(tampered, pre.feature_cols, 10), pre.feature_cols)
    early = make_windows(z, pre.feature_cols, 10).cycle <= 15
    assert np.allclose(base[early].to_numpy(), after[early].to_numpy())
    assert np.isfinite(base.to_numpy()).all()


def test_constant_model_predicts_training_mean():
    m = ConstantModel().fit(None, np.array([10.0, 20.0, 60.0]))
    assert (m.predict(np.zeros((4, 2))) == 30.0).all()


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model("nope")


def test_registry_roundtrip(tmp_path):
    rec = RunRecord(run_id="x-1", subset="FD001", name="n", model="m", config={"a": 1}, split={}, preprocessing={}, seeds=[0])
    save_record(rec, tmp_path)
    assert load_records(tmp_path)[0].run_id == "x-1"
