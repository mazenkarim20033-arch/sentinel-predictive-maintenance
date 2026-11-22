import numpy as np
import pytest

from sentinel.features.preprocess import Preprocessor, PreprocessConfig
from sentinel.features.windows import make_windows
from conftest import REAL_RAW, make_fixture_frame, real_data


def test_constant_sensor_is_dropped(small_frame):
    pre = Preprocessor().fit(small_frame)
    assert "s1" not in pre.feature_cols
    assert len(pre.feature_cols) == 20


def test_training_features_are_standardised(small_frame):
    pre = Preprocessor().fit(small_frame)
    z = pre.transform(small_frame)
    assert np.allclose(z[pre.feature_cols].mean(), 0, atol=1e-6)
    assert np.allclose(z[pre.feature_cols].std(ddof=0), 1, atol=1e-6)


def test_transform_before_fit_raises(small_frame):
    with pytest.raises(RuntimeError):
        Preprocessor().transform(small_frame)


def test_transform_does_not_change_fitted_state(small_frame):
    pre = Preprocessor().fit(small_frame)
    means_before = pre.means_.copy()
    other = make_fixture_frame([20, 25], seed=9)
    pre.transform(other * 1.0)
    assert pre.means_.equals(means_before)


def test_transform_is_row_wise_and_causal(small_frame):
    pre = Preprocessor().fit(small_frame)
    full = pre.transform(small_frame)
    head = pre.transform(small_frame[small_frame["cycle"] <= 10])
    merged = full[full["cycle"] <= 10].reset_index(drop=True)
    assert np.allclose(head[pre.feature_cols].to_numpy(), merged[pre.feature_cols].to_numpy())


def test_save_load_roundtrip_is_identical(small_frame, tmp_path):
    pre = Preprocessor().fit(small_frame)
    pre.save(tmp_path / "pre.joblib")
    loaded = Preprocessor.load(tmp_path / "pre.joblib")
    assert np.array_equal(pre.transform(small_frame)[pre.feature_cols], loaded.transform(small_frame)[loaded.feature_cols])


def test_regime_normalisation_removes_regime_offsets():
    frame = make_fixture_frame([60, 70, 80, 90], n_regimes=2)
    pre = Preprocessor(PreprocessConfig(n_regimes=2)).fit(frame)
    z = pre.transform(frame)
    per_regime = z.groupby("regime")[pre.feature_cols].mean()
    assert np.allclose(per_regime.to_numpy(), 0, atol=1e-6)
    assert z["regime"].nunique() == 2


def test_window_shapes_and_padding(small_frame):
    pre = Preprocessor().fit(small_frame)
    z = pre.transform(small_frame)
    w = make_windows(z, pre.feature_cols, window=10)
    assert w.X.shape == (len(z), 10, len(pre.feature_cols))
    first = np.where((w.engine == 1) & (w.cycle == 1))[0][0]
    assert w.mask[first].sum() == 1
    assert np.allclose(w.X[first][:-1], w.X[first][-1])
    full = np.where((w.engine == 1) & (w.cycle == 20))[0][0]
    assert w.mask[full].all()


def test_last_only_returns_one_window_per_engine(small_frame):
    pre = Preprocessor().fit(small_frame)
    w = make_windows(pre.transform(small_frame), pre.feature_cols, window=8, last_only=True)
    assert len(w.X) == small_frame["engine"].nunique()
    assert np.array_equal(w.cycle, small_frame.groupby("engine")["cycle"].max().to_numpy())


def test_no_future_leakage_in_windows(small_frame):
    pre = Preprocessor().fit(small_frame)
    z = pre.transform(small_frame)
    t = 25
    baseline = make_windows(z, pre.feature_cols, window=15)
    tampered = z.copy()
    tampered.loc[tampered["cycle"] > t, pre.feature_cols] += 1000.0
    after = make_windows(tampered, pre.feature_cols, window=15)
    sel = baseline.cycle <= t
    assert np.array_equal(baseline.X[sel], after.X[sel])


def test_window_target_aligned_with_end_cycle(small_frame):
    from sentinel.data.labels import add_train_rul

    labelled = add_train_rul(small_frame)
    pre = Preprocessor().fit(labelled)
    w = make_windows(pre.transform(labelled).assign(rul=labelled["rul"].to_numpy()), pre.feature_cols, 5, "rul")
    life = labelled.groupby("engine")["cycle"].max().to_dict()
    assert all(y == life[e] - c for e, c, y in zip(w.engine, w.cycle, w.y))


@real_data
def test_real_fd004_preprocessing_end_to_end():
    from sentinel.data.labels import add_test_rul, add_train_rul
    from sentinel.data.loader import load_subset

    data = load_subset(REAL_RAW, "FD004")
    train = add_train_rul(data.train, cap=125)
    pre = Preprocessor(PreprocessConfig(n_regimes=6)).fit(train)
    test = add_test_rul(data.test, data.test_final_rul)
    w = make_windows(pre.transform(test).assign(rul=test["rul"].to_numpy()), pre.feature_cols, 30, "rul", last_only=True)
    assert w.X.shape == (248, 30, 17)
    assert np.isfinite(w.X).all()
    assert np.array_equal(w.y, data.test_final_rul.to_numpy(dtype=np.float32))
    assert w.mask.sum(axis=1).min() == 19
