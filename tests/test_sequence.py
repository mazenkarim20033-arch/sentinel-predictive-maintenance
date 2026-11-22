import numpy as np
import pytest

from conftest import pcfg
from sentinel.data.labels import add_train_rul
from sentinel.features.preprocess import Preprocessor
from sentinel.features.windows import make_windows
from sentinel.models.sequence import SeqConfig, SequenceRegressor


@pytest.fixture
def data(small_frame):
    lab = add_train_rul(small_frame)
    pre = Preprocessor(pcfg()).fit(lab)
    z = pre.transform(lab).assign(rul=lab["rul"].to_numpy())
    return make_windows(z, pre.feature_cols, 10, "rul")


def tiny(kind="gru"):
    return SequenceRegressor(SeqConfig(kind=kind, hidden=8, layers=1, max_epochs=4, batch_size=64))


@pytest.mark.parametrize("kind", ["gru", "lstm"])
def test_fit_predict_shapes_and_finite(data, kind):
    m = tiny(kind).fit(data.X, data.mask, np.minimum(data.y, 125), data.asset)
    pred = m.predict(data.X, data.mask)
    assert pred.shape == (len(data.X),)
    assert np.isfinite(pred).all()
    assert 1 <= m.best_epoch_ <= 4


def test_same_seed_is_reproducible(data):
    a = tiny().fit(data.X, data.mask, data.y, data.asset, seed=3).predict(data.X, data.mask)
    b = tiny().fit(data.X, data.mask, data.y, data.asset, seed=3).predict(data.X, data.mask)
    assert np.allclose(a, b)


def test_training_loss_decreases(data):
    m = SequenceRegressor(SeqConfig(hidden=16, layers=1, max_epochs=12, patience=12, batch_size=64))
    m.fit(data.X, data.mask, np.minimum(data.y, 125), data.asset)
    assert m.history_[-1]["train_loss"] < m.history_[0]["train_loss"]


def test_prediction_depends_only_on_its_own_window(data):
    m = tiny().fit(data.X, data.mask, data.y, data.asset)
    base = m.predict(data.X, data.mask)
    perturbed = data.X.copy()
    perturbed[50:] += 100.0
    after = m.predict(perturbed, data.mask)
    assert np.allclose(base[:50], after[:50])


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        SeqConfig(kind="rnn")
