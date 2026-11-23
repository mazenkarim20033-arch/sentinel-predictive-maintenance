import numpy as np
import pandas as pd
import pytest

from sentinel.data.labels import add_train_rul
from sentinel.inference.bundle import BundleError, ModelBundle
from conftest import fixture_spec, make_fixture_frame


@pytest.fixture(scope="module")
def bundle():
    train = add_train_rul(make_fixture_frame([60, 70, 80, 90, 100, 110, 120, 130, 140, 150], seed=3))
    return ModelBundle.build(fixture_spec(), train, window=10)


@pytest.fixture(scope="module")
def history():
    return make_fixture_frame([80], seed=9).drop(columns=["asset"])


def test_bundle_output_schema_and_ranges(bundle, history):
    out = bundle.predict_history(history)
    assert len(out) == len(history)
    assert {"time", "rul", "lo80", "hi80", "lo90", "hi90", "risk", "risk_instant", "anomaly_score", "anomaly_alert"} <= set(out.columns)
    assert out["rul"].between(0, bundle.cap).all()
    assert (out["lo80"] <= out["rul"] + 1e-9).mean() > 0.5
    assert (out["hi80"] >= out["lo80"]).all() and out["lo80"].min() >= 0 and out["hi80"].max() <= bundle.cap
    assert set(out["risk"]) <= {0, 1, 2} and out["anomaly_score"].notna().all()


def test_prediction_at_t_is_unchanged_when_future_rows_are_altered(bundle, history):
    t = 40
    base = bundle.predict_history(history)
    tampered = history.copy()
    sig = [c for c in history.columns if c.startswith("s")]
    tampered.loc[tampered["time"] > t, sig] += 1000.0
    after = bundle.predict_history(tampered)
    cols = [c for c in base.columns if c != "time"]
    assert np.allclose(base.loc[base["time"] <= t, cols].to_numpy(dtype=float), after.loc[after["time"] <= t, cols].to_numpy(dtype=float))
    assert not np.allclose(base["rul"].to_numpy(), after["rul"].to_numpy())


def test_truncated_history_gives_the_same_prediction_as_full_history(bundle, history):
    full = bundle.predict_history(history)
    cut = bundle.predict_history(history[history["time"] <= 45])
    cols = [c for c in full.columns if c != "time"]
    assert np.allclose(full.iloc[:45][cols].to_numpy(dtype=float), cut[cols].to_numpy(dtype=float))


def test_input_validation(bundle, history):
    with pytest.raises(BundleError, match="missing columns"):
        bundle.predict_history(history.drop(columns=["s4"]))
    bad = history.copy()
    bad.loc[3, "s4"] = np.nan
    with pytest.raises(BundleError, match="non-finite"):
        bundle.predict_history(bad)
    dup = pd.concat([history, history.iloc[[0]]])
    with pytest.raises(BundleError, match="duplicate"):
        bundle.predict_history(dup)
    two = history.copy()
    two["asset"] = [1, 2] * (len(two) // 2)
    with pytest.raises(BundleError, match="one asset"):
        bundle.predict_history(two)


def test_save_load_roundtrip_and_incompatible_files(bundle, history, tmp_path):
    bundle.save(tmp_path / "b.joblib")
    loaded = ModelBundle.load(tmp_path / "b.joblib")
    assert np.allclose(bundle.predict_history(history)["rul"], loaded.predict_history(history)["rul"])
    (tmp_path / "bad.joblib").write_bytes(b"not a bundle")
    with pytest.raises(BundleError):
        ModelBundle.load(tmp_path / "bad.joblib")
    import joblib

    joblib.dump({"format_version": 999, "bundle": bundle}, tmp_path / "old.joblib")
    with pytest.raises(BundleError, match="compatible"):
        ModelBundle.load(tmp_path / "old.joblib")
