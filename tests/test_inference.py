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


@pytest.mark.parametrize("tail", [1, 2, 3, 5, 30, 79, 80, 500])
def test_tail_gives_exactly_the_last_rows_of_the_full_computation(bundle, history, tail):
    full = bundle.predict_history(history)
    part = bundle.predict_history(history, tail=tail)
    k = min(tail, len(full))
    assert len(part) == k
    assert np.array_equal(part["time"].to_numpy(), full["time"].to_numpy()[-k:])
    for col in full.columns:
        assert np.allclose(part[col].to_numpy(dtype=float), full[col].to_numpy(dtype=float)[-k:], atol=1e-9), col


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_tail_on_very_short_histories(bundle, history, n):
    short = history.iloc[:n]
    full = bundle.predict_history(short)
    part = bundle.predict_history(short, tail=1)
    assert len(part) == 1 and np.allclose(part.iloc[0].to_numpy(dtype=float), full.iloc[-1].to_numpy(dtype=float))


def test_batched_last_prediction_matches_per_asset_computation(bundle):
    frames = {a: make_fixture_frame([n], seed=a).drop(columns=["asset"]) for a, n in zip((3, 7, 11, 20), (80, 31, 4, 2))}
    batch = bundle.predict_last_many(frames).set_index("asset")
    assert list(batch.index) == [3, 7, 11, 20]
    for a, f in frames.items():
        single = bundle.predict_history(f, tail=1).iloc[0]
        for col in single.index:
            assert float(batch.loc[a, col]) == pytest.approx(float(single[col]), abs=1e-9), (a, col)


def test_explain_tail_matches_full_explanation(bundle, history):
    c_full, raw_full, base = bundle.explain(history)
    c_tail, raw_tail, base_t = bundle.explain(history, tail=1)
    assert len(c_tail) == 1 and base == base_t and c_tail["time"].iloc[0] == c_full["time"].iloc[-1]
    assert np.allclose(c_tail.drop(columns="time").to_numpy(), c_full.drop(columns="time").to_numpy()[-1:], atol=1e-9)
    assert raw_tail[0] == pytest.approx(raw_full[-1])


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


def test_anomaly_score_saturates_instead_of_overflowing(bundle, history):
    sig = [c for c in history.columns if c.startswith("s")]
    extreme = history.copy()
    extreme[sig] = extreme[sig] * 1e7  # a few 1e8: inside the accepted range, but squared z-scores far exceed the saturation level
    out = bundle.predict_history(extreme)
    assert np.isfinite(out["anomaly_score"]).all() and out["anomaly_score"].max() <= 1e12
    assert out["anomaly_alert"].iloc[-1]
    with pytest.raises(BundleError, match="supported range"):
        bundle.predict_history(history.assign(s4=1e10))


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
