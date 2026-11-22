import numpy as np
import pandas as pd
import pytest

from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import (
    DataValidationError,
    load_subset,
    validate_frame,
)
from sentinel.data.splits import grouped_kfold_assets, select_assets, split_assets
from sentinel.spec import CONFIGS_DIR, DatasetSpec, SourceSpec, load_spec
from conftest import COLUMNS, REAL_RAW, fixture_spec, make_fixture_frame, real_data

SPEC = fixture_spec()


def test_load_subset_parses_and_groups_assets(raw_dir):
    data = load_subset(raw_dir, "FD001")
    assert list(data.train.columns) == COLUMNS
    assert data.train["asset"].nunique() == 10
    assert (data.train.groupby("asset")["time"].max().to_numpy() == np.arange(30, 130, 10)).all()
    assert len(data.test_final_rul) == data.test["asset"].nunique()


def test_missing_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.nasa.gov"):
        load_subset(tmp_path, "FD001")


def test_unknown_subset_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_subset(tmp_path, "FD999")


def test_missing_spec_column_rejected(small_frame):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_frame(small_frame.drop(columns=["s4"]), SPEC, "x")


def test_wrong_raw_column_count_rejected(raw_dir):
    (raw_dir / "train_FD001.txt").write_text("1 1 0.1 0.2 0.3\n")
    with pytest.raises(DataValidationError, match="columns"):
        load_subset(raw_dir, "FD001")


def test_nan_and_inf_rejected(small_frame):
    bad = small_frame.copy()
    bad.loc[3, "s4"] = np.nan
    with pytest.raises(DataValidationError, match="missing"):
        validate_frame(bad, SPEC, "x")
    bad = small_frame.copy()
    bad.loc[3, "s4"] = np.inf
    with pytest.raises(DataValidationError, match="infinite"):
        validate_frame(bad, SPEC, "x")


def test_non_contiguous_time_rejected_only_when_spec_requires_it(small_frame):
    bad = small_frame.drop(index=5).reset_index(drop=True)
    with pytest.raises(DataValidationError, match="time steps"):
        validate_frame(bad, SPEC, "x")
    relaxed = fixture_spec(contiguous_time=False)
    assert len(validate_frame(bad, relaxed, "x")) == len(bad)


def test_unordered_time_rejected_even_when_gaps_allowed(small_frame):
    bad = small_frame.copy()
    bad.loc[[0, 1], "time"] = [2, 1]
    with pytest.raises(DataValidationError):
        validate_frame(bad, fixture_spec(contiguous_time=False), "x")


@pytest.mark.parametrize("subset,regimes", [("FD001", 1), ("FD002", 6), ("FD003", 1), ("FD004", 6)])
def test_shipped_cmapss_specs_are_valid(subset, regimes):
    spec = load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
    assert spec.n_regimes == regimes
    assert len(spec.signals) == 21 and len(spec.conditions) == 3
    assert spec.columns[:2] == ["asset", "time"]


def test_spec_rejects_bad_definitions():
    src = SourceSpec(adapter="cmapss")
    with pytest.raises(ValueError, match="overlap"):
        DatasetSpec(name="x", signals=["a", "b"], conditions=["b"], source=src)
    with pytest.raises(ValueError, match="reserved"):
        DatasetSpec(name="x", signals=["time"], source=src)
    with pytest.raises(ValueError, match="condition"):
        DatasetSpec(name="x", signals=["a"], n_regimes=3, source=src)
    with pytest.raises(ValueError):
        DatasetSpec(name="x", signals=[], source=src)


def test_rul_count_mismatch_rejected(raw_dir):
    (raw_dir / "RUL_FD001.txt").write_text("10\n11\n")
    with pytest.raises(DataValidationError, match="RUL_FD001"):
        load_subset(raw_dir, "FD001")


def test_train_rul_counts_down_to_zero(small_frame):
    out = add_train_rul(small_frame, cap=25)
    last_rows = out.groupby("asset").tail(1)
    assert (last_rows["rul"] == 0).all()
    first = out[out["time"] == 1].set_index("asset")
    assert (first["rul"] == small_frame.groupby("asset")["time"].max() - 1).all()
    assert out["rul_capped"].max() == 25


def test_test_rul_extends_from_final_value(raw_dir):
    data = load_subset(raw_dir, "FD001")
    out = add_test_rul(data.test, data.test_final_rul)
    last = out.groupby("asset").tail(1).set_index("asset")["rul"]
    assert np.array_equal(last.to_numpy(), data.test_final_rul.to_numpy())
    e1 = out[out["asset"] == 1]
    assert (np.diff(e1["rul"]) == -1).all()


def test_engine_split_is_disjoint_and_complete(small_frame):
    s = split_assets(small_frame, val_fraction=0.3, seed=1)
    assert set(s["train"]).isdisjoint(s["val"])
    assert set(s["train"]) | set(s["val"]) == set(small_frame["asset"].unique())
    tr, va = select_assets(small_frame, s["train"]), select_assets(small_frame, s["val"])
    assert set(tr["asset"]).isdisjoint(set(va["asset"]))


def test_split_is_reproducible_and_seed_dependent(small_frame):
    assert split_assets(small_frame, seed=3) == split_assets(small_frame, seed=3)
    assert split_assets(small_frame, seed=3)["val"] != split_assets(small_frame, seed=4)["val"]


def test_kfold_validation_folds_partition_engines():
    frame = make_fixture_frame(list(range(30, 130, 5)))
    folds = grouped_kfold_assets(frame, n_splits=5)
    val_ids = [e for f in folds for e in f["val"]]
    assert sorted(val_ids) == sorted(frame["asset"].unique().tolist())
    assert all(set(f["train"]).isdisjoint(f["val"]) for f in folds)


@real_data
@pytest.mark.parametrize(
    "name,n_train,n_test,rows_train",
    [("FD001", 100, 100, 20631), ("FD002", 260, 259, 53759), ("FD003", 100, 100, 24720), ("FD004", 249, 248, 61249)],
)
def test_official_files_match_published_structure(name, n_train, n_test, rows_train):
    data = load_subset(REAL_RAW, name)
    assert data.train["asset"].nunique() == n_train
    assert data.test["asset"].nunique() == n_test
    assert len(data.train) == rows_train
    assert pd.Series(data.test_final_rul).min() >= 0
