import numpy as np
import pandas as pd
import pytest

from sentinel.data.labels import add_test_rul, add_train_rul
from sentinel.data.loader import (
    COLUMNS,
    DataValidationError,
    load_subset,
    validate_frame,
)
from sentinel.data.splits import grouped_kfold_engines, select_engines, split_engines
from conftest import REAL_RAW, make_fixture_frame, real_data


def test_load_subset_parses_and_groups_engines(raw_dir):
    data = load_subset(raw_dir, "FD001")
    assert list(data.train.columns) == COLUMNS
    assert data.train["engine"].nunique() == 10
    assert (data.train.groupby("engine")["cycle"].max().to_numpy() == np.arange(30, 130, 10)).all()
    assert len(data.test_final_rul) == data.test["engine"].nunique()


def test_missing_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data.nasa.gov"):
        load_subset(tmp_path, "FD001")


def test_unknown_subset_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_subset(tmp_path, "FD999")


def test_wrong_column_count_rejected(small_frame):
    with pytest.raises(DataValidationError, match="columns"):
        validate_frame(small_frame.iloc[:, :20], "x")


def test_nan_and_inf_rejected(small_frame):
    bad = small_frame.copy()
    bad.loc[3, "s4"] = np.nan
    with pytest.raises(DataValidationError, match="missing"):
        validate_frame(bad, "x")
    bad = small_frame.copy()
    bad.loc[3, "s4"] = np.inf
    with pytest.raises(DataValidationError, match="infinite"):
        validate_frame(bad, "x")


def test_non_contiguous_cycles_rejected(small_frame):
    bad = small_frame.drop(index=5).reset_index(drop=True)
    with pytest.raises(DataValidationError, match="cycles"):
        validate_frame(bad, "x")


def test_rul_count_mismatch_rejected(raw_dir):
    (raw_dir / "RUL_FD001.txt").write_text("10\n11\n")
    with pytest.raises(DataValidationError, match="RUL_FD001"):
        load_subset(raw_dir, "FD001")


def test_train_rul_counts_down_to_zero(small_frame):
    out = add_train_rul(small_frame, cap=25)
    last_rows = out.groupby("engine").tail(1)
    assert (last_rows["rul"] == 0).all()
    first = out[out["cycle"] == 1].set_index("engine")
    assert (first["rul"] == small_frame.groupby("engine")["cycle"].max() - 1).all()
    assert out["rul_capped"].max() == 25


def test_test_rul_extends_from_final_value(raw_dir):
    data = load_subset(raw_dir, "FD001")
    out = add_test_rul(data.test, data.test_final_rul)
    last = out.groupby("engine").tail(1).set_index("engine")["rul"]
    assert np.array_equal(last.to_numpy(), data.test_final_rul.to_numpy())
    e1 = out[out["engine"] == 1]
    assert (np.diff(e1["rul"]) == -1).all()


def test_engine_split_is_disjoint_and_complete(small_frame):
    s = split_engines(small_frame, val_fraction=0.3, seed=1)
    assert set(s["train"]).isdisjoint(s["val"])
    assert set(s["train"]) | set(s["val"]) == set(small_frame["engine"].unique())
    tr, va = select_engines(small_frame, s["train"]), select_engines(small_frame, s["val"])
    assert set(tr["engine"]).isdisjoint(set(va["engine"]))


def test_split_is_reproducible_and_seed_dependent(small_frame):
    assert split_engines(small_frame, seed=3) == split_engines(small_frame, seed=3)
    assert split_engines(small_frame, seed=3)["val"] != split_engines(small_frame, seed=4)["val"]


def test_kfold_validation_folds_partition_engines():
    frame = make_fixture_frame(list(range(30, 130, 5)))
    folds = grouped_kfold_engines(frame, n_splits=5)
    val_ids = [e for f in folds for e in f["val"]]
    assert sorted(val_ids) == sorted(frame["engine"].unique().tolist())
    assert all(set(f["train"]).isdisjoint(f["val"]) for f in folds)


@real_data
@pytest.mark.parametrize(
    "name,n_train,n_test,rows_train",
    [("FD001", 100, 100, 20631), ("FD002", 260, 259, 53759), ("FD003", 100, 100, 24720), ("FD004", 249, 248, 61249)],
)
def test_official_files_match_published_structure(name, n_train, n_test, rows_train):
    data = load_subset(REAL_RAW, name)
    assert data.train["engine"].nunique() == n_train
    assert data.test["engine"].nunique() == n_test
    assert len(data.train) == rows_train
    assert pd.Series(data.test_final_rul).min() >= 0
