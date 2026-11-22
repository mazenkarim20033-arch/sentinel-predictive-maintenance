"""Battery adapter tests. The cycle structures below are synthetic fixtures for unit tests only."""
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from sentinel.data.battery import first_crossing, parse_cycles
from sentinel.data.labels import add_train_rul
from sentinel.data.loader import DataValidationError, load_dataset
from sentinel.data.splits import leave_one_asset_out
from sentinel.spec import CONFIGS_DIR, DatasetSpec, SourceSpec, load_spec
from conftest import real_data

REAL_BATTERY = CONFIGS_DIR.parent / "data" / "raw" / "battery" / "extracted"


def charge(dur=1000.0, tmax=30.0):
    return {"type": "charge", "ambient_temperature": 24, "data": {
        "Time": np.array([0.0, dur]), "Temperature_measured": np.array([25.0, tmax])}}


def discharge(cap, dur=3000.0):
    v = np.linspace(4.2, 2.7, 50)
    return {"type": "discharge", "ambient_temperature": 24, "data": {
        "Voltage_measured": v, "Time": np.linspace(0, dur, 50), "Temperature_measured": np.linspace(24, 36, 50),
        "Capacity": np.array(cap)}}


def synthetic_cell(caps):
    cycles = []
    for i, c in enumerate(caps):
        cycles += [charge(1000 + i), discharge(c, 3000 - 5 * i)]
    return cycles


def test_parse_cycles_pairs_each_discharge_with_preceding_charge():
    df = parse_cycles(synthetic_cell([2.0, 1.9, 1.8]))
    assert list(df["time"]) == [1, 2, 3]
    assert list(df["chg_dur_s"]) == [1000.0, 1001.0, 1002.0]
    assert df["t_to_3p5v_s"].iloc[0] < df["dur_s"].iloc[0]
    assert df["t_rise"].iloc[0] == pytest.approx(12.0)


def test_parse_cycles_ignores_impedance_cycles():
    cycles = synthetic_cell([2.0, 1.9])
    cycles.insert(1, {"type": "impedance", "ambient_temperature": 24, "data": {}})
    assert len(parse_cycles(cycles)) == 2


def test_first_crossing_needs_two_of_three_low_readings_so_single_glitch_is_ignored():
    caps = pd.Series([1.8, 1.8, 0.0, 1.8, 1.8, 1.4, 1.4, 1.4])
    assert first_crossing(caps, 1.5) == 7
    assert first_crossing(pd.Series([1.8, 1.8, 0.0, 1.8, 1.8, 1.8]), 1.5) is None
    assert first_crossing(pd.Series([1.9, 1.8, 1.7]), 1.5) is None


def spec_for(cells, thr):
    return DatasetSpec(name="battery_fixture", signals=["dur_s", "v_mean"], n_regimes=1, window=5,
                       source=SourceSpec(adapter="nasa_battery", params={"cells": cells, "eol_capacity_ah": thr}))


def write_mat(path, key, caps):
    cyc = synthetic_cell(caps)
    savemat(path, {key: {"cycle": np.array([{k: v for k, v in c.items()} for c in cyc], dtype=object)}})


def test_adapter_truncates_at_end_of_life_and_labels_rul(tmp_path, monkeypatch):
    import sentinel.data.battery as battery

    cells = {"B1": [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3], "B2": [2.0, 1.95, 1.9, 1.85, 1.8, 1.75]}
    monkeypatch.setattr(battery, "find_cell_file", lambda root, cell: cell)
    monkeypatch.setattr(battery, "cell_cycles", lambda cell: parse_cycles(synthetic_cell(cells[cell])))
    data = load_dataset(spec_for(["B1", "B2"], 1.5), tmp_path)
    assert list(data.asset_info["reached_eol"]) == [True, False]
    assert data.asset_info.loc[0, "eol_time"] == 7
    train = add_train_rul(data.train)
    assert set(train["asset"]) == {1}
    assert train["time"].max() == 7 and train["rul"].iloc[-1] == 0 and train["rul"].iloc[0] == 6
    assert data.test.empty


def test_adapter_fails_when_no_cell_reaches_threshold(tmp_path, monkeypatch):
    import sentinel.data.battery as battery

    monkeypatch.setattr(battery, "find_cell_file", lambda root, cell: cell)
    monkeypatch.setattr(battery, "cell_cycles", lambda cell: parse_cycles(synthetic_cell([2.0, 1.9, 1.8])))
    with pytest.raises(DataValidationError, match="end-of-life"):
        load_dataset(spec_for(["B1"], 1.0), tmp_path)


def test_missing_cell_file_gives_source_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="phm-datasets"):
        load_dataset(spec_for(["B9999"], 1.5), tmp_path)


def test_mat_file_roundtrip_parses(tmp_path):
    from sentinel.data.battery import cell_cycles

    write_mat(tmp_path / "BX.mat", "BX", [2.0, 1.9, 1.8])
    df = cell_cycles(tmp_path / "BX.mat")
    assert len(df) == 3 and df["capacity_ah"].round(2).tolist() == [2.0, 1.9, 1.8]


def test_leave_one_asset_out_holds_out_each_asset_once():
    frame = pd.DataFrame({"asset": [1, 1, 3, 3, 7], "time": [1, 2, 1, 2, 1]})
    folds = leave_one_asset_out(frame)
    assert [f["val"] for f in folds] == [[1], [3], [7]]
    assert all(set(f["train"]).isdisjoint(f["val"]) for f in folds)


@pytest.mark.skipif(not REAL_BATTERY.exists(), reason="NASA battery files not present")
@pytest.mark.parametrize("name", ["battery_nasa_sensors", "battery_nasa_with_capacity"])
def test_real_battery_specs_load_four_cells(name):
    data = load_dataset(load_spec(CONFIGS_DIR / f"{name}.yaml"), REAL_BATTERY)
    assert data.asset_info["reached_eol"].all()
    assert data.asset_info["eol_time"].tolist() == [100, 77, 127, 75]
    assert len(data.train) == 379 and not data.train.isna().any().any()
