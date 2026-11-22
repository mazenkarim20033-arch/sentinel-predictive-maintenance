import pandas as pd


def add_train_rul(train: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    out = train.copy()
    last = out.groupby("engine")["cycle"].transform("max")
    out["rul"] = (last - out["cycle"]).astype(float)
    if cap is not None:
        out["rul_capped"] = out["rul"].clip(upper=cap)
    return out


def add_test_rul(test: pd.DataFrame, final_rul: pd.Series, cap: int | None = None) -> pd.DataFrame:
    """True RUL at every observed test cycle: RUL at the last cycle plus cycles elapsed since."""
    out = test.copy()
    last = out.groupby("engine")["cycle"].transform("max")
    out["rul"] = out["engine"].map(final_rul).astype(float) + (last - out["cycle"])
    if cap is not None:
        out["rul_capped"] = out["rul"].clip(upper=cap)
    return out
