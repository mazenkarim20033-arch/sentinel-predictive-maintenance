from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

RESERVED = {"asset", "time", "rul", "rul_capped", "regime"}
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class SourceSpec(BaseModel):
    adapter: Literal["cmapss", "nasa_battery"]
    params: dict = {}


class DatasetSpec(BaseModel):
    """Describes one kind of machine. Shared pipeline code reads columns only from here."""

    name: str
    description: str = ""
    signals: list[str] = Field(min_length=1)
    conditions: list[str] = []
    n_regimes: int = Field(1, ge=1)
    rul_cap: int | None = None
    window: int = Field(30, ge=1)
    contiguous_time: bool = True
    time_unit: str = "step"
    source: SourceSpec

    @model_validator(mode="after")
    def _check(self) -> "DatasetSpec":
        cols = [*self.signals, *self.conditions]
        if len(set(cols)) != len(cols):
            raise ValueError("signals and conditions must be unique and not overlap")
        if RESERVED & set(cols):
            raise ValueError(f"column names {sorted(RESERVED & set(cols))} are reserved")
        if self.n_regimes > 1 and not self.conditions:
            raise ValueError("n_regimes > 1 needs condition columns")
        return self

    @property
    def columns(self) -> list[str]:
        return ["asset", "time", *self.conditions, *self.signals]


def load_spec(path: str | Path) -> DatasetSpec:
    return DatasetSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def cmapss_spec(subset: str) -> DatasetSpec:
    return load_spec(CONFIGS_DIR / f"cmapss_{subset}.yaml")
