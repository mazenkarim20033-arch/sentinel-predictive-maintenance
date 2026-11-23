"""Data and model access for the API. Real services read files from disk; tests inject in-memory ones."""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from sentinel.data.labels import add_test_rul
from sentinel.data.loader import DataValidationError, load_dataset
from sentinel.inference.bundle import BundleError, ModelBundle
from sentinel.registry.records import RUNS_DIR
from sentinel.spec import CONFIGS_DIR, DatasetSpec, load_spec

ROOT = Path(__file__).resolve().parents[3]
SUBSET_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")


class ServiceError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


@dataclass
class FleetData:
    """Held-out assets with their observed history. `frame` carries the ground-truth `rul` column, which the API only
    exposes in eval mode and never passes to a model."""

    spec: DatasetSpec
    frame: pd.DataFrame
    index: dict[int, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self):
        self.index = {int(a): g.sort_values("time").reset_index(drop=True) for a, g in self.frame.groupby("asset")}


class Services(Protocol):
    runs_dir: Path

    def spec(self, subset: str) -> DatasetSpec: ...
    def subsets(self) -> list[dict]: ...
    def bundle(self, subset: str) -> ModelBundle: ...
    def fleet(self, subset: str) -> FleetData: ...


class DiskServices:
    def __init__(self, bundle_dir: Path | None = None, data_dir: Path | None = None, configs_dir: Path | None = None, runs_dir: Path | None = None):
        self.bundle_dir = bundle_dir or ROOT / "artifacts" / "bundles"
        self.data_dir = data_dir or ROOT / "data" / "raw" / "extracted"
        self.configs_dir = configs_dir or CONFIGS_DIR
        self.runs_dir = runs_dir or RUNS_DIR
        self._bundles: dict[str, ModelBundle] = {}
        self._fleets: dict[str, FleetData] = {}

    def _names(self) -> list[str]:
        return sorted(p.stem.removeprefix("cmapss_") for p in self.configs_dir.glob("cmapss_*.yaml"))

    def spec(self, subset: str) -> DatasetSpec:
        if not SUBSET_RE.match(subset) or subset not in self._names():
            raise ServiceError(404, f"unknown subset {subset!r}")
        return load_spec(self.configs_dir / f"cmapss_{subset}.yaml")

    def subsets(self) -> list[dict]:
        out = []
        for name in self._names():
            spec = self.spec(name)
            out.append({"subset": name, "description": spec.description, "signals": len(spec.signals), "regimes": spec.n_regimes,
                        "model_available": (self.bundle_dir / f"{name}.joblib").exists(),
                        "data_available": (self.data_dir / f"test_{name}.txt").exists()})
        return out

    def bundle(self, subset: str) -> ModelBundle:
        self.spec(subset)
        if subset not in self._bundles:
            path = self.bundle_dir / f"{subset}.joblib"
            if not path.exists():
                raise ServiceError(503, f"model bundle for {subset} not found; build it with scripts/build_bundle.py")
            try:
                self._bundles[subset] = ModelBundle.load(path)
            except BundleError as exc:
                raise ServiceError(503, str(exc)) from exc
        return self._bundles[subset]

    def fleet(self, subset: str) -> FleetData:
        spec = self.spec(subset)
        if subset not in self._fleets:
            try:
                data = load_dataset(spec, self.data_dir)
            except FileNotFoundError as exc:
                raise ServiceError(503, f"dataset files for {subset} not available: {exc}") from exc
            except DataValidationError as exc:
                raise ServiceError(503, f"dataset for {subset} failed validation: {exc}") from exc
            self._fleets[subset] = FleetData(spec=spec, frame=add_test_rul(data.test, data.test_final_rul))
        return self._fleets[subset]


class MemoryServices:
    """Test double: bundles and fleets supplied directly."""

    def __init__(self, specs: dict[str, DatasetSpec], bundles: dict[str, ModelBundle], fleets: dict[str, FleetData], runs_dir: Path):
        self._specs, self._bundles, self._fleets, self.runs_dir = specs, bundles, fleets, runs_dir

    def spec(self, subset: str) -> DatasetSpec:
        if subset not in self._specs:
            raise ServiceError(404, f"unknown subset {subset!r}")
        return self._specs[subset]

    def subsets(self) -> list[dict]:
        return [{"subset": n, "description": s.description, "signals": len(s.signals), "regimes": s.n_regimes,
                 "model_available": n in self._bundles, "data_available": n in self._fleets} for n, s in self._specs.items()]

    def bundle(self, subset: str) -> ModelBundle:
        self.spec(subset)
        if subset not in self._bundles:
            raise ServiceError(503, f"model bundle for {subset} not found; build it with scripts/build_bundle.py")
        return self._bundles[subset]

    def fleet(self, subset: str) -> FleetData:
        self.spec(subset)
        if subset not in self._fleets:
            raise ServiceError(503, f"dataset files for {subset} not available")
        return self._fleets[subset]
