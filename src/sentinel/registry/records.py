import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

RUNS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "runs"


class RunRecord(BaseModel):
    run_id: str
    created_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    subset: str
    name: str
    model: str
    config: dict
    split: dict
    preprocessing: dict
    seeds: list[int]
    cv_metrics: dict = {}
    test_metrics: dict | None = None
    train_seconds: float | None = None
    latency_single_ms: float | None = None
    model_size_bytes: int | None = None
    artifact_path: str | None = None
    training_curve_path: str | None = None
    hardware: dict = {}
    notes: str = ""


def hardware_info() -> dict:
    import lightgbm
    import numpy
    import sklearn
    import torch

    return {
        "torch": torch.__version__,
        "processor": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": "cpu",
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def save_record(record: RunRecord, runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.run_id}.json"
    path.write_text(record.model_dump_json(indent=2))
    return path


def load_records(runs_dir: Path = RUNS_DIR) -> list[RunRecord]:
    return [RunRecord.model_validate(json.loads(p.read_text())) for p in sorted(runs_dir.glob("*.json"))]
