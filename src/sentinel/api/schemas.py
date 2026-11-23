from typing import Literal

from pydantic import BaseModel, Field

Provenance = Literal["dataset", "prediction", "ground_truth", "synthetic", "illustrative", "measured_evaluation"]
Risk = Literal["normal", "warning", "critical"]
CAP_NOTE = "RUL is capped at the training horizon; a value at the cap means 'at least that many cycles remain'."
RISK_NOTE = "Risk levels are experimental thresholds on predicted RUL, not calibrated failure probabilities."


class Interval(BaseModel):
    level: float
    lower: float
    upper: float


class PredictionPoint(BaseModel):
    time: int
    rul: float
    interval_80: Interval
    interval_90: Interval
    risk: Risk
    risk_instant: Risk
    anomaly_score: float
    anomaly_alert: bool


class Prediction(BaseModel):
    provenance: Literal["prediction"] = "prediction"
    subset: str
    point: PredictionPoint
    rul_cap: int
    anomaly_threshold: float
    notes: list[str]


class PredictRequest(BaseModel):
    subset: str = Field(pattern=r"^[A-Za-z0-9_]{1,40}$")
    history: list[dict[str, float]] = Field(min_length=1, max_length=3000)


class RulResponse(BaseModel):
    provenance: Literal["prediction"] = "prediction"
    subset: str
    time: int
    rul: float
    interval_80: Interval
    interval_90: Interval
    rul_cap: int
    notes: list[str]


class RiskResponse(BaseModel):
    provenance: Literal["prediction"] = "prediction"
    subset: str
    time: int
    risk: Risk
    risk_instant: Risk
    rul: float
    notes: list[str]


class AnomalyResponse(BaseModel):
    provenance: Literal["prediction"] = "prediction"
    subset: str
    time: int
    anomaly_score: float
    anomaly_threshold: float
    anomaly_alert: bool
    notes: list[str]


class SubsetInfo(BaseModel):
    subset: str
    description: str
    signals: int
    regimes: int
    model_available: bool
    data_available: bool


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    subsets: list[SubsetInfo]


class FleetSummary(BaseModel):
    total: int
    normal: int
    warning: int
    critical: int
    anomalous: int


class FleetEngine(BaseModel):
    engine_id: int
    observed_cycles: int
    rul: float
    interval_80: Interval
    risk: Risk
    anomaly_alert: bool


class FleetResponse(BaseModel):
    provenance: Literal["prediction"] = "prediction"
    subset: str
    note: str
    summary: FleetSummary
    total_matching: int
    engines: list[FleetEngine]


class SeriesOut(BaseModel):
    time: list[int]
    rul: list[float]
    lower_80: list[float]
    upper_80: list[float]
    lower_90: list[float]
    upper_90: list[float]
    risk: list[Risk]
    anomaly_score: list[float]
    anomaly_alert: list[bool]


class EngineDetail(BaseModel):
    subset: str
    engine_id: int
    observed_cycles: int
    signals_provenance: Literal["dataset"] = "dataset"
    signals: dict[str, list[float]]
    predictions_provenance: Literal["prediction"] = "prediction"
    predictions: SeriesOut
    anomaly_threshold: float
    ground_truth: "GroundTruthSeries | None" = None
    notes: list[str]


class GroundTruthSeries(BaseModel):
    provenance: Literal["ground_truth"] = "ground_truth"
    time: list[int]
    rul: list[float]


class GroundTruthPoint(BaseModel):
    provenance: Literal["ground_truth"] = "ground_truth"
    rul: float


class ReplayState(BaseModel):
    subset: str
    engine_id: int
    cycle: int
    observed_cycles: int
    mode: Literal["blind", "eval"]
    sensors_provenance: Literal["dataset"] = "dataset"
    sensors: dict[str, float]
    prediction: Prediction
    history: SeriesOut
    ground_truth: GroundTruthPoint | None
    notes: list[str]


class ModelInfo(BaseModel):
    subset: str
    available: bool
    meta: dict
    notes: list[str]


class EvaluationRow(BaseModel):
    model: str
    cv_cut_rmse: float | None
    test_rmse: float | None
    test_rmse_ci95: list[float] | None
    test_nasa_score: float | None
    fit_seconds: float | None
    latency_ms: float | None
    size_kb: float | None


class EvaluationResponse(BaseModel):
    provenance: Literal["measured_evaluation"] = "measured_evaluation"
    subset: str
    leaderboard: list[EvaluationRow]
    notes: list[str]


EngineDetail.model_rebuild()
