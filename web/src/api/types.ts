export type Risk = "normal" | "warning" | "critical";

export interface Interval { level: number; lower: number; upper: number }

export interface PredictionPoint {
  time: number;
  rul: number;
  interval_80: Interval;
  interval_90: Interval;
  risk: Risk;
  risk_instant: Risk;
  anomaly_score: number;
  anomaly_alert: boolean;
}

export interface Prediction {
  provenance: "prediction";
  subset: string;
  point: PredictionPoint;
  rul_cap: number;
  anomaly_threshold: number;
  notes: string[];
}

export interface SubsetInfo {
  subset: string;
  description: string;
  signals: number;
  regimes: number;
  model_available: boolean;
  data_available: boolean;
}

export interface Health { status: "ok" | "degraded"; subsets: SubsetInfo[] }

export interface FleetSummary { total: number; normal: number; warning: number; critical: number; anomalous: number }

export interface FleetEngine {
  engine_id: number;
  observed_cycles: number;
  rul: number;
  interval_80: Interval;
  risk: Risk;
  anomaly_alert: boolean;
}

export interface FleetResponse {
  provenance: "prediction";
  subset: string;
  note: string;
  summary: FleetSummary;
  total_matching: number;
  engines: FleetEngine[];
}

export interface SeriesOut {
  time: number[];
  rul: number[];
  lower_80: number[];
  upper_80: number[];
  lower_90: number[];
  upper_90: number[];
  risk: Risk[];
  anomaly_score: number[];
  anomaly_alert: boolean[];
}

export interface GroundTruthSeries { provenance: "ground_truth"; time: number[]; rul: number[] }
export interface GroundTruthPoint { provenance: "ground_truth"; rul: number }

export interface EngineDetail {
  subset: string;
  engine_id: number;
  observed_cycles: number;
  signals: Record<string, number[]>;
  predictions: SeriesOut;
  anomaly_threshold: number;
  ground_truth: GroundTruthSeries | null;
  notes: string[];
}

export interface ReplayState {
  subset: string;
  engine_id: number;
  cycle: number;
  observed_cycles: number;
  mode: "blind" | "eval";
  sensors: Record<string, number>;
  prediction: Prediction;
  history: SeriesOut;
  ground_truth: GroundTruthPoint | null;
  notes: string[];
}

export interface Contribution { signal: string; contribution: number; value: number }

export interface ExplainResponse {
  provenance: "prediction";
  subset: string;
  engine_id: number;
  cycle: number;
  base_value: number;
  raw_prediction: number;
  rul: number;
  contributions: Contribution[];
  notes: string[];
}

export interface ModelInfo { subset: string; available: boolean; meta: Record<string, unknown>; notes: string[] }

export interface EvaluationRow {
  model: string;
  cv_cut_rmse: number | null;
  test_rmse: number | null;
  test_rmse_ci95: number[] | null;
  test_nasa_score: number | null;
  fit_seconds: number | null;
  latency_ms: number | null;
  size_kb: number | null;
}

export interface EvaluationResponse {
  provenance: "measured_evaluation";
  subset: string;
  leaderboard: EvaluationRow[];
  notes: string[];
}
