import { vi } from "vitest";
import type { EngineDetail, EvaluationResponse, ExplainResponse, FleetEngine, FleetResponse, ReplayState, Risk, SeriesOut, SubsetInfo } from "../api/types";

export const SUBSET: SubsetInfo = {
  subset: "FD001", description: "NASA C-MAPSS turbofan degradation, FD001 (single operating condition). Simulated data.",
  signals: 3, regimes: 1, model_available: true, data_available: true,
};

const iv = (lo: number, hi: number, level = 0.8) => ({ level, lower: lo, upper: hi });

export function engine(id: number, rul: number, risk: Risk, anomaly = false): FleetEngine {
  return { engine_id: id, observed_cycles: 120 + id, rul, interval_80: iv(rul - 3, rul + 5), risk, anomaly_alert: anomaly };
}

export const ENGINES: FleetEngine[] = [
  engine(34, 5, "critical", true), engine(76, 9, "critical", true), engine(12, 22, "warning"), engine(8, 28, "warning"),
  engine(3, 90, "normal"), engine(5, 120, "normal"),
];

export function fleet(engines: FleetEngine[] = ENGINES, total = ENGINES.length): FleetResponse {
  const count = (r: Risk) => ENGINES.filter((e) => e.risk === r).length;
  return {
    provenance: "prediction", subset: "FD001", note: "State at last cycle.",
    summary: { total: ENGINES.length, normal: count("normal"), warning: count("warning"), critical: count("critical"), anomalous: ENGINES.filter((e) => e.anomaly_alert).length },
    total_matching: total, engines,
  };
}

export function series(n: number): SeriesOut {
  const time = Array.from({ length: n }, (_, i) => i + 1);
  return {
    time, rul: time.map((t) => Math.max(0, 100 - 10 * t)), lower_80: time.map(() => 1), upper_80: time.map(() => 120), lower_90: time.map(() => 0), upper_90: time.map(() => 125),
    risk: time.map(() => "normal" as Risk), anomaly_score: time.map(() => 0.4), anomaly_alert: time.map(() => false),
  };
}

export function engineDetail(id: number, n: number, truth = false): EngineDetail {
  const time = Array.from({ length: n }, (_, i) => i + 1);
  return {
    subset: "FD001", engine_id: id, observed_cycles: n, signals: { s2: time.map((t) => t * 1.5), s3: time.map((t) => 10 - t), s4: time.map(() => 5) },
    predictions: series(n), anomaly_threshold: 1.2,
    ground_truth: truth ? { provenance: "ground_truth", time, rul: time.map((t) => 200 - t) } : null, notes: [],
  };
}

export function replay(id: number, cycle: number, mode: "blind" | "eval", n: number): ReplayState {
  return {
    subset: "FD001", engine_id: id, cycle, observed_cycles: n, mode, sensors: { s2: cycle * 1.5, s3: 10 - cycle, s4: 5 },
    prediction: { provenance: "prediction", subset: "FD001", rul_cap: 125, anomaly_threshold: 1.2, notes: [],
      point: { time: cycle, rul: 100 - 10 * cycle, interval_80: iv(1, 120), interval_90: iv(0, 125, 0.9), risk: "normal", risk_instant: "normal", anomaly_score: 0.4, anomaly_alert: false } },
    history: series(cycle), ground_truth: mode === "eval" ? { provenance: "ground_truth", rul: 200 - cycle } : null, notes: [],
  };
}

export const EXPLAIN: ExplainResponse = {
  provenance: "prediction", subset: "FD001", engine_id: 34, cycle: 120, base_value: 90, raw_prediction: 5, rul: 5,
  contributions: [{ signal: "s11", contribution: -40.2, value: 1.1 }, { signal: "s4", contribution: -30.1, value: 2.2 }, { signal: "s12", contribution: 8.5, value: 0.3 }],
  notes: [],
};

export const EVALUATION: EvaluationResponse = {
  provenance: "measured_evaluation", subset: "FD001",
  leaderboard: [
    { model: "lightgbm_window50", cv_cut_rmse: 15.1, test_rmse: 13.75, test_rmse_ci95: [11.6, 15.6], test_nasa_score: 296, fit_seconds: 2.8, latency_ms: 1.4, size_kb: 1132 },
    { model: "constant", cv_cut_rmse: 39, test_rmse: 43.07, test_rmse_ci95: [38.6, 47.4], test_nasa_score: 33629, fit_seconds: 0, latency_ms: 0, size_kb: 0 },
  ],
  notes: ["Measured on the official test set."],
};

type Handler = unknown | ((url: URL) => unknown | Promise<unknown>);
export type Routes = Record<string, Handler>;
export const calls: string[] = [];

export function ok(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Replaces fetch with a router keyed by pathname. A handler returns JSON, or a Response for errors. */
export function mockFetch(routes: Routes) {
  calls.length = 0;
  const fn = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input), "http://localhost");
    calls.push(url.pathname + url.search);
    const key = Object.keys(routes).find((k) => url.pathname === k || url.pathname === `/api${k}`);
    if (!key) return ok({ detail: `no mock for ${url.pathname}` }, 404);
    const h = routes[key];
    const out = typeof h === "function" ? await (h as (u: URL) => unknown)(url) : h;
    return out instanceof Response ? out : ok(out);
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

export const baseRoutes: Routes = {
  "/subsets": [SUBSET],
  "/fleet/FD001": () => fleet(),
  "/models": [{ subset: "FD001", available: true, meta: { n_training_assets: 80, n_calibration_assets: 20, trained_utc: "2026-11-22T12:00:00" }, notes: [] }],
  "/evaluation/FD001": EVALUATION,
  "/explain/FD001/34": EXPLAIN,
};
