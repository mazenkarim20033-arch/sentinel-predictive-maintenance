/** Same interface as the live API client, backed by recorded outputs (see scripts/export_snapshot.py).
 *  Filtering, sorting and cycle slicing follow the server's rules; a replay at cycle t returns the recording up to t. */
import { ApiError } from "./errors";
import type { EngineDetail, EvaluationResponse, ExplainResponse, FleetEngine, FleetResponse, Health, ModelInfo, ReplayState, Risk, SeriesOut, SubsetInfo } from "./types";

const ROOT = `${import.meta.env.BASE_URL ?? "/"}snapshot/`;
const cache = new Map<string, Promise<unknown>>();
/** Test hook: forget every loaded file. */
export const clearSnapshotCache = () => cache.clear();

interface EngineFile {
  engine_id: number;
  observed_cycles: number;
  signals: Record<string, number[]>;
  predictions: SeriesOut & { risk_instant: Risk[] };
  truth: { time: number[]; rul: number[] };
}
interface ExplainFile { base_value: number; signals: string[]; raw: number[]; contrib: number[][] }
interface Meta { rul_cap: number; anomaly_threshold: number; engine_notes: string[]; replay_notes: string[]; prediction_notes: string[]; explain_notes: string[]; fleet_note: string }

function load<T>(path: string, notFound: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(path, fetch(`${ROOT}${path}`).then(async (res) => {
      if (res.status === 404 || !res.ok) throw new ApiError(404, notFound);
      return res.json();
    }).catch((e) => {
      cache.delete(path);
      throw e instanceof ApiError ? e : new ApiError(0, "The recorded demo data could not be loaded.", e);
    }));
  }
  return cache.get(path) as Promise<T>;
}

const engineFile = (subset: string, id: number) => load<EngineFile>(`engines/${subset}/${id}.json`, `engine ${id} not found`);
const metaFor = async (subset: string) => (await load<Record<string, Meta>>("meta.json", "snapshot metadata missing"))[subset];
const interval = (lo: number[], hi: number[], i: number, level: number) => ({ level, lower: lo[i], upper: hi[i] });

function series(p: EngineFile["predictions"], upTo: number): SeriesOut {
  const n = upTo;
  return {
    time: p.time.slice(0, n), rul: p.rul.slice(0, n), lower_80: p.lower_80.slice(0, n), upper_80: p.upper_80.slice(0, n),
    lower_90: p.lower_90.slice(0, n), upper_90: p.upper_90.slice(0, n), risk: p.risk.slice(0, n), anomaly_score: p.anomaly_score.slice(0, n),
    anomaly_alert: p.anomaly_alert.slice(0, n),
  };
}

export const snapshotApi = {
  health: async (): Promise<Health> => ({ status: "ok", subsets: await load<SubsetInfo[]>("subsets.json", "no snapshot") }),
  subsets: (_signal?: AbortSignal) => load<SubsetInfo[]>("subsets.json", "no snapshot"),

  fleet: async (subset: string, p: { risk?: Risk; anomalous?: boolean; search?: string; sort?: "rul" | "engine_id"; limit?: number; offset?: number } = {}, _signal?: AbortSignal): Promise<FleetResponse> => {
    const file = await load<{ engines: FleetEngine[] }>(`fleet/${subset}.json`, `unknown subset ${subset}`);
    const all = file.engines;
    const count = (r: Risk) => all.filter((e) => e.risk === r).length;
    let view = all.filter((e) => (p.risk ? e.risk === p.risk : true) && (p.anomalous === undefined ? true : e.anomaly_alert === p.anomalous) && (p.search ? String(e.engine_id).includes(p.search) : true));
    view = [...view].sort(p.sort === "engine_id" ? (a, b) => a.engine_id - b.engine_id : (a, b) => a.rul - b.rul);
    const offset = p.offset ?? 0;
    return {
      provenance: "prediction", subset, note: (await metaFor(subset)).fleet_note,
      summary: { total: all.length, normal: count("normal"), warning: count("warning"), critical: count("critical"), anomalous: all.filter((e) => e.anomaly_alert).length },
      total_matching: view.length, engines: view.slice(offset, offset + (p.limit ?? 100)),
    };
  },

  engine: async (subset: string, id: number, includeTruth = false, _signal?: AbortSignal): Promise<EngineDetail> => {
    const [f, meta] = await Promise.all([engineFile(subset, id), metaFor(subset)]);
    return {
      subset, engine_id: id, observed_cycles: f.observed_cycles, signals: f.signals, predictions: series(f.predictions, f.observed_cycles),
      anomaly_threshold: meta.anomaly_threshold, ground_truth: includeTruth ? { provenance: "ground_truth", time: f.truth.time, rul: f.truth.rul } : null, notes: meta.engine_notes,
    };
  },

  replay: async (subset: string, id: number, cycle: number, mode: "blind" | "eval", _signal?: AbortSignal): Promise<ReplayState> => {
    const [f, meta] = await Promise.all([engineFile(subset, id), metaFor(subset)]);
    if (!Number.isInteger(cycle) || cycle < 1) throw new ApiError(422, "cycle must be a whole number of at least 1");
    if (cycle > f.observed_cycles) throw new ApiError(422, `cycle ${cycle} is beyond the ${f.observed_cycles} observed cycles of engine ${id}`);
    const p = f.predictions;
    const i = cycle - 1;
    return {
      subset, engine_id: id, cycle, observed_cycles: f.observed_cycles, mode,
      sensors: Object.fromEntries(Object.entries(f.signals).map(([k, v]) => [k, v[i]])),
      prediction: {
        provenance: "prediction", subset, rul_cap: meta.rul_cap, anomaly_threshold: meta.anomaly_threshold, notes: meta.prediction_notes,
        point: { time: cycle, rul: p.rul[i], interval_80: interval(p.lower_80, p.upper_80, i, 0.8), interval_90: interval(p.lower_90, p.upper_90, i, 0.9),
                 risk: p.risk[i], risk_instant: p.risk_instant[i], anomaly_score: p.anomaly_score[i], anomaly_alert: p.anomaly_alert[i] },
      },
      history: series(p, cycle),
      ground_truth: mode === "eval" ? { provenance: "ground_truth", rul: f.truth.rul[i] } : null,
      notes: meta.replay_notes,
    };
  },

  explain: async (subset: string, id: number, cycle?: number, _signal?: AbortSignal): Promise<ExplainResponse> => {
    const [f, e, meta] = await Promise.all([engineFile(subset, id), load<ExplainFile>(`explain/${subset}/${id}.json`, `engine ${id} not found`), metaFor(subset)]);
    const t = cycle ?? f.observed_cycles;
    if (t < 1 || t > f.observed_cycles) throw new ApiError(422, `cycle ${t} is beyond the ${f.observed_cycles} observed cycles of engine ${id}`);
    const i = t - 1;
    const contributions = e.signals.map((s, k) => ({ signal: s, contribution: e.contrib[i][k], value: f.signals[s][i] })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    return { provenance: "prediction", subset, engine_id: id, cycle: t, base_value: e.base_value, raw_prediction: e.raw[i], rul: Math.min(Math.max(e.raw[i], 0), meta.rul_cap), contributions, notes: meta.explain_notes };
  },

  models: (_signal?: AbortSignal) => load<ModelInfo[]>("models.json", "no snapshot"),
  evaluation: (subset: string, _signal?: AbortSignal) => load<EvaluationResponse>(`evaluation/${subset}.json`, `no evaluation records for ${subset}`),
};
