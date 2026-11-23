import type { EngineDetail, EvaluationResponse, ExplainResponse, FleetResponse, Health, ModelInfo, ReplayState, Risk, SubsetInfo } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";
export const API_DOWN = "Cannot reach the SENTINEL API. Start it with: uvicorn sentinel.api.app:app --port 8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

type Params = Record<string, string | number | boolean | undefined>;

async function request<T>(path: string, params?: Params, signal?: AbortSignal): Promise<T> {
  const query = Object.entries(params ?? {})
    .filter(([, v]) => v !== undefined && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}${query ? `?${query}` : ""}`, { signal });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(0, API_DOWN);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ") || detail;
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", undefined, signal),
  subsets: (signal?: AbortSignal) => request<SubsetInfo[]>("/subsets", undefined, signal),
  fleet: (subset: string, p: { risk?: Risk; anomalous?: boolean; search?: string; sort?: "rul" | "engine_id"; limit?: number; offset?: number } = {}, signal?: AbortSignal) =>
    request<FleetResponse>(`/fleet/${subset}`, p, signal),
  engine: (subset: string, id: number, includeTruth = false, signal?: AbortSignal) =>
    request<EngineDetail>(`/engines/${subset}/${id}`, { include_truth: includeTruth || undefined }, signal),
  replay: (subset: string, id: number, cycle: number, mode: "blind" | "eval", signal?: AbortSignal) =>
    request<ReplayState>(`/replay/${subset}/${id}`, { cycle, mode }, signal),
  explain: (subset: string, id: number, cycle?: number, signal?: AbortSignal) => request<ExplainResponse>(`/explain/${subset}/${id}`, { cycle }, signal),
  models: (signal?: AbortSignal) => request<ModelInfo[]>("/models", undefined, signal),
  evaluation: (subset: string, signal?: AbortSignal) => request<EvaluationResponse>(`/evaluation/${subset}`, undefined, signal),
};
