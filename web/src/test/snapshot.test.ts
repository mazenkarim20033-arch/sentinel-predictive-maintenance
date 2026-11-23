import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import { liveApi } from "../api/client";
import { clearSnapshotCache, snapshotApi } from "../api/snapshot";
import type { Risk } from "../api/types";

const N = 6;
const time = Array.from({ length: N }, (_, i) => i + 1);
const risks: Risk[] = ["normal", "normal", "warning", "warning", "critical", "critical"];

function engineFile(id: number) {
  return {
    engine_id: id, observed_cycles: N,
    signals: { s2: time.map((t) => t * 1.5), s3: time.map((t) => 10 - t) },
    predictions: {
      time, rul: [100, 80, 60, 25, 12, 4], lower_80: [90, 70, 50, 20, 8, 1], upper_80: [110, 90, 70, 30, 16, 8], lower_90: [85, 65, 45, 15, 5, 0], upper_90: [115, 95, 75, 35, 20, 10],
      risk: risks, risk_instant: risks, anomaly_score: [0.1, 0.2, 0.3, 1.5, 2, 3], anomaly_alert: [false, false, false, false, true, true],
    },
    truth: { time, rul: [200, 150, 90, 40, 10, 3] },
  };
}

const FLEET = [
  { engine_id: 3, observed_cycles: 6, rul: 90, interval_80: { level: 0.8, lower: 80, upper: 100 }, risk: "normal", anomaly_alert: false },
  { engine_id: 34, observed_cycles: 6, rul: 4, interval_80: { level: 0.8, lower: 1, upper: 8 }, risk: "critical", anomaly_alert: true },
  { engine_id: 12, observed_cycles: 6, rul: 22, interval_80: { level: 0.8, lower: 15, upper: 30 }, risk: "warning", anomaly_alert: false },
  { engine_id: 340, observed_cycles: 6, rul: 70, interval_80: { level: 0.8, lower: 60, upper: 80 }, risk: "normal", anomaly_alert: false },
];

const FILES: Record<string, unknown> = {
  "subsets.json": [{ subset: "FDX", description: "fixture", signals: 2, regimes: 1, model_available: true, data_available: true }],
  "fleet/FDX.json": { subset: "FDX", engines: FLEET },
  "engines/FDX/34.json": engineFile(34),
  "explain/FDX/34.json": { base_value: 90, signals: ["s2", "s3"], raw: [95, 85, 60, 25, 12, -3], contrib: [[5, 0], [-5, 0], [-10, -20], [-30, -35], [-40, -38], [-60, -33]] },
  "meta.json": { FDX: { rul_cap: 125, anomaly_threshold: 1.2, engine_notes: ["e"], replay_notes: ["r"], prediction_notes: ["p"], explain_notes: ["x"], fleet_note: "fleet note" } },
};

function stubFetch() {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const key = String(url).replace(/^.*snapshot\//, ""); // independent of the site's base path
    return key in FILES ? new Response(JSON.stringify(FILES[key]), { status: 200 }) : new Response("not found", { status: 404 });
  }));
}
beforeEach(() => clearSnapshotCache());
afterEach(() => vi.unstubAllGlobals());

describe("snapshot client (recorded outputs behind the live API's interface)", () => {
  it("filters, sorts, pages and summarises the fleet like the server", async () => {
    stubFetch();
    const all = await snapshotApi.fleet("FDX", { limit: 10 });
    expect(all.summary).toEqual({ total: 4, normal: 2, warning: 1, critical: 1, anomalous: 1 });
    expect(all.engines.map((e) => e.engine_id)).toEqual([34, 12, 340, 3]);
    expect(all.note).toBe("fleet note");
    expect((await snapshotApi.fleet("FDX", { risk: "normal" })).total_matching).toBe(2);
    expect((await snapshotApi.fleet("FDX", { anomalous: true })).engines.map((e) => e.engine_id)).toEqual([34]);
    expect((await snapshotApi.fleet("FDX", { search: "34" })).engines.map((e) => e.engine_id)).toEqual([34, 340]);
    expect((await snapshotApi.fleet("FDX", { sort: "engine_id" })).engines.map((e) => e.engine_id)).toEqual([3, 12, 34, 340]);
    const page = await snapshotApi.fleet("FDX", { sort: "engine_id", limit: 2, offset: 2 });
    expect(page.engines.map((e) => e.engine_id)).toEqual([34, 340]);
    expect(page.total_matching).toBe(4);
  });

  it("hides ground truth unless asked and rejects unknown engines", async () => {
    stubFetch();
    expect((await snapshotApi.engine("FDX", 34)).ground_truth).toBeNull();
    const withTruth = await snapshotApi.engine("FDX", 34, true);
    expect(withTruth.ground_truth).toMatchObject({ provenance: "ground_truth", rul: [200, 150, 90, 40, 10, 3] });
    await expect(snapshotApi.engine("FDX", 999)).rejects.toMatchObject({ status: 404 });
  });

  it("replays a recording up to the chosen cycle and nothing beyond it", async () => {
    stubFetch();
    const r = await snapshotApi.replay("FDX", 34, 3, "blind");
    expect(r.history.time).toEqual([1, 2, 3]);
    expect(r.history.rul).toEqual([100, 80, 60]);
    expect(r.prediction.point).toMatchObject({ time: 3, rul: 60, risk: "warning", anomaly_alert: false });
    expect(r.sensors).toEqual({ s2: 4.5, s3: 7 });
    expect(r.ground_truth).toBeNull();
    expect(JSON.stringify(r)).not.toContain("200");
    const ev = await snapshotApi.replay("FDX", 34, 3, "eval");
    expect(ev.ground_truth).toEqual({ provenance: "ground_truth", rul: 90 });
    expect(ev.prediction).toEqual(r.prediction);
  });

  it("validates the cycle", async () => {
    stubFetch();
    await expect(snapshotApi.replay("FDX", 34, 0, "blind")).rejects.toMatchObject({ status: 422 });
    await expect(snapshotApi.replay("FDX", 34, 7, "blind")).rejects.toThrowError(/beyond the 6 observed cycles/);
    await expect(snapshotApi.explain("FDX", 34, 9)).rejects.toMatchObject({ status: 422 });
  });

  it("explains a cycle with contributions sorted by size and the output clipped to the cap", async () => {
    stubFetch();
    const e = await snapshotApi.explain("FDX", 34, 6);
    expect(e.contributions.map((c) => c.signal)).toEqual(["s2", "s3"]);
    expect(e.contributions[0]).toMatchObject({ contribution: -60, value: 9 });
    expect(e.rul).toBe(0);
    expect(e.raw_prediction).toBe(-3);
    expect((await snapshotApi.explain("FDX", 34)).cycle).toBe(6);
  });

  it("reports missing recordings as errors instead of crashing", async () => {
    stubFetch();
    await expect(snapshotApi.evaluation("NOPE")).rejects.toBeInstanceOf(ApiError);
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("offline"); }));
    await expect(snapshotApi.models()).rejects.toMatchObject({ status: 0 });
  });
});

const SNAP = resolve(process.cwd(), "public", "snapshot");
const live = describe.skipIf(!import.meta.env.VITE_API_BASE || !existsSync(resolve(SNAP, "manifest.json")));

live("recorded demo equals the live API", () => {
  const near = (a: number, b: number, tol: number) => expect(Math.abs(a - b)).toBeLessThanOrEqual(tol);

  function dispatchFetch() {
    const real = globalThis.fetch;
    vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("snapshot/") && !url.startsWith("http")) {
        const p = resolve(SNAP, url.replace(/^.*snapshot\//, ""));
        return Promise.resolve(existsSync(p) ? new Response(readFileSync(p, "utf8"), { status: 200 }) : new Response("nf", { status: 404 }));
      }
      return real(input, { ...init, signal: undefined });
    });
  }

  it("fleet state matches for every FD001 engine", async () => {
    dispatchFetch();
    const [snap, liveFleet] = await Promise.all([snapshotApi.fleet("FD001", { limit: 500, sort: "engine_id" }), liveApi.fleet("FD001", { limit: 500, sort: "engine_id" })]);
    expect(snap.summary).toEqual(liveFleet.summary);
    expect(snap.engines.length).toBe(100);
    snap.engines.forEach((e, i) => {
      const l = liveFleet.engines[i];
      expect(e.engine_id).toBe(l.engine_id);
      expect(e.risk).toBe(l.risk);
      expect(e.anomaly_alert).toBe(l.anomaly_alert);
      near(e.rul, l.rul, 1e-3);
    });
  }, 120_000);

  it("replay and explanation match at the first, a middle and the last cycle", async () => {
    dispatchFetch();
    for (const [id, cycles] of [[34, [1, 2, 60, 203]], [7, [1, 50, 140]]] as [number, number[]][]) {
      for (const cycle of cycles) {
        const [s, l] = await Promise.all([snapshotApi.replay("FD001", id, cycle, "eval"), liveApi.replay("FD001", id, cycle, "eval")]);
        near(s.prediction.point.rul, l.prediction.point.rul, 2e-3);
        near(s.prediction.point.interval_90.upper, l.prediction.point.interval_90.upper, 2e-3);
        near(s.prediction.point.anomaly_score, l.prediction.point.anomaly_score, 2e-4);
        expect(s.prediction.point.risk).toBe(l.prediction.point.risk);
        expect(s.prediction.point.risk_instant).toBe(l.prediction.point.risk_instant);
        expect(s.prediction.point.anomaly_alert).toBe(l.prediction.point.anomaly_alert);
        expect(s.history.risk).toEqual(l.history.risk);
        expect(s.history.anomaly_alert).toEqual(l.history.anomaly_alert);
        s.history.rul.forEach((v, i) => near(v, l.history.rul[i], 2e-3));
        near(s.ground_truth!.rul, l.ground_truth!.rul, 0.06);
        Object.keys(l.sensors).forEach((k) => near(s.sensors[k], l.sensors[k], 1e-3));
        const [se, le] = await Promise.all([snapshotApi.explain("FD001", id, cycle), liveApi.explain("FD001", id, cycle)]);
        near(se.base_value, le.base_value, 1e-3);
        expect(se.contributions.map((c) => c.signal).slice(0, 3)).toEqual(le.contributions.map((c) => c.signal).slice(0, 3));
        se.contributions.forEach((c) => near(c.contribution, le.contributions.find((x) => x.signal === c.signal)!.contribution, 2e-3));
      }
    }
  }, 180_000);
});
