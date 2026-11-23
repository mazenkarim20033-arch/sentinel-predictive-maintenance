import { Area, CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Risk, SeriesOut } from "../api/types";

export const WARN_LEVEL = 30;
export const CRIT_LEVEL = 15;
const AXIS = { fontSize: 11, fill: "var(--ink-2)" };
const GRID = "var(--rule)";
export const TOOLTIP = { background: "var(--surface)", border: "1px solid var(--rule)", color: "var(--ink)", fontSize: 12 };
export const SERIES_COLORS = ["#1f5fa8", "#b8760d", "#2b7a66", "#bf3b2a", "#6b4fa0", "#0e7c86"];

interface RulPoint { time: number; rul: number; band80: [number, number]; band90: [number, number]; truth?: number }

export function rulRows(s: SeriesOut, truth?: { time: number[]; rul: number[] } | null, upTo?: number): RulPoint[] {
  const truthAt = new Map((truth?.time ?? []).map((t, i) => [t, truth!.rul[i]]));
  const rows: RulPoint[] = [];
  s.time.forEach((t, i) => {
    if (upTo !== undefined && t > upTo) return;
    rows.push({ time: t, rul: s.rul[i], band80: [s.lower_80[i], s.upper_80[i]], band90: [s.lower_90[i], s.upper_90[i]], truth: truthAt.get(t) === undefined ? undefined : Math.min(truthAt.get(t)!, 130) });
  });
  return rows;
}

export function RulChart({ rows, showTruth, label }: { rows: RulPoint[]; showTruth: boolean; label: string }) {
  return (
    <figure style={{ margin: 0 }} aria-label={label}>
      <div className="legend" aria-hidden="true">
        <span style={{ color: "var(--pred)" }}>Predicted remaining life</span>
        <span className="band" style={{ color: "var(--pred)" }}>80% and 90% interval</span>
        {showTruth && <span className="dashed" style={{ color: "var(--truth)" }}>True remaining life (ground truth, capped at 130 for display)</span>}
      </div>
      <div className="chart">
        <ResponsiveContainer>
          <ComposedChart data={rows} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="time" tick={AXIS} tickLine={false} minTickGap={32} label={{ value: "cycle", position: "insideBottomRight", offset: -2, fontSize: 11, fill: "var(--ink-2)" }} />
            <YAxis domain={[0, 130]} ticks={[0, 30, 60, 90, 125]} tick={AXIS} tickLine={false} width={34} />
            <Tooltip contentStyle={TOOLTIP} labelStyle={{ color: "var(--ink-2)" }} formatter={(v: unknown) => (Array.isArray(v) ? `${Number(v[0]).toFixed(0)} to ${Number(v[1]).toFixed(0)}` : Number(v).toFixed(1))} labelFormatter={(l) => `Cycle ${l}`} />
            <ReferenceLine y={WARN_LEVEL} stroke="var(--warning)" strokeDasharray="4 4" />
            <ReferenceLine y={CRIT_LEVEL} stroke="var(--critical)" strokeDasharray="4 4" />
            <Area type="monotone" dataKey="band90" stroke="none" fill="var(--pred)" fillOpacity={0.1} isAnimationActive={false} name="90% interval" />
            <Area type="monotone" dataKey="band80" stroke="none" fill="var(--pred)" fillOpacity={0.2} isAnimationActive={false} name="80% interval" />
            <Line type="monotone" dataKey="rul" stroke="var(--pred)" strokeWidth={2.2} dot={false} isAnimationActive={false} name="Predicted" />
            {showTruth && <Line type="monotone" dataKey="truth" stroke="var(--truth)" strokeWidth={1.6} strokeDasharray="5 4" dot={false} isAnimationActive={false} name="Ground truth" />}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export function AnomalyChart({ time, score, threshold }: { time: number[]; score: number[]; threshold: number }) {
  const rows = time.map((t, i) => ({ time: t, score: score[i] }));
  return (
    <figure style={{ margin: 0 }} aria-label="Anomaly score over time">
      <div className="chart short">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="time" tick={AXIS} tickLine={false} minTickGap={32} />
            <YAxis tick={AXIS} tickLine={false} width={34} />
            <Tooltip contentStyle={TOOLTIP} labelStyle={{ color: "var(--ink-2)" }} formatter={(v: unknown) => Number(v).toFixed(2)} labelFormatter={(l) => `Cycle ${l}`} />
            <ReferenceLine y={threshold} stroke="var(--critical)" strokeDasharray="4 4" label={{ value: "alert level", fontSize: 11, fill: "var(--critical)", position: "insideTopLeft" }} />
            <Line type="monotone" dataKey="score" stroke="var(--ink)" strokeWidth={1.6} dot={false} isAnimationActive={false} name="Anomaly score" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export function scaleToUnit(values: number[]): number[] {
  if (values.length === 0) return [];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return values.map((v) => (hi === lo ? 0.5 : (v - lo) / (hi - lo)));
}

export function SignalChart({ time, signals, names }: { time: number[]; signals: Record<string, number[]>; names: string[] }) {
  const scaled = Object.fromEntries(names.map((n) => [n, scaleToUnit(signals[n] ?? [])]));
  const rows = time.map((t, i) => ({ time: t, ...Object.fromEntries(names.map((n) => [n, scaled[n][i]])) }));
  return (
    <figure style={{ margin: 0 }} aria-label="Sensor signals over time, each scaled to 0 to 1">
      <div className="chart">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="time" tick={AXIS} tickLine={false} minTickGap={32} />
            <YAxis domain={[0, 1]} tick={AXIS} tickLine={false} width={34} />
            <Tooltip contentStyle={TOOLTIP} labelStyle={{ color: "var(--ink-2)" }} formatter={(v: unknown) => Number(v).toFixed(2)} labelFormatter={(l) => `Cycle ${l}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {names.map((n, i) => (
              <Line key={n} type="monotone" dataKey={n} stroke={SERIES_COLORS[i % SERIES_COLORS.length]} strokeWidth={1.7} dot={false} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

export function CompareChart({ series }: { series: { id: number; time: number[]; rul: number[] }[] }) {
  const maxT = Math.max(...series.map((s) => Math.max(...s.time)));
  const rows = Array.from({ length: maxT }, (_, i) => {
    const row: Record<string, number> = { time: i + 1 };
    series.forEach((s) => {
      const k = s.time.indexOf(i + 1);
      if (k >= 0) row[`e${s.id}`] = s.rul[k];
    });
    return row;
  });
  return (
    <figure style={{ margin: 0 }} aria-label="Predicted remaining life of the selected engines">
      <div className="chart">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="time" tick={AXIS} tickLine={false} minTickGap={32} />
            <YAxis domain={[0, 130]} ticks={[0, 30, 60, 90, 125]} tick={AXIS} tickLine={false} width={34} />
            <Tooltip contentStyle={TOOLTIP} labelStyle={{ color: "var(--ink-2)" }} formatter={(v: unknown) => Number(v).toFixed(1)} labelFormatter={(l) => `Cycle ${l}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <ReferenceLine y={WARN_LEVEL} stroke="var(--warning)" strokeDasharray="4 4" />
            <ReferenceLine y={CRIT_LEVEL} stroke="var(--critical)" strokeDasharray="4 4" />
            {series.map((s, i) => (
              <Line key={s.id} type="monotone" dataKey={`e${s.id}`} name={`Engine ${s.id}`} stroke={SERIES_COLORS[i % SERIES_COLORS.length]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

const RISK_FILL: Record<Risk, string> = { normal: "var(--normal)", warning: "var(--warning)", critical: "var(--critical)" };

export function RiskStrip({ risk, anomaly, total }: { risk: Risk[]; anomaly: boolean[]; total: number }) {
  const n = Math.max(total, risk.length, 1);
  const w = 1000 / n;
  const counts = { normal: 0, warning: 0, critical: 0 } as Record<Risk, number>;
  risk.forEach((r) => (counts[r] += 1));
  return (
    <svg className="strip" viewBox="0 0 1000 46" role="img" preserveAspectRatio="none" style={{ height: 46 }}
      aria-label={`Warning timeline: ${counts.normal} cycles normal, ${counts.warning} warning, ${counts.critical} critical, ${anomaly.filter(Boolean).length} with an anomaly alert`}>
      <rect x="0" y="0" width="1000" height="46" fill="var(--paper-2)" />
      {risk.map((r, i) => <rect key={i} x={i * w} y={16} width={w + 0.4} height={30} fill={RISK_FILL[r]} />)}
      {anomaly.map((a, i) => (a ? <rect key={`a${i}`} x={i * w} y={2} width={Math.max(w, 1.5)} height={11} fill="var(--ink)" /> : null))}
    </svg>
  );
}

export function ExplainBars({ items, max = 8 }: { items: { signal: string; contribution: number }[]; max?: number }) {
  const top = items.slice(0, max);
  const peak = Math.max(...top.map((i) => Math.abs(i.contribution)), 1e-9);
  return (
    <div className="bars" role="list" aria-label="Signal contributions to the predicted remaining life, in cycles">
      {top.map((i) => (
        <div className="bar-row" role="listitem" key={i.signal}>
          <span>{i.signal}</span>
          <div className="bar-track" aria-hidden="true">
            <div className={`bar-fill ${i.contribution >= 0 ? "pos" : "neg"}`} style={{ width: `${(Math.abs(i.contribution) / peak) * 50}%` }} />
          </div>
          <span style={{ textAlign: "right" }}>{i.contribution >= 0 ? "+" : "−"}{Math.abs(i.contribution).toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}
