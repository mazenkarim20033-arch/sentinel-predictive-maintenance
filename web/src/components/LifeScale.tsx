import { useNavigate } from "react-router-dom";
import type { FleetEngine, Risk } from "../api/types";

const STROKE: Record<Risk, string> = { normal: "var(--normal)", warning: "var(--warning)", critical: "var(--critical)" };
const W = 640;
const PAD = 10;
const MAX = 125;
const x = (rul: number) => PAD + (Math.min(rul, MAX) / MAX) * (W - 2 * PAD);

/** Every engine as a tick on one remaining-life axis. Ticks at the right edge have at least 125 cycles left. */
export function LifeScale({ engines, cap = MAX }: { engines: FleetEngine[]; cap?: number }) {
  const navigate = useNavigate();
  const sorted = [...engines].sort((a, b) => a.rul - b.rul);
  return (
    <svg className="scale" viewBox={`0 0 ${W} 132`} role="group" aria-label={`Remaining life of ${engines.length} engines on one axis from 0 to ${cap} cycles`}>
      {[15, 30].map((t) => (
        <g key={t}>
          <line className="limit" x1={x(t)} x2={x(t)} y1={16} y2={84} />
          <text x={x(t) + (t === 15 ? -4 : 4)} y={11} textAnchor={t === 15 ? "end" : "start"}>{t === 15 ? "critical below 15" : "warning below 30"}</text>
        </g>
      ))}
      {sorted.map((e, i) => {
        const row = i % 4;
        const open = () => navigate(`/engines/${e.engine_id}`);
        const name = `Engine ${e.engine_id}: about ${e.rul.toFixed(0)} cycles left (80% interval ${e.interval_80.lower.toFixed(0)} to ${e.interval_80.upper.toFixed(0)}), ${e.risk}`;
        return (
          <g key={e.engine_id} className="tick" tabIndex={0} role="link" aria-label={name}
            onClick={open} onKeyDown={(ev) => (ev.key === "Enter" || ev.key === " ") && open()}>
            <title>{name}</title>
            <line x1={x(e.rul)} x2={x(e.rul)} y1={22 + row * 9} y2={50 + row * 9} stroke={STROKE[e.risk]} strokeWidth={2.2} />
          </g>
        );
      })}
      <line className="axis" x1={PAD} x2={W - PAD} y1={96} y2={96} />
      {[0, 15, 30, 60, 90, 125].map((t) => (
        <g key={t}>
          <line className="axis" x1={x(t)} x2={x(t)} y1={96} y2={101} />
          <text x={x(t)} y={116} textAnchor={t === 0 ? "start" : t === 125 ? "end" : "middle"}>{t === 125 ? "125 or more" : t}</text>
        </g>
      ))}
      <text x={W / 2} y={129} textAnchor="middle">predicted cycles left</text>
    </svg>
  );
}
