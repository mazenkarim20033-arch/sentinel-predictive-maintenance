import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { ReplayState } from "../api/types";
import { Margin, Provenance, RiskBadge } from "../components/Badges";
import { RiskStrip, RulChart, SignalChart, rulRows } from "../components/Charts";
import { ErrorBox, Loading } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import { useSubset } from "../state/Subset";

type Mode = "blind" | "eval";
const SPEEDS = [1, 2, 5, 10];

export function SimulationLab() {
  const { subset } = useSubset();
  const [search, setSearch] = useSearchParams();
  const initial = Number(search.get("engine"));
  const [engineId, setEngineId] = useState(Number.isInteger(initial) && initial > 0 ? initial : 1);
  const [draft, setDraft] = useState(String(engineId));
  const [mode, setMode] = useState<Mode>("blind");

  const load = (e: FormEvent) => {
    e.preventDefault();
    const n = Number(draft);
    if (Number.isInteger(n) && n > 0) {
      setEngineId(n);
      setSearch({ engine: String(n) });
    }
  };
  const engine = useAsync((s) => api.engine(subset, engineId, mode === "eval", s), [subset, engineId, mode]);

  return (
    <div className="with-margin">
      <div>
        <header className="page-head">
          <h1>Simulation lab</h1>
          <p className="lede">Replay a held-out engine one cycle at a time. At every step the model sees only the cycles up to that point.</p>
        </header>
        <div className="toolbar">
          <form onSubmit={load} className="toolbar" style={{ margin: 0 }}>
            <label className="field">Engine number
              <input type="number" min={1} value={draft} onChange={(e) => setDraft(e.target.value)} style={{ width: "7rem" }} />
            </label>
            <button type="submit" className="primary">Load engine</button>
          </form>
          <div className="field">
            Mode
            <div className="segmented" role="group" aria-label="Replay mode">
              <button aria-pressed={mode === "blind"} onClick={() => setMode("blind")}>Blind</button>
              <button aria-pressed={mode === "eval"} onClick={() => setMode("eval")}>Evaluation</button>
            </div>
          </div>
        </div>
        <p className="mode-note">
          {mode === "blind"
            ? "Blind mode: like a live operation, the true remaining life is not shown."
            : "Evaluation mode: the true remaining life is shown next to the prediction so you can judge the error. It is never given to the model."}
        </p>

        {engine.status === "loading" && <Loading label={`Loading engine ${engineId}`} />}
        {engine.status === "error" && (
          <ErrorBox title={engine.error instanceof ApiError && engine.error.status === 404 ? `Engine ${engineId} is not in ${subset}` : "The engine could not be loaded"} error={engine.error} onRetry={engine.reload} />
        )}
        {engine.status === "ready" && <Player key={`${subset}-${engineId}-${mode}`} subset={subset} engine={engine.data} mode={mode} />}
      </div>
      <Margin>
        <p><strong>Why the replay is honest.</strong> Each step asks the server for the prediction at that cycle. The server cuts the history before running the model, so later measurements cannot leak in.</p>
        <p><strong>What is drawn.</strong> Only cycles up to the current one. Sensor scaling also uses only those cycles.</p>
        <p><strong>Ground truth.</strong> Visible in evaluation mode only, and labelled as such.</p>
        <p><strong>Held-out engines only.</strong> Training engines are not replayed because the model has already seen them.</p>
      </Margin>
    </div>
  );
}

function Player({ subset, engine, mode }: { subset: string; engine: Awaited<ReturnType<typeof api.engine>>; mode: Mode }) {
  const total = engine.observed_cycles;
  const names = Object.keys(engine.signals);
  const [cycle, setCycle] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [chosen, setChosen] = useState(names.slice(0, 3));
  const [replay, setReplay] = useState<ReplayState | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api.replay(subset, engine.engine_id, cycle, mode, controller.signal).then(
      (r) => { setReplay(r); setError(null); },
      (e: Error) => { if (e.name !== "AbortError") { setError(e); setPlaying(false); } },
    );
    return () => controller.abort();
  }, [subset, engine.engine_id, cycle, mode]);

  useEffect(() => {
    if (!playing) return;
    if (cycle >= total) { setPlaying(false); return; }
    const timer = window.setTimeout(() => setCycle((c) => Math.min(c + 1, total)), 1000 / speed);
    return () => window.clearTimeout(timer);
  }, [playing, cycle, speed, total]);

  const visible = useMemo(() => {
    const signals: Record<string, number[]> = {};
    for (const n of names) signals[n] = engine.signals[n].slice(0, cycle);
    return { time: Array.from({ length: cycle }, (_, i) => i + 1), signals };
  }, [engine, names, cycle]);
  const truth = mode === "eval" && engine.ground_truth
    ? { time: engine.ground_truth.time.filter((t) => t <= cycle), rul: engine.ground_truth.rul.filter((_, i) => engine.ground_truth!.time[i] <= cycle) }
    : null;
  const point = replay?.prediction.point;
  const shown = replay !== null && replay.cycle === cycle;

  return (
    <>
      <section className="section">
        <div className="transport" role="group" aria-label="Replay controls">
          <button onClick={() => { setPlaying(false); setCycle(1); }} disabled={cycle === 1 && !playing}>Restart</button>
          <button onClick={() => { setPlaying(false); setCycle((c) => Math.max(1, c - 1)); }} disabled={cycle <= 1}>Step back</button>
          {playing
            ? <button className="primary" onClick={() => setPlaying(false)}>Pause</button>
            : <button className="primary" onClick={() => { if (cycle >= total) setCycle(1); setPlaying(true); }}>{cycle >= total ? "Replay again" : "Play"}</button>}
          <button onClick={() => { setPlaying(false); setCycle((c) => Math.min(total, c + 1)); }} disabled={cycle >= total}>Step forward</button>
          <label className="field" style={{ flexDirection: "row", alignItems: "center", gap: "0.5rem" }}>Speed
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
              {SPEEDS.map((s) => <option key={s} value={s}>{s} cycle{s === 1 ? "" : "s"} per second</option>)}
            </select>
          </label>
        </div>
        <label className="field" style={{ marginTop: "0.8rem" }}>Cycle {cycle} of {total}
          <input type="range" min={1} max={total} value={cycle} onChange={(e) => { setPlaying(false); setCycle(Number(e.target.value)); }} aria-label="Replay position" />
        </label>
      </section>

      {error && <ErrorBox title="The replay step failed" error={error} />}
      {!replay && !error && <Loading label="Computing the first prediction" />}
      {replay && point && (
        <>
          <section className="section" aria-live="polite" aria-busy={!shown}>
            <div className="readout">
              <div><span className="label">Cycle</span><span className="big">{point.time}</span></div>
              <div><span className="label">Predicted cycles left</span><span className="big">{point.rul.toFixed(0)}</span></div>
              <div><span className="label">80% interval</span><span className="big" style={{ fontSize: "1.6rem" }}>{point.interval_80.lower.toFixed(0)} to {point.interval_80.upper.toFixed(0)}</span></div>
              <div><span className="label">Status</span><RiskBadge risk={point.risk} /></div>
              <div><span className="label">Anomaly</span><span className={`flag ${point.anomaly_alert ? "on" : ""}`}>{point.anomaly_alert ? "Alert" : "None"}</span></div>
              {replay.ground_truth && (
                <div>
                  <span className="label">True cycles left <Provenance kind="ground_truth" /></span>
                  <span className="big" data-testid="truth-value">{replay.ground_truth.rul.toFixed(0)}</span>
                  <span className="unit">error {(point.rul - Math.min(replay.ground_truth.rul, replay.prediction.rul_cap)).toFixed(0)} vs capped truth</span>
                </div>
              )}
            </div>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.4rem" }}>
              <Provenance kind="prediction" />{mode === "eval" && <Provenance kind="ground_truth" />}
            </div>
            <RulChart rows={rulRows(replay.history, truth, cycle)} showTruth={mode === "eval"} label="Predicted remaining life so far" />
            <h3 style={{ marginTop: "1rem" }}>Warning timeline</h3>
            <RiskStrip risk={replay.history.risk} anomaly={replay.history.anomaly_alert} total={total} />
          </section>

          <section className="section">
            <div className="section-head"><h2>Sensors so far</h2><Provenance kind="dataset" /></div>
            <fieldset style={{ border: 0, padding: 0, margin: "0 0 0.5rem" }}>
              <legend className="sr-only">Choose signals to plot</legend>
              <div className="checks">
                {names.map((n) => (
                  <label key={n}><input type="checkbox" checked={chosen.includes(n)} onChange={() => setChosen((c) => (c.includes(n) ? c.filter((x) => x !== n) : c.length < 6 ? [...c, n] : c))} /> {n}</label>
                ))}
              </div>
            </fieldset>
            {chosen.length > 0 && <SignalChart time={visible.time} signals={visible.signals} names={chosen} />}
            <h3 style={{ marginTop: "1rem" }}>Readings at cycle {cycle}</h3>
            <div className="sensors">
              {Object.entries(replay.sensors).slice(0, 12).map(([k, v]) => <span key={k}>{k} <b>{v.toFixed(2)}</b></span>)}
            </div>
          </section>
        </>
      )}
    </>
  );
}
