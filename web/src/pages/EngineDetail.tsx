import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { Margin, Provenance, RiskBadge } from "../components/Badges";
import { AnomalyChart, ExplainBars, RiskStrip, RulChart, SignalChart, rulRows } from "../components/Charts";
import { ErrorBox, Loading } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import { useSubset } from "../state/Subset";

export function EngineDetail() {
  const { subset } = useSubset();
  const params = useParams();
  const id = Number(params.engineId);
  const [truth, setTruth] = useState(false);
  const detail = useAsync((s) => api.engine(subset, id, truth, s), [subset, id, truth]);

  if (!Number.isInteger(id)) return <ErrorBox title="That is not an engine number" error={new Error("Use a whole number, for example /engines/34.")} />;
  if (detail.status === "loading") return <Loading label={`Loading engine ${id}`} />;
  if (detail.status === "error") {
    const missing = detail.error instanceof ApiError && detail.error.status === 404;
    return (
      <div>
        <ErrorBox title={missing ? `Engine ${id} is not in ${subset}` : "The engine could not be loaded"} error={detail.error} onRetry={missing ? undefined : detail.reload} />
        <p><Link to="/fleet">Back to the fleet</Link></p>
      </div>
    );
  }
  return <Ready key={`${subset}-${id}`} d={detail.data} truth={truth} setTruth={setTruth} />;
}

function Ready({ d, truth, setTruth }: { d: Awaited<ReturnType<typeof api.engine>>; truth: boolean; setTruth: (v: boolean) => void }) {
  const names = Object.keys(d.signals);
  const [chosen, setChosen] = useState<string[]>(names.slice(0, 3));
  const [cycle, setCycle] = useState(d.observed_cycles);
  const last = d.predictions;
  const i = last.time.length - 1;
  const explain = useAsync((s) => api.explain(d.subset, d.engine_id, cycle, s), [d.subset, d.engine_id, cycle]);

  return (
    <div className="with-margin">
      <div>
        <header className="page-head">
          <h1>Engine {d.engine_id}</h1>
          <p className="lede"><Link to="/fleet">Fleet</Link> · {d.observed_cycles} cycles observed in {d.subset} · <Link to={`/lab?engine=${d.engine_id}`}>Replay this engine</Link></p>
        </header>

        <section className="section">
          <div className="readout">
            <div><span className="label">Predicted cycles left at cycle {last.time[i]}</span><span className="big">{last.rul[i].toFixed(0)}</span><span className="unit">cycles</span></div>
            <div><span className="label">80% interval</span><span className="big" style={{ fontSize: "1.6rem" }}>{last.lower_80[i].toFixed(0)} to {last.upper_80[i].toFixed(0)}</span></div>
            <div><span className="label">Status</span><RiskBadge risk={last.risk[i]} /></div>
            <div><span className="label">Anomaly</span><span className={`flag ${last.anomaly_alert[i] ? "on" : ""}`}>{last.anomaly_alert[i] ? "Alert" : "None"}</span></div>
          </div>
          <div className="section-head">
            <h2>Remaining life over time</h2>
            <label className="checks"><span style={{ display: "inline-flex", gap: "0.4rem", alignItems: "center" }}>
              <input type="checkbox" checked={truth} onChange={(e) => setTruth(e.target.checked)} /> Show ground truth for comparison
            </span></label>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.4rem" }}>
            <Provenance kind="prediction" />{truth && <Provenance kind="ground_truth" />}
          </div>
          <RulChart rows={rulRows(d.predictions, d.ground_truth)} showTruth={truth && d.ground_truth !== null} label="Predicted remaining life with intervals" />
          <h3 style={{ marginTop: "1rem" }}>Warning timeline</h3>
          <RiskStrip risk={last.risk} anomaly={last.anomaly_alert} total={last.time.length} />
          <p className="flag" style={{ marginTop: "0.3rem" }}>Colour and shape show normal, warning and critical for each cycle; the dark ticks above mark anomaly alerts.</p>
        </section>

        <section className="section">
          <h2>Anomaly score</h2>
          <Provenance kind="prediction" />
          <AnomalyChart time={last.time} score={last.anomaly_score} threshold={d.anomaly_threshold} />
        </section>

        <section className="section">
          <div className="section-head"><h2>Sensor signals</h2><Provenance kind="dataset" /></div>
          <fieldset style={{ border: 0, padding: 0, margin: "0 0 0.5rem" }}>
            <legend className="sr-only">Choose signals to plot</legend>
            <div className="checks">
              {names.map((n) => (
                <label key={n}><input type="checkbox" checked={chosen.includes(n)} onChange={() => setChosen((c) => (c.includes(n) ? c.filter((x) => x !== n) : c.length < 6 ? [...c, n] : c))} /> {n}</label>
              ))}
            </div>
          </fieldset>
          {chosen.length === 0 ? <p className="flag">Choose up to six signals to plot.</p> : <SignalChart time={last.time} signals={d.signals} names={chosen} />}
          <p className="flag">Each signal is scaled to 0 to 1 over this engine's observed history so shapes can be compared.</p>
        </section>

        <section className="section">
          <div className="section-head"><h2>Why this prediction</h2><Provenance kind="prediction" /></div>
          <label className="field" style={{ maxWidth: "28rem", marginBottom: "0.8rem" }}>Explain the prediction made at cycle {cycle}
            <input type="range" min={1} max={d.observed_cycles} value={cycle} onChange={(e) => setCycle(Number(e.target.value))} aria-label="Cycle to explain" />
          </label>
          {explain.status === "loading" && <Loading label="Computing attributions" />}
          {explain.status === "error" && <ErrorBox title="The explanation could not be loaded" error={explain.error} onRetry={explain.reload} />}
          {explain.status === "ready" && (
            <>
              <p style={{ color: "var(--ink-2)" }}>
                At cycle {explain.data.cycle} the model starts from a typical value of {explain.data.base_value.toFixed(0)} cycles and these signals move it to {explain.data.rul.toFixed(0)}. Green raises the estimate, red lowers it.
              </p>
              <ExplainBars items={explain.data.contributions} />
            </>
          )}
        </section>
      </div>
      <Margin>
        <p><strong>Prediction versus truth.</strong> Ground truth is hidden by default. When shown, it is only for comparison; the model never receives it.</p>
        <p><strong>Intervals are marginal.</strong> They are calibrated on average across engines, and are least reliable in mid-life.</p>
        <p><strong>Explanations are associations.</strong> They show how this model uses its inputs, not what physically caused the wear. Signals that move together share credit.</p>
        <p><strong>Anomaly alerts are descriptive.</strong> The detector flags unusual sensor patterns; its accuracy on real faults is untested.</p>
      </Margin>
    </div>
  );
}
