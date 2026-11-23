import { Link } from "react-router-dom";
import { api } from "../api/client";
import { Margin, Provenance, RiskBadge } from "../components/Badges";
import { LifeScale } from "../components/LifeScale";
import { Empty, ErrorBox, Loading } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import { useSubset } from "../state/Subset";

export function Dashboard() {
  const { subset, info } = useSubset();
  const fleet = useAsync((s) => api.fleet(subset, { limit: 500, sort: "rul" }, s), [subset]);

  return (
    <div className="with-margin">
      <div>
        <header className="page-head">
          <h1>Fleet health</h1>
          <p className="lede">Held-out engines from {subset}, shown at the last cycle each was observed. {info?.description}</p>
        </header>
        {fleet.status === "loading" && <Loading label="Loading fleet state" />}
        {fleet.status === "error" && <ErrorBox title="The fleet could not be loaded" error={fleet.error} onRetry={fleet.reload} />}
        {fleet.status === "ready" && <Ready data={fleet.data} />}
      </div>
      <Margin>
        <p><strong>What you are looking at.</strong> Every figure on this page is a model prediction made from sensor history that ended at each engine's last observed cycle.</p>
        <p><strong>Capped at 125.</strong> The model cannot predict beyond 125 cycles, so the right edge of the scale means "at least 125".</p>
        <p><strong>Thresholds are experimental.</strong> Warning below 30 cycles and critical below 15 are working levels for this project, not maintenance standards, and not failure probabilities.</p>
        <p><strong>Simulated data.</strong> These engines come from NASA's C-MAPSS simulator, not from real aircraft.</p>
      </Margin>
    </div>
  );
}

function Ready({ data }: { data: Awaited<ReturnType<typeof api.fleet>> }) {
  const { summary, engines } = data;
  const attention = engines.filter((e) => e.risk !== "normal").slice(0, 10);
  return (
    <>
      <section className="section">
        <h2 className="headline" style={{ marginBottom: "0.4rem" }}>
          <span className="crit">{summary.critical} critical</span>, <span className="warn">{summary.warning} warning</span>, and {summary.normal} normal across {summary.total} engines.
        </h2>
        <p style={{ color: "var(--ink-2)" }}>
          {summary.anomalous} engine{summary.anomalous === 1 ? "" : "s"} currently show{summary.anomalous === 1 ? "s" : ""} an anomaly alert. Each tick is one engine; select a tick to open it.
        </p>
        <div style={{ margin: "0.6rem 0 0.4rem" }}><Provenance kind="prediction" /></div>
        <LifeScale engines={engines} />
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Engines that need attention</h2>
          <Link to="/fleet">Open the full fleet</Link>
        </div>
        {attention.length === 0 ? (
          <Empty title="No engine is in warning or critical state.">Every held-out engine is predicted to have more than 30 cycles left.</Empty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th scope="col">Engine</th><th scope="col">Status</th><th scope="col" className="num">Predicted cycles left</th><th scope="col" className="num">80% interval</th><th scope="col">Anomaly</th><th scope="col" className="num">Observed cycles</th></tr>
              </thead>
              <tbody>
                {attention.map((e) => (
                  <tr key={e.engine_id}>
                    <td><Link to={`/engines/${e.engine_id}`}>Engine {e.engine_id}</Link></td>
                    <td><RiskBadge risk={e.risk} /></td>
                    <td className="num">{e.rul.toFixed(0)}</td>
                    <td className="num">{e.interval_80.lower.toFixed(0)} to {e.interval_80.upper.toFixed(0)}</td>
                    <td><span className={`flag ${e.anomaly_alert ? "on" : ""}`}>{e.anomaly_alert ? "Alert" : "None"}</span></td>
                    <td className="num">{e.observed_cycles}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
