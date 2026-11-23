import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import { Margin, Provenance } from "../components/Badges";
import { TOOLTIP } from "../components/Charts";
import { Empty, ErrorBox, Loading } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import { useSubset } from "../state/Subset";

const f = (x: number | null | undefined, d = 1) => (x === null || x === undefined ? "n/a" : x.toFixed(d));

export function ModelLab() {
  const { subset } = useSubset();
  const ev = useAsync((s) => api.evaluation(subset, s), [subset]);
  const models = useAsync((s) => api.models(s), []);
  const meta = models.status === "ready" ? models.data.find((m) => m.subset === subset)?.meta : undefined;

  return (
    <div className="with-margin">
      <div>
        <header className="page-head">
          <h1>Model lab</h1>
          <p className="lede">How each model performed on {subset}'s official test engines, with the cost of running it.</p>
        </header>
        {ev.status === "loading" && <Loading label="Loading results" />}
        {ev.status === "error" && <ErrorBox title="Results could not be loaded" error={ev.error} onRetry={ev.reload} />}
        {ev.status === "ready" && (ev.data.leaderboard.length === 0 ? <Empty title="No recorded results for this dataset." /> : <Board data={ev.data} />)}
        {meta && (
          <section className="section">
            <h2>Model served by this app</h2>
            <p style={{ color: "var(--ink-2)" }}>
              LightGBM trained on {String(meta.n_training_assets)} engines, with {String(meta.n_calibration_assets)} further engines held out to calibrate the intervals and the anomaly level.
              Built {String(meta.trained_utc ?? "").slice(0, 10)}.
            </p>
          </section>
        )}
      </div>
      <Margin>
        <p><strong>Read the error bars.</strong> Each dataset has only 100 to 259 test engines, so models whose bars overlap cannot be told apart.</p>
        <p><strong>Lower is better.</strong> RMSE is in cycles; the NASA score punishes late predictions more than early ones.</p>
        <p><strong>Cost matters.</strong> A model that matches the best within error but trains in seconds is often the better choice.</p>
        <p><strong>Not a ranking of methods.</strong> Hyperparameters were not tuned, and the sequence models were tried on one dataset only.</p>
      </Margin>
    </div>
  );
}

function Board({ data }: { data: Awaited<ReturnType<typeof api.evaluation>> }) {
  const rows = data.leaderboard;
  const chart = rows.map((r) => ({ model: r.model, rmse: r.test_rmse ?? 0, err: r.test_rmse_ci95 && r.test_rmse !== null ? [r.test_rmse - r.test_rmse_ci95[0], r.test_rmse_ci95[1] - r.test_rmse] : [0, 0] }));
  return (
    <>
      <section className="section">
        <div className="section-head"><h2>Test error by model</h2><Provenance kind="measured_evaluation" /></div>
        <figure style={{ margin: 0 }} aria-label="Test RMSE per model with 95% confidence intervals">
          <div className="chart" style={{ height: Math.max(220, rows.length * 34) }}>
            <ResponsiveContainer>
              <BarChart data={chart} layout="vertical" margin={{ top: 4, right: 24, bottom: 0, left: 8 }}>
                <CartesianGrid stroke="var(--rule)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: "var(--ink-2)" }} tickLine={false} />
                <YAxis type="category" dataKey="model" width={130} tick={{ fontSize: 12, fill: "var(--ink)" }} tickLine={false} />
                <Tooltip contentStyle={TOOLTIP} formatter={(v: unknown) => Number(v).toFixed(2)} />
                <Bar dataKey="rmse" fill="var(--measured)" isAnimationActive={false} name="Test RMSE">
                  <ErrorBar dataKey="err" width={5} stroke="var(--ink)" direction="x" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </figure>
      </section>
      <section className="section">
        <h2>Leaderboard</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th scope="col">Model</th><th scope="col" className="num">Test RMSE</th><th scope="col" className="num">95% interval</th><th scope="col" className="num">Dev CV RMSE</th><th scope="col" className="num">NASA score</th><th scope="col" className="num">Fit time (s)</th><th scope="col" className="num">Latency (ms)</th><th scope="col" className="num">Size (KB)</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model}>
                  <th scope="row" style={{ fontWeight: 500 }}>{r.model}</th>
                  <td className="num">{f(r.test_rmse, 2)}</td>
                  <td className="num">{r.test_rmse_ci95 ? `${f(r.test_rmse_ci95[0])} to ${f(r.test_rmse_ci95[1])}` : "n/a"}</td>
                  <td className="num">{f(r.cv_cut_rmse, 2)}</td>
                  <td className="num">{r.test_nasa_score === null ? "n/a" : Math.round(r.test_nasa_score).toLocaleString()}</td>
                  <td className="num">{f(r.fit_seconds)}</td>
                  <td className="num">{f(r.latency_ms, 2)}</td>
                  <td className="num">{r.size_kb === null ? "n/a" : Math.round(r.size_kb).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {data.notes.map((n) => <p key={n} className="flag" style={{ marginTop: "0.5rem" }}>{n}</p>)}
      </section>
    </>
  );
}
