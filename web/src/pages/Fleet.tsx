import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Risk } from "../api/types";
import { Margin, Provenance, RiskBadge } from "../components/Badges";
import { CompareChart } from "../components/Charts";
import { Empty, ErrorBox, Loading } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import { useSubset } from "../state/Subset";

const PAGE = 25;
const FILTERS: { value: Risk | "all"; label: string }[] = [
  { value: "all", label: "All" }, { value: "critical", label: "Critical" }, { value: "warning", label: "Warning" }, { value: "normal", label: "Normal" },
];

export function Fleet() {
  const { subset } = useSubset();
  const [risk, setRisk] = useState<Risk | "all">("all");
  const [anomalous, setAnomalous] = useState(false);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<"rul" | "engine_id">("rul");
  const [page, setPage] = useState(0);
  const [picked, setPicked] = useState<number[]>([]);

  const fleet = useAsync(
    (s) => api.fleet(subset, { risk: risk === "all" ? undefined : risk, anomalous: anomalous || undefined, search: search || undefined, sort, limit: PAGE, offset: page * PAGE }, s),
    [subset, risk, anomalous, search, sort, page],
  );
  const reset = (fn: () => void) => { fn(); setPage(0); };
  const toggle = (id: number) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : p.length < 4 ? [...p, id] : p));

  return (
    <div className="with-margin">
      <div>
        <header className="page-head">
          <h1>Fleet explorer</h1>
          <p className="lede">Filter and sort held-out engines, then select up to four to compare how their predicted remaining life evolved.</p>
        </header>
        <div className="toolbar">
          <div className="field">
            Status
            <div className="segmented" role="group" aria-label="Filter by status">
              {FILTERS.map((f) => (
                <button key={f.value} aria-pressed={risk === f.value} onClick={() => reset(() => setRisk(f.value))}>{f.label}</button>
              ))}
            </div>
          </div>
          <label className="field">Engine number
            <input type="search" value={search} maxLength={10} placeholder="e.g. 34" onChange={(e) => reset(() => setSearch(e.target.value.trim()))} />
          </label>
          <label className="field">Sort by
            <select value={sort} onChange={(e) => reset(() => setSort(e.target.value as "rul" | "engine_id"))}>
              <option value="rul">Fewest cycles left</option>
              <option value="engine_id">Engine number</option>
            </select>
          </label>
          <label className="checks" style={{ alignSelf: "center" }}>
            <span style={{ display: "inline-flex", gap: "0.4rem", alignItems: "center" }}>
              <input type="checkbox" checked={anomalous} onChange={(e) => reset(() => setAnomalous(e.target.checked))} /> Anomaly alerts only
            </span>
          </label>
        </div>

        {fleet.status === "loading" && <Loading label="Loading engines" />}
        {fleet.status === "error" && <ErrorBox title="The fleet could not be loaded" error={fleet.error} onRetry={fleet.reload} />}
        {fleet.status === "ready" && (
          <>
            <div style={{ marginBottom: "0.5rem" }}><Provenance kind="prediction" /> <span className="flag">{fleet.data.total_matching} matching of {fleet.data.summary.total} engines</span></div>
            {fleet.data.engines.length === 0 ? (
              <Empty title="No engine matches these filters.">Clear the engine number or choose "All" to see the whole fleet.</Empty>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th scope="col"><span className="sr-only">Compare</span></th><th scope="col">Engine</th><th scope="col">Status</th><th scope="col" className="num">Cycles left</th><th scope="col" className="num">80% interval</th><th scope="col">Anomaly</th><th scope="col" className="num">Observed</th></tr>
                  </thead>
                  <tbody>
                    {fleet.data.engines.map((e) => (
                      <tr key={e.engine_id}>
                        <td><input type="checkbox" aria-label={`Compare engine ${e.engine_id}`} checked={picked.includes(e.engine_id)} onChange={() => toggle(e.engine_id)} disabled={!picked.includes(e.engine_id) && picked.length >= 4} /></td>
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
            <div className="pager">
              <button onClick={() => setPage((p) => p - 1)} disabled={page === 0}>Previous</button>
              <span>Page {page + 1} of {Math.max(1, Math.ceil(fleet.data.total_matching / PAGE))}</span>
              <button onClick={() => setPage((p) => p + 1)} disabled={(page + 1) * PAGE >= fleet.data.total_matching}>Next</button>
            </div>
          </>
        )}

        {picked.length > 0 && (
          <section className="section">
            <div className="section-head">
              <h2>Comparing {picked.length} engine{picked.length === 1 ? "" : "s"}</h2>
              <button onClick={() => setPicked([])}>Clear selection</button>
            </div>
            <Compare ids={picked} subset={subset} />
          </section>
        )}
      </div>
      <Margin>
        <p><strong>Sorted by risk first.</strong> The default order puts the engines with the fewest predicted cycles left at the top.</p>
        <p><strong>Intervals.</strong> The 80% interval is calibrated to cover the true remaining life about 80% of the time on average, but it is less reliable in mid-life than near the start or end.</p>
        <p><strong>Comparison.</strong> Each line is the prediction made at that cycle using only the history up to it.</p>
      </Margin>
    </div>
  );
}

function Compare({ ids, subset }: { ids: number[]; subset: string }) {
  const key = useMemo(() => ids.join(","), [ids]);
  const data = useAsync(async (signal) => Promise.all(ids.map((id) => api.engine(subset, id, false, signal))), [subset, key]);
  if (data.status === "loading") return <Loading label="Loading comparison" />;
  if (data.status === "error") return <ErrorBox title="The comparison could not be loaded" error={data.error} onRetry={data.reload} />;
  return (
    <>
      <div style={{ marginBottom: "0.4rem" }}><Provenance kind="prediction" /></div>
      <CompareChart series={data.data.map((d) => ({ id: d.engine_id, time: d.predictions.time, rul: d.predictions.rul }))} />
    </>
  );
}
