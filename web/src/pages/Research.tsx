import { api } from "../api/client";
import { Margin, Provenance } from "../components/Badges";
import { ErrorBox, Loading } from "../components/States";
import { useAsync } from "../hooks/useAsync";

export function Research() {
  const subsets = useAsync((s) => api.subsets(s), []);
  return (
    <div className="with-margin">
      <div>
        <header className="page-head">
          <h1>Research and evaluation</h1>
          <p className="lede">What SENTINEL is built on, how it was tested, and what it does not show. The full write-up is in the repository's docs folder.</p>
        </header>

        <section className="section">
          <h2>Data</h2>
          <p>The engines come from NASA's C-MAPSS turbofan degradation data set, which is simulated. Each engine runs to failure in the training data; test engines stop before failure and come with one known remaining life each. The four subsets differ in operating conditions and fault modes.</p>
          {subsets.status === "loading" && <Loading label="Loading datasets" />}
          {subsets.status === "error" && <ErrorBox title="Datasets could not be listed" error={subsets.error} onRetry={subsets.reload} />}
          {subsets.status === "ready" && (
            <div className="table-wrap">
              <table>
                <thead><tr><th scope="col">Dataset</th><th scope="col">Description</th><th scope="col" className="num">Operating regimes</th><th scope="col">Served here</th></tr></thead>
                <tbody>
                  {subsets.data.map((s) => (
                    <tr key={s.subset}><th scope="row" style={{ fontWeight: 500 }}>{s.subset}</th><td>{s.description.replace(/^NASA C-MAPSS turbofan degradation, /, "")}</td><td className="num">{s.regimes}</td><td>{s.model_available && s.data_available ? "Yes" : "Not built"}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="section">
          <h2>Method</h2>
          <p>Models are trained on engine histories split by engine, never by row, so no engine appears on both sides of a split. Normalisation is fitted per operating regime on training engines only. Predictions use a causal window of the last 50 cycles. The official test sets were evaluated once per final configuration.</p>
          <p>Risk levels come from thresholds on the predicted remaining life. Intervals come from conformalised quantile regression calibrated on engines held out from fitting. Anomaly alerts come from a statistical detector fitted on the early life of training engines. Explanations are exact TreeSHAP attributions.</p>
        </section>

        <section className="section">
          <h2>What this does not show</h2>
          <ul style={{ paddingLeft: "1.2rem", maxWidth: "68ch" }}>
            <li>That the methods work on real engines: all data are simulated.</li>
            <li>That a trained model carries over to other conditions or machines. Models trained on single-condition data failed on multi-condition data in the transfer tests.</li>
            <li>Calibrated failure probabilities. Risk levels are experimental thresholds.</li>
            <li>Interval coverage at every remaining-life value. Coverage holds on average but is weaker in mid-life.</li>
            <li>Accuracy of anomaly detection on real faults. It was measured only on faults injected into the data.</li>
            <li>That one model beats the others: differences between the top models are inside the test confidence intervals.</li>
            <li>That explanations reveal causes. They describe how the model uses its inputs.</li>
          </ul>
        </section>

        <section className="section">
          <div className="section-head"><h2>How values are labelled</h2></div>
          <p style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
            <Provenance kind="prediction" /><Provenance kind="ground_truth" /><Provenance kind="dataset" /><Provenance kind="measured_evaluation" /><Provenance kind="illustrative" />
          </p>
          <p>Every chart and figure carries one of these labels. Nothing in this app is presented as a measured result unless it was recorded from an experiment; illustrative values, if any ever appear, are marked as such.</p>
        </section>

        <section className="section">
          <h2>Reproducing the results</h2>
          <p>Configuration, seeds, splits, metrics, hardware and timings for every experiment are stored as run records in the repository, and the report is generated from them. The README lists the commands.</p>
        </section>
      </div>
      <Margin>
        <p><strong>Where to read more.</strong> The repository's docs folder holds the experiments write-up, the model card, the dataset card and the limitations list.</p>
        <p><strong>Not for maintenance decisions.</strong> This is a research demonstration, not a certified system.</p>
      </Margin>
    </div>
  );
}
