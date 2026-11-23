/** Frontend against the real API process (dataset -> model -> backend -> frontend). Skipped unless VITE_API_BASE points at a
 *  running server; `python scripts/run_e2e.py` starts one and runs this file. */
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { App } from "../App";

const BASE = import.meta.env.VITE_API_BASE as string | undefined;
const live = describe.skipIf(!BASE);
const LONG = { timeout: 90_000 };

function renderApp(path: string) {
  window.localStorage.clear();
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

live("frontend against the real API", () => {
  // jsdom's AbortSignal is not accepted by Node's fetch; abort behaviour is covered by the mocked tests.
  beforeAll(() => {
    const nodeFetch = globalThis.fetch;
    vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => nodeFetch(input, { ...init, signal: undefined }));
  });

  it("shows the real FD001 fleet on the dashboard and opens an engine with an explanation", async () => {
    renderApp("/");
    expect(await screen.findByText(/across 100 engines/, undefined, LONG)).toBeInTheDocument();
    expect(screen.getByRole("group", { name: /remaining life of 100 engines/i })).toBeInTheDocument();
    const table = screen.getByRole("table");
    const first = within(table).getAllByRole("link")[0];
    fireEvent.click(first);
    expect(await screen.findByRole("heading", { name: /^Engine \d+$/ }, LONG)).toBeInTheDocument();
    expect(await screen.findByText(/the model starts from a typical value/, undefined, LONG)).toBeInTheDocument();
    expect(screen.queryByText("Ground truth, evaluation only")).not.toBeInTheDocument();
  }, 180_000);

  it("replays a real engine: blind hides truth, evaluation shows the dataset's true value", async () => {
    renderApp("/lab?engine=34");
    expect(await screen.findByText(/Cycle 1 of 203/, undefined, LONG)).toBeInTheDocument();
    for (let i = 0; i < 3; i++) fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(await screen.findByText("Cycle 4 of 203", undefined, LONG)).toBeInTheDocument();
    expect(screen.queryByTestId("truth-value")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Evaluation" }));
    const truth = await screen.findByTestId("truth-value", undefined, LONG);
    const direct = await (await fetch(`${BASE}/replay/FD001/34?cycle=1&mode=eval`)).json();
    await waitFor(() => expect(Number(truth.textContent)).toBe(Math.round(direct.ground_truth.rul)), LONG);
    fireEvent.click(screen.getByRole("button", { name: "Blind" }));
    await waitFor(() => expect(screen.queryByTestId("truth-value")).not.toBeInTheDocument(), LONG);
  }, 180_000);

  it("lists the recorded benchmark results in the model lab", async () => {
    renderApp("/models");
    expect(await screen.findByRole("heading", { name: "Leaderboard" }, LONG)).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /lightgbm_window50/ })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /constant/ })).toBeInTheDocument();
  }, 180_000);
});
