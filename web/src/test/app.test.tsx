import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { ApiError, api } from "../api/client";
import { baseRoutes, calls, engineDetail, engine, fleet, mockFetch, ok, replay, SUBSET } from "./mockApi";

function renderApp(path = "/") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("dashboard", () => {
  it("shows the fleet summary, the scale and only engines that need attention", async () => {
    mockFetch(baseRoutes);
    renderApp("/");
    expect(await screen.findByText(/2 critical/)).toBeInTheDocument();
    expect(screen.getByText(/2 warning/)).toBeInTheDocument();
    expect(screen.getByText(/2 normal/)).toBeInTheDocument();
    expect(screen.getByRole("group", { name: /remaining life of 6 engines/i })).toBeInTheDocument();
    const table = screen.getByRole("table");
    expect(within(table).getByRole("link", { name: "Engine 34" })).toBeInTheDocument();
    expect(within(table).getByRole("link", { name: "Engine 12" })).toBeInTheDocument();
    expect(within(table).queryByRole("link", { name: "Engine 3" })).not.toBeInTheDocument();
    expect(screen.getByText("Model prediction")).toBeInTheDocument();
    expect(screen.getByText(/Simulated NASA C-MAPSS data/)).toBeInTheDocument();
  });

  it("shows a loading state while the fleet request is pending", async () => {
    mockFetch({ ...baseRoutes, "/fleet/FD001": () => new Promise(() => {}) });
    renderApp("/");
    expect(await screen.findByText("Loading fleet state…")).toBeInTheDocument();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows the server's message and retries after a failure", async () => {
    let attempts = 0;
    mockFetch({
      ...baseRoutes,
      "/fleet/FD001": () => (++attempts === 1 ? ok({ detail: "model bundle for FD001 not found; build it with scripts/build_bundle.py" }, 503) : fleet()),
    });
    renderApp("/");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The fleet could not be loaded");
    expect(alert).toHaveTextContent("scripts/build_bundle.py");
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(/2 critical/)).toBeInTheDocument();
    expect(attempts).toBe(2);
  });

  it("shows a clear message when the API cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("network down"); }));
    renderApp("/");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The API is not reachable");
    expect(alert).toHaveTextContent("uvicorn sentinel.api.app:app");
  });

  it("explains what to do when no dataset is ready", async () => {
    mockFetch({ "/subsets": [{ ...SUBSET, model_available: false }] });
    renderApp("/");
    expect(await screen.findByRole("alert")).toHaveTextContent("No dataset is ready to serve");
  });

  it("says so when nothing needs attention", async () => {
    mockFetch({ ...baseRoutes, "/fleet/FD001": () => ({ ...fleet(), engines: [engine(3, 90, "normal")] }) });
    renderApp("/");
    expect(await screen.findByText("No engine is in warning or critical state.")).toBeInTheDocument();
  });
});

describe("fleet explorer and engine selection", () => {
  it("filters by status through the API, handles empty results, and opens an engine", async () => {
    mockFetch({
      ...baseRoutes,
      "/fleet/FD001": (url: URL) => (url.searchParams.get("risk") === "warning" ? { ...fleet([]), total_matching: 0 } : fleet()),
      "/engines/FD001/34": () => engineDetail(34, 6),
    });
    renderApp("/fleet");
    expect(await screen.findByRole("link", { name: "Engine 34" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Warning" }));
    expect(await screen.findByText("No engine matches these filters.")).toBeInTheDocument();
    expect(calls.some((c) => c.includes("risk=warning"))).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "All" }));
    fireEvent.click(await screen.findByRole("link", { name: "Engine 34" }));
    expect(await screen.findByRole("heading", { name: "Engine 34" })).toBeInTheDocument();
    expect(calls.some((c) => c.startsWith("/api/engines/FD001/34") || c.startsWith("/engines/FD001/34"))).toBe(true);
    expect(await screen.findByText(/At cycle 120 the model starts from/)).toBeInTheDocument();
  });

  it("limits comparison to four engines", async () => {
    mockFetch({ ...baseRoutes, "/engines/FD001/34": () => engineDetail(34, 5), "/engines/FD001/76": () => engineDetail(76, 5) });
    renderApp("/fleet");
    await screen.findByRole("link", { name: "Engine 34" });
    for (const id of [34, 76, 12, 8]) fireEvent.click(screen.getByRole("checkbox", { name: `Compare engine ${id}` }));
    expect(screen.getByRole("checkbox", { name: "Compare engine 3" })).toBeDisabled();
    expect(screen.getByText("Comparing 4 engines")).toBeInTheDocument();
  });

  it("reports an unknown engine without crashing", async () => {
    mockFetch({ ...baseRoutes, "/engines/FD001/999": () => ok({ detail: "engine 999 not found" }, 404) });
    renderApp("/engines/999");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Engine 999 is not in FD001");
    expect(screen.getByRole("link", { name: "Back to the fleet" })).toBeInTheDocument();
  });

  it("keeps ground truth off the engine page until asked", async () => {
    mockFetch({ ...baseRoutes, "/engines/FD001/34": (u: URL) => engineDetail(34, 6, u.searchParams.get("include_truth") === "true") });
    renderApp("/engines/34");
    await screen.findByRole("heading", { name: "Engine 34" });
    expect(screen.queryByText("Ground truth, evaluation only")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /Show ground truth/ }));
    await waitFor(() => expect(screen.getByText("Ground truth, evaluation only")).toBeInTheDocument());
  });
});

describe("simulation lab", () => {
  const N = 5;
  function labRoutes() {
    return {
      ...baseRoutes,
      "/engines/FD001/7": (u: URL) => engineDetail(7, N, u.searchParams.get("include_truth") === "true"),
      "/replay/FD001/7": (u: URL) => replay(7, Number(u.searchParams.get("cycle")), u.searchParams.get("mode") as "blind" | "eval", N),
    };
  }
  const cycleCalls = () => calls.filter((c) => c.includes("/replay/")).map((c) => Number(new URL(c, "http://x").searchParams.get("cycle")));

  it("steps forward and back one cycle at a time and never asks beyond the observed history", async () => {
    mockFetch(labRoutes());
    renderApp("/lab?engine=7");
    expect(await screen.findByText("Cycle 1 of 5")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Step back" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(await screen.findByText("Cycle 2 of 5")).toBeInTheDocument();
    await waitFor(() => expect(calls.some((c) => c.includes("cycle=2"))).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "Step back" }));
    expect(await screen.findByText("Cycle 1 of 5")).toBeInTheDocument();
    for (let i = 0; i < 8; i++) fireEvent.click(screen.getByRole("button", { name: "Step forward" }));
    expect(await screen.findByText("Cycle 5 of 5")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Step forward" })).toBeDisabled();
    expect(Math.max(...cycleCalls())).toBeLessThanOrEqual(N);
  });

  it("plays automatically, stops at the end, and can restart", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    mockFetch(labRoutes());
    renderApp("/lab?engine=7");
    await screen.findByText("Cycle 1 of 5");
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByRole("button", { name: "Pause" })).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(600); });
    expect(await screen.findByText("Cycle 2 of 5")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(screen.getByText("Cycle 2 of 5")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    for (let i = 0; i < 6; i++) await act(async () => { await vi.advanceTimersByTimeAsync(600); });
    expect(await screen.findByText("Cycle 5 of 5")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Replay again" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Restart" }));
    expect(await screen.findByText("Cycle 1 of 5")).toBeInTheDocument();
  });

  it("hides ground truth in blind mode and shows it, labelled, in evaluation mode", async () => {
    mockFetch(labRoutes());
    renderApp("/lab?engine=7");
    await screen.findByText("Cycle 1 of 5");
    expect(screen.queryByTestId("truth-value")).not.toBeInTheDocument();
    expect(screen.queryByText("Ground truth, evaluation only")).not.toBeInTheDocument();
    expect(calls.filter((c) => c.includes("/engines/")).every((c) => !c.includes("include_truth"))).toBe(true);
    expect(calls.filter((c) => c.includes("/replay/")).every((c) => c.includes("mode=blind"))).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Evaluation" }));
    expect(await screen.findByTestId("truth-value")).toHaveTextContent("199");
    expect(screen.getAllByText("Ground truth, evaluation only").length).toBeGreaterThan(0);
    expect(calls.some((c) => c.includes("include_truth=true"))).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Blind" }));
    await waitFor(() => expect(screen.queryByTestId("truth-value")).not.toBeInTheDocument());
  });

  it("shows an error for an unknown engine and a failed replay step", async () => {
    mockFetch({ ...labRoutes(), "/engines/FD001/404": () => ok({ detail: "engine 404 not found" }, 404) });
    renderApp("/lab?engine=404");
    expect(await screen.findByRole("alert")).toHaveTextContent("Engine 404 is not in FD001");
    vi.unstubAllGlobals();
    mockFetch({ ...labRoutes(), "/replay/FD001/7": () => ok({ detail: "cycle 1 is beyond the observed history" }, 422) });
    renderApp("/lab?engine=7");
    await waitFor(() => expect(screen.getAllByRole("alert").some((a) => /replay step failed/.test(a.textContent ?? ""))).toBe(true));
  });
});

describe("model lab and research", () => {
  it("lists recorded results with their uncertainty and cost", async () => {
    mockFetch(baseRoutes);
    renderApp("/models");
    expect(await screen.findByRole("heading", { name: "Leaderboard" })).toBeInTheDocument();
    const row = screen.getByRole("row", { name: /lightgbm_window50/ });
    expect(within(row).getByText("13.75")).toBeInTheDocument();
    expect(within(row).getByText("11.6 to 15.6")).toBeInTheDocument();
    expect(screen.getByText("Measured benchmark")).toBeInTheDocument();
    expect(screen.getByText(/trained on 80 engines/i)).toBeInTheDocument();
  });

  it("shows an error when results are missing", async () => {
    mockFetch({ ...baseRoutes, "/evaluation/FD001": () => ok({ detail: "no evaluation records for FD001" }, 404) });
    renderApp("/models");
    expect(await screen.findByRole("alert")).toHaveTextContent("no evaluation records");
  });

  it("states the limitations and the labelling scheme", async () => {
    mockFetch(baseRoutes);
    renderApp("/research");
    expect(await screen.findByRole("heading", { name: "What this does not show" })).toBeInTheDocument();
    expect(screen.getByText(/all data are simulated/i)).toBeInTheDocument();
    for (const label of ["Model prediction", "Ground truth, evaluation only", "Dataset signal", "Measured benchmark", "Illustrative"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
  });
});

describe("api client", () => {
  it("turns server errors into messages", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ok({ detail: [{ msg: "Input should be 'blind' or 'eval'" }, { msg: "cycle must be >= 1" }] }, 422)));
    await expect(api.replay("FD001", 1, 0, "blind")).rejects.toMatchObject({ status: 422, message: "Input should be 'blind' or 'eval'; cycle must be >= 1" });
    vi.stubGlobal("fetch", vi.fn(async () => new Response("gateway", { status: 502 })));
    await expect(api.health()).rejects.toThrowError(new ApiError(502, "Request failed (502)"));
  });

  it("omits empty query parameters", async () => {
    const f = vi.fn(async (_input: RequestInfo | URL) => ok({}));
    vi.stubGlobal("fetch", f);
    await api.fleet("FD001", { risk: undefined, search: "", limit: 25, offset: 0 });
    expect(String(f.mock.calls[0][0])).toBe("/api/fleet/FD001?limit=25&offset=0");
  });
});
