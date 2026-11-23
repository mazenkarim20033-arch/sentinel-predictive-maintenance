/** Automated accessibility checks with axe-core on every page. Colour contrast needs real layout, so it is checked in a browser
 *  (see docs/TESTING.md); everything else axe can verify in jsdom is verified here. */
import axe from "axe-core";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { baseRoutes, engineDetail, mockFetch, replay } from "./mockApi";

const routes = {
  ...baseRoutes,
  "/engines/FD001/34": (u: URL) => engineDetail(34, 8, u.searchParams.get("include_truth") === "true"),
  "/replay/FD001/34": (u: URL) => replay(34, Number(u.searchParams.get("cycle")), u.searchParams.get("mode") as "blind" | "eval", 8),
};

async function violations(container: HTMLElement) {
  const result = await axe.run(container, { rules: { "color-contrast": { enabled: false } } });
  return result.violations.map((v) => `${v.id}: ${v.help} (${v.nodes.map((n) => n.target.join(" ")).slice(0, 3).join("; ")})`);
}

const PAGES: [string, string, RegExp | string][] = [
  ["dashboard", "/", /critical/],
  ["fleet", "/fleet", "Fleet explorer"],
  ["engine", "/engines/34", "Engine 34"],
  ["simulation lab", "/lab?engine=34", "Simulation lab"],
  ["model lab", "/models", "Leaderboard"],
  ["research", "/research", "What this does not show"],
];

beforeEach(() => window.localStorage.clear());
afterEach(() => vi.unstubAllGlobals());

describe("accessibility (axe-core, colour contrast excluded)", () => {
  it.each(PAGES)("%s has no detectable violations", async (_name, path, ready) => {
    mockFetch(routes);
    const { container } = render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
    if (typeof ready === "string") await screen.findByRole("heading", { name: ready });
    else await screen.findByText(ready);
    if (path.startsWith("/engines")) await screen.findByText(/the model starts from/);
    expect(await violations(container)).toEqual([]);
  });

  it("negative control: the checker does flag a known-bad element", async () => {
    const { container } = render(<div><img src="x.png" /><button /><input type="text" /></div>);
    const found = await violations(container);
    expect(found.length).toBeGreaterThanOrEqual(3);
    expect(found.join(" ")).toMatch(/image-alt/);
    expect(found.join(" ")).toMatch(/button-name/);
  });

  it("error and loading states are announced and have no violations", async () => {
    mockFetch({ ...baseRoutes, "/fleet/FD001": () => new Response(JSON.stringify({ detail: "model bundle for FD001 not found" }), { status: 503 }) });
    const { container } = render(<MemoryRouter><App /></MemoryRouter>);
    expect(await screen.findByRole("alert")).toHaveTextContent("model bundle for FD001 not found");
    expect(await violations(container)).toEqual([]);
  });

  it("every form control and button has an accessible name", async () => {
    mockFetch(routes);
    render(<MemoryRouter initialEntries={["/lab?engine=34"]}><App /></MemoryRouter>);
    await screen.findByText(/Cycle 1 of 8/);
    for (const el of [...document.querySelectorAll("button, input, select")]) {
      const name = el.getAttribute("aria-label") || el.textContent?.trim() || (el as HTMLInputElement).labels?.[0]?.textContent?.trim();
      expect(name, el.outerHTML.slice(0, 80)).toBeTruthy();
    }
  });
});
