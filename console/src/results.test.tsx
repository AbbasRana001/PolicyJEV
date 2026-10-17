import { render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { LABELS, SYSTEM_IDS, SYSTEM_NAMES, parseResults } from "./api";
import type { ResultsPayload } from "./api";
import { BUDGET_METRIC, ResultsView, budgetSeries } from "./components/ResultsView";
import { formatCount, words } from "./format";
import { budgetLevels } from "./scales";
import { FIXTURES, mockMedia } from "./test-utils";

const results = parseResults(FIXTURES.results) as ResultsPayload;

beforeEach(() => mockMedia({ narrow: false, reduced: true }));

describe("average precision by label", () => {
  it("draws a group per label with a bar per system", () => {
    render(<ResultsView results={results} />);
    const groups = [...document.querySelectorAll<HTMLElement>("#apc .apg")];
    expect(groups.map((group) => group.dataset["label"])).toEqual([...LABELS]);
    for (const group of groups) expect(group.querySelectorAll(".apb")).toHaveLength(3);
  });

  it("sizes each bar by its average precision and tags it with the value", () => {
    render(<ResultsView results={results} />);
    const rows = results.per_label_ap ?? [];
    const highest = Math.max(...rows.map((row) => row.ap_hi));
    for (const row of rows) {
      const bar = document.querySelector<HTMLElement>(`#apc .apg[data-label="${row.label}"] .apb[data-system="${row.system}"]`) as HTMLElement;
      const height = Number.parseFloat(bar.style.height);
      expect(height).toBeGreaterThan((row.ap / 1) * 100 - 1e-9);
      expect(height).toBeLessThanOrEqual(100);
      expect(bar.querySelector(".vv")?.textContent).toBe(row.ap.toFixed(3));
      expect(bar.querySelector(".vv")?.textContent).toMatch(/^0\.\d{3}$/);
      expect(bar.title).toContain(SYSTEM_NAMES[row.system]);
    }
    expect(highest).toBeLessThanOrEqual(1);
  });

  it("puts the interval whiskers inside the axis and says how many positives each label has", () => {
    render(<ResultsView results={results} />);
    for (const row of results.per_label_ap ?? []) {
      const bar = document.querySelector<HTMLElement>(`#apc .apg[data-label="${row.label}"] .apb[data-system="${row.system}"]`) as HTMLElement;
      const whisker = bar.querySelector<HTMLElement>(".wh") as HTMLElement;
      expect(Number.parseFloat(whisker.style.bottom)).toBeCloseTo((row.ap_lo / row.ap) * 100, 6);
      expect(Number.parseFloat(whisker.style.height)).toBeCloseTo(((row.ap_hi - row.ap_lo) / row.ap) * 100, 6);
    }
    const first = results.per_label_ap?.[0];
    expect(screen.getByText(`${formatCount(first?.positives ?? 0)} positives`)).toBeTruthy();
    expect(screen.getAllByText(words("severe_toxic")).length).toBeGreaterThan(0);
  });

  it("describes the comparison from the intervals of the paired differences", () => {
    render(<ResultsView results={results} />);
    expect(screen.getByText(/JEV trails both baselines on/)).toBeTruthy();
  });
});

describe("recall of the rare labels against the review budget", () => {
  it("draws a line and a dot per budget for each system", () => {
    render(<ResultsView results={results} />);
    const levels = budgetLevels(results.review_budget ?? [], BUDGET_METRIC);
    for (const id of SYSTEM_IDS) {
      const group = document.querySelector(`#lc g[data-system="${id}"]`) as SVGElement;
      expect(group.querySelectorAll("path")).toHaveLength(1);
      expect(group.querySelectorAll("circle")).toHaveLength(levels.length);
    }
    const labels = [...document.querySelectorAll("#lc text")].map((node) => node.textContent);
    for (const level of levels) expect(labels).toContain(`${level}%`);
  });

  it("takes each series from the rows for that system at that budget", () => {
    const { levels, series } = budgetSeries(results.review_budget ?? [], BUDGET_METRIC);
    for (const item of series) {
      levels.forEach((level, index) => {
        const row = results.review_budget?.find((entry) => entry.metric === BUDGET_METRIC && entry.comparison === item.system && entry.budget_pct === level);
        expect(item.values[index]).toBe(row?.value ?? null);
      });
    }
  });

  it("makes the lines visible when motion is reduced (no hidden start state)", () => {
    render(<ResultsView results={results} />);
    const path = document.querySelector("#lc path") as SVGPathElement;
    expect(path.style.strokeDashoffset).toBe("0");
  });

  it("labels the highest and lowest system at the smallest budget", () => {
    render(<ResultsView results={results} />);
    const { series } = budgetSeries(results.review_budget ?? [], BUDGET_METRIC);
    const first = series.map((item) => ({ system: item.system, value: item.values[0] ?? 0 }));
    const top = [...first].sort((a, b) => b.value - a.value)[0];
    const bottom = [...first].sort((a, b) => a.value - b.value)[0];
    const labelled = [...document.querySelectorAll("#lc text")].map((node) => node.textContent ?? "");
    expect(labelled.some((text) => text.startsWith(SYSTEM_NAMES[top?.system ?? "jev"]) && text.endsWith((top?.value ?? 0).toFixed(3)))).toBe(true);
    expect(labelled.some((text) => text.startsWith(SYSTEM_NAMES[bottom?.system ?? "jev"]) && text.endsWith((bottom?.value ?? 0).toFixed(3)))).toBe(true);
  });

  it("states the budget range and the finding from the data", () => {
    render(<ResultsView results={results} />);
    const levels = budgetLevels(results.review_budget ?? [], BUDGET_METRIC);
    expect(screen.getByText(new RegExp(`top-scored ${levels[0]}% to ${levels[levels.length - 1]}% of traffic`))).toBeTruthy();
    expect(screen.getByText(/JEV leads both baselines at/)).toBeTruthy();
    expect(screen.getByText(/comparisons were not pre-specified/)).toBeTruthy();
  });
});

describe("when data is missing", () => {
  it("shows Data not available for each chart whose section is absent", () => {
    const { per_label_ap: _ap, review_budget: _budget, ...rest } = results;
    render(<ResultsView results={{ ...rest, missing_sections: ["per_label_ap", "review_budget"] }} />);
    const notes = screen.getAllByText(/Data not available/);
    expect(notes).toHaveLength(2);
    expect(document.getElementById("apc")).toBeNull();
    expect(document.getElementById("lc")).toBeNull();
  });

  it("shows how to start the API when the results could not be loaded", () => {
    render(<ResultsView results={null} />);
    const section = screen.getByText("Average precision by label").closest(".sec") as HTMLElement;
    expect(within(section).getByText(/python -m policyjev.api/)).toBeTruthy();
  });

  it("does not claim a lead for JEV in the heading without the data", () => {
    render(<ResultsView results={undefined} />);
    expect(screen.queryByText(/JEV leads both baselines on rare-label recall/)).toBeNull();
  });
});
