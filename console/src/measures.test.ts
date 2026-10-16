import { describe, expect, it } from "vitest";

import { SYSTEM_IDS, parseResults } from "./api";
import type { PolicyContrastRow, ResultsPayload } from "./api";
import { comparisonRows } from "./components/ComparisonPlot";
import resultsFixture from "./fixtures/results.json";
import {
  BASELINES,
  COMPARISON_MEASURES,
  FIVE_MEASURES,
  bestIndex,
  measureValues,
  normaliseContrast,
} from "./measures";

const payload = parseResults(resultsFixture) as ResultsPayload;

describe("normaliseContrast", () => {
  const row = { difference: 0.04, ci_low: 0.03, ci_high: 0.06 };

  it("flips the sign where lower is better and swaps the ends of the interval", () => {
    expect(normaliseContrast(row, "lower")).toEqual({ difference: -0.04, low: -0.06, high: -0.03 });
  });

  it("leaves a measure where higher is better as it is", () => {
    expect(normaliseContrast(row, "higher")).toEqual({ difference: 0.04, low: 0.03, high: 0.06 });
  });

  it("keeps the interval ordered after the flip (low never above high)", () => {
    const flipped = normaliseContrast({ difference: -0.02, ci_low: -0.05, ci_high: 0.01 }, "lower");
    expect(flipped).toEqual({ difference: 0.02, low: -0.01, high: 0.05 });
    expect((flipped.low ?? 0) <= (flipped.high ?? 0)).toBe(true);
  });

  it("passes missing values through", () => {
    expect(normaliseContrast({ difference: null, ci_low: null, ci_high: null }, "lower")).toEqual({
      difference: null,
      low: null,
      high: null,
    });
    expect(normaliseContrast({ difference: 0.1, ci_low: null, ci_high: 0.2 }, "lower")).toEqual({
      difference: -0.1,
      low: -0.2,
      high: null,
    });
  });

  it("is its own inverse for lower-is-better measures", () => {
    const once = normaliseContrast(row, "lower");
    const twice = normaliseContrast({ difference: once.difference, ci_low: once.low, ci_high: once.high }, "lower");
    expect(twice).toEqual({ difference: row.difference, low: row.ci_low, high: row.ci_high });
  });
});

describe("the comparison puts JEV-worse on the left for every measure and baseline", () => {
  const summary = payload.policy_summary ?? [];
  const contrasts = payload.policy_contrasts ?? [];

  it("matches the sign of the difference between the two systems' own values", () => {
    for (const measure of COMPARISON_MEASURES) {
      for (const baseline of BASELINES) {
        const values = measureValues(summary, measure.key, ["jev", baseline.system]);
        const jev = values[0];
        const other = values[1];
        expect(jev).not.toBeNull();
        expect(other).not.toBeNull();
        if (jev === null || jev === undefined || other === null || other === undefined) continue;
        const row = contrasts.find((item) => item.contrast === baseline.contrast && item.metric === measure.metric);
        expect(row).toBeDefined();
        if (row === undefined) continue;
        const normalised = normaliseContrast(row, measure.better);
        const jevIsWorse = measure.better === "lower" ? jev > other : jev < other;
        expect((normalised.difference ?? 0) < 0).toBe(jevIsWorse);
        // the flip is a sign change only: the size is the one in the CSV
        expect(Math.abs(normalised.difference ?? 0)).toBeCloseTo(Math.abs(row.difference ?? 0), 12);
      }
    }
  });

  it("builds a row for every measure with a lane per baseline, in the order of the page", () => {
    const rows = comparisonRows(contrasts);
    expect(rows.map((row) => row.key)).toEqual(COMPARISON_MEASURES.map((measure) => measure.metric));
    for (const row of rows) expect(row.lanes.map((lane) => lane.system)).toEqual(BASELINES.map((b) => b.system));
  });

  it("leaves out a lane whose contrast is not in the data", () => {
    const partial: PolicyContrastRow[] = contrasts.filter((row) => row.contrast !== "jev - distilbert");
    for (const row of comparisonRows(partial)) expect(row.lanes.map((lane) => lane.system)).toEqual(["logreg_word"]);
  });
});

describe("bestIndex and measureValues", () => {
  it("finds the lowest or highest value and skips missing ones", () => {
    expect(bestIndex([0.3, 0.1, 0.2], "lower")).toBe(1);
    expect(bestIndex([0.3, 0.1, 0.2], "higher")).toBe(0);
    expect(bestIndex([null, 0.5, null], "lower")).toBe(1);
    expect(bestIndex([null, null], "higher")).toBeNull();
  });

  it("reads the five measures for the three systems in system order", () => {
    for (const measure of FIVE_MEASURES) {
      const values = measureValues(payload.policy_summary ?? [], measure.key, SYSTEM_IDS);
      expect(values).toHaveLength(3);
      expect(values.every((value) => value !== null)).toBe(true);
    }
  });
});
