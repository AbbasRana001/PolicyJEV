import { describe, expect, it } from "vitest";

import { parseResults } from "./api";
import type { PolicyLabelRow, ResultsPayload } from "./api";
import resultsFixture from "./fixtures/results.json";
import {
  agreementAxisMin,
  apAxisMax,
  budgetLevels,
  contrastPosition,
  contrastRange,
  measureAxisMax,
  tenths,
} from "./scales";

const payload = parseResults(resultsFixture) as ResultsPayload;

describe("measureAxisMax", () => {
  it("is 1 where higher is better", () => {
    expect(measureAxisMax([0.2, 0.8], "higher")).toBe(1);
  });

  it("sits just above the largest value where lower is better", () => {
    for (const largest of [0.05, 0.115, 0.173, 0.4]) {
      const max = measureAxisMax([largest, largest / 2, null], "lower");
      expect(max).toBeGreaterThan(largest);
      expect(max).toBeLessThanOrEqual(largest * 1.25);
    }
  });

  it("never exceeds 1 and copes with no data", () => {
    expect(measureAxisMax([0.99], "lower")).toBe(1);
    expect(measureAxisMax([null, null], "lower")).toBe(1);
  });
});

describe("the comparison axis", () => {
  const rows = [
    { difference: -0.1, low: -0.14, high: -0.05 },
    { difference: 0.02, low: null, high: 0.1 },
  ];

  it("covers every plotted value with a margin", () => {
    const range = contrastRange(rows);
    expect(range).toBeGreaterThan(0.14);
    expect(range).toBeLessThan(0.17);
  });

  it("puts zero in the middle and the ends at plus and minus the range", () => {
    expect(contrastPosition(0, 0.2)).toBe(50);
    expect(contrastPosition(-0.2, 0.2)).toBe(0);
    expect(contrastPosition(0.2, 0.2)).toBe(100);
    expect(contrastPosition(0.1, 0.2)).toBeCloseTo(75, 10);
  });

  it("falls back to a small range when nothing is plotted", () => {
    expect(contrastRange([])).toBeGreaterThan(0);
  });

  it("keeps every real interval on the plot", () => {
    const real = (payload.policy_contrasts ?? []).flatMap((row) => [row.difference, row.ci_low, row.ci_high]);
    const range = contrastRange(real.map((value) => ({ difference: value, low: null, high: null })));
    for (const value of real) {
      if (value === null) continue;
      expect(Math.abs(value)).toBeLessThan(range);
    }
  });
});

describe("the validation-versus-test axis", () => {
  function row(values: Partial<PolicyLabelRow>): PolicyLabelRow {
    return {
      system: "jev",
      label: "toxic",
      positives: 1,
      action_enabled: true,
      recall_target: 0.9,
      precision_target: 0.9,
      validation_recall_at_low: null,
      test_recall_at_low: null,
      validation_precision_at_high: null,
      test_action_precision: null,
      ...values,
    };
  }

  it("starts a tenth below the lowest value, with room for the dot", () => {
    expect(agreementAxisMin([row({ test_action_precision: 0.623 })])).toBe(0.5);
    expect(agreementAxisMin([row({ test_action_precision: 0.78 })])).toBe(0.7);
  });

  it("is shared: it comes from the lowest value in either plot", () => {
    const rows = [row({ test_action_precision: 0.9 }), row({ test_recall_at_low: 0.62 })];
    expect(agreementAxisMin(rows)).toBe(0.5);
  });

  it("never goes below zero", () => {
    expect(agreementAxisMin([row({ test_action_precision: 0.01 })])).toBe(0);
  });

  it("includes every value in the real data", () => {
    const rows = payload.policy_per_label ?? [];
    const min = agreementAxisMin(rows);
    for (const item of rows) {
      for (const value of [item.validation_recall_at_low, item.test_recall_at_low, item.validation_precision_at_high, item.test_action_precision]) {
        if (value !== null) expect(value).toBeGreaterThan(min);
      }
    }
  });

  it("makes ticks every tenth from the first tenth at or above the minimum", () => {
    expect(tenths(0.5, 1)).toEqual([0.5, 0.6, 0.7, 0.8, 0.9, 1]);
    expect(tenths(0.55, 1)).toEqual([0.6, 0.7, 0.8, 0.9, 1]);
  });
});

describe("the average-precision and budget axes", () => {
  it("rounds the highest interval end up to a tenth", () => {
    const rows = payload.per_label_ap ?? [];
    const max = apAxisMax(rows);
    expect(Math.max(...rows.map((row) => row.ap_hi))).toBeLessThanOrEqual(max);
    expect(max).toBeLessThanOrEqual(1);
    expect(Math.round(max * 10) / 10).toBe(max);
  });

  it("lists the budgets of a metric in ascending order without repeats", () => {
    const levels = budgetLevels(payload.review_budget ?? [], "recall_rare");
    expect(levels.length).toBeGreaterThan(1);
    expect([...levels].sort((a, b) => a - b)).toEqual(levels);
    expect(new Set(levels).size).toBe(levels.length);
  });
});
