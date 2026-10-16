import { describe, expect, it } from "vitest";

import { parseResults } from "./api";
import type { ResultsPayload } from "./api";
import { SYSTEM_NAMES } from "./api";
import { LABEL_ASCENT, LABEL_CHAR_WIDTH, LABEL_HEIGHT, budgetGeometry, layoutBudgetLabels, leaderFor, obstaclesFor, placeLabel, rectTouches, segmentHitsRect } from "./chartLabels";
import type { Rect } from "./chartLabels";
import { BUDGET_METRIC, budgetSeries } from "./components/ResultsView";
import resultsFixture from "./fixtures/results.json";

const payload = parseResults(resultsFixture) as ResultsPayload;
const { levels, series } = budgetSeries(payload.review_budget ?? [], BUDGET_METRIC);

// card widths the chart gets at the page widths the console is checked at (360 to 1440 px)
const WIDTHS = [300, 328, 343, 519, 520, 668, 712, 968, 1184];

describe("segmentHitsRect", () => {
  const box: Rect = { x: 10, y: 10, w: 20, h: 10 };

  it("sees a line that crosses the box, or has an end inside it", () => {
    expect(segmentHitsRect([0, 15, 40, 15], box)).toBe(true);
    expect(segmentHitsRect([15, 15, 100, 100], box)).toBe(true);
    expect(segmentHitsRect([0, 0, 40, 40], box)).toBe(true);
  });

  it("does not see a line that passes by", () => {
    expect(segmentHitsRect([0, 0, 40, 0], box)).toBe(false);
    expect(segmentHitsRect([0, 40, 40, 30], box)).toBe(false);
    expect(segmentHitsRect([0, 0, 5, 40], box)).toBe(false);
    expect(segmentHitsRect([35, 0, 35, 40], box)).toBe(false);
  });

  it("handles a line that stops short of the box", () => {
    expect(segmentHitsRect([0, 15, 9, 15], box)).toBe(false);
  });
});

describe("the direct labels at the smallest budget", () => {
  const first = series.map((item) => ({ system: item.system, value: item.values[0] ?? 0 }));
  const ranked = [...first].sort((a, b) => b.value - a.value);
  const highest = ranked[0];

  for (const width of WIDTHS) {
    it(`sit clear of every line and dot at a chart width of ${width}px`, () => {
      const { geometry, labels } = layoutBudgetLabels(series, levels.length, width);
      const obstacles = obstaclesFor(series, geometry);
      expect(labels.map((label) => label.system)).toEqual(ranked.map((item) => item.system));
      expect(labels).toHaveLength(3);
      for (const label of labels) {
        expect(label.clear, label.text).toBe(true);
        expect(rectTouches(label.rect, obstacles), label.text).toBe(false);
        // and no label touches another label
        for (const other of labels) {
          if (other === label) continue;
          const apart = label.rect.x + label.rect.w <= other.rect.x || other.rect.x + other.rect.w <= label.rect.x || label.rect.y + label.rect.h <= other.rect.y || other.rect.y + other.rect.h <= label.rect.y;
          expect(apart, `${label.text} / ${other.text}`).toBe(true);
        }
        expect(label.rect.x).toBeGreaterThanOrEqual(0);
        expect(label.rect.x + label.rect.w).toBeLessThanOrEqual(geometry.width);
        expect(label.rect.y).toBeGreaterThanOrEqual(0);
        expect(label.rect.y + label.rect.h).toBeLessThanOrEqual(geometry.height);
      }
    });
  }

  it("keeps the highest label above its point and the lowest below its point", () => {
    const { geometry, labels } = layoutBudgetLabels(series, levels.length, 343);
    const top = labels[0];
    const bottom = labels[labels.length - 1];
    expect(top && top.baseline < geometry.y(top.value)).toBe(true);
    expect(bottom && bottom.rect.y > geometry.y(bottom.value)).toBe(true);
  });

  it("states the value it labels to three decimals, with the leading zero", () => {
    const { labels } = layoutBudgetLabels(series, levels.length, 700);
    for (const label of labels) expect(label.text.endsWith(label.value.toFixed(3))).toBe(true);
    for (const label of labels) expect(label.text).toMatch(/ 0\.\d{3}$/);
  });

  it("is what the old fixed offset was not: that offset ran into the line at phone width", () => {
    const width = 343;
    const geometry = budgetGeometry(width, levels.length);
    const obstacles = obstaclesFor(series, geometry);
    const top = highest;
    const text = `JEV ${(top?.value ?? 0).toFixed(3)}`;
    const w = text.length * LABEL_CHAR_WIDTH;
    const baseline = geometry.y(top?.value ?? 0) - 10; // where the label used to sit
    const old: Rect = { x: geometry.x(0) + 10, y: baseline - LABEL_ASCENT, w, h: LABEL_HEIGHT };
    expect(rectTouches(old, obstacles)).toBe(true);
  });

  it("puts a label somewhere inside the chart even when nothing is clear, and says so", () => {
    const geometry = budgetGeometry(300, 2);
    const wall = { segments: Array.from({ length: 80 }, (_, i): [number, number, number, number] => [0, i * 4, 400, i * 4]), dots: [], rects: [] };
    const placed = placeLabel({ x: 60, y: 120 }, "JEV 0.307", ["above"], wall, geometry);
    expect(placed.clear).toBe(false);
    expect(placed.rect.y).toBeGreaterThanOrEqual(0);
  });

  it("makes no labels when only one system has a value", () => {
    const { labels } = layoutBudgetLabels([{ system: "jev", values: [0.3, 0.4] }], 2, 700);
    expect(labels).toEqual([]);
  });
});

import { render } from "@testing-library/react";
import { createElement } from "react";
import { tickStyle } from "./components/Explorer";
import { FlowBars, labelFits } from "./components/SummaryCharts";

describe("labels that used to be cut off at narrow widths", () => {
  it("keeps the first and last slider tick inside the box", () => {
    expect(tickStyle(0)).toEqual({ left: "0%", transform: "none" });
    expect(tickStyle(1)).toEqual({ left: "100%", transform: "translateX(-100%)" });
    expect(tickStyle(0.5)).toEqual({ left: "50%" });
  });

  it("prints a flow percentage only where it fits (10.1% of a 520px bar did not)", () => {
    expect(labelFits("10.1%", 0.101 * 520)).toBe(false);
    expect(labelFits("76.3%", 0.763 * 520)).toBe(true);
    expect(labelFits("13.6%", 0.136 * 1000)).toBe(true);
  });

  it("falls back to the share cutoff before a width is known", () => {
    const { container } = render(
      createElement(FlowBars, { summary: [{ system: "jev", n_rows: 100, n_allow: 70, n_review: 22, n_action: 8, m1_review_share: null, m2_action_precision: null, m3_miss_rate: null, m4_rare_miss_rate: null, action_coverage: null }] }),
    );
    const text = [...container.querySelectorAll(".fl-bar > div")].map((d) => d.textContent);
    expect(text).toEqual(["70.0%", "22.0%", ""]);
  });
});

describe("direct labels are attached to their own points and use the legend's names", () => {
  for (const width of WIDTHS) {
    it(`at ${width}px: full names, adjacent or on a leader line, no label over another or over a leader`, () => {
      const { geometry, labels } = layoutBudgetLabels(series, levels.length, width);
      for (const label of labels) {
        expect(label.text).toBe(`${SYSTEM_NAMES[label.system]} ${label.value.toFixed(3)}`);
        const anchor = { x: geometry.x(0), y: geometry.y(label.value) };
        if (label.leader === null) {
          expect(leaderFor(anchor, label.rect)).toBeNull();
        } else {
          // the leader starts at this label's own point and ends on this label's box
          expect(label.leader[0]).toBe(anchor.x);
          expect(label.leader[1]).toBe(anchor.y);
          expect(segmentHitsRect(label.leader, { x: label.rect.x - 0.5, y: label.rect.y - 0.5, w: label.rect.w + 1, h: label.rect.h + 1 })).toBe(true);
        }
        for (const other of labels) {
          if (other === label || label.leader === null) continue;
          expect(segmentHitsRect(label.leader, other.rect), `${label.text} leader over ${other.text}`).toBe(false);
        }
      }
    });
  }
});
