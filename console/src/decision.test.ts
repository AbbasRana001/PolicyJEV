import { describe, expect, it } from "vitest";

import { LABELS, parseThresholds } from "./api";
import type { Decision, Label } from "./api";
import {
  decideComment,
  decideLabel,
  probabilityFromSlider,
  sqrtPosition,
  systemThresholds,
  zoneWidths,
} from "./decision";
import type { SystemThresholds } from "./decision";
import parity from "./fixtures/decision_parity.json";
import thresholdsResponse from "./fixtures/thresholds.json";

const LOW = 0.1;
const HIGH = 0.6;

function uniform(p: number): Record<Label, number> {
  return Object.fromEntries(LABELS.map((label) => [label, p])) as Record<Label, number>;
}

function allLabels(low: number, high: number | null): SystemThresholds {
  return Object.fromEntries(LABELS.map((label) => [label, { low, high }])) as SystemThresholds;
}

describe("decideLabel at the boundaries", () => {
  it("allows below LOW", () => {
    expect(decideLabel(0, LOW, HIGH)).toBe("ALLOW");
    expect(decideLabel(LOW - 1e-12, LOW, HIGH)).toBe("ALLOW");
  });

  it("sends p equal to LOW to human review", () => {
    expect(decideLabel(LOW, LOW, HIGH)).toBe("HUMAN REVIEW");
  });

  it("keeps review up to, but not including, HIGH", () => {
    expect(decideLabel(HIGH - 1e-12, LOW, HIGH)).toBe("HUMAN REVIEW");
  });

  it("acts at p equal to HIGH and above", () => {
    expect(decideLabel(HIGH, LOW, HIGH)).toBe("ACTION");
    expect(decideLabel(1, LOW, HIGH)).toBe("ACTION");
  });

  it("never acts when HIGH is null, even at p = 1", () => {
    expect(decideLabel(1, LOW, null)).toBe("HUMAN REVIEW");
    expect(decideLabel(0.99, LOW, null)).toBe("HUMAN REVIEW");
    expect(decideLabel(LOW - 1e-12, LOW, null)).toBe("ALLOW");
  });
});

describe("decideComment across the six labels", () => {
  const rules = allLabels(LOW, HIGH);

  it("allows a comment when no label reaches LOW and names no triggering label", () => {
    expect(decideComment(uniform(0), rules)).toEqual({ decision: "ALLOW", labels: [] });
  });

  it("takes the most severe outcome and lists only the labels that set it", () => {
    const probabilities = { ...uniform(0), toxic: LOW, insult: HIGH, obscene: HIGH, threat: 0.3 };
    expect(decideComment(probabilities, rules)).toEqual({ decision: "ACTION", labels: ["obscene", "insult"] });
  });

  it("lists every label that reaches review when none reaches action", () => {
    const result = decideComment(uniform(LOW), allLabels(LOW, null));
    expect(result.decision).toBe("HUMAN REVIEW");
    expect(result.labels).toEqual([...LABELS]);
  });

  it("gives each of the six labels the power to set the outcome alone", () => {
    for (const label of LABELS) {
      const review = decideComment({ ...uniform(0), [label]: LOW }, rules);
      expect(review).toEqual({ decision: "HUMAN REVIEW", labels: [label] });
      const action = decideComment({ ...uniform(0), [label]: HIGH }, rules);
      expect(action).toEqual({ decision: "ACTION", labels: [label] });
    }
  });

  it("uses each label's own thresholds", () => {
    const mixed = { ...allLabels(LOW, HIGH), threat: { low: 0.5, high: null } } as SystemThresholds;
    expect(decideComment({ ...uniform(0), threat: 0.4 }, mixed).decision).toBe("ALLOW");
    expect(decideComment({ ...uniform(0), threat: 1 }, mixed).decision).toBe("HUMAN REVIEW");
  });
});

describe("systemThresholds", () => {
  const rows = parseThresholds(thresholdsResponse) ?? [];

  it("parses the real /api/thresholds response", () => {
    expect(rows.length).toBe(18);
  });

  it("returns null HIGH exactly where automatic action is not enabled", () => {
    for (const row of rows) {
      const rules = systemThresholds(rows, row.system);
      expect(rules).not.toBeNull();
      expect(rules?.[row.label].high === null).toBe(!row.action_enabled);
      expect(rules?.[row.label].low).toBe(row.low);
    }
  });

  it("returns null when a label is missing", () => {
    expect(systemThresholds(rows.filter((row) => row.label !== "threat"), "jev")).toBeNull();
  });
});

describe("the rule agrees with the policy engine for every system and label", () => {
  // decision_parity.json is written from the real /api/thresholds response and policy/engine.py by
  // tests/test_decision_parity.py, which also fails if the file is out of date.
  const rows = parseThresholds(parity.thresholds) ?? [];
  const rules = new Map<string, SystemThresholds | null>();
  for (const system of new Set(rows.map((row) => row.system))) rules.set(system, systemThresholds(rows, system));

  it("covers every system and label of the real response", () => {
    expect(rows.length).toBe(18);
    expect(rules.size).toBe(3);
  });

  it("matches the engine just below and at each threshold, per label", () => {
    expect(parity.label_cases.length).toBeGreaterThan(18 * 3);
    for (const item of parity.label_cases) {
      const threshold = rules.get(item.system)?.[item.label as Label];
      expect(threshold, `${item.system} ${item.label}`).toBeDefined();
      if (threshold === undefined) continue;
      expect(decideLabel(item.p, threshold.low, threshold.high), `${item.system} ${item.label} p=${item.p}`).toBe(
        item.decision as Decision,
      );
    }
  });

  it("has a probe just below and exactly at LOW and HIGH for every system and label", () => {
    for (const row of rows) {
      const cases = parity.label_cases.filter((item) => item.system === row.system && item.label === row.label);
      expect(cases.some((item) => item.p === row.low && item.decision === "HUMAN REVIEW")).toBe(true);
      expect(cases.some((item) => item.p < (row.low ?? 0) && item.decision === "ALLOW")).toBe(true);
      if (row.action_enabled) {
        expect(cases.some((item) => item.p === row.high && item.decision === "ACTION")).toBe(true);
        expect(cases.some((item) => item.p < (row.high ?? 0) && item.decision === "HUMAN REVIEW")).toBe(true);
      } else {
        expect(cases.some((item) => item.decision === "ACTION")).toBe(false);
      }
    }
  });

  it("matches the engine for whole comments, including the labels that triggered", () => {
    expect(parity.comment_cases.length).toBeGreaterThan(18);
    for (const item of parity.comment_cases) {
      const system = rules.get(item.system);
      expect(system).toBeTruthy();
      if (!system) continue;
      const result = decideComment(item.probabilities as Record<Label, number>, system);
      expect(result.decision).toBe(item.decision);
      expect(result.labels).toEqual(item.triggering_labels);
    }
  });
});

describe("square-root scale", () => {
  it("maps 0 and 1 to the ends and 0.25 to the middle", () => {
    expect(sqrtPosition(0)).toBe(0);
    expect(sqrtPosition(1)).toBe(1);
    expect(sqrtPosition(0.25)).toBeCloseTo(0.5, 12);
  });

  it("clamps and ignores non-finite values", () => {
    expect(sqrtPosition(-1)).toBe(0);
    expect(sqrtPosition(4)).toBe(1);
    expect(sqrtPosition(Number.NaN)).toBe(0);
  });

  it("makes the slider linear in the square root", () => {
    expect(probabilityFromSlider(0)).toBe(0);
    expect(probabilityFromSlider(1000)).toBe(1);
    expect(probabilityFromSlider(500)).toBeCloseTo(0.25, 12);
    expect(sqrtPosition(probabilityFromSlider(333))).toBeCloseTo(0.333, 12);
  });

  it("splits the track into zones that add up to the whole", () => {
    const withAction = zoneWidths({ low: 0.04, high: 0.64 });
    expect(withAction.allow).toBeCloseTo(20, 10);
    expect(withAction.review).toBeCloseTo(60, 10);
    expect(withAction.action).toBeCloseTo(20, 10);
    const reviewOnly = zoneWidths({ low: 0.04, high: null });
    expect(reviewOnly.allow).toBeCloseTo(20, 10);
    expect(reviewOnly.review).toBeCloseTo(80, 10);
    expect(reviewOnly.action).toBe(0);
  });
});
