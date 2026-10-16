import type { PolicyContrastRow, PolicySummaryRow, SystemId } from "./api";

export type Better = "lower" | "higher";

export type SummaryKey =
  | "m1_review_share"
  | "m3_miss_rate"
  | "m4_rare_miss_rate"
  | "m2_action_precision"
  | "action_coverage";

export interface MeasureSpec {
  key: SummaryKey;
  name: string;
  description: string;
  better: Better;
}

/** The five measures of the policy test summary, in the order the page shows them. */
export const FIVE_MEASURES: readonly MeasureSpec[] = [
  {
    key: "m1_review_share",
    name: "Review share",
    description: "Of all comments, the share sent to human review. Lower is better.",
    better: "lower",
  },
  {
    key: "m3_miss_rate",
    name: "Harmful comments allowed",
    description: "Of comments with at least one positive label, the share that were allowed. Lower is better.",
    better: "lower",
  },
  {
    key: "m4_rare_miss_rate",
    name: "Rare-label comments allowed",
    description: "Of comments positive for threat, severe toxic or identity hate, the share that were allowed. Lower is better.",
    better: "lower",
  },
  {
    key: "m2_action_precision",
    name: "Precision of automatic action",
    description:
      "Of comments that received automatic action, the share where at least one label that reached its HIGH threshold is truly positive. Higher is better.",
    better: "higher",
  },
  {
    key: "action_coverage",
    name: "Harmful comments actioned",
    description:
      "Of comments with at least one positive label, the share that received automatic action. Higher is better.",
    better: "higher",
  },
];

/** The four measures that have a paired comparison (policy_test_contrasts.csv has no coverage row). */
export const COMPARISON_MEASURES: readonly (MeasureSpec & { metric: string })[] = [
  "m1_review_share",
  "m3_miss_rate",
  "m4_rare_miss_rate",
  "m2_action_precision",
].map((metric) => {
  const spec = FIVE_MEASURES.find((item) => item.key === metric);
  if (spec === undefined) throw new Error(`unknown measure ${metric}`);
  return { ...spec, metric };
});

/** The two baselines JEV is compared with, as named in the contrast column of the CSV. */
export const BASELINES: readonly { system: SystemId; contrast: string; text: string }[] = [
  { system: "logreg_word", contrast: "jev - logreg_word", text: "logistic regression" },
  { system: "distilbert", contrast: "jev - distilbert", text: "DistilBERT" },
];

export interface NormalisedContrast {
  /** JEV minus the baseline, flipped where lower is better: negative always means JEV is worse. */
  difference: number | null;
  low: number | null;
  high: number | null;
}

/**
 * Put a paired contrast on one direction for the comparison plot.
 *
 * policy_test_contrasts.csv holds JEV minus the baseline. For the measures where lower is better
 * (review share and the two miss rates) a positive difference means JEV is worse, so the sign is
 * flipped, and because negating swaps the ends of an interval, the low and high ends swap too.
 * For precision, where higher is better, the row is used as it is. After this, a value left of
 * zero (red on the plot) always means JEV is worse and a value right of zero means JEV is better.
 */
export function normaliseContrast(
  row: Pick<PolicyContrastRow, "difference" | "ci_low" | "ci_high">,
  better: Better,
): NormalisedContrast {
  if (better === "higher") {
    return { difference: row.difference, low: row.ci_low, high: row.ci_high };
  }
  const negate = (value: number | null): number | null => (value === null ? null : -value);
  return { difference: negate(row.difference), low: negate(row.ci_high), high: negate(row.ci_low) };
}

/** The value of one measure for each system, in system order. */
export function measureValues(
  summary: readonly PolicySummaryRow[],
  key: SummaryKey,
  systems: readonly SystemId[],
): (number | null)[] {
  return systems.map((system) => summary.find((row) => row.system === system)?.[key] ?? null);
}

/** Index of the best value, ignoring missing values; null if there is none. */
export function bestIndex(values: readonly (number | null)[], better: Better): number | null {
  let best: number | null = null;
  values.forEach((value, index) => {
    if (value === null) return;
    const current = best === null ? null : values[best];
    if (current === null || current === undefined) {
      best = index;
    } else if (better === "lower" ? value < current : value > current) {
      best = index;
    }
  });
  return best;
}
