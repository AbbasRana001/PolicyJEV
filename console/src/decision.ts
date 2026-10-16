import { LABELS } from "./api";
import { joinList, words } from "./format";
import type { Decision, Label, SystemId, ThresholdRow } from "./api";

/**
 * The moderation rule, in one place for the Score page and the Policy explorer.
 *
 * For one label: p < LOW is ALLOW, LOW <= p < HIGH is HUMAN REVIEW, p >= HIGH is ACTION. A null
 * HIGH means automatic action is not enabled for the label, so it never gives ACTION. A comment
 * gets the most severe outcome across its labels. The same rule is in
 * src/policyjev/policy/engine.py; tests/test_decision_parity.py and decision.test.ts keep the two
 * in step.
 */

/** Thresholds of one label; high is null when automatic action is not enabled. */
export interface LabelThresholds {
  low: number;
  high: number | null;
}

export type SystemThresholds = Record<Label, LabelThresholds>;

export const DECISION_RANK: Record<Decision, number> = { ALLOW: 0, "HUMAN REVIEW": 1, ACTION: 2 };

export const DECISION_WORDS: Record<Decision, string> = {
  ALLOW: "Allow",
  "HUMAN REVIEW": "Human review",
  ACTION: "Action",
};

/** Zone class from the design: za, zr and zx set the colour of a decision. */
export const ZONE_CLASS: Record<Decision, string> = { ALLOW: "za", "HUMAN REVIEW": "zr", ACTION: "zx" };

/** Decision for one label. Boundaries are inclusive on the upper zone (p equal to LOW is review). */
export function decideLabel(probability: number, low: number, high: number | null): Decision {
  if (high !== null && probability >= high) return "ACTION";
  if (probability >= low) return "HUMAN REVIEW";
  return "ALLOW";
}

export interface CommentDecision {
  decision: Decision;
  /** Labels whose own outcome equals the decision; empty for ALLOW. */
  labels: Label[];
}

/** Most severe outcome across the six labels, with the labels that set it. */
export function decideComment(
  probabilities: Record<Label, number>,
  thresholds: SystemThresholds,
): CommentDecision {
  let best: Decision = "ALLOW";
  let labels: Label[] = [];
  for (const label of LABELS) {
    const { low, high } = thresholds[label];
    const outcome = decideLabel(probabilities[label], low, high);
    if (DECISION_RANK[outcome] > DECISION_RANK[best]) {
      best = outcome;
      labels = [label];
    } else if (outcome === best && outcome !== "ALLOW") {
      labels.push(label);
    }
  }
  return { decision: best, labels };
}

/**
 * The sentence that says why a comment got its decision, shared by the Score tiles and the explorer:
 * "No label reached its review threshold." or "Review triggered by x and y." or "Action triggered by x.".
 */
export function decisionWhy(decision: CommentDecision): string {
  if (decision.decision === "ALLOW") return "No label reached its review threshold.";
  const verb = decision.decision === "ACTION" ? "Action" : "Review";
  return `${verb} triggered by ${joinList(decision.labels.map(words))}.`;
}

/** The thresholds of one system, or null if any label is missing or has no LOW. */
export function systemThresholds(rows: readonly ThresholdRow[], system: SystemId): SystemThresholds | null {
  const result: Partial<SystemThresholds> = {};
  for (const label of LABELS) {
    const row = rows.find((item) => item.system === system && item.label === label);
    if (row === undefined || row.low === null) return null;
    result[label] = { low: row.low, high: row.action_enabled ? row.high : null };
  }
  return result as SystemThresholds;
}

/** Position on a square-root scale, as a fraction of the track from 0 to 1. */
export function sqrtPosition(probability: number): number {
  if (!Number.isFinite(probability)) return 0;
  return Math.sqrt(Math.min(1, Math.max(0, probability)));
}

/** Zone widths for one label on the square-root scale, as percentages of the track. */
export function zoneWidths(thresholds: LabelThresholds): { allow: number; review: number; action: number } {
  const low = sqrtPosition(thresholds.low) * 100;
  const high = thresholds.high === null ? 100 : Math.max(low, sqrtPosition(thresholds.high) * 100);
  return { allow: low, review: high - low, action: thresholds.high === null ? 0 : 100 - high };
}

/** Slider position (0 to 1000) to probability; the slider is linear in the square root. */
export function probabilityFromSlider(position: number): number {
  return (position / 1000) ** 2;
}
