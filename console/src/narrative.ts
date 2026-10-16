import { LABELS, SYSTEM_IDS, SYSTEM_NAMES } from "./api";
import type { ApDifferenceRow, BudgetRow, Label, PolicyLabelRow, PolicySummaryRow, SystemId } from "./api";
import { capitalise, formatCount, joinList, words } from "./format";
import { BASELINES } from "./measures";
import type { NormalisedContrast } from "./measures";

/**
 * Sentences that state findings are built here from the numbers the API returns, so the page text
 * cannot drift from the data. Each function falls back to plainer wording when the data does not
 * show the pattern it describes.
 */

/*
 * Every comparison in this file is exact: a value is below a target, or below its validation value,
 * or a confidence interval is above, below or around zero, with no margin of our own. The one number
 * shown to the reader is rounded only for printing.
 */

/** System name for the middle of a sentence (logistic regression is lower case there). */
function inSentence(system: SystemId): string {
  return system === "logreg_word" ? "logistic regression" : SYSTEM_NAMES[system];
}

function extreme(
  rows: readonly PolicySummaryRow[],
  pick: (row: PolicySummaryRow) => number | null,
  which: "min" | "max",
): SystemId | null {
  let best: PolicySummaryRow | null = null;
  let bestValue = 0;
  for (const row of rows) {
    const value = pick(row);
    if (value === null) continue;
    if (best === null || (which === "min" ? value < bestValue : value > bestValue)) {
      best = row;
      bestValue = value;
    }
  }
  return best === null ? null : best.system;
}

export interface Statement {
  headline: string;
  detail: string;
}

/** Heading and lead paragraph of the test-set section, from the three systems' summary rows. */
export function summaryStatement(summary: readonly PolicySummaryRow[], labelRows: readonly PolicyLabelRow[] = []): Statement {
  const total = summary[0]?.n_rows;
  const lead = total === undefined ? "Same thresholds, scored once." : `All ${formatCount(total)} labelled test comments, same thresholds, scored once.`;
  const leastReview = extreme(summary, (row) => row.m1_review_share, "min");
  const fewestMissed = extreme(summary, (row) => row.m3_miss_rate, "min");
  const mostReview = extreme(summary, (row) => row.m1_review_share, "max");
  const mostMissed = extreme(summary, (row) => row.m3_miss_rate, "max");
  const mostPrecise = extreme(summary, (row) => row.m2_action_precision, "max");
  if (
    leastReview === null || fewestMissed === null || mostReview === null || mostMissed === null ||
    mostPrecise === null
  ) {
    return { headline: "Policy results on the test set", detail: lead };
  }
  // Point estimates only: the heading says which value was lowest, not that the difference was established.
  const headline =
    leastReview === fewestMissed
      ? `On the test set, ${SYSTEM_NAMES[leastReview]} had the lowest review share and the lowest share of harmful comments allowed`
      : `On the test set, ${SYSTEM_NAMES[leastReview]} had the lowest review share and ${SYSTEM_NAMES[fewestMissed]} the lowest share of harmful comments allowed`;
  const sets = automationLabels(labelRows);
  const differentSets = new Set(SYSTEM_IDS.map((id) => (sets[id] ?? []).join())).size > 1;
  const caveat = differentSets ? ", but each system automates a different set of labels" : "";
  let cost: string;
  if (mostReview === mostMissed && mostPrecise === mostReview) {
    cost = `${SYSTEM_NAMES[mostReview]} needed the most review and missed the most. Its automatic actions had the highest precision${caveat}.`;
  } else {
    const first =
      mostReview === mostMissed
        ? `${SYSTEM_NAMES[mostReview]} needed the most review and missed the most.`
        : `${SYSTEM_NAMES[mostReview]} needed the most review and ${SYSTEM_NAMES[mostMissed]} missed the most.`;
    cost = `${first} ${capitalise(SYSTEM_NAMES[mostPrecise])}'s automatic actions had the highest precision${caveat}.`;
  }
  const untested = "Only JEV was compared with each baseline; the difference between logistic regression and DistilBERT was not tested.";
  return { headline, detail: `${lead} ${cost} ${untested}` };
}

/** How many of the plotted intervals reach the middle line, as a sentence. */
export function comparisonNote(rows: readonly NormalisedContrast[]): string {
  const withInterval = rows.filter((row) => row.low !== null && row.high !== null);
  if (withInterval.length === 0) return "";
  const touching = withInterval.filter((row) => (row.low ?? 0) <= 0 && (row.high ?? 0) >= 0).length;
  return touching === 0
    ? "No interval touches the middle line."
    : `${touching} of ${withInterval.length} intervals touch the middle line.`;
}

export interface Reading {
  tone: "good" | "bad";
  text: string;
}

function range2(values: readonly number[]): string {
  const low = Math.min(...values).toFixed(2);
  const high = Math.max(...values).toFixed(2);
  return low === high ? low : `${low} to ${high}`;
}

type Outcome = "fell" | "held";

/** Fell means the test value is below the validation value; equal or above is held. */
function outcomeOf(validation: number, test: number): Outcome {
  return test < validation ? "fell" : "held";
}

export interface PrecisionGroups {
  byLabel: Map<Label, { validation: number; test: number }[]>;
  /** Labels where every system with automatic action had test precision below validation precision. */
  fell: Label[];
  /** Labels where none did (test equal to or above validation for every such system). */
  held: Label[];
  mixed: Label[];
  systemCount: number;
}

/** Precision at HIGH, where automatic action is on, grouped by label and compared with validation exactly. */
export function precisionGroups(rows: readonly PolicyLabelRow[]): PrecisionGroups {
  const byLabel = new Map<Label, { validation: number; test: number }[]>();
  for (const row of rows) {
    if (!row.action_enabled || row.validation_precision_at_high === null || row.test_action_precision === null) continue;
    const list = byLabel.get(row.label) ?? [];
    list.push({ validation: row.validation_precision_at_high, test: row.test_action_precision });
    byLabel.set(row.label, list);
  }
  const fell: Label[] = [];
  const held: Label[] = [];
  const mixed: Label[] = [];
  for (const [label, list] of byLabel) {
    const outcomes = new Set(list.map((item) => outcomeOf(item.validation, item.test)));
    if (outcomes.size > 1) mixed.push(label);
    else if (outcomes.has("fell")) fell.push(label);
    else held.push(label);
  }
  const inLabelOrder = (x: Label, y: Label): number => LABELS.indexOf(x) - LABELS.indexOf(y);
  fell.sort(inLabelOrder);
  held.sort(inLabelOrder);
  mixed.sort(inLabelOrder);
  return { byLabel, fell, held, mixed, systemCount: new Set(rows.map((row) => row.system)).size };
}

/** One sentence for the Score page: the validation targets were missed on the test set, in the words of the results. */
export function thresholdsOnTestSentence(rows: readonly PolicyLabelRow[] | undefined): string {
  const base = "Thresholds were fitted on validation data";
  if (rows === undefined) return `${base}; how they held on the test set is shown on the Policy page.`;
  const { fell, byLabel, systemCount } = precisionGroups(rows);
  if (fell.length === 0) return `${base}; how they held on the test set is shown on the Policy page.`;
  const everySystem = fell.every((label) => (byLabel.get(label)?.length ?? 0) === systemCount);
  return `${base}, and its validation targets were missed on the test set: precision of automatic action fell on ${joinList(fell.map(words))} ${
    everySystem ? `for all ${systemCount === 3 ? "three" : systemCount} systems` : "where it is on"
  }.`;
}

/** The labels each system can act on automatically (those with a HIGH threshold), in label order. */
export function automationLabels(rows: readonly PolicyLabelRow[]): Partial<Record<SystemId, Label[]>> {
  const out: Partial<Record<SystemId, Label[]>> = {};
  for (const system of SYSTEM_IDS) {
    const labels = rows.filter((row) => row.system === system && row.action_enabled).map((row) => row.label);
    if (rows.some((row) => row.system === system)) out[system] = LABELS.filter((label) => labels.includes(label));
  }
  return out;
}

/** Footnote naming each system's automatic-action labels; null if every system has the same ones. */
export function automationFootnote(rows: readonly PolicyLabelRow[]): string | null {
  const sets = automationLabels(rows);
  const systems = SYSTEM_IDS.filter((id) => sets[id] !== undefined);
  if (systems.length < 2 || new Set(systems.map((id) => (sets[id] ?? []).join())).size === 1) return null;
  const parts = systems.map((id) => {
    const labels = sets[id] ?? [];
    return `${inSentence(id)} on ${labels.length === 0 ? "no label" : joinList(labels.map(words))}`;
  });
  return `Automatic action is enabled on different labels for each system: ${parts.join("; ")}. Precision of automatic action and harmful comments actioned are therefore not like-for-like.`;
}

/** Three plain-language readings of the promised-versus-delivered plots. */
export function agreementReadings(rows: readonly PolicyLabelRow[]): Reading[] {
  const readings: Reading[] = [];

  const { byLabel, fell, held, mixed, systemCount } = precisionGroups(rows);
  const valuesFor = (labels: readonly Label[], field: "validation" | "test"): number[] =>
    labels.flatMap((label) => (byLabel.get(label) ?? []).map((item) => item[field]));
  if (fell.length > 0) {
    const everySystem = fell.every((label) => (byLabel.get(label)?.length ?? 0) === systemCount);
    readings.push({
      tone: "bad",
      text: `Precision at the action threshold fell on ${joinList(fell.map(words))} ${
        everySystem ? `for all ${systemCount === 3 ? "three" : systemCount} systems` : "where it is on"
      }: ${range2(valuesFor(fell, "validation"))} on validation, ${range2(valuesFor(fell, "test"))} on test.`,
    });
  }
  if (held.length > 0) {
    readings.push({
      tone: "good",
      text: `${capitalise(joinList(held.map(words)))} held where automatic action is on: ${range2(valuesFor(held, "test"))} on test.`,
    });
  }
  if (mixed.length > 0) {
    readings.push({
      tone: "bad",
      text: `On ${joinList(mixed.map(words))}, precision at the action threshold fell for some systems and held for others.`,
    });
  }

  const recall = recallReading(rows);
  if (recall !== null) readings.push(recall);
  return readings;
}

/**
 * Recall at the review threshold against the target, per system, from the test values. A label counts
 * as below the target when its test recall is strictly below the recall target of that row.
 * The "lowest" is the smallest test recall of the system, whether or not it is below the target.
 */
export function recallReading(rows: readonly PolicyLabelRow[]): Reading | null {
  const parts: { system: SystemId; below: number; of: number; lowest: { value: number; label: Label } }[] = [];
  for (const system of SYSTEM_IDS) {
    let below = 0;
    let counted = 0;
    let lowest: { value: number; label: Label } | null = null;
    for (const row of rows) {
      if (row.system !== system || row.test_recall_at_low === null || row.recall_target === null) continue;
      counted += 1;
      if (row.test_recall_at_low < row.recall_target) below += 1;
      if (lowest === null || row.test_recall_at_low < lowest.value) lowest = { value: row.test_recall_at_low, label: row.label };
    }
    if (lowest !== null) parts.push({ system, below, of: counted, lowest });
  }
  if (parts.length === 0) return null;
  const targets = new Set(rows.map((row) => row.recall_target).filter((value): value is number => value !== null));
  const [only] = [...targets];
  const target = targets.size === 1 && only !== undefined ? `the ${only.toFixed(2)} target` : "its target";
  const clauses = parts.map(
    (part, index) =>
      `${part.below} of ${part.of}${index === 0 ? " labels" : ""} for ${inSentence(part.system)} (lowest ${part.lowest.value.toFixed(3)}, ${words(part.lowest.label)})`,
  );
  const list = clauses.length === 1 ? (clauses[0] ?? "") : `${clauses.slice(0, -1).join(", ")} and ${clauses[clauses.length - 1] ?? ""}`;
  return {
    tone: parts.some((part) => part.below > 0) ? "bad" : "good",
    text: `Recall at the review threshold was below ${target} on ${list}.`,
  };
}

type Relation = "ahead" | "behind" | "level";

/** Whether a difference (JEV minus a baseline) with its interval is clearly above, below or around zero. */
export function relation(lo: number | null, hi: number | null): Relation | null {
  if (lo === null || hi === null) return null;
  if (lo > 0) return "ahead";
  if (hi < 0) return "behind";
  return "level";
}

const RELATION_PHRASE: Record<Relation, string> = { ahead: "ahead of", behind: "behind", level: "level with" };

/** Average-precision finding for JEV label by label, from the paired differences with intervals. */
export function apFinding(differences: readonly ApDifferenceRow[], labels: readonly Label[]): string {
  const groups: Record<"behind" | "ahead" | "level", Label[]> = { behind: [], ahead: [], level: [] };
  const mixed: { label: Label; lr: Relation; db: Relation }[] = [];
  for (const label of labels) {
    const pick = (contrast: string): Relation | null => {
      const row = differences.find((item) => item.label === label && item.contrast === contrast);
      return row === undefined ? null : relation(row.lo, row.hi);
    };
    const lr = pick(BASELINES[0]?.contrast ?? "");
    const db = pick(BASELINES[1]?.contrast ?? "");
    if (lr === null || db === null) continue;
    if (lr === db) groups[lr].push(label);
    else mixed.push({ label, lr, db });
  }
  const sentences: string[] = [];
  if (groups.behind.length > 0) {
    sentences.push(`JEV trails both baselines on ${joinList(groups.behind.map(words))}.`);
  }
  if (groups.ahead.length > 0) {
    sentences.push(`JEV leads both baselines on ${joinList(groups.ahead.map(words))}.`);
  }
  if (groups.level.length > 0) {
    sentences.push(
      `On ${joinList(groups.level.map(words))} the test set cannot separate it from either baseline.`,
    );
  }
  for (const item of mixed) {
    sentences.push(
      `On ${words(item.label)} it is ${RELATION_PHRASE[item.lr]} logistic regression and ${RELATION_PHRASE[item.db]} DistilBERT.`,
    );
  }
  return sentences.join(" ");
}

/** Review-budget finding for the rare labels, from the paired differences with intervals. */
export function budgetFinding(rows: readonly BudgetRow[], metric: string): string {
  const budgets = [...new Set(rows.filter((row) => row.metric === metric).map((row) => row.budget_pct))].sort((a, b) => a - b);
  const steps = budgets.flatMap((budget) => {
    const pick = (contrast: string): Relation | null => {
      const row = rows.find((item) => item.metric === metric && item.budget_pct === budget && item.comparison === contrast);
      return row === undefined ? null : relation(row.lo, row.hi);
    };
    const lr = pick(BASELINES[0]?.contrast ?? "");
    const db = pick(BASELINES[1]?.contrast ?? "");
    return lr === null || db === null ? [] : [{ budget, lr, db }];
  });
  const groups: { lr: Relation; db: Relation; budgets: number[] }[] = [];
  for (const step of steps) {
    const last = groups[groups.length - 1];
    if (last !== undefined && last.lr === step.lr && last.db === step.db) last.budgets.push(step.budget);
    else groups.push({ lr: step.lr, db: step.db, budgets: [step.budget] });
  }
  return groups
    .map((group, index) => {
      const subject = index === 0 ? "JEV" : "It";
      const at = joinList(group.budgets.map((budget) => `${budget}%`));
      if (group.lr === group.db) {
        const verb = { ahead: "leads both baselines", behind: "trails both baselines", level: "is level with both baselines" }[group.lr];
        return `${subject} ${verb} at ${at}.`;
      }
      const lead = index === 0 ? "JEV" : "it";
      return `At ${at} ${lead} is ${RELATION_PHRASE[group.lr]} logistic regression and ${RELATION_PHRASE[group.db]} DistilBERT.`;
    })
    .join(" ");
}

/**
 * The status caveat for the review-budget finding, read from the status column of the paired differences:
 * which budgets were not pre-specified (and why) and which were. Empty if the rows carry other statuses.
 */
export function budgetStatusNote(rows: readonly BudgetRow[], metric: string): string {
  const budgets = [...new Set(rows.filter((row) => row.metric === metric).map((row) => row.budget_pct))].sort((a, b) => a - b);
  const status = (budget: number, contrast: string): string | undefined =>
    rows.find((row) => row.metric === metric && row.budget_pct === budget && row.comparison === contrast)?.status;
  const lr = BASELINES[0]?.contrast ?? "";
  const db = BASELINES[1]?.contrast ?? "";
  const notPrespecified = budgets.filter((b) => status(b, lr) === "descriptive" && status(b, db) === "hypothesis generated on validation");
  if (notPrespecified.length === 0) return "";
  const prespecified = budgets.filter((b) => status(b, lr) === "pre-specified" && status(b, db) === "pre-specified");
  const at = (list: number[]): string => joinList(list.map((b) => `${b}%`));
  const only = prespecified.length > 0 ? ` Only the ${at(prespecified)} contrasts were pre-specified.` : "";
  return `The ${at(notPrespecified)} comparisons were not pre-specified: against logistic regression the difference is descriptive, and against DistilBERT it is a hypothesis generated on validation.${only}`;
}

/** Recall and precision targets shared by all rows, or null if they are missing. */
export function policyTargets(rows: readonly PolicyLabelRow[] | undefined): { recall: number; precision: number } | null {
  const row = rows?.find((item) => item.recall_target !== null && item.precision_target !== null);
  if (row === undefined || row.recall_target === null || row.precision_target === null) return null;
  return { recall: row.recall_target, precision: row.precision_target };
}

/**
 * Heading for the review-budget chart. It names a lead only as far as it holds: JEV clearly ahead of both
 * baselines (interval above zero against each) at the smallest budgets only, or at every budget.
 */
export function budgetHeadline(rows: readonly BudgetRow[], metric: string): string {
  const budgets = [...new Set(rows.filter((row) => row.metric === metric).map((row) => row.budget_pct))].sort((a, b) => a - b);
  const aheadOfBoth = budgets.map((budget) =>
    BASELINES.every((baseline) => {
      const row = rows.find((item) => item.metric === metric && item.budget_pct === budget && item.comparison === baseline.contrast);
      return row !== undefined && relation(row.lo, row.hi) === "ahead";
    }),
  );
  const leading = aheadOfBoth.findIndex((value) => !value);
  const prefix = leading === -1 ? aheadOfBoth.length : leading;
  const laterAhead = aheadOfBoth.slice(prefix).some(Boolean);
  if (prefix === 0 || laterAhead) return "Recall of rare-label comments as the review budget grows";
  return prefix === aheadOfBoth.length
    ? "JEV has higher rare-label recall at every review budget"
    : "JEV has higher rare-label recall only at the smallest review budgets";
}
