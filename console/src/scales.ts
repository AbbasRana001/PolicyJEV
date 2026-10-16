import type { ApRow, BudgetRow, PolicyLabelRow } from "./api";
import type { Better, NormalisedContrast } from "./measures";

/** Axis end for a measure bar: 0 to 1 where higher is better, a little above the largest value otherwise. */
export function measureAxisMax(values: readonly (number | null)[], better: Better): number {
  if (better === "higher") return 1;
  const largest = Math.max(0, ...values.filter((value): value is number => value !== null));
  if (largest === 0) return 1;
  return Math.min(1, Math.ceil(largest * 1.15 * 50) / 50);
}

/** Half-width of the comparison plot: the largest interval end, with a margin, to two decimals. */
export function contrastRange(rows: readonly NormalisedContrast[]): number {
  let largest = 0;
  for (const row of rows) {
    for (const value of [row.difference, row.low, row.high]) {
      if (value !== null) largest = Math.max(largest, Math.abs(value));
    }
  }
  return largest === 0 ? 0.1 : Math.ceil(largest * 1.08 * 100) / 100;
}

/** Position of a value on the comparison plot as a percentage; zero is always the middle. */
export function contrastPosition(value: number, range: number): number {
  return ((value + range) / (2 * range)) * 100;
}

/** Lower end of the validation-versus-test plots, shared by both: a tenth below the lowest value. */
export function agreementAxisMin(rows: readonly PolicyLabelRow[]): number {
  const values = rows.flatMap((row) => [
    row.validation_recall_at_low,
    row.test_recall_at_low,
    row.validation_precision_at_high,
    row.test_action_precision,
    row.precision_target,
    row.recall_target,
  ]);
  const lowest = Math.min(1, ...values.filter((value): value is number => value !== null));
  return Math.max(0, Math.floor((lowest - 0.05) * 10) / 10);
}

/** Upper end of the average-precision bars: the highest interval end rounded up to a tenth. */
export function apAxisMax(rows: readonly ApRow[]): number {
  const highest = Math.max(0, ...rows.map((row) => row.ap_hi));
  return highest === 0 ? 1 : Math.min(1, Math.ceil(highest * 10) / 10);
}

/** Budgets (percent of traffic) present for a metric, ascending. */
export function budgetLevels(rows: readonly BudgetRow[], metric: string): number[] {
  return [...new Set(rows.filter((row) => row.metric === metric).map((row) => row.budget_pct))].sort(
    (a, b) => a - b,
  );
}

/** Ticks every tenth from the first multiple of 0.1 at or above min, up to max. */
export function tenths(min: number, max: number): number[] {
  const ticks: number[] = [];
  for (let v = Math.ceil(min * 10 - 1e-9) / 10; v <= max + 1e-9; v += 0.1) ticks.push(Math.round(v * 10) / 10);
  return ticks;
}
