/*
 * Number formats used on every page, one per kind of value:
 *   probabilities and thresholds  4 decimals (f4), e.g. 0.0954
 *   metrics, averages, recalls     3 decimals (f3), e.g. 0.759, always with the leading zero
 *   shares of comments             1 decimal as a percentage (percent1), e.g. 13.6%
 *   targets                        whole percent (percent0) in sentences, 2 decimals (toFixed(2)) beside a score
 * Rounding is JavaScript's toFixed: half up on the exact value of the number, so an exact tie such as
 * 0.03125 prints as 0.0313. The end-to-end checker rounds the same way.
 */
const MINUS = "−";

/** Four decimals, as in the probability readouts. */
export function f4(value: number): string {
  return value.toFixed(4);
}

/** Three decimals, as in the result tables. */
export function f3(value: number): string {
  return value.toFixed(3);
}

/** Three decimals with an explicit sign (a true minus sign for negatives). */
export function signed3(value: number): string {
  if (value > 0) return `+${f3(value)}`;
  if (value < 0) return `${MINUS}${f3(Math.abs(value))}`;
  return f3(0);
}

export function formatCount(value: number): string {
  return value.toLocaleString("en-US");
}

/** Percent with one decimal, for shares of comments. */
export function percent1(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** A whole-number percent for a target such as 0.9. */
export function percent0(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** Label as words: severe_toxic becomes severe toxic. */
export function words(label: string): string {
  return label.replace(/_/g, " ");
}

/** A list in words: a, b and c. */
export function joinList(items: readonly string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1] ?? ""}`;
}

/** Sentence case for the first character. */
export function capitalise(text: string): string {
  return text.length === 0 ? text : `${text[0]?.toUpperCase() ?? ""}${text.slice(1)}`;
}
