import type { SystemId } from "./api";
import { f3 } from "./format";
import { SYSTEM_NAMES } from "./api";

/** Geometry and direct-label placement for the recall-against-review-budget chart. */

export const CHART_MARGIN = { l: 64, r: 18, t: 16, b: 40 } as const;

/** Height of a label box, the part of it above the baseline, and a width per character (12px text). */
export const LABEL_HEIGHT = 14;
export const LABEL_ASCENT = 11;
export const LABEL_CHAR_WIDTH = 7.4;
/** Space kept between a label and any line or dot. */
export const LABEL_CLEARANCE = 4;
/** Where the x-axis tick labels start, measured up from the bottom of the chart. */
const AXIS_LABEL_ZONE = 27;

export interface Geometry {
  width: number;
  height: number;
  innerWidth: number;
  innerHeight: number;
  x: (index: number) => number;
  y: (value: number) => number;
}

export function budgetGeometry(measuredWidth: number, levelCount: number): Geometry {
  const width = Math.max(300, measuredWidth || 700);
  const height = width < 520 ? 300 : 330;
  const innerWidth = width - CHART_MARGIN.l - CHART_MARGIN.r;
  const innerHeight = height - CHART_MARGIN.t - CHART_MARGIN.b;
  return {
    width,
    height,
    innerWidth,
    innerHeight,
    x: (index) => CHART_MARGIN.l + (levelCount < 2 ? innerWidth / 2 : (innerWidth * index) / (levelCount - 1)),
    y: (value) => CHART_MARGIN.t + innerHeight * (1 - value),
  };
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type Segment = readonly [number, number, number, number];

export interface Obstacles {
  segments: Segment[];
  dots: { x: number; y: number; r: number }[];
  /** Boxes already taken, such as labels placed earlier. */
  rects: Rect[];
}

/** True if the segment passes through the rectangle (Liang-Barsky clipping). */
export function segmentHitsRect([x1, y1, x2, y2]: Segment, rect: Rect): boolean {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const p = [-dx, dx, -dy, dy];
  const q = [x1 - rect.x, rect.x + rect.w - x1, y1 - rect.y, rect.y + rect.h - y1];
  let enter = 0;
  let leave = 1;
  for (let i = 0; i < 4; i += 1) {
    const pi = p[i] ?? 0;
    const qi = q[i] ?? 0;
    if (pi === 0) {
      if (qi < 0) return false;
    } else {
      const t = qi / pi;
      if (pi < 0) {
        if (t > leave) return false;
        if (t > enter) enter = t;
      } else {
        if (t < enter) return false;
        if (t < leave) leave = t;
      }
    }
  }
  return true;
}

function grow(rect: Rect, by: number): Rect {
  return { x: rect.x - by, y: rect.y - by, w: rect.w + 2 * by, h: rect.h + 2 * by };
}

/** True if the rectangle, with its clearance, touches a line or a dot. */
export function rectTouches(rect: Rect, obstacles: Obstacles, clearance = LABEL_CLEARANCE): boolean {
  const padded = grow(rect, clearance);
  if (obstacles.segments.some((segment) => segmentHitsRect(segment, padded))) return true;
  if (
    obstacles.rects.some(
      (other) => padded.x < other.x + other.w && padded.x + padded.w > other.x && padded.y < other.y + other.h && padded.y + padded.h > other.y,
    )
  ) {
    return true;
  }
  return obstacles.dots.some((dot) => {
    const nearestX = Math.min(Math.max(dot.x, padded.x), padded.x + padded.w);
    const nearestY = Math.min(Math.max(dot.y, padded.y), padded.y + padded.h);
    return (dot.x - nearestX) ** 2 + (dot.y - nearestY) ** 2 <= dot.r ** 2;
  });
}

export interface PlacedLabel {
  text: string;
  system: SystemId;
  value: number;
  x: number;
  baseline: number;
  rect: Rect;
  /** False only if no position within the chart cleared every line and dot. */
  clear: boolean;
  /** A short line from the point to the label, present when the label is not adjacent to its point. */
  leader: Segment | null;
}

/** The nearest point of a box to a given point. */
function nearestOnRect(point: { x: number; y: number }, rect: Rect): { x: number; y: number } {
  return { x: Math.min(Math.max(point.x, rect.x), rect.x + rect.w), y: Math.min(Math.max(point.y, rect.y), rect.y + rect.h) };
}

/** A leader from a point to its label, or null if the label is adjacent. */
export function leaderFor(anchor: { x: number; y: number }, rect: Rect): Segment | null {
  const near = nearestOnRect(anchor, rect);
  return Math.hypot(near.x - anchor.x, near.y - anchor.y) <= ADJACENT ? null : [anchor.x, anchor.y, near.x, near.y];
}

/** How far to the right of its point a label may start, tried in this order. */
const SHIFTS = [10, 26, 44, 66, 92, 124, 160, 200, 240, 280];

/** A label whose box is no further than this from its point counts as adjacent; further out it gets a leader line. */
export const ADJACENT = 8;

/**
 * Put a label next to a point, above or below it, as close as it can go without touching a line, a dot or
 * another label: the gap grows, and at each gap the label may start a little further to the right.
 */
export function placeLabel(
  anchor: { x: number; y: number },
  text: string,
  sides: readonly ("above" | "below")[],
  obstacles: Obstacles,
  geometry: Geometry,
): Omit<PlacedLabel, "system" | "value"> {
  const w = text.length * LABEL_CHAR_WIDTH;
  const withLeader = <T extends { rect: Rect }>(found: T): T & { leader: Segment | null } => ({ ...found, leader: leaderFor(anchor, found.rect) });
  const top = 2;
  const bottom = geometry.height - AXIS_LABEL_ZONE;
  const first = sides[0] ?? "above";
  let fallback: { x: number; baseline: number; rect: Rect } | null = null;
  for (let gap = 8; gap < geometry.height; gap += 2) {
    for (const shift of SHIFTS) {
      const x = Math.min(anchor.x + shift, geometry.width - 2 - w);
      for (const side of sides) {
        const baseline = side === "above" ? anchor.y - gap : anchor.y + gap + LABEL_ASCENT;
        const rect: Rect = { x, y: baseline - LABEL_ASCENT, w, h: LABEL_HEIGHT };
        if (rect.y < top || rect.y + rect.h > bottom) continue;
        if (fallback === null && shift === SHIFTS[0] && side === first) fallback = { x, baseline, rect };
        if (rectTouches(rect, obstacles)) continue;
        // a leader must not run over a label that is already placed
        const leader = leaderFor(anchor, rect);
        if (leader !== null && obstacles.rects.some((other) => segmentHitsRect(leader, grow(other, 2)))) continue;
        return withLeader({ text, x, baseline, rect, clear: true });
      }
    }
  }
  const last = fallback ?? { x: anchor.x + 10, baseline: anchor.y - 12, rect: { x: anchor.x + 10, y: anchor.y - 12 - LABEL_ASCENT, w, h: LABEL_HEIGHT } };
  return withLeader({ text, x: last.x, baseline: last.baseline, rect: last.rect, clear: false });
}

export interface LabelSeries {
  system: SystemId;
  values: (number | null)[];
}

export function obstaclesFor(series: readonly LabelSeries[], geometry: Geometry): Obstacles {
  const segments: Segment[] = [];
  const dots: Obstacles["dots"] = [];
  for (const item of series) {
    const points = item.values.flatMap((value, index) => (value === null ? [] : [{ x: geometry.x(index), y: geometry.y(value) }]));
    const radius = item.system === "jev" ? 5 : 3.5;
    for (const point of points) dots.push({ x: point.x, y: point.y, r: radius + 1 });
    for (let i = 1; i < points.length; i += 1) {
      const a = points[i - 1];
      const b = points[i];
      if (a !== undefined && b !== undefined) segments.push([a.x, a.y, b.x, b.y]);
    }
  }
  return { segments, dots, rects: [] };
}

/**
 * The direct labels at the smallest budget: one for every system that has a value there, the highest above
 * its point, the lowest below its point, any in between on whichever side is free (none if only one system).
 */
export function layoutBudgetLabels(series: readonly LabelSeries[], levelCount: number, measuredWidth: number): {
  geometry: Geometry;
  labels: PlacedLabel[];
} {
  const geometry = budgetGeometry(measuredWidth, levelCount);
  const first = series
    .map((item) => ({ system: item.system, value: item.values[0] ?? null }))
    .filter((item): item is { system: SystemId; value: number } => item.value !== null)
    .sort((a, b) => b.value - a.value);
  const labels: PlacedLabel[] = [];
  if (first.length < 2) return { geometry, labels };
  const obstacles = obstaclesFor(series, geometry);
  first.forEach((item, index) => {
    const sides: ("above" | "below")[] = index === 0 ? ["above"] : index === first.length - 1 ? ["below"] : ["above", "below"];
    const placed = placeLabel(
      { x: geometry.x(0), y: geometry.y(item.value) },
      `${SYSTEM_NAMES[item.system]} ${f3(item.value)}`,
      sides,
      obstacles,
      geometry,
    );
    obstacles.rects.push(placed.rect);
    if (placed.leader !== null) obstacles.segments.push(placed.leader);
    labels.push({ system: item.system, value: item.value, ...placed });
  });
  return { geometry, labels };
}
