import { LABELS, SYSTEM_IDS, SYSTEM_NAMES } from "../api";
import type { ApRow, BudgetRow, Label, ResultsPayload, SystemId } from "../api";
import { f3, formatCount, joinList, words } from "../format";
import { useElementWidth, useReducedMotion } from "../hooks";
import { apFinding, budgetFinding, budgetHeadline } from "../narrative";
import { CHART_MARGIN, layoutBudgetLabels } from "../chartLabels";
import { apAxisMax, budgetLevels } from "../scales";
import { SYSTEM_COLOUR, systemStyle } from "../theme";
import { part, renderPart } from "./parts";
import { SystemKey } from "./SummaryCharts";

/** Average precision by label: one bar per system with a thin line for the 95% interval. */
export function ApChart({ rows }: { rows: readonly ApRow[] }) {
  const max = apAxisMax(rows);
  const labels = LABELS.filter((label) => rows.some((row) => row.label === label));
  return (
    <div className="apc" id="apc">
      {labels.map((label) => {
        const group = SYSTEM_IDS.map((id) => rows.find((row) => row.label === label && row.system === id));
        const positives = group.find((row) => row !== undefined)?.positives;
        return (
          <div className="apg" key={label} data-label={label}>
            <div className="apbars">
              {group.map((row, index) => {
                if (row === undefined) return null;
                const height = (row.ap / max) * 100;
                return (
                  <div
                    key={row.system}
                    className="apb"
                    data-system={row.system}
                    style={systemStyle(row.system, { height: `${height}%`, animationDelay: `${index * 60}ms` })}
                    title={`${SYSTEM_NAMES[row.system]} ${f3(row.ap)} [${f3(row.ap_lo)}, ${f3(row.ap_hi)}]`}
                  >
                    <i
                      className="wh"
                      style={{
                        bottom: `${(row.ap_lo / row.ap) * 100}%`,
                        height: `${((row.ap_hi - row.ap_lo) / row.ap) * 100}%`,
                      }}
                    />
                    <span className="vv">{f3(row.ap)}</span>
                    <span className="sr-only">
                      {SYSTEM_NAMES[row.system]} {words(label)}: {f3(row.ap)}, 95% interval {f3(row.ap_lo)} to {f3(row.ap_hi)}
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="gl">{words(label)}</div>
            <div className="gn">{positives === undefined ? "" : `${formatCount(positives)} positives`}</div>
          </div>
        );
      })}
    </div>
  );
}

interface Series {
  system: SystemId;
  values: (number | null)[];
}

/** Rows of one metric for the three systems, ordered by budget level. */
export function budgetSeries(rows: readonly BudgetRow[], metric: string): { levels: number[]; series: Series[] } {
  const levels = budgetLevels(rows, metric);
  const series = SYSTEM_IDS.map((system) => ({
    system,
    values: levels.map((level) => rows.find((row) => row.metric === metric && row.comparison === system && row.budget_pct === level)?.value ?? null),
  }));
  return { levels, series };
}

/** Recall of rare-label comments against the share of traffic reviewed, one line per system. */
export function BudgetChart({ rows, metric }: { rows: readonly BudgetRow[]; metric: string }) {
  const [ref, measured] = useElementWidth<HTMLDivElement>();
  const reduced = useReducedMotion();
  const { levels, series } = budgetSeries(rows, metric);
  const { geometry, labels } = layoutBudgetLabels(series, levels.length, measured);
  const { width, height, innerWidth, x, y } = geometry;

  return (
    <div className="line-chart" id="lc" ref={ref}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Recall of rare-label comments by review budget for three systems">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={CHART_MARGIN.l} x2={width - CHART_MARGIN.r} y1={y(tick)} y2={y(tick)} stroke="#272932" />
            <text x={CHART_MARGIN.l - 10} y={y(tick) + 4} textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {levels.map((level, index) => (
          <text key={level} x={x(index)} y={height - 14} textAnchor="middle">
            {level}%
          </text>
        ))}
        <text x={CHART_MARGIN.l + innerWidth / 2} y={height} textAnchor="middle" style={{ fontSize: 11 }}>
          Share of traffic reviewed
        </text>
        <text transform="rotate(-90)" x={-(CHART_MARGIN.t + geometry.innerHeight / 2)} y={12} textAnchor="middle" style={{ fontSize: 11 }} data-axis-title="y">
          Share of rare-label comments caught
        </text>
        {series.map((item) => {
          const points = item.values.flatMap((value, index) => (value === null ? [] : [{ px: x(index), py: y(value), value, level: levels[index] ?? 0 }]));
          if (points.length === 0) return null;
          const path = points.map(({ px, py }, index) => `${index === 0 ? "M" : "L"}${px.toFixed(1)} ${py.toFixed(1)}`).join(" ");
          const isJev = item.system === "jev";
          return (
            <g key={item.system} data-system={item.system}>
              <path
                d={path}
                fill="none"
                stroke={SYSTEM_COLOUR[item.system]}
                strokeWidth={isJev ? 3.5 : 2.2}
                strokeLinecap="round"
                strokeLinejoin="round"
                pathLength={1}
                style={{
                  strokeDasharray: 1,
                  strokeDashoffset: reduced ? 0 : 1,
                  animation: reduced ? "none" : "dash 1.1s var(--ease) forwards",
                }}
              />
              {points.map(({ px, py, value, level }) => (
                <circle key={level} cx={px} cy={py} r={isJev ? 5 : 3.5} fill={SYSTEM_COLOUR[item.system]} data-level={level} data-value={value}>
                  <title>{`${SYSTEM_NAMES[item.system]} at ${level}%: ${f3(value)}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
        {labels.map((label) =>
          label.leader === null ? null : (
            <line
              key={`leader-${label.system}`}
              data-leader={label.system}
              x1={label.leader[0]}
              y1={label.leader[1]}
              x2={label.leader[2]}
              y2={label.leader[3]}
              stroke={SYSTEM_COLOUR[label.system]}
              strokeWidth={1}
              opacity={0.7}
            />
          ),
        )}
        {labels.map((label) => (
          <text
            key={label.system}
            data-direct-label={label.system}
            x={label.x}
            y={label.baseline}
            style={{
              fill: SYSTEM_COLOUR[label.system],
              fontWeight: label.system === "jev" ? 600 : 400,
              // a halo in the page colour keeps the text legible if a line comes near it
              stroke: "var(--bg)",
              strokeWidth: 4,
              strokeLinejoin: "round",
              paintOrder: "stroke",
            }}
          >
            {label.text}
          </text>
        ))}
      </svg>
    </div>
  );
}

/** The metric whose curves the page draws: recall of the rare labels (threat, severe toxic, identity hate). */
export const BUDGET_METRIC = "recall_rare";

export function ResultsView({ results }: { results: ResultsPayload | null | undefined }) {
  const ap = part(results, (payload) => payload.per_label_ap);
  const budget = part(results, (payload) => payload.review_budget);
  const differences = results?.ap_differences;
  const levels = budget.state === "ok" ? budgetLevels(budget.data, BUDGET_METRIC) : [];
  const finding = budget.state === "ok" ? budgetFinding(budget.data, BUDGET_METRIC) : "";
  const apText = differences === undefined ? "" : apFinding(differences, LABELS as readonly Label[]);

  return (
    <>
      <h1>Results</h1>
      <p className="lede">
        Ranking quality and review-budget results from the single test run. JEV is zero-shot: it was not trained
        on Jigsaw, but its calibration and thresholds were fitted on validation data.
      </p>

      <div className="sec">
        <h2>Average precision by label</h2>
        <p className="sub">
          How well each system ranks true positives above negatives, from 0 to 1. Thin lines are 95% intervals.
          {apText === "" ? "" : ` ${apText}`}
        </p>
        <SystemKey />
        {renderPart(ap, (rows) => (
          <ApChart rows={rows} />
        ))}
      </div>

      <div className="sec">
        <h2>{budget.state === "ok" ? budgetHeadline(budget.data, BUDGET_METRIC) : "Recall of rare-label comments as the review budget grows"}</h2>
        <p className="sub">
          Share of comments with a rare label (threat, severe toxic, identity hate) caught when moderators review
          only the top-scored {levels.length === 0 ? "" : `${levels[0]}% to ${levels[levels.length - 1]}% `}of
          traffic. {finding}
        </p>
        <SystemKey />
        {renderPart(budget, (rows) => (
          <>
            <BudgetChart rows={rows} metric={BUDGET_METRIC} />
            <p className="note" data-note="budget-spacing">
              The review budgets ({joinList(budgetLevels(rows, BUDGET_METRIC).map((level) => `${level}%`))}) are spaced
              evenly along the axis, not to scale.
            </p>
          </>
        ))}
      </div>
    </>
  );
}
