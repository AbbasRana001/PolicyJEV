import { SYSTEM_NAMES } from "../api";
import type { PolicyContrastRow } from "../api";
import { signed3 } from "../format";
import { BASELINES, COMPARISON_MEASURES, normaliseContrast } from "../measures";
import type { NormalisedContrast } from "../measures";
import { contrastPosition, contrastRange } from "../scales";
import { SYSTEM_COLOUR, systemStyle } from "../theme";

export interface PlotRow {
  key: string;
  name: string;
  hint: string;
  lanes: { system: (typeof BASELINES)[number]["system"]; text: string; value: NormalisedContrast }[];
}

/** One plot row per measure: JEV against each baseline, flipped so left always means JEV is worse. */
export function comparisonRows(contrasts: readonly PolicyContrastRow[]): PlotRow[] {
  return COMPARISON_MEASURES.map((measure) => ({
    key: measure.metric,
    name: measure.name,
    hint: measure.better === "lower" ? "Lower is better" : "Higher is better",
    lanes: BASELINES.flatMap((baseline) => {
      const row = contrasts.find((item) => item.contrast === baseline.contrast && item.metric === measure.metric);
      return row === undefined
        ? []
        : [{ system: baseline.system, text: baseline.text, value: normaliseContrast(row, measure.better) }];
    }),
  }));
}

/** Every plotted value, for the shared axis and for the sentence about the middle line. */
export function plottedValues(rows: readonly PlotRow[]): NormalisedContrast[] {
  return rows.flatMap((row) => row.lanes.map((lane) => lane.value));
}

export function ComparisonPlot({ rows }: { rows: readonly PlotRow[] }) {
  const range = contrastRange(plottedValues(rows));
  const position = (value: number): number => contrastPosition(value, range);
  return (
    <div id="cmp-plot">
      <div className="mkey">
        {BASELINES.map((baseline) => (
          <span key={baseline.system}>
            <i style={{ background: SYSTEM_COLOUR[baseline.system] }} />
            JEV against {baseline.text}
          </span>
        ))}
      </div>
      <div className="crhead">
        <span />
        <div className="t">
          <span>JEV worse</span>
          <span>JEV better</span>
        </div>
        <span className="signed">JEV minus baseline, signed so that negative means JEV is worse.</span>
      </div>
      <div id="cmp">
        {rows.map((row) => (
          <div className="cr" key={row.key} data-measure={row.key}>
            <div className="cn">
              <b>{row.name}</b>
              <span>{row.hint}</span>
            </div>
            <div
              className="ctrack"
              role="img"
              aria-label={`${row.name}: ${row.lanes
                .map((lane) =>
                  lane.value.difference === null
                    ? `against ${lane.text} not defined`
                    : `against ${lane.text} ${signed3(lane.value.difference)}`,
                )
                .join(", ")}. Left of the middle line means JEV is worse.`}
            >
              <i className="zero" />
              {row.lanes.map((lane, index) => {
                const { difference, low, high } = lane.value;
                return (
                  <div className="lane" key={lane.system} style={systemStyle(lane.system, { top: index === 0 ? 20 : 46 })}>
                    {low !== null && high !== null && (
                      <i
                        className="iv"
                        style={{
                          left: `${position(low)}%`,
                          width: `${Math.max(0.4, position(high) - position(low))}%`,
                          animationDelay: `${index * 120}ms`,
                        }}
                      />
                    )}
                    {difference !== null && <i className="dt" style={{ left: `${position(difference)}%` }} />}
                  </div>
                );
              })}
            </div>
            <div className="nn">
              {row.lanes.map((lane) => (
                <p key={lane.system} style={systemStyle(lane.system)}>
                  <i />
                  <b>{lane.value.difference === null ? "–" : signed3(lane.value.difference)}</b>
                  {lane.value.low !== null && lane.value.high !== null && (
                    <span>
                      [{signed3(lane.value.low)}, {signed3(lane.value.high)}]
                    </span>
                  )}
                  <span className="sr-only">against {SYSTEM_NAMES[lane.system]}</span>
                </p>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
