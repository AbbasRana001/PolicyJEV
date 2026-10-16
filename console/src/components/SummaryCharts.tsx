import { SYSTEM_IDS, SYSTEM_NAMES } from "../api";
import { useElementWidth } from "../hooks";
import type { PolicySummaryRow, SystemId } from "../api";
import { ZONE_CLASS } from "../decision";
import { f3, formatCount, percent1 } from "../format";
import { FIVE_MEASURES, bestIndex, measureValues } from "../measures";
import { measureAxisMax } from "../scales";
import { cssVars, systemStyle, SYSTEM_COLOUR } from "../theme";

/** Smallest share of the bar that still gets a printed percentage. */
export const FLOW_LABEL_MIN_PERCENT = 9;

/** Room a printed percentage needs inside its segment: text plus the segment's side padding, in pixels. */
export function labelFits(text: string, segmentPixels: number): boolean {
  return segmentPixels >= text.length * 8.5 + 22;
}

function FlowBarSegments({ parts, total, name }: { parts: readonly { decision: "ALLOW" | "HUMAN REVIEW" | "ACTION"; name: string; count: number; delay: number }[]; total: number; name: string }) {
  const [ref, width] = useElementWidth<HTMLDivElement>();
  return (
    <div ref={ref} className="fl-bar" role="img" aria-label={`${name}: ${parts.map((p) => `${p.name} ${formatCount(p.count)}`).join(", ")}`}>
      {parts.map((p) => {
        const share = (p.count / total) * 100;
        const text = `${share.toFixed(1)}%`;
        // measured: print the percentage only if it fits; before the bar has a width, fall back to a share cutoff
        const fits = width > 0 ? labelFits(text, (share / 100) * width) : share >= FLOW_LABEL_MIN_PERCENT;
        return (
          <div key={p.decision} className={ZONE_CLASS[p.decision]} data-flow={p.decision} style={{ width: `${share}%`, animationDelay: `${p.delay}ms` }} title={`${p.name}: ${formatCount(p.count)} comments`}>
            {fits ? text : ""}
          </div>
        );
      })}
    </div>
  );
}

/** Where the comments went: allow, review and action, as bars that fill each system's row. */
export function FlowBars({ summary }: { summary: readonly PolicySummaryRow[] }) {
  return (
    <div className="flow" id="flow">
      {SYSTEM_IDS.map((id, index) => {
        const row = summary.find((item) => item.system === id);
        if (row === undefined) return null;
        const total = row.n_allow + row.n_review + row.n_action;
        if (total === 0) return null;
        const parts = [
          { decision: "ALLOW", name: "Allow", count: row.n_allow, delay: index * 110 },
          { decision: "HUMAN REVIEW", name: "Human review", count: row.n_review, delay: index * 110 + 120 },
          { decision: "ACTION", name: "Action", count: row.n_action, delay: index * 110 + 240 },
        ] as const;
        return (
          <div className="fl" key={id} style={systemStyle(id)} data-system={id}>
            <div className="fl-name">
              <i />
              {SYSTEM_NAMES[id]}
            </div>
            <FlowBarSegments parts={parts} total={total} name={SYSTEM_NAMES[id]} />
            <div className="fl-n">
              <span className="za">
                <i />
                Allow {formatCount(row.n_allow)}
              </span>
              <span className="zr">
                <i />
                Human review {formatCount(row.n_review)} ({percent1(row.n_review / total)})
              </span>
              <span className="zx">
                <i />
                Action {formatCount(row.n_action)} ({percent1(row.n_action / total)})
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** The key that names the three systems by colour. */
export function SystemKey({ aside }: { aside?: string }) {
  return (
    <div className="mkey">
      {SYSTEM_IDS.map((id) => (
        <span key={id}>
          <i style={{ background: SYSTEM_COLOUR[id] }} />
          {SYSTEM_NAMES[id]}
        </span>
      ))}
      {aside !== undefined && <span className="small aside">{aside}</span>}
    </div>
  );
}

/** Five measures, one row each: a dot per system on a track, the best one larger, and the values. */
/** The two measures that depend on which labels a system can act on automatically (docs/results_policy.md). */
const NOT_LIKE_FOR_LIKE = ["m2_action_precision", "action_coverage"];

export function FiveMeasures({ summary, footnote = null }: { summary: readonly PolicySummaryRow[]; footnote?: string | null }) {
  return (
    <div id="metrics">
      {FIVE_MEASURES.map((measure) => {
        const values = measureValues(summary, measure.key, SYSTEM_IDS);
        const best = bestIndex(values, measure.better);
        const max = measureAxisMax(values, measure.better);
        return (
          <div className="mr" key={measure.key} data-measure={measure.key}>
            <div className="mn">
              <b>{measure.name}</b>
              {footnote !== null && NOT_LIKE_FOR_LIKE.includes(measure.key) && (
                <em className="flag" data-flag="not-like-for-like">Different labels per system</em>
              )}
              <span>{measure.description}</span>
            </div>
            <div className="trackbox">
            <div className="dtrack" style={cssVars({ "--gd": measure.better === "lower" ? "90deg" : "270deg" })}>
              {SYSTEM_IDS.map((id: SystemId, index) => {
                const value = values[index];
                if (value === null || value === undefined) return null;
                return (
                  <i
                    key={id}
                    className={`pd${best === index ? " best" : ""}`}
                    style={systemStyle(id, cssVars({ "--f": value / max, animationDelay: `${index * 90}ms` }))}
                    title={`${SYSTEM_NAMES[id]} ${f3(value)}`}
                    data-system={id}
                  />
                );
              })}
              {measure.better === "lower" ? (
                <>
                  <span className="end" style={{ left: 12 }}>better</span>
                  <span className="end" style={{ right: 12 }}>worse</span>
                </>
              ) : (
                <>
                  <span className="end" style={{ left: 12 }}>worse</span>
                  <span className="end" style={{ right: 12 }}>better</span>
                </>
              )}
            </div>
            <div className="scale" data-scale={max} aria-hidden="true">
              <span>0</span>
              <span>{max.toFixed(2)}</span>
            </div>
            </div>
            <div className="vals">
              {SYSTEM_IDS.map((id, index) => {
                const value = values[index];
                return (
                  <span key={id} className={`vc${best === index ? " best" : ""}`} style={systemStyle(id)} data-system={id} data-best={best === index ? "true" : undefined}>
                    <i />
                    {value === null || value === undefined ? "–" : f3(value)}
                  </span>
                );
              })}
            </div>
          </div>
        );
      })}
      {footnote !== null && (
        <p className="note" data-note="automation">
          {footnote}
        </p>
      )}
    </div>
  );
}
