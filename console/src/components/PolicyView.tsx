import { SYSTEM_IDS } from "../api";
import type { PolicySummaryRow, ResultsPayload, ThresholdRow } from "../api";
import type { PolicyLabelRow } from "../api";
import { joinList, words } from "../format";
import { precisionGroups } from "../narrative";
import { automationFootnote, comparisonNote, policyTargets, summaryStatement } from "../narrative";
import { Agreement } from "./Agreement";
import { ComparisonPlot, comparisonRows, plottedValues } from "./ComparisonPlot";
import { Explorer, targetWords } from "./Explorer";
import { part, renderPart } from "./parts";
import { FiveMeasures, FlowBars, SystemKey } from "./SummaryCharts";

interface PolicyViewProps {
  thresholds: ThresholdRow[] | null | undefined;
  results: ResultsPayload | null | undefined;
}

function usableSummary(summary: readonly PolicySummaryRow[]): PolicySummaryRow[] {
  return summary.filter((row) => SYSTEM_IDS.includes(row.system));
}

/** What the cards show, in a sentence with counts taken from the rows: where precision and recall fell short. */
function shown(targets: { recall: number; precision: number } | null, rows: readonly PolicyLabelRow[]): string {
  const parts: string[] = [];
  const { fell, byLabel, systemCount } = precisionGroups(rows);
  if (fell.length > 0) {
    const every = fell.every((label) => (byLabel.get(label)?.length ?? 0) === systemCount);
    parts.push(`precision at the action threshold fell on ${joinList(fell.map(words))} ${every ? `for all ${systemCount === 3 ? "three" : systemCount} systems` : "where it is on"}`);
  }
  const counted = rows.filter((row) => row.test_recall_at_low !== null && row.recall_target !== null);
  const below = counted.filter((row) => (row.test_recall_at_low ?? 1) < (row.recall_target ?? 0)).length;
  if (counted.length > 0 && targets !== null) {
    parts.push(`recall at the review threshold was below the ${targets.recall.toFixed(2)} target on ${below} of ${counted.length} system-label pairs`);
  }
  return parts.length === 0 ? "" : ` On the test set, ${parts.join(", and ")}.`;
}

/** The paragraph about how far to trust the targets; the target values come from the results. */
export function trustText(targets: { recall: number; precision: number } | null, rows: readonly PolicyLabelRow[] = []): string {
  let subject: string;
  if (targets === null) subject = "The recall and precision targets";
  else if (targets.recall === targets.precision) subject = `The ${targets.recall.toFixed(2)} recall and precision targets`;
  else subject = `The ${targets.recall.toFixed(2)} recall and ${targets.precision.toFixed(2)} precision targets`;
  return `${subject} are assumptions about what a moderation team might want, not findings. The thresholds were fitted on the validation split of train.csv and applied once to the official test set.${shown(targets, rows)} The cause of the difference is not established. Before enabling automatic action in a real deployment, thresholds would need to be re-estimated on recent data from that deployment.`;
}

/** Names the dashed target line: one number if both targets are the same, otherwise one per measure. */
export function targetLinePhrase(targets: { recall: number; precision: number } | null): string {
  if (targets === null) return "target";
  return targets.recall === targets.precision
    ? `${targets.precision.toFixed(2)} target`
    : `target for each measure (${targets.precision.toFixed(2)} for precision, ${targets.recall.toFixed(2)} for recall)`;
}

export function PolicyView({ thresholds, results }: PolicyViewProps) {
  const labelRows = part(results, (payload) => payload.policy_per_label);
  const summary = part(results, (payload) => payload.policy_summary);
  const contrasts = part(results, (payload) => payload.policy_contrasts);
  const targets = policyTargets(labelRows.state === "ok" ? labelRows.data : undefined);
  const copy = targetWords(targets);
  const targetPhrase = targetLinePhrase(targets);

  const statement = summary.state === "ok" ? summaryStatement(usableSummary(summary.data), labelRows.state === "ok" ? labelRows.data : []) : null;

  return (
    <>
      <h1>Policy</h1>
      <p className="lede">
        Every label has a LOW threshold and, where one was found, a HIGH threshold. Below LOW a comment is
        allowed, from LOW up to HIGH a person reviews it, and at or above HIGH the system acts automatically.
        A label with no HIGH threshold is never acted on automatically. A comment gets the most severe outcome
        across its six labels.
      </p>

      <div className="sec">
        <h2>Try a probability</h2>
        <p className="sub">
          The thresholds are fixed. Drag across any bar, or use the slider, to set a probability and see what
          the policy would do with it: the same value is applied to all six labels. Each system has its own
          thresholds, fitted on validation data only, with the same {copy.recall} recall and {copy.precision}{" "}
          precision targets for every system.
        </p>
        <Explorer thresholds={thresholds} targets={targets} supportFloor={results?.policy_fit?.support_floor ?? null} />
        <p className="note">The scale is square root, so small probabilities get room.</p>
      </div>

      <div className="sec">
        <h2>{statement === null ? "Policy results on the test set" : statement.headline}</h2>
        <p className="sub">{statement === null ? "Same thresholds, scored once on the test set." : statement.detail}</p>

        <h3 className="grp">Where the comments went</h3>
        {renderPart(summary, (rows) => (
          <FlowBars summary={usableSummary(rows)} />
        ))}

        <h3 className="grp" style={{ marginTop: 40 }}>
          Five measures
        </h3>
        {summary.state === "ok" && <SystemKey aside="The larger dot marks the best of the three." />}
        {summary.state === "ok" && (
          <p className="note" data-note="scales">
            Each row has its own scale, from 0 to the value printed at the right-hand end of its track.
          </p>
        )}
        {renderPart(summary, (rows) => (
          <FiveMeasures summary={usableSummary(rows)} footnote={labelRows.state === "ok" ? automationFootnote(labelRows.data) : null} />
        ))}
      </div>

      <div className="sec">
        <h2>How JEV compares, measure by measure</h2>
        {renderPart(contrasts, (rows) => {
          const plot = comparisonRows(rows);
          const note = comparisonNote(plottedValues(plot));
          return (
            <>
              <p className="sub">
                The dot is the difference on the same test comments and the line is the 95% interval. Direction is
                normalised: the red side means JEV is worse, the lime side means JEV is better. {note} Precision is
                not like-for-like, because each system has automatic action on a different set of labels. Harmful
                comments actioned has no comparison here: the paired contrasts were specified for the review share,
                precision and the two miss rates only.
              </p>
              <ComparisonPlot rows={plot} />
            </>
          );
        })}
      </div>

      <div className="sec">
        <h2>Validation against test results at the fitted thresholds</h2>
        <p className="sub">
          Open circles are validation and filled circles are the test set. The dashed line is the {targetPhrase} and
          the number on the right is the change from validation to test.
        </p>
        {renderPart(labelRows, (rows) => (
          <Agreement rows={rows} />
        ))}
      </div>

      <div className="sec">
        <div className="finding">
          <h3>How far to trust the targets</h3>
          <p className="sub" style={{ margin: "6px 0 0" }}>
            {trustText(targets, labelRows.state === "ok" ? labelRows.data : [])}
          </p>
        </div>
      </div>
    </>
  );
}
