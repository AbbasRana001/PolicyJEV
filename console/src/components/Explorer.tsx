import { useRef, useState } from "react";
import type { PointerEvent } from "react";

import { LABELS, SYSTEM_IDS, SYSTEM_NAMES } from "../api";
import type { Label, SystemId, ThresholdRow } from "../api";
import {
  DECISION_WORDS,
  ZONE_CLASS,
  decideComment,
  decideLabel,
  decisionWhy,
  probabilityFromSlider,
  sqrtPosition,
  systemThresholds,
  zoneWidths,
} from "../decision";
import { capitalise, f4, formatCount, joinList, percent0, words } from "../format";
import { cssVars } from "../theme";
import { DataNotAvailable } from "./DataNotAvailable";
import { Segmented } from "./Segmented";

export const DEFAULT_SLIDER = 224;
const SCALE_TICKS = [0, 0.01, 0.05, 0.25, 0.5, 1];

/** A tick label centred on its position, except at the two ends, where it stays inside the box (not cut off). */
export function tickStyle(position: number): { left: string; transform?: string } {
  if (position <= 0) return { left: "0%", transform: "none" };
  if (position >= 1) return { left: "100%", transform: "translateX(-100%)" };
  return { left: `${position * 100}%` };
}

interface ExplorerProps {
  thresholds: ThresholdRow[] | null | undefined;
  targets: { recall: number; precision: number } | null;
  /** The fewest validation comments HIGH had to be supported by, from the API; null if not known. */
  supportFloor: number | null;
}

/** The two target percentages for the copy, or neutral wording if the targets are not known. */
export function targetWords(targets: ExplorerProps["targets"]): { recall: string; precision: string } {
  return targets === null
    ? { recall: "the recall target", precision: "the precision target" }
    : { recall: percent0(targets.recall), precision: percent0(targets.precision) };
}

export function Explorer({ thresholds, targets, supportFloor }: ExplorerProps) {
  const [system, setSystem] = useState<SystemId>("logreg_word");
  const [slider, setSlider] = useState(DEFAULT_SLIDER);
  const dragging = useRef<HTMLElement | null>(null);
  const copy = targetWords(targets);

  const probability = probabilityFromSlider(slider);
  const rules = thresholds ? systemThresholds(thresholds, system) : null;

  const setFromPointer = (event: PointerEvent<HTMLElement>, bar: HTMLElement): void => {
    const box = bar.getBoundingClientRect();
    if (box.width === 0) return;
    const fraction = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    setSlider(Math.round(fraction * 1000));
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>): void => {
    const bar = (event.target as HTMLElement).closest<HTMLElement>(".xbar");
    if (bar === null) return;
    dragging.current = bar;
    bar.setPointerCapture?.(event.pointerId);
    setFromPointer(event, bar);
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>): void => {
    if (dragging.current !== null) setFromPointer(event, dragging.current);
  };
  const endDrag = (): void => {
    dragging.current = null;
  };

  const noHigh = rules === null ? [] : LABELS.filter((label) => rules[label].high === null);
  const outcome =
    rules === null
      ? null
      : decideComment(Object.fromEntries(LABELS.map((label) => [label, probability])) as Record<Label, number>, rules);

  return (
    <div className="explore">
      <div className="ex-top">
        <Segmented
          label="System"
          options={SYSTEM_IDS.map((id) => ({ id, text: SYSTEM_NAMES[id] }))}
          value={system}
          onChange={setSystem}
        />
        <div className={`outcome ${outcome === null ? "" : ZONE_CLASS[outcome.decision]}`} aria-live="polite" data-outcome={outcome?.decision}>
          {outcome === null ? (
            <span className="ot">Thresholds not loaded.</span>
          ) : (
            <>
              <span className="ow">{DECISION_WORDS[outcome.decision]}</span>
              <span className="ot">{decisionWhy(outcome)}</span>
            </>
          )}
        </div>
      </div>

      <div className="xr xscrub">
        <div className="lbl">
          Probability
          <br />
          <span className="small">applied to every label</span>
        </div>
        <div>
          <input
            className="pr"
            id="probe"
            type="range"
            min={0}
            max={1000}
            step={1}
            value={slider}
            aria-label="Probability every label scores"
            aria-valuetext={`probability ${f4(probability)}`}
            onChange={(event) => setSlider(Number(event.target.value))}
          />
          <div className="pticks" aria-hidden="true">
            {SCALE_TICKS.map((tick) => (
              <span key={tick} style={tickStyle(sqrtPosition(tick))}>
                {tick}
              </span>
            ))}
          </div>
        </div>
        <div className="xend">
          <span className="pval" data-probability>
            {f4(probability)}
          </span>
        </div>
      </div>

      {rules === null ? (
        <DataNotAvailable detail="The thresholds could not be loaded from the API." />
      ) : (
        <div key={system} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endDrag} onPointerCancel={endDrag}>
          {LABELS.map((label) => {
            const rule = rules[label];
            const widths = zoneWidths(rule);
            const decision = decideLabel(probability, rule.low, rule.high);
            const lowAt = widths.allow;
            const highAt = widths.allow + widths.review;
            return (
              <div className="xr" key={label} data-l={label} data-decision={decision}>
                <div className="lbl">{words(label)}</div>
                <div>
                  <div className="xbar" aria-hidden="true">
                    <div className="zs">
                      <div className={`za${decision === "ALLOW" ? " hit" : ""}`} data-z="allow" style={{ width: `${widths.allow}%` }} />
                      <div className={`zr${decision === "HUMAN REVIEW" ? " hit" : ""}`} data-z="review" style={{ width: `${widths.review}%` }} />
                      {rule.high !== null && (
                        <div className={`zx${decision === "ACTION" ? " hit" : ""}`} data-z="action" style={{ width: `${widths.action}%` }} />
                      )}
                    </div>
                    <i className="mk" style={cssVars({ left: `${sqrtPosition(probability) * 100}%` })} />
                  </div>
                  <div className="xsc">
                    <span style={{ left: `${lowAt}%` }}>LOW {f4(rule.low)}</span>
                    {rule.high !== null ? (
                      <span style={{ left: `${highAt}%` }}>HIGH {f4(rule.high)}</span>
                    ) : (
                      <span className="ro">no HIGH threshold</span>
                    )}
                  </div>
                </div>
                <div className="xend">
                  <span className={`dchip ${ZONE_CLASS[decision]}`}>{DECISION_WORDS[decision]}</span>
                  <span className="nt">{rule.high !== null ? "Can trigger action" : "Cannot trigger action"}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="zlegend">
        <p className="za">
          <b>
            <i />
            Allow
          </b>
          The probability is below LOW. Fitted so that at least {copy.recall} of true positives on validation
          are at or above it.
        </p>
        <p className="zr">
          <b>
            <i />
            Human review
          </b>
          From LOW up to HIGH. A person looks at the comment and decides.
        </p>
        <p className="zx">
          <b>
            <i />
            Action
          </b>
          At or above HIGH, where precision on validation was at least {copy.precision} with{" "}
          {supportFloor === null ? "enough comments" : `at least ${formatCount(supportFloor)} comments`}.{" "}
          {noHigh.length === 0
            ? "Every label has a HIGH threshold for this system."
            : `${capitalise(joinList(noHigh.map(words)))} ${noHigh.length === 1 ? "has" : "have"} no HIGH threshold for this system: no threshold met that.`}
        </p>
      </div>
    </div>
  );
}
