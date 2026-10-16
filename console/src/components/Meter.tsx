import { decideLabel, sqrtPosition, ZONE_CLASS } from "../decision";
import type { LabelThresholds } from "../decision";
import { f4 } from "../format";
import { cssVars } from "../theme";

export const METER_CELLS = 40;

/** Number of lit cells for a probability on the square-root scale. */
export function litCells(probability: number): number {
  return Math.round(sqrtPosition(probability) * METER_CELLS);
}

/** The zone of cell i: the decision for the probability at the centre of that cell. */
export function cellZone(index: number, thresholds: LabelThresholds): string {
  const centre = ((index + 0.5) / METER_CELLS) ** 2;
  return ZONE_CLASS[decideLabel(centre, thresholds.low, thresholds.high)];
}

interface MeterProps {
  thresholds: LabelThresholds;
  /** Calibrated probability to light up to; null leaves every cell dim. */
  probability: number | null;
  label: string;
}

/**
 * A 40-cell bar on a square-root scale. The colour of each cell is the decision that probability
 * would trigger for this label; lit cells show the score; ticks mark LOW and HIGH.
 */
export function Meter({ thresholds, probability, label }: MeterProps) {
  const lit = probability === null ? 0 : litCells(probability);
  const title =
    thresholds.high === null
      ? `LOW ${f4(thresholds.low)}, no HIGH threshold`
      : `LOW ${f4(thresholds.low)}, HIGH ${f4(thresholds.high)}`;
  return (
    <div className="meter" title={title} role="img" aria-label={`${label}. ${title}`}>
      {Array.from({ length: METER_CELLS }, (_, index) => (
        <i
          key={index}
          data-lit={index < lit ? "true" : undefined}
          className={`c ${cellZone(index, thresholds)}${index < lit ? " lit" : ""}${index === lit - 1 && lit > 0 ? " peak" : ""}`}
          style={cssVars({ "--i": index })}
        />
      ))}
      <b className="tk" style={{ left: `${sqrtPosition(thresholds.low) * 100}%` }} />
      {thresholds.high !== null && <b className="tk" style={{ left: `${sqrtPosition(thresholds.high) * 100}%` }} />}
    </div>
  );
}
