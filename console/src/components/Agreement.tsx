import { useState } from "react";

import { SYSTEM_IDS, SYSTEM_NAMES } from "../api";
import type { PolicyLabelRow, SystemId } from "../api";
import { f3, signed3, words } from "../format";
import { agreementReadings } from "../narrative";
import { agreementAxisMin, tenths } from "../scales";
import { systemStyle } from "../theme";
import { Segmented } from "./Segmented";

type Measure = "precision" | "recall";

interface PlotRow {
  key: string;
  system: SystemId;
  label: string;
  validation: number;
  test: number;
}

/** Rows for one plot: precision at HIGH where automatic action is on, recall at LOW for every label. */
export function agreementRows(rows: readonly PolicyLabelRow[], measure: Measure): PlotRow[] {
  const out: PlotRow[] = [];
  for (const system of SYSTEM_IDS) {
    for (const row of rows.filter((item) => item.system === system)) {
      const validation = measure === "precision" ? row.validation_precision_at_high : row.validation_recall_at_low;
      const test = measure === "precision" ? row.test_action_precision : row.test_recall_at_low;
      if (measure === "precision" && !row.action_enabled) continue;
      if (validation === null || test === null) continue;
      out.push({ key: `${system}|${row.label}`, system, label: words(row.label), validation, test });
    }
  }
  return out;
}

function Dumbbells({ rows, axisMin, target, id }: { rows: readonly PlotRow[]; axisMin: number; target: number | null; id: string }) {
  const position = (value: number): number => ((value - axisMin) / (1 - axisMin)) * 100;
  let last: SystemId | null = null;
  const body = rows.flatMap((row, index) => {
    const items = [];
    if (row.system !== last) {
      last = row.system;
      items.push(
        <div className="fp gh" key={`head-${row.system}`} style={systemStyle(row.system)}>
          <span>
            <i />
            {SYSTEM_NAMES[row.system]}
          </span>
          <span />
          <span />
        </div>,
      );
    }
    const change = row.test - row.validation;
    items.push(
      <div className="fp" key={row.key} style={systemStyle(row.system)} data-row={row.key}>
        <div className="rl">{row.label}</div>
        <div
          className="t"
          role="img"
          aria-label={`${SYSTEM_NAMES[row.system]}, ${row.label}: validation ${f3(row.validation)}, test ${f3(row.test)}`}
        >
          {target !== null && <i className="tgt" style={{ left: `${position(target)}%` }} />}
          <i
            className="ln"
            style={{
              left: `${position(Math.min(row.validation, row.test))}%`,
              width: `${position(Math.max(row.validation, row.test)) - position(Math.min(row.validation, row.test))}%`,
              animationDelay: `${index * 30}ms`,
            }}
          />
          <i className="d v" style={{ left: `${position(row.validation)}%` }} title={`Validation ${f3(row.validation)}`} />
          <i className="d s" style={{ left: `${position(row.test)}%` }} title={`Test ${f3(row.test)}`} />
        </div>
        <div className={`dl${change >= 0 ? " up" : ""}`}>{change >= 0 ? `+${f3(change)}` : signed3(change)}</div>
      </div>,
    );
    return items;
  });
  return (
    <div id={id}>
      <div>{body}</div>
      <div className="axis">
        <span />
        <div className="t">
          {tenths(axisMin, 1).map((tick) => (
            <span key={tick} style={{ left: `${position(tick)}%` }}>
              {tick.toFixed(1)}
            </span>
          ))}
        </div>
        <span />
      </div>
    </div>
  );
}

const MEASURES: readonly { id: Measure; text: string }[] = [
  { id: "precision", text: "Precision of automatic action" },
  { id: "recall", text: "Recall at the review threshold" },
];

/** Validation (open circle) against test (filled circle) for each system and label, plus the readings. */
export function Agreement({ rows }: { rows: readonly PolicyLabelRow[] }) {
  const [measure, setMeasure] = useState<Measure>("precision");
  const axisMin = agreementAxisMin(rows);
  const targetFor = (field: "precision_target" | "recall_target"): number | null =>
    rows.find((row) => row[field] !== null)?.[field] ?? null;
  const readings = agreementReadings(rows);
  return (
    <>
      {readings.length > 0 && (
        <div className="reads">
          {readings.map((reading) => (
            <p key={reading.text} className={reading.tone} data-tone={reading.tone}>
              <i />
              {reading.text}
            </p>
          ))}
        </div>
      )}
      <Segmented label="Measure" variant="simple" options={MEASURES} value={measure} onChange={setMeasure} />
      {measure === "precision" ? (
        <Dumbbells id="fp-precision" rows={agreementRows(rows, "precision")} axisMin={axisMin} target={targetFor("precision_target")} />
      ) : (
        <Dumbbells id="fp-recall" rows={agreementRows(rows, "recall")} axisMin={axisMin} target={targetFor("recall_target")} />
      )}
    </>
  );
}
