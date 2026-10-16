"""Run-to-run variation: compare the first and second JEV passes on the second-pass test rows."""

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.evaluation.calibration import apply_calibrator, calibrator_path, load_calibrators
from policyjev.evaluation.splits import (
    f1_at_thresholds,
    load_split_scores_bundle,
    require_split_access,
)
from policyjev.evaluation.validation_report import average_precision
from policyjev.jev.second_pass import second_pass_filename


def compare_passes(
    y: np.ndarray,
    first_raw: np.ndarray,
    second_raw: np.ndarray,
    first_calibrated: np.ndarray,
    second_calibrated: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Per label (plus mean): probability change, and change in AP and in F1."""
    rows = []
    for j, label in enumerate(LABELS):
        ap_first = average_precision(y[:, j], first_raw[:, j])
        ap_second = average_precision(y[:, j], second_raw[:, j])
        rows.append(
            {
                "label": label,
                "positives": int(y[:, j].sum()),
                "mean_abs_probability_change": float(np.abs(first_raw[:, j] - second_raw[:, j]).mean()),
                "share_rows_changed": float((first_raw[:, j] != second_raw[:, j]).mean()),
                "ap_first": ap_first,
                "ap_second": ap_second,
                "ap_change": ap_second - ap_first,
            }
        )
    f1_first = f1_at_thresholds(first_calibrated, y, thresholds)
    f1_second = f1_at_thresholds(second_calibrated, y, thresholds)
    table = pd.DataFrame(rows)
    table["f1_first"] = f1_first
    table["f1_second"] = f1_second
    table["f1_change"] = f1_second - f1_first

    mean_row = {"label": "mean", "positives": np.nan}
    for column in table.columns:
        if column not in mean_row:
            mean_row[column] = float(np.nanmean(table[column]))
    return pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)


def main() -> None:
    """Write artifacts/metrics/second_pass_report.csv and print it."""
    require_split_access("test")
    config = load_config()
    version = config["jev"]["primary_question_version"]

    second_path = REPO_ROOT / "artifacts" / "predictions" / second_pass_filename(version)
    if not second_path.exists():
        raise SystemExit(f"{second_path.name} not found; run python -m policyjev.jev.second_pass first")
    second = pd.read_csv(second_path, dtype={"id": str}).set_index("id")

    data = load_split_scores_bundle("test", ["jev"])
    position = {comment_id: i for i, comment_id in enumerate(data.ids)}
    missing = [i for i in second.index if i not in position]
    if missing:
        raise SystemExit(f"{len(missing)} second-pass ids are not test rows")
    rows = [position[i] for i in second.index]

    saved = load_calibrators(calibrator_path("jev", version))
    second_raw = second[LABELS].to_numpy(dtype=float)
    second_calibrated = np.column_stack(
        [apply_calibrator(saved["labels"][label], second_raw[:, j]) for j, label in enumerate(LABELS)]
    )

    table = compare_passes(
        data.y[rows],
        data.raw["jev"][rows],
        second_raw,
        data.calibrated["jev"][rows],
        second_calibrated,
        data.thresholds["jev"],
    )
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / "second_pass_report.csv", index=False)

    print(f"JEV question set {version}: first pass (canonical) against second pass on {len(rows)} test rows")
    print("F1 uses thresholds fit on validation; the first pass stays the reported result.")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
