"""Split-aware comparison of the baselines and JEV (validation now, test behind the freeze tag)."""

import argparse

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.evaluation.review_budget import RARE_LABELS
from policyjev.evaluation.splits import (
    SPLITS,
    SplitScores,
    f1_at_thresholds,
    load_split_scores_bundle,
    require_split_access,
)
from policyjev.evaluation.stack import STACK_SYSTEM
from policyjev.evaluation.validation_report import DIFFERENCES, HEADER, SYSTEMS, build_comparison

OUTPUT_NAMES = {"validation": "validation_comparison.csv", "test": "test_comparison.csv"}
STACK_OUTPUT_NAMES = {
    "validation": "validation_comparison_with_stack.csv",
    "test": "test_comparison_with_stack.csv",
}
THRESHOLD_OUTPUT_NAMES = {
    "validation": "validation_threshold_metrics.csv",
    "test": "test_threshold_metrics.csv",
}
STACK_THRESHOLD_OUTPUT_NAMES = {
    "validation": "validation_threshold_metrics_with_stack.csv",
    "test": "test_threshold_metrics_with_stack.csv",
}
FIXED_THRESHOLD = 0.5
BASIS_FITTED = "validation_oof_fit"
BASIS_FIXED = "fixed_0.5"
STACK_NOTE = (
    "jev_stack is a secondary supervised system: a logistic regression on the six JEV probabilities "
    "fit on validation labels. It is not JEV and is never reported as JEV."
)


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall and F1 from counts; undefined ratios are 0.0."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def threshold_metrics(
    y: np.ndarray, calibrated: np.ndarray, thresholds: np.ndarray, system: str, basis: str,
    sample_status: str,
) -> list[dict]:
    """Per-label precision, recall and F1, plus micro F1 and macro F1, at the given thresholds."""
    predicted = calibrated >= np.asarray(thresholds)
    actual = y.astype(bool)
    tp = (predicted & actual).sum(axis=0)
    fp = (predicted & ~actual).sum(axis=0)
    fn = (~predicted & actual).sum(axis=0)

    def row(label: str, positives: float, threshold: float, values: tuple[float, float, float], rare: bool):
        return {
            "threshold_basis": basis,
            "system": system,
            "label": label,
            "is_rare_label": rare,
            "positives": positives,
            "threshold": threshold,
            "precision": values[0],
            "recall": values[1],
            "f1": values[2],
            "sample_status": sample_status,
        }

    rows = [
        row(
            label, int(actual[:, j].sum()), float(np.asarray(thresholds)[j]),
            _prf(tp[j], fp[j], fn[j]), label in RARE_LABELS,
        )
        for j, label in enumerate(LABELS)
    ]
    per_label_f1 = [r["f1"] for r in rows]
    rows.append(
        row("micro", int(actual.sum()), float("nan"), _prf(tp.sum(), fp.sum(), fn.sum()), False)
    )
    rows.append(
        row("macro", float("nan"), float("nan"),
            (float("nan"), float("nan"), float(np.mean(per_label_f1))), False)
    )
    return rows


def threshold_metrics_table(data: SplitScores, systems: list[str]) -> pd.DataFrame:
    """Point-estimate metrics at the frozen thresholds and at a fixed 0.5, per system."""
    if data.split == "validation":
        fitted_status = "in-sample (thresholds were fit on these rows)"
    else:
        fitted_status = "out-of-sample (thresholds were fit on validation)"
    rows = []
    for system in systems:
        rows += threshold_metrics(
            data.y, data.calibrated[system], data.thresholds[system], system, BASIS_FITTED, fitted_status
        )
        rows += threshold_metrics(
            data.y, data.calibrated[system], np.full(len(LABELS), FIXED_THRESHOLD), system,
            BASIS_FIXED, "not fitted",
        )
    return pd.DataFrame(rows)


def run_split(
    split: str, n_resamples: int, seed: int, confidence: float, with_stack: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The comparison table and the threshold-metrics table for a split (data loaded once)."""
    systems = SYSTEMS + [STACK_SYSTEM] if with_stack else SYSTEMS
    differences = DIFFERENCES + [(STACK_SYSTEM, s) for s in SYSTEMS] if with_stack else DIFFERENCES
    data = load_split_scores_bundle(split, systems)
    comparison = _comparison_from_bundle(data, systems, differences, n_resamples, seed, confidence)
    return comparison, threshold_metrics_table(data, systems)


def comparison_for_split(
    split: str, n_resamples: int, seed: int, confidence: float, with_stack: bool = False
) -> pd.DataFrame:
    """The comparison table for a split, using calibrated probabilities for Brier."""
    return run_split(split, n_resamples, seed, confidence, with_stack)[0]


def _comparison_from_bundle(
    data: SplitScores,
    systems: list[str],
    differences: list[tuple[str, str]],
    n_resamples: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    """Build the comparison table from already loaded split scores."""
    split = data.split
    if split == "validation":
        f1, prefix = data.validation_f1, "f1_in_sample"
    else:
        f1 = {
            system: f1_at_thresholds(data.calibrated[system], data.y, data.thresholds[system])
            for system in systems
        }
        prefix = "f1"
    return build_comparison(
        data.y, data.raw, f1, n_resamples, seed, confidence,
        calibrated=data.calibrated, f1_prefix=prefix, systems=systems, differences=differences,
    )


def comparison_header(split: str) -> str:
    """Console header for a split: its title line, the method notes, and the F1 status."""
    lines = HEADER.split("\n")
    if split == "test":
        lines[0] = "Full-test comparison (all test rows, unweighted)."
    f1_line = (
        "F1 is measured at validation-fitted thresholds and is in-sample."
        if split == "validation"
        else "F1 uses thresholds fit on validation and is out-of-sample."
    )
    return "\n".join(lines[:-1] + [f1_line])


def main() -> None:
    """Write the comparison CSV for a split and print it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()
    require_split_access(args.split)

    config = load_config()
    table, threshold_table = run_split(
        args.split,
        config["bootstrap"]["n_resamples"],
        config["seed"],
        config["bootstrap"]["confidence_level"],
        with_stack=args.with_stack,
    )
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    names = STACK_OUTPUT_NAMES if args.with_stack else OUTPUT_NAMES
    table.to_csv(metrics_dir / names[args.split], index=False)
    threshold_names = STACK_THRESHOLD_OUTPUT_NAMES if args.with_stack else THRESHOLD_OUTPUT_NAMES
    threshold_table.to_csv(metrics_dir / threshold_names[args.split], index=False)

    print(comparison_header(args.split))
    print(f"split: {args.split}; JEV question set {config['jev']['primary_question_version']}")
    if args.with_stack:
        print(STACK_NOTE)
    print(
        "Brier is on calibrated probabilities (out-of-fold on validation, saved calibrators on test); "
        "brier_raw_* is on the raw scores."
    )
    print()
    print(table.round(4).to_string(index=False))
    print()
    print(
        f"Threshold metrics ({threshold_names[args.split]}): point estimates at the frozen per-label "
        f"thresholds (threshold_basis={BASIS_FITTED}) and at a fixed {FIXED_THRESHOLD} on calibrated "
        f"probabilities (threshold_basis={BASIS_FIXED}). macro F1 is the mean of per-label F1; "
        "positives are the raw counts, and rare labels are flagged in is_rare_label."
    )
    if args.split == "validation":
        print("On validation the fitted-threshold numbers are in-sample (see sample_status).")
    print(threshold_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
