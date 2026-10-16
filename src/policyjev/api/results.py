"""Aggregate results for the console, built only from committed metric files (no predictions, no text)."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT
from policyjev.evaluation.results_tables import CONTRASTS, SYSTEMS, load_frames
from policyjev.policy.thresholds import SUPPORT_FLOOR

METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"
PER_LABEL_COLUMNS = {
    "system", "label", "positives", "action_enabled", "action_precision_label", "miss_rate_label",
}
THRESHOLD_COLUMNS = {
    "system", "label", "is_primary", "recall_target", "precision_target",
    "validation_recall_at_low", "validation_precision_at_high",
}


def clean(value: object) -> object:
    """JSON-safe scalar: NaN and infinity become None, numpy types become Python types."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def records(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Rows of the selected columns as plain dictionaries."""
    return [{c: clean(row[c]) for c in columns} for _, row in frame.iterrows()]


def row_for(frame: pd.DataFrame, label: str) -> pd.Series | None:
    """The row of a comparison table for a label (or 'mean'), if present."""
    match = frame[frame["label"] == label]
    return None if match.empty else match.iloc[0]


def _is_true(value: object) -> bool:
    """Read a boolean column that may have been stored as text."""
    return str(value).strip().lower() == "true"


def policy_label_rows(per_label: pd.DataFrame, thresholds: pd.DataFrame) -> list[dict]:
    """Validation and test values per system and label at the primary targets.

    Validation recall at LOW and validation precision at HIGH come from policy_thresholds.csv
    (primary rows). Test precision at HIGH is action_precision_label from
    policy_test_per_label.csv, and test recall at LOW is one minus its miss_rate_label (the
    positives that fell below LOW). Nothing is recomputed from predictions.
    """
    primary = thresholds[thresholds["is_primary"].map(_is_true)]
    rows = []
    for _, test in per_label.iterrows():
        match = primary[(primary["system"] == test["system"]) & (primary["label"] == test["label"])]
        if match.empty:
            continue
        validation = match.iloc[0]
        miss = float(test["miss_rate_label"])
        rows.append(
            {
                "system": test["system"],
                "label": test["label"],
                "positives": clean(test["positives"]),
                "action_enabled": _is_true(test["action_enabled"]),
                "recall_target": clean(validation["recall_target"]),
                "precision_target": clean(validation["precision_target"]),
                "validation_recall_at_low": clean(validation["validation_recall_at_low"]),
                "test_recall_at_low": clean(1.0 - miss) if math.isfinite(miss) else None,
                "validation_precision_at_high": clean(validation["validation_precision_at_high"]),
                "test_action_precision": clean(test["action_precision_label"]),
            }
        )
    return rows


def build_results_payload(frames: dict[str, pd.DataFrame | None]) -> dict:
    """The aggregate payload; sections whose source files are missing are listed, not guessed."""
    payload: dict = {}
    missing: list[str] = []
    comparison = frames.get("comparison")

    if comparison is None:
        missing += ["mean_ap", "per_label_ap", "ap_differences"]
    else:
        mean = row_for(comparison, "mean")
        payload["mean_ap"] = [
            {
                "system": s,
                "ap": clean(mean[f"ap_{s}"]),
                "ap_lo": clean(mean[f"ap_{s}_lo"]),
                "ap_hi": clean(mean[f"ap_{s}_hi"]),
            }
            for s in SYSTEMS
        ]
        payload["per_label_ap"] = [
            {
                "label": label,
                "positives": clean(row_for(comparison, label)["positives"]),
                "system": s,
                "ap": clean(row_for(comparison, label)[f"ap_{s}"]),
                "ap_lo": clean(row_for(comparison, label)[f"ap_{s}_lo"]),
                "ap_hi": clean(row_for(comparison, label)[f"ap_{s}_hi"]),
            }
            for label in LABELS
            for s in SYSTEMS
        ]
        payload["ap_differences"] = [
            {
                "label": label,
                "contrast": f"{a} - {b}",
                "difference": clean(row_for(comparison, label)[f"diff_ap_{a}_minus_{b}"]),
                "lo": clean(row_for(comparison, label)[f"diff_ap_{a}_minus_{b}_lo"]),
                "hi": clean(row_for(comparison, label)[f"diff_ap_{a}_minus_{b}_hi"]),
            }
            for label in LABELS + ["mean"]
            for a, b, _ in CONTRASTS
        ]

    summary = frames.get("policy_summary")
    if summary is None:
        missing.append("policy_summary")
    else:
        payload["policy_summary"] = records(
            summary,
            [
                "system", "n_rows", "n_failed_rows", "n_allow", "n_review", "n_action", "allow_share",
                "m1_review_share", "m2_action_precision", "m3_miss_rate", "m4_rare_miss_rate",
                "action_share", "action_coverage",
            ],
        )

    contrasts = frames.get("policy_contrasts")
    if contrasts is None:
        missing.append("policy_contrasts")
    else:
        payload["policy_contrasts"] = records(
            contrasts,
            ["contrast", "metric", "difference", "ci_low", "ci_high", "resamples_used", "resamples_left_out"],
        )

    per_label = frames.get("policy_per_label")
    thresholds = frames.get("policy_thresholds")
    if (
        per_label is None
        or thresholds is None
        or not PER_LABEL_COLUMNS <= set(per_label.columns)
        or not THRESHOLD_COLUMNS <= set(thresholds.columns)
    ):
        missing.append("policy_per_label")
    else:
        payload["policy_per_label"] = policy_label_rows(per_label, thresholds)

    budget = frames.get("budget")
    if budget is None:
        missing.append("review_budget")
    else:
        payload["review_budget"] = records(
            budget, ["budget_pct", "metric", "comparison", "value", "lo", "hi", "status"]
        )

    # The minimum number of validation comments HIGH had to be supported by when it was fitted.
    payload["policy_fit"] = {"support_floor": SUPPORT_FLOOR}

    payload["missing_sections"] = missing
    return payload


def load_results_payload(metrics_dir: Path = METRICS_DIR) -> dict:
    """Read the committed metric files and build the payload."""
    return build_results_payload(load_frames(metrics_dir))
