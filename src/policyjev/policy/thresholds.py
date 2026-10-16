"""Policy thresholds (LOW and HIGH per label) selected on validation data only."""

import argparse
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config

SYSTEMS = ["logreg_word", "distilbert", "jev"]
RECALL_GRID = [0.80, 0.90, 0.95]
PRECISION_GRID = [0.80, 0.90, 0.95]
PRIMARY_RECALL = 0.90
PRIMARY_PRECISION = 0.90
SUPPORT_FLOOR = 30
TOLERANCE = 1e-9
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "policy" / "policy_thresholds.csv"
COLUMNS = [
    "system",
    "label",
    "recall_target",
    "precision_target",
    "is_primary",
    "low",
    "high",
    "action_enabled",
    "validation_positives",
    "validation_recall_at_low",
    "validation_precision_at_high",
    "validation_flagged_at_high",
    "validation_review_share_label",
]


@dataclass(frozen=True)
class LabelThresholds:
    """LOW and HIGH for one label, with the validation quantities at those thresholds."""

    low: float
    high: float
    action_enabled: bool
    positives: int
    recall_at_low: float
    precision_at_high: float
    flagged_at_high: int
    review_share: float


def select_label_thresholds(
    probabilities: np.ndarray,
    y: np.ndarray,
    recall_target: float,
    precision_target: float,
    support_floor: int = SUPPORT_FLOOR,
) -> LabelThresholds:
    """Select LOW and HIGH for one label from validation probabilities and labels.

    Candidates are the distinct probability values; a comment at or above a threshold
    counts as flagged. HIGH is the lowest candidate with precision at or above the target
    and at least support_floor flagged comments (infinity, with ACTION disabled, if none
    qualifies). LOW is the highest candidate with recall at or above the target; if LOW is
    above HIGH it is set to HIGH.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y).astype(int)
    total_positives = int(y.sum())
    if total_positives == 0:
        raise ValueError("a label with no validation positives cannot be thresholded")

    values, inverse = np.unique(p, return_inverse=True)
    counts = np.bincount(inverse)
    positives = np.bincount(inverse, weights=y)
    flagged = np.cumsum(counts[::-1])[::-1]
    true_positives = np.cumsum(positives[::-1])[::-1]

    high_ok = (true_positives >= precision_target * flagged - TOLERANCE) & (flagged >= support_floor)
    high = float(values[high_ok][0]) if high_ok.any() else math.inf
    low_ok = true_positives >= recall_target * total_positives - TOLERANCE
    low = float(values[low_ok][-1])
    if low > high:
        low = high

    at_low = p >= low
    recall_at_low = float(y[at_low].sum() / total_positives)
    action_enabled = math.isfinite(high)
    if action_enabled:
        at_high = p >= high
        flagged_at_high = int(at_high.sum())
        precision_at_high = float(y[at_high].sum() / flagged_at_high)
    else:
        flagged_at_high = 0
        precision_at_high = float("nan")
    return LabelThresholds(
        low=low,
        high=high,
        action_enabled=action_enabled,
        positives=total_positives,
        recall_at_low=recall_at_low,
        precision_at_high=precision_at_high,
        flagged_at_high=flagged_at_high,
        review_share=float(at_low.mean()),
    )


def build_policy_table(
    calibrated: Mapping[str, np.ndarray],
    y: np.ndarray,
    recall_grid: Iterable[float] = RECALL_GRID,
    precision_grid: Iterable[float] = PRECISION_GRID,
    support_floor: int = SUPPORT_FLOOR,
) -> pd.DataFrame:
    """Thresholds per system, grid point and label; the primary point is flagged."""
    rows = []
    for system, probabilities in calibrated.items():
        for recall_target in recall_grid:
            for precision_target in precision_grid:
                is_primary = (
                    math.isclose(recall_target, PRIMARY_RECALL)
                    and math.isclose(precision_target, PRIMARY_PRECISION)
                )
                for j, label in enumerate(LABELS):
                    chosen = select_label_thresholds(
                        probabilities[:, j], y[:, j], recall_target, precision_target, support_floor
                    )
                    rows.append(
                        {
                            "system": system,
                            "label": label,
                            "recall_target": recall_target,
                            "precision_target": precision_target,
                            "is_primary": is_primary,
                            "low": chosen.low,
                            "high": chosen.high,
                            "action_enabled": chosen.action_enabled,
                            "validation_positives": chosen.positives,
                            "validation_recall_at_low": chosen.recall_at_low,
                            "validation_precision_at_high": chosen.precision_at_high,
                            "validation_flagged_at_high": chosen.flagged_at_high,
                            "validation_review_share_label": chosen.review_share,
                        }
                    )
    return pd.DataFrame(rows, columns=COLUMNS)


def thresholds_for(
    table: pd.DataFrame, system: str, recall_target: float, precision_target: float
) -> pd.DataFrame:
    """The label, low, high rows of one system at one grid point (in label order)."""
    selected = table[
        (table["system"] == system)
        & np.isclose(table["recall_target"], recall_target)
        & np.isclose(table["precision_target"], precision_target)
    ]
    if len(selected) != len(LABELS):
        raise ValueError(f"expected {len(LABELS)} rows for {system} at {recall_target}/{precision_target}")
    return selected.set_index("label").loc[LABELS, ["low", "high"]].reset_index()


def refuse_test_paths(paths: Iterable[str | Path]) -> None:
    """Exit if any path, as given, contains 'test' (this CLI reads validation inputs only)."""
    for path in paths:
        if "test" in str(path).lower():
            raise SystemExit(f"refusing path {str(path)!r}: policy thresholds use validation inputs only")


def main() -> None:
    """Select policy thresholds on validation data and write artifacts/policy/policy_thresholds.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    from policyjev.evaluation.calibration import configured_methods
    from policyjev.evaluation.splits import load_split_scores_bundle, prediction_path

    version = load_config()["jev"]["primary_question_version"]
    refuse_test_paths(
        [args.out] + [prediction_path(s, "validation", version) for s in SYSTEMS]
    )

    data = load_split_scores_bundle("validation", SYSTEMS, configured_methods())
    table = build_policy_table({s: data.calibrated[s] for s in SYSTEMS}, data.y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    primary = table[table["is_primary"]]
    print(f"wrote {len(table)} rows to {out}")
    print(primary.drop(columns=["is_primary", "recall_target", "precision_target"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
