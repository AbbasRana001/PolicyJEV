"""Where JEV and DistilBERT agree and disagree, per label, at validation-fitted thresholds."""

import argparse

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.evaluation.splits import (
    SPLITS,
    load_split_scores_bundle,
    load_texts,
    require_split_access,
)

CATEGORIES = ["both_right", "only_jev_right", "only_distilbert_right", "both_wrong"]
ROW_CLASSES = {"positive": 1, "negative": 0}


def category_masks(
    y_col: np.ndarray, jev_pred: np.ndarray, distilbert_pred: np.ndarray
) -> dict[str, np.ndarray]:
    """Boolean masks of the four agreement categories for one label."""
    jev_right = jev_pred == y_col
    distilbert_right = distilbert_pred == y_col
    return {
        "both_right": jev_right & distilbert_right,
        "only_jev_right": jev_right & ~distilbert_right,
        "only_distilbert_right": ~jev_right & distilbert_right,
        "both_wrong": ~jev_right & ~distilbert_right,
    }


def disagreement_counts(
    y: np.ndarray, jev_pred: np.ndarray, distilbert_pred: np.ndarray
) -> pd.DataFrame:
    """Counts per label, separately for positive and negative rows."""
    rows = []
    for j, label in enumerate(LABELS):
        masks = category_masks(y[:, j], jev_pred[:, j], distilbert_pred[:, j])
        for row_class, value in ROW_CLASSES.items():
            in_class = y[:, j] == value
            rows.append(
                {
                    "label": label,
                    "row_class": row_class,
                    **{c: int((masks[c] & in_class).sum()) for c in CATEGORIES},
                }
            )
    return pd.DataFrame(rows)


def sample_examples(
    y: np.ndarray,
    jev_pred: np.ndarray,
    distilbert_pred: np.ndarray,
    n_per_group: int,
    seed: int,
) -> list[tuple[str, str, str, int]]:
    """Seeded random row indices per (label, row class, category): (label, class, category, row)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for j, label in enumerate(LABELS):
        masks = category_masks(y[:, j], jev_pred[:, j], distilbert_pred[:, j])
        for row_class, value in ROW_CLASSES.items():
            for category in CATEGORIES:
                candidates = np.flatnonzero(masks[category] & (y[:, j] == value))
                take = min(n_per_group, len(candidates))
                for row in rng.choice(candidates, size=take, replace=False):
                    chosen.append((label, row_class, category, int(row)))
    return chosen


def format_examples(
    examples: list[tuple[str, str, str, int]],
    ids: list[str],
    texts: dict[str, str],
    jev_scores: np.ndarray,
    distilbert_scores: np.ndarray,
    max_chars: int = 400,
) -> list[str]:
    """Printable lines for the chosen rows (text is only ever printed, never saved)."""
    lines = []
    for label, row_class, category, row in examples:
        j = LABELS.index(label)
        text = " ".join(texts[ids[row]].split())[:max_chars]
        lines.append(
            f"[{label} | {row_class} | {category}] id={ids[row]} "
            f"jev={jev_scores[row, j]:.3f} distilbert={distilbert_scores[row, j]:.3f}\n    {text}"
        )
    return lines


def main() -> None:
    """Write artifacts/metrics/disagreement_<split>.csv; optionally print example rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--show", type=int, default=0, metavar="N")
    args = parser.parse_args()
    require_split_access(args.split)

    config = load_config()
    data = load_split_scores_bundle(args.split, ["distilbert", "jev"])
    jev_pred = data.calibrated["jev"] >= data.thresholds["jev"]
    distilbert_pred = data.calibrated["distilbert"] >= data.thresholds["distilbert"]

    table = disagreement_counts(data.y, jev_pred.astype(int), distilbert_pred.astype(int))
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / f"disagreement_{args.split}.csv", index=False)

    print(
        f"split: {args.split}; JEV question set {config['jev']['primary_question_version']}; "
        "thresholds fit on out-of-fold calibrated validation scores"
    )
    print(table.to_string(index=False))

    if args.show > 0:
        examples = sample_examples(
            data.y, jev_pred.astype(int), distilbert_pred.astype(int), args.show, config["seed"]
        )
        texts = load_texts(args.split, [data.ids[row] for *_, row in examples])
        print()
        print("\n".join(
            format_examples(
                examples, data.ids, texts, data.calibrated["jev"], data.calibrated["distilbert"]
            )
        ))


if __name__ == "__main__":
    main()
