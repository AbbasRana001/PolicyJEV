"""Qualitative error review on validation: ids and counts on disk, text only in the terminal.

Per label, seeded samples of up to 25 comments from four categories, at the validation-fitted
thresholds on calibrated scores: (a) JEV false negatives that both baselines catch, (b) JEV hits
that both baselines miss, (c) JEV-only false positives, and (d) possible label noise, where all
three systems agree with each other against the gold label.
"""

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

MAX_PER_GROUP = 25
CATEGORIES = [
    "jev_false_negative_both_baselines_catch",
    "jev_hit_both_baselines_miss",
    "jev_only_false_positive",
    "possible_label_noise",
]
WARNING = (
    "CONTENT WARNING: the comments below come from the Jigsaw dataset and contain abusive, "
    "hateful and threatening language. They are printed to the terminal only."
)


def category_masks(
    y_col: np.ndarray, logreg: np.ndarray, distilbert: np.ndarray, jev: np.ndarray
) -> dict[str, np.ndarray]:
    """Boolean masks of the four review categories for one label (predictions are 0/1)."""
    gold = np.asarray(y_col).astype(bool)
    a, b, c = (np.asarray(p).astype(bool) for p in (logreg, distilbert, jev))
    return {
        CATEGORIES[0]: gold & ~c & a & b,
        CATEGORIES[1]: gold & c & ~a & ~b,
        CATEGORIES[2]: ~gold & c & ~a & ~b,
        CATEGORIES[3]: (~gold & a & b & c) | (gold & ~a & ~b & ~c),
    }


def review_samples(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    ids: list[str],
    seed: int,
    max_per_group: int = MAX_PER_GROUP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Counts per label and category, and the sampled ids (label, category, id)."""
    rng = np.random.default_rng(seed)
    counts, sampled = [], []
    for j, label in enumerate(LABELS):
        masks = category_masks(
            y[:, j], predictions["logreg_word"][:, j], predictions["distilbert"][:, j],
            predictions["jev"][:, j],
        )
        for category in CATEGORIES:
            candidates = np.flatnonzero(masks[category])
            take = min(max_per_group, len(candidates))
            chosen = rng.choice(candidates, size=take, replace=False) if take else []
            counts.append(
                {"label": label, "category": category, "count": len(candidates), "sampled": take}
            )
            sampled.extend(
                {"label": label, "category": category, "id": ids[int(row)]} for row in chosen
            )
    return pd.DataFrame(counts), pd.DataFrame(sampled, columns=["label", "category", "id"])


def format_review(
    sampled: pd.DataFrame, texts: dict[str, str], scores: dict[str, np.ndarray],
    id_position: dict[str, int], show: int, max_chars: int = 400,
) -> list[str]:
    """Printable lines for the first `show` sampled comments of each label and category."""
    lines = [WARNING]
    for (label, category), group in sampled.groupby(["label", "category"], sort=False):
        j = LABELS.index(label)
        lines.append(f"\n== {label} | {category} ==")
        for comment_id in group["id"].head(show):
            row = id_position[comment_id]
            text = " ".join(texts[comment_id].split())[:max_chars]
            values = " ".join(f"{s}={scores[s][row, j]:.3f}" for s in scores)
            lines.append(f"id={comment_id} {values}\n    {text}")
    return lines


def main() -> None:
    """Write ids and counts to artifacts/metrics; optionally print example text."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--show", type=int, default=0, metavar="N")
    args = parser.parse_args()
    require_split_access(args.split)

    config = load_config()
    systems = ["logreg_word", "distilbert", "jev"]
    data = load_split_scores_bundle(args.split, systems)
    predictions = {
        s: (data.calibrated[s] >= data.thresholds[s]).astype(int) for s in systems
    }
    counts, sampled = review_samples(data.y, predictions, data.ids, config["seed"])

    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "validation" else f"_{args.split}"
    sampled.to_csv(metrics_dir / f"error_review_ids{suffix}.csv", index=False)
    counts.to_csv(metrics_dir / f"error_review_counts{suffix}.csv", index=False)

    print(
        f"split: {args.split}; JEV question set {config['jev']['primary_question_version']}; "
        "thresholds fit on out-of-fold calibrated validation scores"
    )
    print(counts.to_string(index=False))

    if args.show > 0:
        texts = load_texts(args.split, sampled["id"].tolist())
        position = {comment_id: i for i, comment_id in enumerate(data.ids)}
        print()
        print("\n".join(format_review(sampled, texts, data.calibrated, position, args.show)))


if __name__ == "__main__":
    main()
