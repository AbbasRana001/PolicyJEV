"""Summarize cached pilot judgments against the pilot's true labels."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.jev.cache import JevCache
from policyjev.jev.client import cache_key, normalize_text
from policyjev.jev.runner import _load_questions, _load_rows


def collect_judgments(
    rows: Sequence[tuple[str, str]],
    questions: dict[str, dict],
    question_set_version: str,
    model: str,
    cache: JevCache,
) -> tuple[pd.DataFrame, int]:
    """Return (probabilities indexed by id, count of rows missing from the cache)."""
    records: dict[str, dict[str, float]] = {}
    missing = 0
    for comment_id, text in rows:
        key = cache_key(normalize_text(text), question_set_version, model, questions)
        stored = cache.get(key)
        if stored is None:
            missing += 1
        else:
            records[comment_id] = stored["answers"]
    probs = pd.DataFrame.from_dict(records, orient="index", columns=list(questions))
    probs.index.name = "id"
    return probs, missing


def pilot_report_table(probs: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per label from aligned probability and label frames."""
    rows = []
    for label in LABELS:
        p = probs[label].to_numpy(dtype=float)
        y = labels.loc[probs.index, label].to_numpy(dtype=int)
        positives = p[y == 1]
        negatives = p[y == 0]
        has_both = 0 < y.sum() < len(y)
        rows.append(
            {
                "label": label,
                "n_pos": int(y.sum()),
                "average_precision": (
                    float(average_precision_score(y, p)) if has_both else float("nan")
                ),
                "mean_prob_positives": (
                    float(positives.mean()) if len(positives) else float("nan")
                ),
                "mean_prob_negatives": (
                    float(negatives.mean()) if len(negatives) else float("nan")
                ),
                "min_prob": float(p.min()) if len(p) else float("nan"),
                "median_prob": float(np.median(p)) if len(p) else float("nan"),
                "max_prob": float(p.max()) if len(p) else float("nan"),
                "share_exactly_0_or_1": (
                    float(((p == 0.0) | (p == 1.0)).mean()) if len(p) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Write artifacts/metrics/jev_pilot_report.csv from the cached pilot judgments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config()["jev"]
    version, questions = _load_questions(args.questions)
    rows = _load_rows("pilot", None)
    labels = load_train().set_index("id")[LABELS]

    with JevCache(REPO_ROOT / cfg["cache_path"]) as cache:
        probs, missing = collect_judgments(rows, questions, version, cfg["model"], cache)

    report = pilot_report_table(probs, labels)
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(metrics_dir / "jev_pilot_report.csv", index=False)

    print(f"question set version: {version}, judged comments: {len(probs)}")
    print(report.round(4).to_string(index=False))
    print(f"pilot comments missing from the cache: {missing}")


if __name__ == "__main__":
    main()
