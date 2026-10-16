"""Check whether repeated JEV calls on the same input return identical probabilities."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import load_config
from policyjev.jev.client import JevClient, get_api_key
from policyjev.jev.runner import _load_questions, _load_rows


def check_determinism(
    client: JevClient,
    rows: Sequence[tuple[str, str]],
    questions: dict[str, dict],
    repeats: int,
) -> pd.DataFrame:
    """Call the client directly `repeats` times per comment; never touches the cache."""
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    question_ids = list(questions)
    spread = np.zeros((len(rows), len(question_ids)))
    for i, (_, text) in enumerate(rows):
        results = [client.evaluate(text, questions) for _ in range(repeats)]
        runs = np.array([[r.answers[q] for q in question_ids] for r in results])
        spread[i] = runs.max(axis=0) - runs.min(axis=0)
    return pd.DataFrame(
        {
            "question": question_ids,
            "max_abs_difference": spread.max(axis=0) if len(rows) else 0.0,
            "comments_with_difference": (spread > 0).sum(axis=0),
            "comments": len(rows),
        }
    )


def main() -> None:
    """Run the determinism check on the first n pilot comments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config()["jev"]
    _, questions = _load_questions(args.questions)
    rows = _load_rows("pilot", args.n)

    client = JevClient.from_config(cfg, get_api_key())
    report = check_determinism(client, rows, questions, args.repeats)
    print(f"comments: {len(rows)}, repeats per comment: {args.repeats}")
    print("these calls bypass the cache, so their cost is not counted toward the hard stop")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
