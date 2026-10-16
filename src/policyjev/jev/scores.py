"""Export cached JEV judgments for a set as a prediction file (ids and probabilities only)."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.freeze_manifest import verify_files
from policyjev.jev.cache import JevCache
from policyjev.jev.pilot_report import collect_judgments
from policyjev.jev.runner import SETS, _check_guards, _load_questions, _load_rows

OUTPUT_PREFIXES = {"val_full": "jev_validation", "test_full": "jev_test"}


def scores_filename(set_name: str, version: str) -> str:
    """File name of the exported scores for a set and question set version."""
    prefix = OUTPUT_PREFIXES.get(set_name, f"jev_{set_name}")
    return f"{prefix}_{version}.csv"


def load_scores(
    rows: Sequence[tuple[str, str]],
    questions: dict[str, dict],
    question_set_version: str,
    model: str,
    cache: JevCache,
) -> pd.DataFrame:
    """Return id plus one probability column per label, in the order of rows."""
    absent = [label for label in LABELS if label not in questions]
    if absent:
        raise ValueError(f"question set has no question for: {', '.join(absent)}")

    probs, missing = collect_judgments(rows, questions, question_set_version, model, cache)
    if missing:
        raise ValueError(
            f"{missing} of {len(rows)} rows are not in the cache for question set "
            f"{question_set_version} and model {model}; score them with "
            "python -m policyjev.jev.runner first"
        )
    ordered = probs.loc[[comment_id for comment_id, _ in rows], LABELS]
    return ordered.reset_index()


def main() -> None:
    """Write artifacts/predictions/jev_*.csv from the cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", dest="set_name", choices=SETS, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()["jev"]
    version, questions = _load_questions(args.questions)
    _check_guards(args.set_name, version)
    if args.set_name == "test_full":
        verify_files([args.questions])
    rows = _load_rows(args.set_name, None)

    with JevCache(REPO_ROOT / cfg["cache_path"]) as cache:
        scores = load_scores(rows, questions, version, cfg["model"], cache)

    out_path = args.out or (
        REPO_ROOT / "artifacts" / "predictions" / scores_filename(args.set_name, version)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_path, index=False, float_format="%.8g")
    print(f"wrote {len(scores)} rows to {out_path}")


if __name__ == "__main__":
    main()
