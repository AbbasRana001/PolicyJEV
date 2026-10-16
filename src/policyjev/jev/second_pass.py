"""Second JEV scoring pass on a fixed random subset of test rows (run-to-run variation).

Results go to the separate second_pass table; the main judgments table is never read or
written, and the first pass stays canonical. Spend counts toward the shared hard stop.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from policyjev.config import REPO_ROOT, load_config
from policyjev.evaluation.splits import load_labeled_frame, require_split_access
from policyjev.freeze_manifest import verify_files
from policyjev.jev.cache import JevCache, SecondPassStore
from policyjev.jev.client import HARD_STOP_ENV, JevClient, get_api_key, get_spend_setting
from policyjev.jev.runner import (
    _check_guards,
    _load_questions,
    count_uncached,
    format_summary,
    run_comments,
)
from policyjev.jev.scores import load_scores


def select_rows(ids: Sequence[str], n_rows: int, seed: int) -> list[str]:
    """Fixed random subset: a seeded permutation of the sorted ids; prefixes are nested."""
    ordered = sorted(ids)
    order = np.random.default_rng(seed).permutation(len(ordered))
    return [ordered[i] for i in order[:n_rows]]


def second_pass_filename(version: str) -> str:
    """File name of the exported second-pass scores."""
    return f"jev_test_{version}_second_pass.csv"


def main() -> None:
    """Run (or dry-run) the second pass and export its scores when complete."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config()
    cfg = config["jev"]
    version, questions = _load_questions(args.questions)
    _check_guards("test_full", version)
    require_split_access("test")
    verify_files([args.questions])

    test = load_labeled_frame("test")[["id", "comment_text"]]
    texts = dict(zip(test["id"], test["comment_text"]))
    selected = select_rows(
        list(texts), cfg["second_pass_rows"], config["seed"] + cfg["second_pass_seed_offset"]
    )
    if args.limit is not None:
        selected = selected[: args.limit]
    rows = [(i, texts[i]) for i in selected]

    with JevCache(REPO_ROOT / cfg["cache_path"]) as cache:
        store = SecondPassStore(cache)
        if args.dry_run:
            uncached_rows, uncached_texts = count_uncached(rows, questions, version, cfg["model"], store)
            print(f"second-pass rows selected: {len(rows)}")
            print(f"not yet scored: {uncached_rows} ({uncached_texts} unique texts)")
            print(f"spend so far: ${cache.total_spend_usd():.4f}")
            return

        hard_stop_usd = get_spend_setting(HARD_STOP_ENV)
        client = JevClient.from_config(cfg, get_api_key())
        counts = run_comments(
            rows, questions, version, client, store, hard_stop_usd, cfg["max_workers"],
            progress=lambda line: print(line, flush=True),
        )
        print(format_summary(counts, cache.total_spend_usd(), hard_stop_usd))

        if args.limit is None and counts.failed == 0 and counts.not_attempted == 0:
            scores = load_scores(rows, questions, version, cfg["model"], store)
            out_path = REPO_ROOT / "artifacts" / "predictions" / second_pass_filename(version)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scores.sort_values("id").to_csv(out_path, index=False, float_format="%.8g")
            print(f"exported {len(scores)} rows to {out_path}")
        elif args.limit is not None:
            print("partial run (--limit): no export")

    if counts.aborted_auth:
        sys.exit("stopped: the API rejected the key")
    if counts.stopped_for_budget:
        sys.exit("stopped: hard spending limit reached")
    if counts.failed:
        sys.exit(f"{counts.failed} rows failed; rerun to retry them")


if __name__ == "__main__":
    main()
