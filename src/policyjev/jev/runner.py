"""Resumable, budget-gated batch runs of JEV over a fixed set of comments."""

import argparse
import logging
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from policyjev.config import REPO_ROOT, load_config
from policyjev.data.load import load_test, load_train
from policyjev.data.split import load_split_ids
from policyjev.freeze_manifest import verify_files
from policyjev.jev.cache import JevCache
from policyjev.jev.client import (
    HARD_STOP_ENV,
    AttemptRecord,
    JevAuthError,
    JevClient,
    JevError,
    build_questions,
    cache_key,
    get_api_key,
    get_spend_setting,
    normalize_text,
)

SETS = ["pilot", "val_full", "test_full"]
QUESTIONS_TAG = "questions-frozen"
ALT_QUESTIONS_TAG = "questions-alt-frozen"
PROTOCOL_TAG = "protocol-frozen"
QUESTION_VERSION_TAGS = {
    "v1": QUESTIONS_TAG,
    "v2": ALT_QUESTIONS_TAG,
    "v3": ALT_QUESTIONS_TAG,
}
PROGRESS_EVERY = 1000
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunCounts:
    """Row-level outcome of a run, plus request-level usage statistics."""

    cached: int
    newly_done: int
    failed: int
    not_attempted: int
    stopped_for_budget: bool
    aborted_auth: bool
    model_mismatch: int = 0
    ok_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spend_usd: float = 0.0
    latencies_s: tuple[float, ...] = ()


@dataclass
class _RunStats:
    """Thread-safe accumulator of per-run usage statistics."""

    ok_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spend_usd: float = 0.0
    model_mismatch: int = 0
    latencies_s: list[float] = field(default_factory=list)
    _warned: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_attempt(self, record: AttemptRecord) -> None:
        """Fold one HTTP attempt into the totals."""
        with self._lock:
            self.spend_usd += record.cost_usd
            if record.outcome == "ok":
                self.ok_requests += 1
                self.input_tokens += record.input_tokens
                self.output_tokens += record.output_tokens
                self.latencies_s.append(record.latency_s)

    def add_model(self, reported: str, pinned: str) -> None:
        """Count a model-id mismatch and warn about the first one."""
        if reported == pinned:
            return
        with self._lock:
            self.model_mismatch += 1
            first = not self._warned
            self._warned = True
        if first:
            logger.warning(
                "API reported model %r but %r is pinned; further mismatches are "
                "counted in the run summary, not logged",
                reported,
                pinned,
            )


def _group_by_key(
    rows: Iterable[tuple[str, str]], questions: dict[str, dict], version: str, model: str
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Map each cache key to its comment ids and its normalized text."""
    ids_by_key: dict[str, list[str]] = {}
    text_by_key: dict[str, str] = {}
    for comment_id, text in rows:
        normalized = normalize_text(text)
        key = cache_key(normalized, version, model, questions)
        ids_by_key.setdefault(key, []).append(comment_id)
        text_by_key.setdefault(key, normalized)
    return ids_by_key, text_by_key


def count_uncached(
    rows: Iterable[tuple[str, str]],
    questions: dict[str, dict],
    question_set_version: str,
    model: str,
    cache: JevCache,
) -> tuple[int, int]:
    """Return (uncached rows, uncached unique texts) without any network call."""
    ids_by_key, _ = _group_by_key(rows, questions, question_set_version, model)
    missing = [k for k in ids_by_key if cache.get(k) is None]
    return sum(len(ids_by_key[k]) for k in missing), len(missing)


def run_comments(
    rows: Iterable[tuple[str, str]],
    questions: dict[str, dict],
    question_set_version: str,
    client: JevClient,
    cache: JevCache,
    hard_stop_usd: float,
    max_workers: int,
    progress: Callable[[str], None] | None = None,
    progress_every: int = PROGRESS_EVERY,
) -> RunCounts:
    """Judge every uncached row once, logging every attempt and honoring the hard stop."""
    started = time.monotonic()
    next_report = progress_every
    ids_by_key, text_by_key = _group_by_key(
        rows, questions, question_set_version, client.model
    )
    pending = [k for k in ids_by_key if cache.get(k) is None]
    pending_set = set(pending)
    cached_rows = sum(len(v) for k, v in ids_by_key.items() if k not in pending_set)
    stats = _RunStats()

    def process(key: str) -> str:
        comment_id = ids_by_key[key][0]

        def on_attempt(record: AttemptRecord) -> None:
            stats.add_attempt(record)
            cache.log_call(
                comment_id,
                record.attempt,
                record.outcome,
                record.http_status,
                record.input_tokens,
                record.output_tokens,
                record.cost_usd,
                record.latency_s,
                record.error,
            )

        try:
            result = client.evaluate(text_by_key[key], questions, on_attempt=on_attempt)
        except JevAuthError:
            return "auth_error"
        except JevError:
            return "failed"
        stats.add_model(result.model_reported, client.model)
        cache.put(
            key,
            comment_id,
            question_set_version,
            client.model,
            result.model_reported,
            result.answers,
            result.input_tokens,
            result.output_tokens,
            result.latency_s,
        )
        return "done"

    newly_done = failed = 0
    stopped_for_budget = aborted = False
    next_index = 0
    in_flight: dict[Future, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            while not (stopped_for_budget or aborted) and len(in_flight) < max_workers:
                if next_index >= len(pending):
                    break
                if cache.total_spend_usd() >= hard_stop_usd:
                    stopped_for_budget = True
                    break
                key = pending[next_index]
                next_index += 1
                in_flight[executor.submit(process, key)] = key
            if not in_flight:
                break
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                key = in_flight.pop(future)
                outcome = future.result()
                n_rows = len(ids_by_key[key])
                if outcome == "done":
                    newly_done += n_rows
                    while progress is not None and newly_done >= next_report:
                        rate = newly_done / max(time.monotonic() - started, 1e-9)
                        progress(
                            f"progress: {newly_done} rows done, "
                            f"spend this run ${stats.spend_usd:.4f}, {rate:.1f} rows/s"
                        )
                        next_report += progress_every
                else:
                    failed += n_rows
                    if outcome == "auth_error":
                        aborted = True

    not_attempted = sum(len(ids_by_key[k]) for k in pending[next_index:])
    return RunCounts(
        cached=cached_rows,
        newly_done=newly_done,
        failed=failed,
        not_attempted=not_attempted,
        stopped_for_budget=stopped_for_budget and not_attempted > 0,
        aborted_auth=aborted,
        model_mismatch=stats.model_mismatch,
        ok_requests=stats.ok_requests,
        input_tokens=stats.input_tokens,
        output_tokens=stats.output_tokens,
        spend_usd=stats.spend_usd,
        latencies_s=tuple(stats.latencies_s),
    )


def tag_exists(tag: str) -> bool:
    """True if the git tag exists in the repository."""
    try:
        output = subprocess.run(
            ["git", "tag", "--list", tag],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return tag in output.split()


def _check_guards(set_name: str, version: str) -> None:
    """Refuse frozen-only sets until the tags for the question set version exist."""
    if version not in QUESTION_VERSION_TAGS:
        known = ", ".join(sorted(QUESTION_VERSION_TAGS))
        sys.exit(f"unknown question set version {version!r}; known versions: {known}")
    version_tag = QUESTION_VERSION_TAGS[version]
    if set_name != "pilot" and not tag_exists(version_tag):
        sys.exit(
            f"refusing {set_name} with question set {version}: "
            f"git tag '{version_tag}' does not exist"
        )
    if set_name == "test_full" and not tag_exists(PROTOCOL_TAG):
        sys.exit(f"refusing {set_name}: git tag '{PROTOCOL_TAG}' does not exist")


def _load_rows(set_name: str, limit: int | None) -> list[tuple[str, str]]:
    """Return (id, comment_text) pairs for the set, in saved order."""
    if set_name == "val_full":
        _, validation_ids = load_split_ids(PROCESSED_DIR)
        ids = validation_ids[:limit] if limit is not None else validation_ids
        texts = load_train().set_index("id")["comment_text"]
        missing = [i for i in ids if i not in texts.index]
        if missing:
            sys.exit(f"{len(missing)} validation ids are not in the raw data")
        return [(i, texts.loc[i]) for i in ids]
    if set_name == "test_full":
        test = load_test()
        if limit is not None:
            test = test.iloc[:limit]
        return list(zip(test["id"], test["comment_text"]))

    path = PROCESSED_DIR / "pilot.csv"
    if not path.exists():
        sys.exit(
            "data/processed/pilot.csv does not exist yet; "
            "create the pilot sample from train-split comments first"
        )

    ids = pd.read_csv(path, dtype={"id": str})["id"].tolist()
    if limit is not None:
        ids = ids[:limit]

    train_ids, _ = load_split_ids(PROCESSED_DIR)
    outside = set(ids) - set(train_ids)
    if outside:
        sys.exit(f"pilot.csv has {len(outside)} ids outside the train split")

    texts = load_train().set_index("id")["comment_text"]
    missing = [i for i in ids if i not in texts.index]
    if missing:
        sys.exit(f"{len(missing)} ids in pilot.csv are not in the raw data")
    return [(i, texts.loc[i]) for i in ids]


def _load_questions(path: Path) -> tuple[str, dict[str, dict]]:
    """Read a question file with keys version and questions."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "version" not in data or "questions" not in data:
        sys.exit(f"{path} must contain the keys version and questions")
    return str(data["version"]), build_questions(data["questions"])


def _format_seconds(value: float) -> str:
    """Seconds with three decimals, or n/a."""
    return "n/a" if math.isnan(value) else f"{value:.3f}s"


def format_summary(counts: RunCounts, total_spend_usd: float, hard_stop_usd: float) -> str:
    """Printable run summary: rows, usage, spend and latency for this run."""
    if counts.latencies_s:
        values = np.array(counts.latencies_s, dtype=float)
        p50, p95 = float(np.percentile(values, 50)), float(np.percentile(values, 95))
    else:
        p50 = p95 = float("nan")
    mean_input = counts.input_tokens / counts.ok_requests if counts.ok_requests else float("nan")
    mean_text = "n/a" if math.isnan(mean_input) else f"{mean_input:.1f}"
    lines = [
        f"rows done: {counts.newly_done}",
        f"cached: {counts.cached}",
        f"failed: {counts.failed}",
        f"not attempted: {counts.not_attempted}",
        f"model_mismatch: {counts.model_mismatch}",
        f"successful requests: {counts.ok_requests}",
        f"mean input tokens per request: {mean_text}",
        f"total tokens: {counts.input_tokens + counts.output_tokens} "
        f"(input {counts.input_tokens}, output {counts.output_tokens})",
        f"spend this run: ${counts.spend_usd:.4f}",
        f"spend total (cache): ${total_spend_usd:.4f} of ${hard_stop_usd:.2f}",
        f"latency this run p50: {_format_seconds(p50)}, p95: {_format_seconds(p95)}",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run (or dry-run) JEV over one benchmark set."""
    logging.basicConfig(format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", dest="set_name", choices=SETS, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()["jev"]
    version, questions = _load_questions(args.questions)
    _check_guards(args.set_name, version)
    if args.set_name == "test_full":
        verify_files([args.questions])
    rows = _load_rows(args.set_name, args.limit)

    with JevCache(REPO_ROOT / cfg["cache_path"]) as cache:
        if args.dry_run:
            uncached_rows, uncached_texts = count_uncached(
                rows, questions, version, cfg["model"], cache
            )
            print(f"rows in set: {len(rows)}")
            print(f"uncached rows: {uncached_rows} ({uncached_texts} unique texts)")
            print(f"spend so far: ${cache.total_spend_usd():.4f}")
            return

        hard_stop_usd = get_spend_setting(HARD_STOP_ENV)
        client = JevClient.from_config(cfg, get_api_key())
        counts = run_comments(
            rows,
            questions,
            version,
            client,
            cache,
            hard_stop_usd,
            cfg["max_workers"],
            progress=lambda line: print(line, flush=True),
        )
        print(format_summary(counts, cache.total_spend_usd(), hard_stop_usd))

    if counts.aborted_auth:
        sys.exit("stopped: the API rejected the key")
    if counts.stopped_for_budget:
        sys.exit("stopped: hard spending limit reached")
    if counts.failed:
        sys.exit(f"{counts.failed} rows failed; rerun to retry them")


if __name__ == "__main__":
    main()
