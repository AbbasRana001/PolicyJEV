"""SQLite cache of JEV judgments plus a log of every API call attempt."""

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS judgments (
    cache_key TEXT PRIMARY KEY,
    comment_id TEXT,
    question_set_version TEXT,
    model_requested TEXT,
    model_reported TEXT,
    answers_json TEXT,
    input_tokens INT,
    output_tokens INT,
    latency_s REAL,
    created_utc TEXT
);
CREATE TABLE IF NOT EXISTS second_pass (
    cache_key TEXT PRIMARY KEY,
    comment_id TEXT,
    question_set_version TEXT,
    model_requested TEXT,
    model_reported TEXT,
    answers_json TEXT,
    input_tokens INT,
    output_tokens INT,
    latency_s REAL,
    created_utc TEXT
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY,
    created_utc TEXT,
    comment_id TEXT,
    attempt INT,
    outcome TEXT,
    http_status INT,
    input_tokens INT,
    output_tokens INT,
    cost_usd REAL,
    latency_s REAL,
    error TEXT
);
"""


JUDGMENT_TABLES = ("judgments", "second_pass")


def _now_utc() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class JevCache:
    """Thread-safe SQLite store; the first stored judgment per key is canonical."""

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def __enter__(self) -> "JevCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def get(self, key: str, table: str = "judgments") -> dict | None:
        """Return the stored judgment for key from a judgment table, or None."""
        if table not in JUDGMENT_TABLES:
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            row = self._conn.execute(
                "SELECT comment_id, question_set_version, model_requested, "
                "model_reported, answers_json, input_tokens, output_tokens, "
                f"latency_s, created_utc FROM {table} WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "cache_key": key,
            "comment_id": row[0],
            "question_set_version": row[1],
            "model_requested": row[2],
            "model_reported": row[3],
            "answers": json.loads(row[4]),
            "input_tokens": row[5],
            "output_tokens": row[6],
            "latency_s": row[7],
            "created_utc": row[8],
        }

    def put(
        self,
        key: str,
        comment_id: str,
        question_set_version: str,
        model_requested: str,
        model_reported: str,
        answers: dict[str, float],
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        table: str = "judgments",
    ) -> bool:
        """Store a judgment unless the key exists; return True if it was inserted."""
        if table not in JUDGMENT_TABLES:
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            cursor = self._conn.execute(
                f"INSERT OR IGNORE INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    comment_id,
                    question_set_version,
                    model_requested,
                    model_reported,
                    json.dumps(answers, sort_keys=True),
                    input_tokens,
                    output_tokens,
                    latency_s,
                    _now_utc(),
                ),
            )
            self._conn.commit()
            return cursor.rowcount == 1

    def log_call(
        self,
        comment_id: str,
        attempt: int,
        outcome: str,
        http_status: int | None,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_s: float,
        error: str | None,
    ) -> None:
        """Record one API call attempt."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO calls (created_utc, comment_id, attempt, outcome, "
                "http_status, input_tokens, output_tokens, cost_usd, latency_s, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now_utc(),
                    comment_id,
                    attempt,
                    outcome,
                    http_status,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    latency_s,
                    error,
                ),
            )
            self._conn.commit()

    def total_spend_usd(self) -> float:
        """Cumulative cost of all logged calls."""
        with self._lock:
            (total,) = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM calls"
            ).fetchone()
        return float(total)

    def count_ok_calls(self) -> int:
        """Number of successful API calls logged."""
        with self._lock:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM calls WHERE outcome = 'ok'"
            ).fetchone()
        return int(count)

    def latency_percentiles(self) -> tuple[float, float]:
        """p50 and p95 latency in seconds over successful calls (NaN if none)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT latency_s FROM calls WHERE outcome = 'ok'"
            ).fetchall()
        if not rows:
            return float("nan"), float("nan")
        values = np.array([r[0] for r in rows], dtype=float)
        return float(np.percentile(values, 50)), float(np.percentile(values, 95))


class SecondPassStore:
    """View of the cache that reads and writes the second_pass table but shares call logging."""

    def __init__(self, cache: JevCache) -> None:
        self._cache = cache

    def get(self, key: str) -> dict | None:
        """Return the stored second-pass judgment for key, or None."""
        return self._cache.get(key, table="second_pass")

    def put(self, *args) -> bool:
        """Store a second-pass judgment (same arguments as JevCache.put)."""
        return self._cache.put(*args, table="second_pass")

    def log_call(self, *args, **kwargs) -> None:
        """Record an API call attempt in the shared calls table."""
        self._cache.log_call(*args, **kwargs)

    def total_spend_usd(self) -> float:
        """Cumulative cost across all logged calls, first pass included."""
        return self._cache.total_spend_usd()
