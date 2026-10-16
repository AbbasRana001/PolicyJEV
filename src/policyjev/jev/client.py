"""HTTP client for the TypeSafe JEV systemone endpoint (Noul questions)."""

import hashlib
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from policyjev.config import REPO_ROOT

API_KEY_ENV = "TYPESAFE_API_KEY"
COST_RATE_ENV = "TYPESAFE_COST_RATE"
HARD_STOP_ENV = "TYPESAFE_HARD_STOP"
RETRYABLE_STATUSES = {429, 529}
MAX_BACKOFF_SECONDS = 60.0


class JevError(Exception):
    """Base class for JEV call failures."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class JevAuthError(JevError):
    """The API rejected the key (HTTP 401)."""


class JevValidationError(JevError):
    """The API rejected the request body (HTTP 422)."""


class JevHTTPError(JevError):
    """A non-retryable HTTP error other than 401 or 422."""


class JevRetryExhaustedError(JevError):
    """Retryable failures continued past max_retries."""


class JevResponseError(JevError):
    """The response was missing data or malformed."""


class JevConfigError(RuntimeError):
    """Local configuration problem, such as a missing API key."""


@dataclass(frozen=True)
class AttemptRecord:
    """Outcome of one HTTP attempt."""

    attempt: int
    outcome: str
    http_status: int | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    error: str | None


@dataclass(frozen=True)
class JevResult:
    """A validated JEV judgment."""

    answers: dict[str, float]
    model_reported: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    attempts: int


def normalize_text(text: str) -> str:
    """NFC-normalize, collapse whitespace runs to one space, and strip."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def cache_key(
    text: str, question_set_version: str, model: str, questions: dict[str, dict]
) -> str:
    """SHA-256 of canonical JSON of the normalized text, version, model and questions."""
    payload = {
        "text": normalize_text(text),
        "question_set_version": question_set_version,
        "model": model,
        "questions": questions,
    }
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_criteria(qid: str, criteria: Any) -> str | dict[str, str]:
    """Validate criteria: plain text, or a mapping with the string keys 'true' and 'false'."""
    if isinstance(criteria, str):
        return criteria
    if not isinstance(criteria, dict):
        raise ValueError(f"question {qid!r} criteria must be text or a true/false mapping")
    if any(not isinstance(key, str) for key in criteria):
        raise ValueError(
            f"question {qid!r} criteria keys must be the strings \"true\" and \"false\"; "
            "quote them in YAML, because unquoted true/false parse as booleans"
        )
    if set(criteria) != {"true", "false"}:
        raise ValueError(f"question {qid!r} criteria must have exactly the keys true and false")
    for key, value in criteria.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"question {qid!r} criteria[{key!r}] must be non-empty text")
    return {"true": criteria["true"], "false": criteria["false"]}


def build_questions(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate question specs and return them in the request format."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError("questions must be a non-empty mapping of id to spec")
    questions: dict[str, dict[str, Any]] = {}
    for qid, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"question {qid!r} must be a mapping")
        instructions = spec.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"question {qid!r} needs non-empty instructions")
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        criteria = spec.get("criteria")
        if criteria is not None:
            question["criteria"] = _build_criteria(str(qid), criteria)
        questions[str(qid)] = question
    return questions


def _read_setting(name: str, env_path: Path | None = None) -> str:
    """Read a setting from the environment, else from the repository .env file; empty if unset."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    path = env_path if env_path is not None else REPO_ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            if key.strip() == name:
                value = raw_value.strip().strip('"').strip("'")
                if value:
                    return value
    return ""


def get_api_key(env_path: Path | None = None) -> str:
    """Read the API key from the environment, else from the repository .env file."""
    value = _read_setting(API_KEY_ENV, env_path)
    if value:
        return value
    raise JevConfigError(
        f"{API_KEY_ENV} is not set; put it in the environment or in a .env file "
        "in the repository root (see .env.example)"
    )


def get_spend_setting(name: str, env_path: Path | None = None) -> float:
    """Read a positive number (the cost rate used to track spend, or the spending hard stop) that is never committed."""
    raw = _read_setting(name, env_path)
    if not raw:
        raise JevConfigError(
            f"{name} is not set; put it in the environment or in a .env file "
            "in the repository root (see .env.example)"
        )
    try:
        value = float(raw)
    except ValueError:
        raise JevConfigError(f"{name} must be a number") from None
    if not math.isfinite(value) or value <= 0:
        raise JevConfigError(f"{name} must be a positive number")
    return value


def _is_number(value: Any) -> bool:
    """True for int or float values that are not bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class JevClient:
    """Calls the JEV endpoint with retries; the API key is never exposed in messages."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        cost_rate: float,
        max_retries: int = 5,
        backoff_base_seconds: float = 1.0,
        request_timeout_seconds: float = 30.0,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not api_key:
            raise JevConfigError("an API key is required")
        self._api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.cost_rate = cost_rate
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._session = session
        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()
        self._local = threading.local()

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        api_key: str,
        cost_rate: float | None = None,
        **kwargs: Any,
    ) -> "JevClient":
        """Build a client from the jev block of the project config.

        The cost rate is not in the config: unless given, it is read from the environment or
        .env (TYPESAFE_COST_RATE) and has no default.
        """
        if cost_rate is None:
            cost_rate = get_spend_setting(COST_RATE_ENV)
        return cls(
            api_key=api_key,
            endpoint=cfg["endpoint"],
            model=cfg["model"],
            cost_rate=cost_rate,
            max_retries=cfg["max_retries"],
            backoff_base_seconds=cfg["backoff_base_seconds"],
            request_timeout_seconds=cfg["request_timeout_seconds"],
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"JevClient(endpoint={self.endpoint!r}, model={self.model!r})"

    def _get_session(self) -> Any:
        """Injected session if given, else one requests.Session per thread."""
        if self._session is not None:
            return self._session
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _scrub(self, text: str) -> str:
        """Remove the API key from text."""
        return text.replace(self._api_key, "[redacted]") if self._api_key else text

    def _error_detail(self, response: Any) -> str:
        """The response's error message, scrubbed and truncated."""
        try:
            body = response.json()
        except ValueError:
            return self._scrub(str(getattr(response, "text", ""))[:500])
        detail: Any = body
        if isinstance(body, dict):
            for field in ("detail", "error", "message"):
                if field in body:
                    detail = body[field]
                    break
        text = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
        return self._scrub(text[:500])

    def _backoff_seconds(self, attempt: int, retry_after: str | None) -> float:
        """Retry-After if parseable, else exponential backoff plus jitter."""
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0.0), MAX_BACKOFF_SECONDS)
            except ValueError:
                pass
        base = self.backoff_base_seconds
        delay = base * 2 ** (attempt - 1) + self._rng.uniform(0, base)
        return min(delay, MAX_BACKOFF_SECONDS)

    def evaluate(
        self,
        state: str,
        questions: dict[str, dict],
        on_attempt: Callable[[AttemptRecord], None] | None = None,
    ) -> JevResult:
        """Ask every question about state; every attempt is reported to on_attempt."""

        def record(**fields: Any) -> None:
            if on_attempt is not None:
                on_attempt(AttemptRecord(**fields))

        body = {"state": normalize_text(state), "model": self.model, "questions": questions}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        total_attempts = self.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            started = time.perf_counter()
            try:
                response = self._get_session().post(
                    self.endpoint,
                    json=body,
                    headers=headers,
                    timeout=self.request_timeout_seconds,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                error = self._scrub(f"{type(exc).__name__}: {exc}")[:300]
                record(
                    attempt=attempt, outcome="retryable_error", http_status=None,
                    input_tokens=0, output_tokens=0, cost_usd=0.0,
                    latency_s=time.perf_counter() - started, error=error,
                )
                if attempt < total_attempts:
                    self._sleep(self._backoff_seconds(attempt, None))
                    continue
                raise JevRetryExhaustedError(
                    f"gave up after {attempt} attempts: {error}"
                ) from None

            latency = time.perf_counter() - started
            status = int(response.status_code)

            if 200 <= status < 300:
                return self._parse_success(response, questions, latency, attempt, record)

            detail = self._error_detail(response)
            if status in RETRYABLE_STATUSES or 500 <= status < 600:
                record(
                    attempt=attempt, outcome="retryable_error", http_status=status,
                    input_tokens=0, output_tokens=0, cost_usd=0.0,
                    latency_s=latency, error=detail,
                )
                if attempt < total_attempts:
                    retry_after = getattr(response, "headers", {}).get("Retry-After")
                    self._sleep(self._backoff_seconds(attempt, retry_after))
                    continue
                raise JevRetryExhaustedError(
                    f"gave up after {attempt} attempts (HTTP {status}): {detail}", status
                )

            record(
                attempt=attempt, outcome="fatal_error", http_status=status,
                input_tokens=0, output_tokens=0, cost_usd=0.0,
                latency_s=latency, error=detail,
            )
            if status == 401:
                raise JevAuthError(f"authentication failed (HTTP 401): {detail}", status)
            if status == 422:
                raise JevValidationError(f"request rejected (HTTP 422): {detail}", status)
            raise JevHTTPError(f"request failed (HTTP {status}): {detail}", status)

        raise JevRetryExhaustedError(f"gave up after {total_attempts} attempts")

    def _parse_success(
        self,
        response: Any,
        questions: dict[str, dict],
        latency: float,
        attempt: int,
        record: Callable[..., None],
    ) -> JevResult:
        """Validate a 2xx response and convert it to a JevResult."""
        status = int(response.status_code)

        def fail(message: str, input_tokens: int = 0, output_tokens: int = 0) -> JevResponseError:
            cost = input_tokens * self.cost_rate / 1_000_000
            record(
                attempt=attempt, outcome="invalid_response", http_status=status,
                input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost,
                latency_s=latency, error=message,
            )
            return JevResponseError(message, status)

        try:
            body = response.json()
        except ValueError:
            raise fail("response body is not JSON") from None
        if not isinstance(body, dict):
            raise fail("response body is not a JSON object")

        usage = body.get("usage")
        if not isinstance(usage, dict):
            raise fail("response has no usage block")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if not (isinstance(input_tokens, int) and not isinstance(input_tokens, bool)):
            raise fail("usage.input_tokens is missing or not an integer")
        if not (isinstance(output_tokens, int) and not isinstance(output_tokens, bool)):
            raise fail("usage.output_tokens is missing or not an integer", input_tokens)

        answers_raw = body.get("answers")
        if not isinstance(answers_raw, dict):
            raise fail("response has no answers mapping", input_tokens, output_tokens)

        answers: dict[str, float] = {}
        for qid in questions:
            entry = answers_raw.get(qid)
            value = entry.get("noul") if isinstance(entry, dict) else None
            if not _is_number(value) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise fail(
                    f"answer for question {qid!r} is missing or not a probability in [0, 1]",
                    input_tokens,
                    output_tokens,
                )
            answers[qid] = float(value)

        cost = input_tokens * self.cost_rate / 1_000_000
        model_reported = body.get("model")
        record(
            attempt=attempt, outcome="ok", http_status=status,
            input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost,
            latency_s=latency, error=None,
        )
        return JevResult(
            answers=answers,
            model_reported=model_reported if isinstance(model_reported, str) else "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            cost_usd=cost,
            attempts=attempt,
        )
