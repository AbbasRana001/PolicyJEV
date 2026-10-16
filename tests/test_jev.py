import logging

import pytest
import requests

from policyjev.jev.cache import JevCache
from policyjev.jev.client import (
    JevAuthError,
    JevClient,
    JevConfigError,
    JevResponseError,
    JevValidationError,
    cache_key,
    get_api_key,
)
from policyjev.jev.runner import format_summary, run_comments

SECRET = "sk-test-SECRET-123"
MODEL = "jev-1.13.0"
QUESTIONS = {
    "toxic": {"type": "noul", "instructions": "Is the comment toxic?"},
    "threat": {"type": "noul", "instructions": "Does it threaten?", "criteria": "Explicit harm."},
}


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, body=None, headers: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = "" if body is None else str(body)

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


def ok_body(questions: dict, input_tokens: int = 1000, value: float = 0.25) -> dict:
    """A well-formed success body answering every question."""
    return {
        "model": MODEL,
        "answers": {q: {"type": "noul", "noul": value} for q in questions},
        "usage": {"input_tokens": input_tokens, "output_tokens": 12},
    }


class FakeSession:
    """Returns queued responses, then a success built from the request."""

    def __init__(self, queued: list | None = None, input_tokens: int = 1000) -> None:
        self.queued = list(queued or [])
        self.input_tokens = input_tokens
        self.calls: list[dict] = []
        self.headers_seen: list[dict] = []

    def post(self, url, json, headers, timeout):
        self.calls.append(json)
        self.headers_seen.append(headers)
        if self.queued:
            item = self.queued.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return FakeResponse(200, ok_body(json["questions"], self.input_tokens))


def make_client(session: FakeSession, price: float = 0.5, sleeps: list | None = None) -> JevClient:
    """Client wired to a fake session with no real sleeping."""
    return JevClient(
        api_key=SECRET,
        endpoint="https://example.invalid/v1/systemone",
        model=MODEL,
        price_per_million_input_tokens=price,
        max_retries=3,
        backoff_base_seconds=0.01,
        request_timeout_seconds=5,
        session=session,
        sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
    )


def rows(n: int) -> list[tuple[str, str]]:
    """n distinct synthetic comments."""
    return [(f"c{i}", f"synthetic comment number {i}") for i in range(n)]


def test_successful_call_is_parsed():
    session = FakeSession()
    client = make_client(session)
    result = client.evaluate("  hello\n world ", QUESTIONS)

    assert result.answers == {"toxic": 0.25, "threat": 0.25}
    assert result.model_reported == MODEL
    assert result.input_tokens == 1000
    assert result.output_tokens == 12
    assert result.cost_usd == pytest.approx(1000 * 0.5 / 1_000_000)
    assert session.calls[0]["state"] == "hello world"
    assert session.calls[0]["model"] == MODEL
    assert session.headers_seen[0]["Authorization"] == f"Bearer {SECRET}"


def test_429_then_200_retries_and_succeeds():
    sleeps: list[float] = []
    session = FakeSession([FakeResponse(429, {"detail": "slow down"}, {"Retry-After": "2"})])
    client = make_client(session, sleeps=sleeps)
    attempts = []
    result = client.evaluate("text", QUESTIONS, on_attempt=attempts.append)

    assert len(session.calls) == 2
    assert result.attempts == 2
    assert [a.outcome for a in attempts] == ["retryable_error", "ok"]
    assert sleeps == [2.0]


def test_timeout_is_retried():
    session = FakeSession([requests.Timeout("timed out")])
    result = make_client(session).evaluate("text", QUESTIONS)
    assert len(session.calls) == 2
    assert result.answers["toxic"] == 0.25


def test_401_raises_without_retry_and_hides_key():
    session = FakeSession([FakeResponse(401, {"detail": f"invalid key {SECRET}"})])
    client = make_client(session)

    with pytest.raises(JevAuthError) as info:
        client.evaluate("text", QUESTIONS)

    assert len(session.calls) == 1
    assert SECRET not in str(info.value)
    assert SECRET not in repr(info.value)
    assert SECRET not in repr(client)
    assert SECRET not in str(client)


def test_422_raises_with_field_message():
    session = FakeSession(
        [FakeResponse(422, {"detail": "questions.toxic.instructions: field required"})]
    )
    with pytest.raises(JevValidationError) as info:
        make_client(session).evaluate("text", QUESTIONS)
    assert len(session.calls) == 1
    assert "questions.toxic.instructions" in str(info.value)


def test_missing_answer_raises():
    body = ok_body(QUESTIONS)
    del body["answers"]["threat"]
    session = FakeSession([FakeResponse(200, body)])
    with pytest.raises(JevResponseError):
        make_client(session).evaluate("text", QUESTIONS)


@pytest.mark.parametrize("bad_value", [1.5, -0.1, "0.3", None, True, float("nan")])
def test_malformed_answer_raises(bad_value):
    body = ok_body(QUESTIONS)
    body["answers"]["toxic"]["noul"] = bad_value
    session = FakeSession([FakeResponse(200, body)])
    with pytest.raises(JevResponseError):
        make_client(session).evaluate("text", QUESTIONS)


def test_cache_hit_makes_zero_calls(tmp_path):
    session = FakeSession()
    client = make_client(session)
    with JevCache(tmp_path / "jev.sqlite") as cache:
        first = run_comments(rows(5), QUESTIONS, "v1", client, cache, 3.0, 2)
        n_calls = len(session.calls)
        second = run_comments(rows(5), QUESTIONS, "v1", client, cache, 3.0, 2)

    assert first.newly_done == 5
    assert n_calls == 5
    assert len(session.calls) == n_calls
    assert second.cached == 5
    assert second.newly_done == 0


def test_rerun_after_partial_run_calls_only_missing_rows(tmp_path):
    session = FakeSession()
    client = make_client(session)
    with JevCache(tmp_path / "jev.sqlite") as cache:
        run_comments(rows(6)[:2], QUESTIONS, "v1", client, cache, 3.0, 2)
        assert len(session.calls) == 2
        counts = run_comments(rows(6), QUESTIONS, "v1", client, cache, 3.0, 2)

    assert len(session.calls) == 6
    assert counts.cached == 2
    assert counts.newly_done == 4
    assert counts.failed == 0


def test_failed_rows_are_counted_and_logged(tmp_path):
    body = ok_body(QUESTIONS)
    del body["answers"]["toxic"]
    session = FakeSession([FakeResponse(200, body)])
    client = make_client(session)
    with JevCache(tmp_path / "jev.sqlite") as cache:
        counts = run_comments(rows(3), QUESTIONS, "v1", client, cache, 3.0, 1)
        n_logged = cache._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    assert counts.failed == 1
    assert counts.newly_done == 2
    assert n_logged == 3


def test_cache_key_normalization_and_sensitivity():
    base = cache_key("hello   world\n", "v1", MODEL, QUESTIONS)

    assert base == cache_key(" hello world", "v1", MODEL, QUESTIONS)
    assert base == cache_key("hello\tworld", "v1", MODEL, QUESTIONS)

    changed = {**QUESTIONS, "toxic": {"type": "noul", "instructions": "Is it rude?"}}
    assert base != cache_key("hello world", "v1", MODEL, changed)
    assert base != cache_key("hello world", "v2", MODEL, QUESTIONS)
    assert base != cache_key("hello world", "v1", "jev-1.14.0", QUESTIONS)
    assert base != cache_key("hello world!", "v1", MODEL, QUESTIONS)


def test_budget_gate_stops_new_calls(tmp_path):
    session = FakeSession(input_tokens=1000)
    client = make_client(session, price=1000.0)
    with JevCache(tmp_path / "jev.sqlite") as cache:
        counts = run_comments(rows(5), QUESTIONS, "v1", client, cache, 2.5, 1)
        spend = cache.total_spend_usd()

    assert len(session.calls) == 3
    assert counts.newly_done == 3
    assert counts.not_attempted == 2
    assert counts.stopped_for_budget
    assert spend == pytest.approx(3.0)


def test_budget_already_spent_makes_no_calls(tmp_path):
    session = FakeSession()
    client = make_client(session)
    with JevCache(tmp_path / "jev.sqlite") as cache:
        cache.log_call("x", 1, "ok", 200, 0, 0, 5.0, 0.1, None)
        counts = run_comments(rows(3), QUESTIONS, "v1", client, cache, 3.0, 2)

    assert session.calls == []
    assert counts.stopped_for_budget
    assert counts.not_attempted == 3


def test_first_cached_result_stays_canonical(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        assert cache.put("k", "c1", "v1", MODEL, MODEL, {"toxic": 0.1}, 10, 1, 0.5)
        assert not cache.put("k", "c2", "v1", MODEL, MODEL, {"toxic": 0.9}, 10, 1, 0.5)
        stored = cache.get("k")

    assert stored["answers"] == {"toxic": 0.1}
    assert stored["comment_id"] == "c1"


def test_get_api_key_reads_env_file_and_errors_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("# local secrets\nOTHER=1\nTYPESAFE_API_KEY=abc123\n", encoding="utf-8")
    assert get_api_key(env_file) == "abc123"

    empty_file = tmp_path / "empty.env"
    empty_file.write_text("TYPESAFE_API_KEY=\n", encoding="utf-8")
    with pytest.raises(JevConfigError):
        get_api_key(empty_file)

    with pytest.raises(JevConfigError):
        get_api_key(tmp_path / "missing.env")


def test_model_mismatch_is_counted_and_warned_once(tmp_path, caplog):
    queued = []
    for _ in range(3):
        body = ok_body(QUESTIONS)
        body["model"] = "jev-latest"
        queued.append(FakeResponse(200, body))
    client = make_client(FakeSession(queued))

    with JevCache(tmp_path / "jev.sqlite") as cache:
        with caplog.at_level(logging.WARNING, logger="policyjev.jev.runner"):
            counts = run_comments(rows(3), QUESTIONS, "v1", client, cache, 3.0, 1)
        total_spend = cache.total_spend_usd()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "jev-latest" in warnings[0].getMessage()
    assert counts.newly_done == 3
    assert counts.model_mismatch == 3

    summary = format_summary(counts, total_spend, 3.0)
    assert "model_mismatch: 3" in summary


def test_matching_model_gives_no_warning_and_usage_statistics(tmp_path, caplog):
    client = make_client(FakeSession(input_tokens=1000))

    with JevCache(tmp_path / "jev.sqlite") as cache:
        with caplog.at_level(logging.WARNING, logger="policyjev.jev.runner"):
            counts = run_comments(rows(3), QUESTIONS, "v1", client, cache, 3.0, 1)
        total_spend = cache.total_spend_usd()

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    assert counts.model_mismatch == 0
    assert counts.ok_requests == 3
    assert counts.input_tokens == 3000
    assert counts.output_tokens == 36
    assert counts.spend_usd == pytest.approx(3000 * 0.5 / 1_000_000)
    assert len(counts.latencies_s) == 3

    summary = format_summary(counts, total_spend, 3.0)
    assert "mean input tokens per request: 1000.0" in summary
    assert "total tokens: 3036" in summary
    assert "spend this run:" in summary
    assert "spend total (cache):" in summary


def test_spend_settings_come_from_the_environment_or_env_file_and_have_no_default(monkeypatch, tmp_path):
    from policyjev.jev.client import HARD_STOP_ENV, PRICE_ENV, JevConfigError, get_spend_setting

    env_file = tmp_path / ".env"
    monkeypatch.delenv(HARD_STOP_ENV, raising=False)
    monkeypatch.delenv(PRICE_ENV, raising=False)
    with pytest.raises(JevConfigError, match=HARD_STOP_ENV):
        get_spend_setting(HARD_STOP_ENV, env_file)
    env_file.write_text(f"{HARD_STOP_ENV}=2.5{chr(10)}{PRICE_ENV}=\"0.75\"{chr(10)}", encoding="utf-8")
    assert get_spend_setting(HARD_STOP_ENV, env_file) == 2.5
    assert get_spend_setting(PRICE_ENV, env_file) == 0.75
    monkeypatch.setenv(HARD_STOP_ENV, "4")
    assert get_spend_setting(HARD_STOP_ENV, env_file) == 4.0
    for bad in ("abc", "0", "-1", "nan", "inf"):
        monkeypatch.setenv(HARD_STOP_ENV, bad)
        with pytest.raises(JevConfigError):
            get_spend_setting(HARD_STOP_ENV, env_file)


def test_from_config_needs_a_price_unless_given_one(monkeypatch):
    from pathlib import Path

    from policyjev.jev.client import PRICE_ENV, JevConfigError

    cfg = {"endpoint": "https://example.invalid", "model": "m", "max_retries": 1,
           "backoff_base_seconds": 0.0, "request_timeout_seconds": 5}
    monkeypatch.delenv(PRICE_ENV, raising=False)
    monkeypatch.setattr("policyjev.jev.client.REPO_ROOT", Path("/nonexistent"))
    with pytest.raises(JevConfigError, match=PRICE_ENV):
        JevClient.from_config(cfg, "key")
    monkeypatch.setenv(PRICE_ENV, "0.5")
    assert JevClient.from_config(cfg, "key").price_per_million_input_tokens == 0.5
    assert JevClient.from_config(cfg, "key", price_per_million_input_tokens=0.0).price_per_million_input_tokens == 0.0
