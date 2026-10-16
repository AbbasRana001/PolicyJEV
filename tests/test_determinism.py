import pytest

from policyjev.jev.client import JevClient
from policyjev.jev.determinism import check_determinism

QUESTIONS = {
    "toxic": {"type": "noul", "instructions": "Is the comment toxic?"},
    "threat": {"type": "noul", "instructions": "Does it threaten?"},
}


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    status_code = 200

    def __init__(self, body: dict) -> None:
        self._body = body
        self.headers: dict = {}
        self.text = ""

    def json(self) -> dict:
        return self._body


class FakeSession:
    """Deterministic answers, or a toxic answer that drifts on every call."""

    def __init__(self, noisy: bool) -> None:
        self.noisy = noisy
        self.n_calls = 0

    def post(self, url, json, headers, timeout):
        toxic = 0.1 + 0.01 * self.n_calls if self.noisy else 0.3
        self.n_calls += 1
        values = {"toxic": toxic, "threat": 0.05}
        return FakeResponse(
            {
                "model": "jev-1.13.0",
                "answers": {q: {"type": "noul", "noul": values[q]} for q in json["questions"]},
                "usage": {"input_tokens": 100, "output_tokens": 4},
            }
        )


def make_client(session: FakeSession) -> JevClient:
    """Client using a fake transport."""
    return JevClient(
        api_key="test-key",
        endpoint="https://example.invalid",
        model="jev-1.13.0",
        cost_rate=0.5,
        max_retries=1,
        backoff_base_seconds=0.0,
        session=session,
        sleep=lambda s: None,
    )


ROWS = [("c1", "first synthetic comment"), ("c2", "second synthetic comment")]


def test_deterministic_transport_shows_no_difference():
    session = FakeSession(noisy=False)
    report = check_determinism(make_client(session), ROWS, QUESTIONS, repeats=3).set_index("question")

    assert session.n_calls == 6
    assert report["max_abs_difference"].tolist() == [0.0, 0.0]
    assert report["comments_with_difference"].tolist() == [0, 0]
    assert report["comments"].tolist() == [2, 2]


def test_noisy_transport_reports_differences_per_question():
    session = FakeSession(noisy=True)
    report = check_determinism(make_client(session), ROWS, QUESTIONS, repeats=3).set_index("question")

    assert report.loc["toxic", "max_abs_difference"] == pytest.approx(0.02)
    assert report.loc["toxic", "comments_with_difference"] == 2
    assert report.loc["threat", "max_abs_difference"] == 0.0
    assert report.loc["threat", "comments_with_difference"] == 0


def test_fewer_than_two_repeats_is_rejected():
    with pytest.raises(ValueError):
        check_determinism(make_client(FakeSession(noisy=False)), ROWS, QUESTIONS, repeats=1)
