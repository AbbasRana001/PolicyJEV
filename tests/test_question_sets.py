import json
from pathlib import Path

import pytest
import yaml

from policyjev.config import LABELS, REPO_ROOT
from policyjev.jev.client import JevClient, build_questions, cache_key
from policyjev.jev.runner import _load_questions

CONFIGS = REPO_ROOT / "configs"
MODEL = "jev-1.13.0"


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    status_code = 200

    def __init__(self, body: dict) -> None:
        self._body = body
        self.headers: dict = {}
        self.text = ""

    def json(self) -> dict:
        return self._body


class CapturingSession:
    """Records request bodies and answers every question with 0.5."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def post(self, url, json, headers, timeout):
        self.bodies.append(json)
        return FakeResponse(
            {
                "model": MODEL,
                "answers": {q: {"type": "noul", "noul": 0.5} for q in json["questions"]},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        )


def make_client(session: CapturingSession) -> JevClient:
    """Client using a fake transport."""
    return JevClient(
        api_key="test-key",
        endpoint="https://example.invalid",
        model=MODEL,
        price_per_million_input_tokens=0.5,
        max_retries=1,
        backoff_base_seconds=0.0,
        session=session,
        sleep=lambda s: None,
    )


def test_v2_yaml_criteria_keys_load_as_strings():
    raw = yaml.safe_load((CONFIGS / "questions_v2.yaml").read_text(encoding="utf-8"))

    assert raw["version"] == "v2"
    assert list(raw["questions"]) == LABELS
    for spec in raw["questions"].values():
        assert set(spec["criteria"]) == {"true", "false"}
        assert all(isinstance(key, str) for key in spec["criteria"])


def test_unquoted_boolean_keys_are_rejected():
    raw = yaml.safe_load("toxic:\n  instructions: Is it toxic?\n  criteria:\n    true: yes text\n    false: no text\n")
    assert True in raw["toxic"]["criteria"]

    with pytest.raises(ValueError) as info:
        build_questions(raw)
    assert "quote" in str(info.value)


def test_criteria_must_have_exactly_true_and_false():
    with pytest.raises(ValueError):
        build_questions({"q": {"instructions": "x", "criteria": {"true": "a"}}})
    with pytest.raises(ValueError):
        build_questions({"q": {"instructions": "x", "criteria": {"true": "a", "false": ""}}})
    with pytest.raises(ValueError):
        build_questions({"q": {"instructions": "x", "criteria": 5}})


@pytest.mark.parametrize("name,has_criteria", [("v1", False), ("v2", True), ("v3", False)])
def test_repo_question_files_are_valid(name, has_criteria):
    version, questions = _load_questions(CONFIGS / f"questions_{name}.yaml")

    assert version == name
    assert list(questions) == LABELS
    for question in questions.values():
        assert question["type"] == "noul"
        assert ("criteria" in question) == has_criteria


def test_request_body_sends_criteria_with_string_true_false_keys():
    _, questions = _load_questions(CONFIGS / "questions_v2.yaml")
    session = CapturingSession()

    make_client(session).evaluate("some comment", questions)

    body = json.loads(json.dumps(session.bodies[0]))
    assert set(body["questions"]) == set(LABELS)
    for question in body["questions"].values():
        assert set(question["criteria"]) == {"true", "false"}
    assert '"true": ' in json.dumps(session.bodies[0])
    assert '"false": ' in json.dumps(session.bodies[0])


def test_v1_request_body_has_no_criteria():
    _, questions = _load_questions(CONFIGS / "questions_v1.yaml")
    session = CapturingSession()

    make_client(session).evaluate("some comment", questions)

    assert all("criteria" not in q for q in session.bodies[0]["questions"].values())


def test_same_text_has_different_cache_keys_under_v1_v2_and_v3():
    keys = {}
    for name in ("v1", "v2", "v3"):
        version, questions = _load_questions(CONFIGS / f"questions_{name}.yaml")
        keys[name] = cache_key("the same comment", version, MODEL, questions)

    assert len(set(keys.values())) == 3


def test_cache_key_changes_when_only_criteria_change(tmp_path: Path):
    version, questions = _load_questions(CONFIGS / "questions_v2.yaml")
    original = cache_key("text", version, MODEL, questions)

    changed = json.loads(json.dumps(questions))
    changed["toxic"]["criteria"]["false"] += " Extra sentence."

    assert cache_key("text", version, MODEL, changed) != original
