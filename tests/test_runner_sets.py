import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.jev import runner
from policyjev.jev.cache import JevCache
from policyjev.jev.client import JevClient

QUESTIONS = {"toxic": {"type": "noul", "instructions": "Is the comment toxic?"}}
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


class FakeSession:
    """Always answers 0.25 for every requested question."""

    def __init__(self) -> None:
        self.calls = 0

    def post(self, url, json, headers, timeout):
        self.calls += 1
        return FakeResponse(
            {
                "model": MODEL,
                "answers": {q: {"type": "noul", "noul": 0.25} for q in json["questions"]},
                "usage": {"input_tokens": 500, "output_tokens": 3},
            }
        )


def make_client(session: FakeSession) -> JevClient:
    """Client using a fake transport."""
    return JevClient(
        api_key="test-key",
        endpoint="https://example.invalid",
        model=MODEL,
        cost_rate=0.5,
        max_retries=1,
        backoff_base_seconds=0.0,
        session=session,
        sleep=lambda s: None,
    )


def labeled_frame(ids: list[str]) -> pd.DataFrame:
    """Small frame in the loader format with a distinct text per id."""
    data = {"id": ids, "comment_text": [f"text of {i}" for i in ids]}
    for label in LABELS:
        data[label] = [0] * len(ids)
    return pd.DataFrame(data)


@pytest.fixture
def fake_data(monkeypatch):
    """Replace the data loaders with a tiny train/validation/test world."""
    train_ids = ["t1", "t2", "t3", "t4"]
    validation_ids = ["v1", "v2", "v3"]
    test_ids = ["x1", "x2"]
    monkeypatch.setattr(runner, "load_split_ids", lambda _dir: (train_ids, validation_ids))
    monkeypatch.setattr(
        runner, "load_train", lambda: labeled_frame(train_ids + validation_ids)
    )
    monkeypatch.setattr(runner, "load_test", lambda: labeled_frame(test_ids))
    return train_ids, validation_ids, test_ids


def set_tags(monkeypatch, tags: set[str]) -> None:
    """Pretend that exactly these git tags exist."""
    monkeypatch.setattr(runner, "tag_exists", lambda tag: tag in tags)


def test_val_full_loads_every_validation_row_only(fake_data):
    _, validation_ids, _ = fake_data
    rows = runner._load_rows("val_full", None)

    assert [i for i, _ in rows] == validation_ids
    assert rows[0] == ("v1", "text of v1")
    assert runner._load_rows("val_full", 2) == rows[:2]


def test_test_full_loads_every_labeled_test_row(fake_data):
    _, _, test_ids = fake_data
    rows = runner._load_rows("test_full", None)

    assert [i for i, _ in rows] == test_ids
    assert runner._load_rows("test_full", 1) == rows[:1]


def test_guards_for_full_sets_and_existing_sets(monkeypatch):
    set_tags(monkeypatch, set())
    runner._check_guards("pilot", "v1")
    for name in ("val_full", "test_full"):
        with pytest.raises(SystemExit):
            runner._check_guards(name, "v1")

    set_tags(monkeypatch, {"questions-frozen"})
    runner._check_guards("val_full", "v1")
    with pytest.raises(SystemExit):
        runner._check_guards("test_full", "v1")

    set_tags(monkeypatch, {"questions-frozen", "protocol-frozen"})
    for name in runner.SETS:
        runner._check_guards(name, "v1")

    set_tags(monkeypatch, {"protocol-frozen"})
    with pytest.raises(SystemExit):
        runner._check_guards("test_full", "v1")


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_alternative_question_sets_need_the_alt_tag(monkeypatch, version):
    runner._check_guards("pilot", version)

    set_tags(monkeypatch, {"questions-frozen", "protocol-frozen"})
    for name in ("val_full", "test_full"):
        with pytest.raises(SystemExit) as info:
            runner._check_guards(name, version)
        assert "questions-alt-frozen" in str(info.value)

    set_tags(monkeypatch, {"questions-alt-frozen"})
    runner._check_guards("val_full", version)
    with pytest.raises(SystemExit):
        runner._check_guards("test_full", version)

    set_tags(monkeypatch, {"questions-alt-frozen", "protocol-frozen"})
    runner._check_guards("test_full", version)


def test_v1_does_not_accept_the_alt_tag(monkeypatch):
    set_tags(monkeypatch, {"questions-alt-frozen"})
    with pytest.raises(SystemExit) as info:
        runner._check_guards("val_full", "v1")
    assert "questions-frozen" in str(info.value)


@pytest.mark.parametrize("set_name", ["pilot", "val_full", "test_full"])
def test_unknown_version_is_an_error_for_every_set(monkeypatch, set_name):
    set_tags(monkeypatch, {"questions-frozen", "questions-alt-frozen", "protocol-frozen"})
    with pytest.raises(SystemExit) as info:
        runner._check_guards(set_name, "v9")
    assert "unknown question set version" in str(info.value)


@pytest.mark.parametrize("set_name", ["val_full", "test_full"])
def test_guards_apply_to_dry_run(monkeypatch, tmp_path, set_name):
    set_tags(monkeypatch, set())
    questions = tmp_path / "q.yaml"
    questions.write_text(
        "version: v1\nquestions:\n  toxic:\n    type: noul\n    instructions: x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["runner", "--set", set_name, "--questions", str(questions), "--dry-run"],
    )
    with pytest.raises(SystemExit) as info:
        runner.main()
    assert "refusing" in str(info.value)


def test_rows_scored_for_an_earlier_set_are_not_repeated(tmp_path, fake_data):
    session = FakeSession()
    client = make_client(session)
    with JevCache(tmp_path / "jev.sqlite") as cache:
        subset = runner._load_rows("val_full", 2)
        first = runner.run_comments(subset, QUESTIONS, "v1", client, cache, 3.0, 2)
        calls_after_subset = session.calls

        full = runner._load_rows("val_full", None)
        second = runner.run_comments(full, QUESTIONS, "v1", client, cache, 3.0, 2)
        third = runner.run_comments(full, QUESTIONS, "v1", client, cache, 3.0, 2)

    assert first.newly_done == 2
    assert calls_after_subset == 2
    assert second.cached == 2
    assert second.newly_done == 1
    assert session.calls == 3
    assert third.newly_done == 0
    assert third.cached == 3


def test_progress_lines_every_n_completed_rows(tmp_path):
    session = FakeSession()
    client = make_client(session)
    lines: list[str] = []
    rows = [(f"c{i}", f"synthetic comment {i}") for i in range(5)]
    with JevCache(tmp_path / "jev.sqlite") as cache:
        counts = runner.run_comments(
            rows, QUESTIONS, "v1", client, cache, 3.0, 1, progress=lines.append, progress_every=2
        )

    assert counts.newly_done == 5
    assert len(lines) == 2
    assert lines[0].startswith("progress: 2 rows done")
    assert lines[1].startswith("progress: 4 rows done")
    assert "spend this run $" in lines[0]
    assert "rows/s" in lines[0]


def test_no_progress_callback_means_no_output(tmp_path, capsys):
    client = make_client(FakeSession())
    with JevCache(tmp_path / "jev.sqlite") as cache:
        runner.run_comments(
            [("c1", "one"), ("c2", "two")], QUESTIONS, "v1", client, cache, 3.0, 1,
            progress_every=1,
        )
    assert capsys.readouterr().out == ""
