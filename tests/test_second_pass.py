import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import splits
from policyjev.jev import runner, second_pass
from policyjev.jev import second_pass_report as report
from policyjev.jev.cache import JevCache
from policyjev.jev.client import JevClient, cache_key, normalize_text

MODEL = "jev-1.13.0"
TEST_IDS = [f"t{i:02d}" for i in range(30)]


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
    """Answers every question with 0.25 and counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def post(self, url, json, headers, timeout):
        self.calls += 1
        return FakeResponse(
            {
                "model": MODEL,
                "answers": {q: {"type": "noul", "noul": 0.25} for q in json["questions"]},
                "usage": {"input_tokens": 100, "output_tokens": 4},
            }
        )


def make_client(session: FakeSession) -> JevClient:
    """Client using a fake transport."""
    return JevClient(
        api_key="test-key", endpoint="https://example.invalid", model=MODEL,
        price_per_million_input_tokens=0.5, max_retries=1, backoff_base_seconds=0.0,
        session=session, sleep=lambda s: None,
    )


QUESTIONS_YAML = "version: v2\nquestions:\n" + "".join(
    f"  {label}:\n    type: noul\n    instructions: Question about {label}?\n" for label in LABELS
)


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Tags present, a tiny test split, a fake transport and a temporary repository root."""
    session = FakeSession()
    client = make_client(session)
    tags = {"questions-alt-frozen", "protocol-frozen"}
    monkeypatch.setattr(runner, "tag_exists", lambda tag: tag in tags)
    monkeypatch.setattr(splits, "tag_exists", lambda tag: tag in tags)

    frame = pd.DataFrame(
        {"id": TEST_IDS, "comment_text": [f"synthetic test comment {i}" for i in TEST_IDS],
         **{label: 0 for label in LABELS}}
    )
    monkeypatch.setattr(second_pass, "load_labeled_frame", lambda split: frame)
    monkeypatch.setattr(second_pass, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(second_pass, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(second_pass, "verify_files", lambda paths: None)
    monkeypatch.setattr(
        second_pass, "JevClient", SimpleNamespace(from_config=lambda cfg, key: client)
    )
    questions = tmp_path / "questions_v2.yaml"
    questions.write_text(QUESTIONS_YAML, encoding="utf-8")
    return SimpleNamespace(session=session, tmp=tmp_path, questions=questions, tags=tags, frame=frame)


def run_cli(monkeypatch, world, *extra):
    monkeypatch.setenv("TYPESAFE_HARD_STOP_USD", "3")
    monkeypatch.setenv("TYPESAFE_PRICE_PER_MILLION_INPUT_TOKENS", "0.5")
    monkeypatch.setattr(
        sys, "argv", ["second_pass", "--questions", str(world.questions), *extra]
    )
    second_pass.main()


def open_cache(world):
    from policyjev.config import load_config

    return JevCache(world.tmp / load_config()["jev"]["cache_path"])


def test_select_rows_is_fixed_nested_and_input_order_free():
    first = second_pass.select_rows(TEST_IDS, 10, seed=53)
    again = second_pass.select_rows(list(reversed(TEST_IDS)), 10, seed=53)
    other = second_pass.select_rows(TEST_IDS, 10, seed=54)
    bigger = second_pass.select_rows(TEST_IDS, 20, seed=53)

    assert first == again
    assert first != other
    assert len(set(first)) == 10
    assert bigger[:10] == first
    assert len(second_pass.select_rows(TEST_IDS, 500, seed=53)) == len(TEST_IDS)


def test_second_pass_scores_rows_without_touching_the_main_judgments(monkeypatch, world, capsys):
    with open_cache(world) as cache:
        key = cache_key(
            normalize_text(f"synthetic test comment {TEST_IDS[0]}"), "v2", MODEL,
            runner._load_questions(world.questions)[1],
        )
        cache.put(key, "x", "v2", MODEL, MODEL, {l: 0.9 for l in LABELS}, 10, 1, 0.1)

    run_cli(monkeypatch, world)

    assert world.session.calls == len(TEST_IDS)
    with open_cache(world) as cache:
        n_second = cache._conn.execute("SELECT COUNT(*) FROM second_pass").fetchone()[0]
        n_main = cache._conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
        assert cache.get(key)["answers"]["toxic"] == 0.9
        assert cache.get(key, table="second_pass")["answers"]["toxic"] == 0.25
        assert cache.total_spend_usd() > 0
    assert n_second == len(TEST_IDS)
    assert n_main == 1

    exported = pd.read_csv(
        world.tmp / "artifacts" / "predictions" / "jev_test_v2_second_pass.csv", dtype={"id": str}
    )
    assert exported.columns.tolist() == ["id"] + LABELS
    assert exported["id"].tolist() == sorted(TEST_IDS)
    assert (exported[LABELS] == 0.25).all().all()
    assert "comment_text" not in exported.columns
    assert "model_mismatch" in capsys.readouterr().out


def test_second_pass_is_resumable(monkeypatch, world):
    run_cli(monkeypatch, world, "--limit", "10")
    assert world.session.calls == 10
    assert not (world.tmp / "artifacts" / "predictions" / "jev_test_v2_second_pass.csv").exists()

    run_cli(monkeypatch, world)
    assert world.session.calls == len(TEST_IDS)

    run_cli(monkeypatch, world)
    assert world.session.calls == len(TEST_IDS)
    assert (world.tmp / "artifacts" / "predictions" / "jev_test_v2_second_pass.csv").exists()


def test_dry_run_makes_no_calls_and_needs_no_key(monkeypatch, world, capsys):
    monkeypatch.setattr(
        second_pass, "get_api_key", lambda: (_ for _ in ()).throw(AssertionError("key read"))
    )
    run_cli(monkeypatch, world, "--dry-run")

    out = capsys.readouterr().out
    assert f"second-pass rows selected: {len(TEST_IDS)}" in out
    assert f"not yet scored: {len(TEST_IDS)}" in out
    assert world.session.calls == 0


def test_second_pass_spend_counts_toward_the_hard_stop(monkeypatch, world):
    with open_cache(world) as cache:
        cache.log_call("earlier", 1, "ok", 200, 1, 1, 1000.0, 0.1, None)

    with pytest.raises(SystemExit) as info:
        run_cli(monkeypatch, world)

    assert "hard spending limit" in str(info.value)
    assert world.session.calls == 0


@pytest.mark.parametrize("missing", ["protocol-frozen", "questions-alt-frozen"])
def test_second_pass_requires_both_tags(monkeypatch, world, missing):
    world.tags.discard(missing)

    with pytest.raises(SystemExit) as info:
        run_cli(monkeypatch, world)

    assert missing in str(info.value)
    assert world.session.calls == 0


def test_second_pass_refuses_without_any_tag_and_reads_no_test_data(monkeypatch, world):
    world.tags.clear()
    monkeypatch.setattr(
        second_pass, "load_labeled_frame", lambda split: (_ for _ in ()).throw(AssertionError("read"))
    )
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, world)
    assert world.session.calls == 0


def make_pass_inputs(n: int = 400):
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    first = np.round(np.clip(y * 0.5 + rng.random(y.shape) * 0.5, 0, 1), 2)
    return y, first


def test_identical_passes_show_no_variation():
    y, first = make_pass_inputs()
    table = report.compare_passes(y, first, first.copy(), first, first.copy(), np.full(6, 0.5))

    assert table["label"].tolist() == LABELS + ["mean"]
    for column in ("mean_abs_probability_change", "share_rows_changed", "ap_change", "f1_change"):
        assert np.allclose(table[column], 0.0)


def test_noisy_second_pass_is_measured():
    y, first = make_pass_inputs()
    second = np.round(np.clip(first + np.random.default_rng(1).choice([-0.02, 0.0, 0.02], first.shape), 0, 1), 2)

    table = report.compare_passes(y, first, second, first, second, np.full(6, 0.5))

    toxic = table[table["label"] == "toxic"].iloc[0]
    assert toxic["mean_abs_probability_change"] == pytest.approx(np.abs(first[:, 0] - second[:, 0]).mean())
    assert toxic["share_rows_changed"] == pytest.approx((first[:, 0] != second[:, 0]).mean())
    assert toxic["ap_change"] == pytest.approx(toxic["ap_second"] - toxic["ap_first"])
    assert 0.0 < toxic["mean_abs_probability_change"] < 0.03
    assert table[table["label"] == "mean"]["mean_abs_probability_change"].iloc[0] > 0


def test_report_cli_refuses_without_the_tag(monkeypatch):
    monkeypatch.setattr(splits, "tag_exists", lambda tag: False)
    with pytest.raises(SystemExit) as info:
        report.main()
    assert "protocol-frozen" in str(info.value)
