import pytest

from policyjev.config import LABELS
from policyjev.jev.cache import JevCache
from policyjev.jev.client import cache_key, normalize_text
from policyjev.jev.scores import load_scores

QUESTIONS = {
    label: {"type": "noul", "instructions": f"Question about {label}"} for label in LABELS
}
MODEL = "jev-1.13.0"
VERSION = "v1"
ROWS = [("c3", "third text"), ("c1", "first text"), ("c2", "second text")]


def store(cache: JevCache, comment_id: str, text: str, value: float) -> None:
    """Cache one judgment where every label has the same probability."""
    key = cache_key(normalize_text(text), VERSION, MODEL, QUESTIONS)
    cache.put(key, comment_id, VERSION, MODEL, MODEL, {l: value for l in LABELS}, 100, 4, 0.2)


def test_load_scores_keeps_row_order_and_has_no_text(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        store(cache, "c3", "third text", 0.3)
        store(cache, "c1", "first text", 0.1)
        store(cache, "c2", "second text", 0.2)
        scores = load_scores(ROWS, QUESTIONS, VERSION, MODEL, cache)

    assert list(scores.columns) == ["id"] + LABELS
    assert scores["id"].tolist() == ["c3", "c1", "c2"]
    assert scores["toxic"].tolist() == [0.3, 0.1, 0.2]


def test_load_scores_raises_when_rows_are_missing(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        store(cache, "c3", "third text", 0.3)
        with pytest.raises(ValueError) as info:
            load_scores(ROWS, QUESTIONS, VERSION, MODEL, cache)

    assert "2 of 3 rows are not in the cache" in str(info.value)


def test_load_scores_ignores_other_question_versions(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        for comment_id, text in ROWS:
            store(cache, comment_id, text, 0.5)
        with pytest.raises(ValueError):
            load_scores(ROWS, QUESTIONS, "v2", MODEL, cache)


def test_load_scores_requires_a_question_for_every_label(tmp_path):
    partial = {label: QUESTIONS[label] for label in LABELS[:-1]}
    with JevCache(tmp_path / "jev.sqlite") as cache:
        with pytest.raises(ValueError) as info:
            load_scores(ROWS, partial, VERSION, MODEL, cache)

    assert LABELS[-1] in str(info.value)
