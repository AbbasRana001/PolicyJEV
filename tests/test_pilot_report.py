import math

import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.jev.cache import JevCache
from policyjev.jev.client import cache_key, normalize_text
from policyjev.jev.pilot_report import collect_judgments, pilot_report_table

QUESTIONS = {
    label: {"type": "noul", "instructions": f"Question about {label}"} for label in LABELS
}
MODEL = "jev-1.13.0"
VERSION = "v1"

ROWS = [
    ("c1", "first comment"),
    ("c2", "second comment"),
    ("c3", "third comment"),
    ("c4", "fourth comment, never judged"),
]

ANSWERS = {
    "c1": dict(toxic=0.9, severe_toxic=0.0, obscene=0.5, threat=0.7, insult=0.2, identity_hate=0.0),
    "c2": dict(toxic=0.8, severe_toxic=0.0, obscene=0.4, threat=0.0, insult=0.6, identity_hate=1.0),
    "c3": dict(toxic=0.1, severe_toxic=0.0, obscene=0.3, threat=0.0, insult=0.1, identity_hate=0.0),
}

LABEL_FRAME = pd.DataFrame(
    {
        "toxic": [1, 1, 0],
        "severe_toxic": [0, 0, 0],
        "obscene": [1, 0, 0],
        "threat": [1, 0, 0],
        "insult": [0, 1, 0],
        "identity_hate": [0, 1, 0],
    },
    index=pd.Index(["c1", "c2", "c3"], name="id"),
)


def fill_cache(cache: JevCache) -> None:
    """Store judgments for c1 to c3 only."""
    for comment_id, text in ROWS[:3]:
        key = cache_key(normalize_text(text), VERSION, MODEL, QUESTIONS)
        cache.put(key, comment_id, VERSION, MODEL, MODEL, ANSWERS[comment_id], 100, 4, 0.2)


def test_collect_judgments_counts_missing_rows(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        fill_cache(cache)
        probs, missing = collect_judgments(ROWS, QUESTIONS, VERSION, MODEL, cache)

    assert missing == 1
    assert probs.index.tolist() == ["c1", "c2", "c3"]
    assert probs.loc["c2", "identity_hate"] == 1.0


def test_collect_judgments_ignores_other_question_versions(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        fill_cache(cache)
        probs, missing = collect_judgments(ROWS, QUESTIONS, "v2", MODEL, cache)

    assert missing == 4
    assert probs.empty


def test_pilot_report_table_values(tmp_path):
    with JevCache(tmp_path / "jev.sqlite") as cache:
        fill_cache(cache)
        probs, _ = collect_judgments(ROWS, QUESTIONS, VERSION, MODEL, cache)

    report = pilot_report_table(probs, LABEL_FRAME).set_index("label")

    assert report.index.tolist() == LABELS

    toxic = report.loc["toxic"]
    assert toxic["n_pos"] == 2
    assert toxic["average_precision"] == pytest.approx(1.0)
    assert toxic["mean_prob_positives"] == pytest.approx(0.85)
    assert toxic["mean_prob_negatives"] == pytest.approx(0.1)
    assert toxic["min_prob"] == pytest.approx(0.1)
    assert toxic["median_prob"] == pytest.approx(0.8)
    assert toxic["max_prob"] == pytest.approx(0.9)
    assert toxic["share_exactly_0_or_1"] == 0.0

    severe = report.loc["severe_toxic"]
    assert severe["n_pos"] == 0
    assert math.isnan(severe["average_precision"])
    assert math.isnan(severe["mean_prob_positives"])
    assert severe["mean_prob_negatives"] == 0.0
    assert severe["share_exactly_0_or_1"] == 1.0

    identity = report.loc["identity_hate"]
    assert identity["average_precision"] == pytest.approx(1.0)
    assert identity["share_exactly_0_or_1"] == 1.0
