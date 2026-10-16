import math

import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.policy import engine

THRESHOLDS = pd.DataFrame(
    {
        "label": LABELS,
        "low": [0.2, 0.1, 0.3, 0.05, 0.25, 0.1],
        "high": [0.8, 0.7, 0.9, math.inf, 0.85, 0.6],
    }
)


def probs(**overrides):
    base = {label: 0.0 for label in LABELS}
    base.update(overrides)
    return base


def test_all_labels_below_low_is_allow_with_no_triggers():
    result = engine.decide(probs(), THRESHOLDS)
    assert result == engine.Decision("ALLOW", ())


def test_review_band_and_boundaries_use_greater_or_equal():
    assert engine.decide(probs(toxic=0.2), THRESHOLDS).decision == "HUMAN REVIEW"
    assert engine.decide(probs(toxic=0.199999), THRESHOLDS).decision == "ALLOW"
    assert engine.decide(probs(toxic=0.8), THRESHOLDS).decision == "ACTION"
    assert engine.decide(probs(toxic=0.799999), THRESHOLDS).decision == "HUMAN REVIEW"


def test_most_severe_outcome_wins_and_only_those_labels_trigger():
    result = engine.decide(probs(toxic=0.5, severe_toxic=0.75, insult=0.3), THRESHOLDS)
    assert result.decision == "ACTION"
    assert result.triggering_labels == ("severe_toxic",)

    result = engine.decide(probs(toxic=0.5, obscene=0.4, threat=0.04), THRESHOLDS)
    assert result.decision == "HUMAN REVIEW"
    assert result.triggering_labels == ("toxic", "obscene")

    result = engine.decide(probs(toxic=0.9, identity_hate=0.6, obscene=0.95), THRESHOLDS)
    assert result.decision == "ACTION"
    assert result.triggering_labels == ("toxic", "obscene", "identity_hate")


def test_severity_ordering_is_action_over_review_over_allow():
    order = [engine.ALLOW, engine.HUMAN_REVIEW, engine.ACTION]
    levels = [
        engine.decide(probs(), THRESHOLDS).decision,
        engine.decide(probs(toxic=0.5), THRESHOLDS).decision,
        engine.decide(probs(toxic=0.5, obscene=0.95), THRESHOLDS).decision,
    ]
    assert levels == order


def test_infinite_high_disables_action_for_that_label():
    result = engine.decide(probs(threat=1.0), THRESHOLDS)
    assert result == engine.Decision("HUMAN REVIEW", ("threat",))


def test_low_equal_to_high_has_no_review_band():
    equal = pd.DataFrame({"label": LABELS, "low": [0.5] * 6, "high": [0.5] * 6})
    assert engine.decide(probs(toxic=0.49), equal).decision == "ALLOW"
    assert engine.decide(probs(toxic=0.5), equal) == engine.Decision("ACTION", ("toxic",))


def test_scalar_decision_uses_the_thresholds_label_order():
    reordered = THRESHOLDS.iloc[::-1].reset_index(drop=True)
    result = engine.decide(probs(toxic=0.5, obscene=0.4), reordered)
    assert result.triggering_labels == ("obscene", "toxic")


def test_vectorised_result_equals_the_scalar_function():
    rng = np.random.default_rng(0)
    values = np.round(rng.random((500, len(LABELS))) ** 4, 2)
    values[::7, 3] = 1.0
    frame = pd.DataFrame(values, columns=LABELS, index=[f"c{i}" for i in range(500)])

    result = engine.decide_frame(frame, THRESHOLDS)

    assert list(result.columns) == ["decision", "triggering_labels"]
    assert result.index.tolist() == frame.index.tolist()
    for index, row in frame.iterrows():
        expected = engine.decide(row.to_dict(), THRESHOLDS)
        assert result.loc[index, "decision"] == expected.decision
        assert result.loc[index, "triggering_labels"] == expected.triggering_labels
    assert set(result["decision"]) == {"ALLOW", "HUMAN REVIEW", "ACTION"}


def test_vectorised_boundaries_and_columns_outside_the_thresholds_are_ignored():
    frame = pd.DataFrame(
        {**{label: [0.0, 0.0] for label in LABELS}, "toxic": [0.2, 0.8], "extra": [1.0, 1.0]}
    )
    result = engine.decide_frame(frame, THRESHOLDS)

    assert result["decision"].tolist() == ["HUMAN REVIEW", "ACTION"]
    assert result["triggering_labels"].tolist() == [("toxic",), ("toxic",)]


def test_engine_module_does_no_file_or_network_access():
    import inspect

    source = inspect.getsource(engine)
    for forbidden in ("open(", "read_csv", "to_csv", "requests", "socket", "import os", "pathlib"):
        assert forbidden not in source
