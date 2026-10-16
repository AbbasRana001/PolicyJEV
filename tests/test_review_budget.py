import sys

import numpy as np
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import review_budget as rb
from policyjev.evaluation import splits


def test_budget_metrics_hand_example():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    any_positive = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    rare_positive = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    result = rb.budget_metrics(scores, np.zeros(10), any_positive, rare_positive, [10, 20, 50])

    assert result[0] == pytest.approx([1 / 3, 1.0, 0.0])
    assert result[1] == pytest.approx([1 / 3, 0.5, 0.0])
    assert result[2] == pytest.approx([2 / 3, 2 / 5, 1.0])


def test_tied_scores_are_broken_by_the_tiebreak_vector():
    scores = np.zeros(4)
    any_positive = np.array([0, 0, 0, 1])

    first = rb.budget_metrics(scores, np.array([0.4, 0.3, 0.2, 0.1]), any_positive, any_positive, [25])
    last = rb.budget_metrics(scores, np.array([0.1, 0.2, 0.3, 0.4]), any_positive, any_positive, [25])

    assert first[0, 0] == 1.0
    assert last[0, 0] == 0.0


def test_review_scores_take_the_maximum_over_labels():
    calibrated = np.array([[0.1, 0.4, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.9]])
    assert rb.review_scores(calibrated).tolist() == [0.4, 0.9]


def make_inputs(n: int = 800):
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.06).astype(int)
    signal = np.clip(y * 0.7 + rng.random(y.shape) * 0.3, 0, 1)
    noise = rng.random(y.shape)
    return y, {"logreg_word": noise, "distilbert": signal, "jev": signal.copy()}


def test_table_layout_pairing_and_ordering():
    y, calibrated = make_inputs()
    table = rb.build_review_budget_table(y, calibrated, n_resamples=40, seed=1, confidence=0.95)

    assert len(table) == (3 + 2) * len(rb.BUDGETS_PCT) * len(rb.METRICS)
    assert set(table["comparison"]) == {
        "logreg_word", "distilbert", "jev", "jev - logreg_word", "jev - distilbert",
    }
    assert (table["lo"] <= table["hi"]).all()

    same = table[table["comparison"] == "jev - distilbert"]
    assert np.allclose(same["value"], 0.0)
    assert np.allclose(same["lo"], 0.0) and np.allclose(same["hi"], 0.0)

    gap = table[(table["comparison"] == "jev - logreg_word") & (table["metric"] == "recall_any")]
    assert (gap["value"] > 0).all()

    recall = table[(table["comparison"] == "jev") & (table["metric"] == "recall_any")]
    assert recall.sort_values("budget_pct")["value"].is_monotonic_increasing


def test_table_is_reproducible():
    y, calibrated = make_inputs()
    first = rb.build_review_budget_table(y, calibrated, 20, 5, 0.95)
    again = rb.build_review_budget_table(y, calibrated, 20, 5, 0.95)
    assert first.equals(again)


def test_cli_refuses_test_split_without_the_tag(monkeypatch):
    monkeypatch.setattr(splits, "tag_exists", lambda tag: False)
    monkeypatch.setattr(
        rb, "load_split_scores_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read"))
    )
    monkeypatch.setattr(sys, "argv", ["review_budget", "--split", "test"])

    with pytest.raises(SystemExit) as info:
        rb.main()
    assert "protocol-frozen" in str(info.value)
