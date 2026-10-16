import numpy as np
import pytest

from policyjev.config import LABELS
from policyjev.evaluation.question_set_report import (
    build_question_set_comparison,
    discover_versions,
    select_question_set,
)
from policyjev.jev.scores import scores_filename


def make_inputs(n: int = 400):
    """Labels plus three question sets; v3 is an exact copy of v1, v2 is weaker."""
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    strong = np.clip(y * 0.5 + rng.random(y.shape) * 0.5, 0, 1)
    weak = np.clip(y * 0.1 + rng.random(y.shape) * 0.9, 0, 1)
    return y, {"v1": strong, "v2": weak, "v3": strong.copy()}


def test_comparison_layout_and_baseline_rows():
    y, probs = make_inputs()
    table = build_question_set_comparison(y, probs, n_resamples=40, seed=1, confidence=0.95)

    assert table["version"].unique().tolist() == ["v1", "v2", "v3"]
    assert table[table["version"] == "v1"]["label"].tolist() == LABELS + ["mean"]
    v1 = table[table["version"] == "v1"]
    assert v1["diff_vs_v1"].isna().all()
    assert (table["ap_lo"] <= table["ap_hi"]).all()

    mean_row = table[(table["version"] == "v2") & (table["label"] == "mean")].iloc[0]
    label_rows = table[(table["version"] == "v2") & (table["label"] != "mean")]
    assert mean_row["ap"] == pytest.approx(label_rows["ap"].mean())


def test_paired_differences_against_v1():
    y, probs = make_inputs()
    table = build_question_set_comparison(y, probs, n_resamples=40, seed=1, confidence=0.95)

    identical = table[table["version"] == "v3"]
    assert np.allclose(identical["diff_vs_v1"], 0.0)
    assert np.allclose(identical["diff_vs_v1_lo"], 0.0)
    assert np.allclose(identical["diff_vs_v1_hi"], 0.0)

    weaker = table[table["version"] == "v2"]
    assert (weaker["diff_vs_v1"] < 0).all()
    assert (weaker["diff_vs_v1_hi"] < 0).all()


def test_comparison_is_reproducible_and_needs_the_baseline():
    y, probs = make_inputs()
    first = build_question_set_comparison(y, probs, n_resamples=20, seed=3, confidence=0.95)
    again = build_question_set_comparison(y, probs, n_resamples=20, seed=3, confidence=0.95)
    assert first.equals(again)

    with pytest.raises(ValueError):
        build_question_set_comparison(
            y, {"v2": probs["v2"]}, n_resamples=5, seed=3, confidence=0.95
        )


def test_selection_rule_keeps_v1_below_the_margin():
    winner, message = select_question_set({"v1": 0.541, "v2": 0.5455, "v3": 0.530})
    assert winner == "v1"
    assert "v1 kept" in message


def test_selection_rule_picks_best_alternative_at_or_above_the_margin():
    winner, message = select_question_set({"v1": 0.500, "v2": 0.520, "v3": 0.510})
    assert winner == "v2"
    assert "v2 selected" in message

    winner, _ = select_question_set({"v1": 0.500, "v2": 0.498, "v3": 0.520})
    assert winner == "v3"

    winner, _ = select_question_set({"v1": 0.600, "v2": 0.500})
    assert winner == "v1"

    winner, message = select_question_set({"v1": 0.5})
    assert winner == "v1"
    assert "no alternative" in message


def test_discover_versions_reads_score_file_names(tmp_path):
    for name in ("jev_validation_v2.csv", "jev_validation_v1.csv", "jev_test_v1.csv",
                 "logreg_word_validation.csv", "jev_validation.csv"):
        (tmp_path / name).write_text("id\n", encoding="utf-8")

    assert discover_versions(tmp_path) == ["v1", "v2"]


def test_scores_filename_carries_the_version():
    assert scores_filename("val_full", "v2") == "jev_validation_v2.csv"
    assert scores_filename("test_full", "v1") == "jev_test_v1.csv"
