import numpy as np
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import jev_headroom as jh
from policyjev.evaluation.validation_report import average_precision


def make_inputs(n: int = 500):
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    v1 = np.round(np.clip(y * 0.3 + rng.random(y.shape) * 0.7, 0, 1), 2)
    v2 = np.round(np.clip(y * 0.5 + rng.random(y.shape) * 0.5, 0, 1), 2)
    v3 = np.round(np.clip(y * 0.2 + rng.random(y.shape) * 0.8, 0, 1), 2)
    strong = np.clip(y * 0.7 + rng.random(y.shape) * 0.3, 0, 1)
    baselines = {"logreg_word": strong, "distilbert": strong.copy()}
    return y, {"v1": v1, "v2": v2, "v3": v3}, baselines


def test_candidates_include_training_free_ensemble_and_stacked_models():
    y, probs, baselines = make_inputs()
    candidates = jh.build_candidates(y, probs, baselines, seed=1)

    assert list(candidates) == [
        "v1", "v2", "v3", "ensemble_mean", "stack_v2", "stack_all", "logreg_word", "distilbert",
    ]
    assert np.allclose(candidates["ensemble_mean"], (probs["v1"] + probs["v2"] + probs["v3"]) / 3)
    for name in ("stack_v2", "stack_all"):
        assert candidates[name].shape == y.shape
        assert not np.isnan(candidates[name]).any()
        assert (candidates[name] >= 0).all() and (candidates[name] <= 1).all()


def test_stacking_is_out_of_fold_and_seeded():
    rng = np.random.default_rng(3)
    y_col = (rng.random(400) < 0.3).astype(int)
    features = np.column_stack([y_col + rng.normal(0, 0.8, 400), rng.normal(size=400)])

    first = jh.stacked_out_of_fold(features, y_col, seed=2)
    again = jh.stacked_out_of_fold(features, y_col, seed=2)
    other = jh.stacked_out_of_fold(features, y_col, seed=3)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert average_precision(y_col, first) > 0.5


def test_table_layout_and_paired_differences():
    y, probs, baselines = make_inputs()
    candidates = jh.build_candidates(y, probs, baselines, seed=1)
    table = jh.build_headroom_table(y, candidates, n_resamples=30, seed=1, confidence=0.95)

    assert table["candidate"].unique().tolist() == list(candidates)
    assert len(table) == len(candidates) * (len(LABELS) + 1)
    kinds = table.drop_duplicates("candidate").set_index("candidate")["kind"]
    assert kinds["ensemble_mean"] == "training-free"
    assert kinds["stack_all"].startswith("stacked")
    assert (table["ap_lo"] <= table["ap_hi"]).all()

    v2 = table[table["candidate"] == "v2"]
    assert v2["diff_vs_v2"].isna().all()
    assert v2["diff_vs_logreg_word"].notna().all()

    same = table[table["candidate"] == "distilbert"]
    assert np.allclose(same["diff_vs_logreg_word"], 0.0)
    assert np.allclose(same["diff_vs_logreg_word_lo"], 0.0)
    assert np.allclose(same["diff_vs_logreg_word_hi"], 0.0)

    mean_row = table[(table["candidate"] == "v1") & (table["label"] == "mean")].iloc[0]
    label_rows = table[(table["candidate"] == "v1") & (table["label"] != "mean")]
    assert mean_row["ap"] == pytest.approx(label_rows["ap"].mean())


def test_exploratory_labeling_and_no_test_split_support():
    assert "EXPLORATORY" in jh.HEADER
    assert "training-free" in jh.HEADER
    assert "not JEV-only" in jh.HEADER or "not a JEV-only" in jh.HEADER or "not JEV-only results" in jh.HEADER
    assert "never run on test" in jh.HEADER
    assert "split" not in jh.main.__code__.co_varnames
