import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from policyjev.config import LABELS
from policyjev.evaluation import calibration as cal
from policyjev.evaluation import comparison_report as cr
from policyjev.evaluation import review_budget as rb
from policyjev.evaluation import splits
from policyjev.evaluation import stack as st
from policyjev.evaluation.splits import SplitScores
from policyjev.evaluation.validation_report import SYSTEMS


def make_world(n: int, seed: int):
    """Labels plus noisy JEV-like probabilities and two baseline score sets."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    jev = np.round(np.clip(y * 0.4 + rng.random(y.shape) * 0.6, 0, 1), 2)
    other = {
        "logreg_word": np.clip(y * 0.3 + rng.random(y.shape) * 0.7, 0, 1),
        "distilbert": np.clip(y * 0.6 + rng.random(y.shape) * 0.4, 0, 1),
    }
    return y, jev, other


def test_saved_stack_reproduces_the_fitted_model(tmp_path):
    y, jev, _ = make_world(600, 0)
    models = st.fit_stack(jev, y)
    path = st.stack_path(tmp_path)
    st.save_stack(path, models, "v2")
    saved = st.load_stack(path)

    assert path.name == "jev_stack.json"
    assert saved["system"] == "jev_stack"
    assert saved["question_version"] == "v2"
    assert saved["features"] == LABELS
    assert set(saved["labels"]) == set(LABELS)

    table = pd.DataFrame(jev, columns=LABELS)
    table.insert(0, "id", [f"r{i}" for i in range(len(table))])
    applied = st.apply_stack(table, saved)
    assert applied["id"].tolist() == table["id"].tolist()

    reference = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(jev, y[:, 0])
    assert np.allclose(applied["toxic"], reference.predict_proba(jev)[:, 1], atol=1e-5)

    with pytest.raises(FileNotFoundError):
        st.load_stack(tmp_path / "missing.json")


def test_constant_label_gets_a_constant_prediction():
    y, jev, _ = make_world(200, 1)
    y[:, 3] = 0
    saved = {"labels": st.fit_stack(jev, y)}

    applied = st.apply_stack(pd.DataFrame(jev, columns=LABELS), saved)
    assert (applied["threat"] == 0.0).all()


def test_out_of_fold_stack_is_seeded_and_complete():
    y, jev, _ = make_world(500, 2)
    first = st.stacked_matrix(jev, y, seed=3)
    again = st.stacked_matrix(jev, y, seed=3)
    other = st.stacked_matrix(jev, y, seed=4)

    assert first.shape == y.shape
    assert not np.isnan(first).any()
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def patch_world(monkeypatch, tmp_path, with_saved_stack: bool):
    """Replace the loaders with a tiny validation/test world that includes JEV scores."""
    val_ids = [f"v{i}" for i in range(500)]
    test_ids = [f"t{i}" for i in range(300)]
    y_val, jev_val, other_val = make_world(500, 0)
    y_test, jev_test, other_test = make_world(300, 1)
    raw_val = {"jev": jev_val, **other_val}
    raw_test = {"jev": jev_test, **other_test}
    methods = {"logreg_word": "platt", "distilbert": "platt", "jev": "isotonic"}

    for system in ("logreg_word", "distilbert", "jev"):
        fitted = cal.fit_calibrators(methods[system], raw_val[system], y_val)
        cal.save_calibrators(cal.calibrator_path(system, "v2", tmp_path), system, methods[system], fitted, "v2")
    if with_saved_stack:
        st.save_stack(st.stack_path(tmp_path), st.fit_stack(jev_val, y_val), "v2")

    def frame(ids, y):
        out = pd.DataFrame(y, columns=LABELS)
        out.insert(0, "comment_text", "text")
        out.insert(0, "id", ids)
        return out

    monkeypatch.setattr(splits, "tag_exists", lambda tag: True)
    monkeypatch.setattr(
        splits, "load_labeled_frame",
        lambda split: frame(val_ids, y_val) if split == "validation" else frame(test_ids, y_test),
    )
    monkeypatch.setattr(
        splits, "load_split_scores",
        lambda system, split, ids, version: (raw_val if split == "validation" else raw_test)[system],
    )
    monkeypatch.setattr(splits, "calibrator_path", lambda s, v: cal.calibrator_path(s, "v2", tmp_path))
    monkeypatch.setattr(splits, "stack_path", lambda: st.stack_path(tmp_path))
    monkeypatch.setattr(splits, "thresholds_table_path", lambda: tmp_path / "validation_thresholds.csv")
    monkeypatch.setattr(splits, "verify_files", lambda paths: None)
    return methods, raw_val, raw_test, y_val


def freeze_thresholds(tmp_path, systems, methods):
    """Write the frozen thresholds table from the validation bundle (as the freeze step does)."""
    validation = splits.load_split_scores_bundle("validation", systems, methods)
    splits.write_thresholds_table(validation.thresholds, validation.validation_f1, tmp_path / "validation_thresholds.csv")
    return validation


def test_validation_bundle_includes_the_stack_as_out_of_fold_predictions(monkeypatch, tmp_path):
    methods, raw_val, _, y_val = patch_world(monkeypatch, tmp_path, with_saved_stack=False)

    data = splits.load_split_scores_bundle(
        "validation", ["logreg_word", "distilbert", "jev", "jev_stack"], methods
    )

    assert np.array_equal(data.raw["jev_stack"], st.stacked_matrix(raw_val["jev"], y_val, 42))
    assert np.array_equal(data.calibrated["jev_stack"], data.raw["jev_stack"])
    assert data.thresholds["jev_stack"].shape == (6,)
    assert data.validation_f1["jev_stack"].shape == (6,)


def test_test_bundle_applies_the_saved_stack_to_jev_test_scores(monkeypatch, tmp_path):
    methods, _, raw_test, _ = patch_world(monkeypatch, tmp_path, with_saved_stack=True)
    validation = freeze_thresholds(tmp_path, ["jev", "jev_stack"], methods)

    data = splits.load_split_scores_bundle("test", ["jev", "jev_stack"], methods)

    expected = st.apply_stack(pd.DataFrame(raw_test["jev"], columns=LABELS), st.load_stack(st.stack_path(tmp_path)))
    assert np.allclose(data.raw["jev_stack"], expected.to_numpy())
    assert np.array_equal(data.calibrated["jev_stack"], data.raw["jev_stack"])
    assert np.allclose(data.thresholds["jev_stack"], validation.thresholds["jev_stack"], atol=1e-9)


def test_test_bundle_without_a_saved_stack_fails_clearly(monkeypatch, tmp_path):
    methods, *_ = patch_world(monkeypatch, tmp_path, with_saved_stack=False)
    freeze_thresholds(tmp_path, ["jev", "jev_stack"], methods)
    with pytest.raises(FileNotFoundError):
        splits.load_split_scores_bundle("test", ["jev", "jev_stack"], methods)


def make_four_system_bundle(split: str) -> SplitScores:
    y, jev, other = make_world(300, 0)
    raw = {"jev": jev, "jev_stack": np.clip(jev + 0.1, 0, 1), **other}
    thresholds = {s: np.full(len(LABELS), 0.4) for s in raw}
    f1 = {s: np.linspace(0.3, 0.6, len(LABELS)) for s in raw}
    return SplitScores(split, [f"r{i}" for i in range(300)], y, raw, dict(raw), thresholds, f1)


def test_comparison_report_adds_the_stack_with_paired_contrasts(monkeypatch):
    bundle = make_four_system_bundle("validation")
    monkeypatch.setattr(cr, "load_split_scores_bundle", lambda split, systems: bundle)

    table = cr.comparison_for_split("validation", 20, 1, 0.95, with_stack=True)

    assert "ap_jev_stack" in table.columns
    for other in SYSTEMS:
        assert f"diff_ap_jev_stack_minus_{other}" in table.columns
        assert (table[f"diff_ap_jev_stack_minus_{other}_lo"] <= table[f"diff_ap_jev_stack_minus_{other}_hi"]).all()
    assert "f1_in_sample_jev_stack" in table.columns

    primary_only = cr.comparison_for_split("validation", 20, 1, 0.95)
    assert "ap_jev_stack" not in primary_only.columns


def test_review_budget_adds_the_stack_and_marks_cell_status():
    bundle = make_four_system_bundle("validation")
    table = rb.build_review_budget_table(
        bundle.y, bundle.calibrated, 20, 1, 0.95, with_stack=True
    )

    comparisons = set(table["comparison"])
    assert {"jev_stack", "jev_stack - logreg_word", "jev_stack - distilbert", "jev_stack - jev"} <= comparisons
    assert "status" in table.columns

    assert rb.cell_status("jev - logreg_word", "recall_any", 5) == "pre-specified"
    assert rb.cell_status("jev - distilbert", "recall_rare", 10) == "pre-specified"
    assert rb.cell_status("jev - distilbert", "recall_rare", 1) == "hypothesis generated on validation"
    assert rb.cell_status("jev - distilbert", "recall_rare", 2) == "hypothesis generated on validation"
    assert rb.cell_status("jev - logreg_word", "recall_rare", 1) == "descriptive"
    assert rb.cell_status("jev - distilbert", "precision", 5) == "descriptive"
    assert rb.cell_status("jev - distilbert", "recall_any", 20) == "descriptive"
    assert rb.cell_status("jev_stack - jev", "recall_any", 5) == "descriptive"
    assert rb.cell_status("jev", "recall_any", 5) == "descriptive"

    prespecified = table[table["status"] == "pre-specified"]
    assert len(prespecified) == 2 * 2 * 2
    hypotheses = table[table["status"] == "hypothesis generated on validation"]
    assert len(hypotheses) == 2
