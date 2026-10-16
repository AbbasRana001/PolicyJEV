import sys

import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import comparison_report as cr
from policyjev.evaluation import splits
from policyjev.evaluation.splits import SplitScores
from policyjev.evaluation.validation_report import SYSTEMS, build_comparison


def make_bundle(split: str, n: int = 300) -> SplitScores:
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    raw = {s: np.clip(y * q + rng.random(y.shape) * (1 - q), 0, 1) for s, q in zip(SYSTEMS, (0.2, 0.5, 0.4))}
    calibrated = {s: np.clip(raw[s] * 0.9, 0, 1) for s in SYSTEMS}
    thresholds = {s: np.full(len(LABELS), 0.4) for s in SYSTEMS}
    validation_f1 = {s: np.linspace(0.3, 0.6, len(LABELS)) for s in SYSTEMS}
    return SplitScores(split, [f"r{i}" for i in range(n)], y, raw, calibrated, thresholds, validation_f1)


def test_build_comparison_uses_calibrated_probabilities_for_brier():
    bundle = make_bundle("validation")
    table = build_comparison(
        bundle.y, bundle.raw, bundle.validation_f1, 20, 1, 0.95, calibrated=bundle.calibrated
    )

    assert np.allclose(
        table["brier_jev"][:-1], ((bundle.calibrated["jev"] - bundle.y) ** 2).mean(axis=0)
    )
    assert np.allclose(
        table["brier_raw_jev"][:-1], ((bundle.raw["jev"] - bundle.y) ** 2).mean(axis=0)
    )
    assert "f1_in_sample_jev" in table.columns


def test_build_comparison_without_calibration_is_unchanged():
    bundle = make_bundle("validation")
    table = build_comparison(bundle.y, bundle.raw, bundle.validation_f1, 20, 1, 0.95)

    assert not any(c.startswith("brier_raw_") for c in table.columns)
    assert np.allclose(table["brier_jev"][:-1], ((bundle.raw["jev"] - bundle.y) ** 2).mean(axis=0))


def test_validation_comparison_labels_f1_as_in_sample(monkeypatch):
    monkeypatch.setattr(cr, "load_split_scores_bundle", lambda split, systems: make_bundle(split))

    table = cr.comparison_for_split("validation", 20, 1, 0.95)

    assert "f1_in_sample_distilbert" in table.columns
    assert not any(c.startswith("f1_distilbert") for c in table.columns)
    assert table["f1_in_sample_distilbert"][:-1].tolist() == pytest.approx(
        np.linspace(0.3, 0.6, len(LABELS)).tolist()
    )


def test_test_comparison_uses_validation_thresholds_out_of_sample(monkeypatch):
    bundle = make_bundle("test")
    monkeypatch.setattr(cr, "load_split_scores_bundle", lambda split, systems: bundle)

    table = cr.comparison_for_split("test", 20, 1, 0.95)

    assert "f1_jev" in table.columns
    assert not any(c.startswith("f1_in_sample") for c in table.columns)
    predicted = bundle.calibrated["jev"][:, 0] >= 0.4
    actual = bundle.y[:, 0] == 1
    tp = (predicted & actual).sum()
    expected = 2 * tp / (predicted.sum() + actual.sum())
    assert table.loc[0, "f1_jev"] == pytest.approx(expected)


def test_cli_refuses_test_split_without_the_tag(monkeypatch):
    monkeypatch.setattr(splits, "tag_exists", lambda tag: False)
    monkeypatch.setattr(
        cr, "load_split_scores_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read"))
    )
    monkeypatch.setattr(sys, "argv", ["comparison_report", "--split", "test"])

    with pytest.raises(SystemExit) as info:
        cr.main()
    assert "protocol-frozen" in str(info.value)


def hand_example():
    y = np.zeros((4, 6), dtype=int)
    y[:, 0] = [1, 1, 0, 0]
    calibrated = np.zeros((4, 6))
    calibrated[:, 0] = [0.9, 0.3, 0.6, 0.1]
    return y, calibrated


def test_threshold_metrics_hand_computation():
    y, calibrated = hand_example()
    rows = pd.DataFrame(
        cr.threshold_metrics(y, calibrated, np.full(6, 0.5), "jev", "fixed_0.5", "not fitted")
    ).set_index("label")

    assert list(rows.index) == LABELS + ["micro", "macro"]
    toxic = rows.loc["toxic"]
    assert (toxic["precision"], toxic["recall"], toxic["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert toxic["positives"] == 2
    assert rows.loc["obscene", ["precision", "recall", "f1"]].tolist() == [0.0, 0.0, 0.0]
    assert rows.loc["micro", "f1"] == pytest.approx(0.5)
    assert rows.loc["micro", "positives"] == 2
    assert rows.loc["macro", "f1"] == pytest.approx(0.5 / 6)
    assert np.isnan(rows.loc["macro", "positives"])

    lower = pd.DataFrame(
        cr.threshold_metrics(y, calibrated, np.array([0.2, 0.5, 0.5, 0.5, 0.5, 0.5]), "jev", "x", "s")
    ).set_index("label")
    assert lower.loc["toxic", "precision"] == pytest.approx(2 / 3)
    assert lower.loc["toxic", "recall"] == pytest.approx(1.0)
    assert lower.loc["toxic", "threshold"] == 0.2


def test_macro_f1_is_the_mean_of_per_label_f1():
    bundle = make_bundle("validation")
    table = cr.threshold_metrics_table(bundle, ["jev"])
    fitted = table[table["threshold_basis"] == "validation_oof_fit"].set_index("label")

    assert fitted.loc["macro", "f1"] == pytest.approx(fitted.loc[LABELS, "f1"].mean())
    micro_pred = bundle.calibrated["jev"] >= bundle.thresholds["jev"]
    tp = (micro_pred & (bundle.y == 1)).sum()
    expected = 2 * tp / (micro_pred.sum() + bundle.y.sum())
    assert fitted.loc["micro", "f1"] == pytest.approx(expected)


def test_both_threshold_bases_are_reported_per_system_with_positive_counts():
    bundle = make_bundle("validation")
    table = cr.threshold_metrics_table(bundle, SYSTEMS)

    assert set(table["threshold_basis"]) == {"validation_oof_fit", "fixed_0.5"}
    assert set(table["system"]) == set(SYSTEMS)
    assert len(table) == len(SYSTEMS) * 2 * (len(LABELS) + 2)

    fixed = table[table["threshold_basis"] == "fixed_0.5"]
    assert (fixed[fixed["label"].isin(LABELS)]["threshold"] == 0.5).all()
    fitted = table[(table["threshold_basis"] == "validation_oof_fit") & (table["system"] == "jev")]
    assert np.allclose(fitted[fitted["label"].isin(LABELS)]["threshold"], 0.4)

    label_rows = table[table["label"].isin(LABELS)]
    assert label_rows["positives"].notna().all()
    for label in LABELS:
        expected = int(bundle.y[:, LABELS.index(label)].sum())
        assert (label_rows[label_rows["label"] == label]["positives"] == expected).all()
    rare = set(label_rows[label_rows["is_rare_label"]]["label"])
    assert rare == {"threat", "severe_toxic", "identity_hate"}


def test_sample_status_marks_validation_fits_in_sample_and_test_fits_out_of_sample():
    validation = cr.threshold_metrics_table(make_bundle("validation"), ["jev"])
    test = cr.threshold_metrics_table(make_bundle("test"), ["jev"])

    def status(table, basis):
        return set(table[table["threshold_basis"] == basis]["sample_status"])

    assert all(s.startswith("in-sample") for s in status(validation, "validation_oof_fit"))
    assert all(s.startswith("out-of-sample") for s in status(test, "validation_oof_fit"))
    assert status(validation, "fixed_0.5") == {"not fitted"} == status(test, "fixed_0.5")


def test_run_split_uses_the_bundle_thresholds_for_each_split(monkeypatch):
    seen = []

    def fake_bundle(split, systems):
        seen.append(split)
        return make_bundle(split)

    monkeypatch.setattr(cr, "load_split_scores_bundle", fake_bundle)

    comparison, thresholds = cr.run_split("test", 10, 1, 0.95)

    assert seen == ["test"]
    assert "f1_jev" in comparison.columns
    fitted = thresholds[(thresholds["threshold_basis"] == "validation_oof_fit") & thresholds["label"].isin(LABELS)]
    assert np.allclose(fitted["threshold"], 0.4)
    assert fitted["sample_status"].str.startswith("out-of-sample").all()


def test_run_split_with_stack_adds_the_stack_rows(monkeypatch):
    bundle = make_bundle("validation")
    bundle.raw["jev_stack"] = bundle.calibrated["jev_stack"] = bundle.raw["jev"]
    bundle.thresholds["jev_stack"] = np.full(len(LABELS), 0.4)
    bundle.validation_f1["jev_stack"] = np.linspace(0.3, 0.6, len(LABELS))
    monkeypatch.setattr(cr, "load_split_scores_bundle", lambda split, systems: bundle)

    comparison, thresholds = cr.run_split("validation", 10, 1, 0.95, with_stack=True)

    assert "ap_jev_stack" in comparison.columns
    assert "jev_stack" in set(thresholds["system"])
    primary_only = cr.run_split("validation", 10, 1, 0.95)[1]
    assert "jev_stack" not in set(primary_only["system"])


def test_main_writes_the_comparison_and_the_threshold_metrics(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cr, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cr, "load_split_scores_bundle", lambda split, systems: make_bundle(split))
    monkeypatch.setattr(sys, "argv", ["comparison_report", "--split", "validation"])

    cr.main()

    metrics = tmp_path / "artifacts" / "metrics"
    assert (metrics / "validation_comparison.csv").exists()
    written = pd.read_csv(metrics / "validation_threshold_metrics.csv")
    assert {"threshold_basis", "positives", "is_rare_label", "sample_status"} <= set(written.columns)
    out = capsys.readouterr().out
    assert "in-sample" in out
    assert "macro F1" in out


def test_header_names_the_split():
    validation = cr.comparison_header("validation")
    test = cr.comparison_header("test")

    assert validation.splitlines()[0] == "Full-validation comparison (all validation rows, unweighted)."
    assert test.splitlines()[0] == "Full-test comparison (all test rows, unweighted)."
    assert "Full-validation" not in test
    assert "all validation rows" not in test
    assert validation.splitlines()[1:3] == test.splitlines()[1:3]
    assert validation.splitlines()[-1].endswith("in-sample.")
    assert test.splitlines()[-1] == "F1 uses thresholds fit on validation and is out-of-sample."
