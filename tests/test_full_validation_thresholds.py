import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import thresholds
from policyjev.evaluation.thresholds import fit_full_validation

GRID = np.round(np.arange(1, 100) * 0.01, 10)


def make_labels(n: int = 200) -> pd.DataFrame:
    """Labels with a known number of positives per column."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({label: (rng.random(n) < 0.3).astype(int) for label in LABELS})


def test_full_validation_fit_is_unweighted_and_covers_each_system():
    labels = make_labels()
    perfect = labels.astype(float) * 0.8 + 0.1
    constant = pd.DataFrame({label: np.full(len(labels), 0.5) for label in LABELS})

    table = fit_full_validation({"perfect": perfect, "constant": constant}, labels, GRID)

    assert table["system"].unique().tolist() == ["perfect", "constant"]
    assert len(table) == 2 * len(LABELS)
    perfect_rows = table[table["system"] == "perfect"].set_index("label")
    assert (perfect_rows["f1"] == 1.0).all()
    assert (perfect_rows["positives"] == labels.sum()).all()
    assert np.allclose(perfect_rows["weighted_prevalence"], labels.mean())
    assert ((perfect_rows["threshold"] > 0.1) & (perfect_rows["threshold"] <= 0.9)).all()

    constant_rows = table[table["system"] == "constant"]
    assert (constant_rows["f1"] < 1.0).all()


def test_load_system_probabilities_requires_exact_validation_ids(tmp_path, monkeypatch):
    predictions_dir = tmp_path / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True)
    frame = pd.DataFrame({"id": ["a", "b", "c"], **{label: [0.1, 0.2, 0.3] for label in LABELS}})
    frame.to_csv(predictions_dir / "demo_validation.csv", index=False)
    monkeypatch.setattr(thresholds, "REPO_ROOT", tmp_path)

    aligned = thresholds.load_system_probabilities("demo", ["c", "a", "b"])
    assert aligned.index.tolist() == ["c", "a", "b"]
    assert aligned["toxic"].tolist() == [0.3, 0.1, 0.2]

    with pytest.raises(ValueError):
        thresholds.load_system_probabilities("demo", ["a", "b"])
    with pytest.raises(ValueError):
        thresholds.load_system_probabilities("demo", ["a", "b", "z"])


def test_jev_predictions_are_read_for_the_requested_question_version(tmp_path, monkeypatch):
    predictions_dir = tmp_path / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True)
    for version, value in (("v1", 0.1), ("v2", 0.9)):
        frame = pd.DataFrame({"id": ["a", "b"], **{label: [value, value] for label in LABELS}})
        frame.to_csv(predictions_dir / f"jev_validation_{version}.csv", index=False)
    monkeypatch.setattr(thresholds, "REPO_ROOT", tmp_path)

    assert thresholds.load_system_probabilities("jev", ["a", "b"], "v1")["toxic"].tolist() == [0.1, 0.1]
    assert thresholds.load_system_probabilities("jev", ["a", "b"], "v2")["toxic"].tolist() == [0.9, 0.9]
    assert thresholds.validation_prediction_path("jev", "v2").name == "jev_validation_v2.csv"
    assert thresholds.validation_prediction_path("distilbert").name == "distilbert_validation.csv"

    with pytest.raises(ValueError):
        thresholds.load_system_probabilities("jev", ["a", "b"])
