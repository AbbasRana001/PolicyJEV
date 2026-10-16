import json

import numpy as np
import pandas as pd
import pytest
import yaml

from policyjev.config import LABELS
from policyjev.evaluation import calibration as cal


def make_data(n: int = 1500, seed: int = 0):
    """Labels drawn from true probabilities, with distorted (miscalibrated) raw scores."""
    rng = np.random.default_rng(seed)
    true_p = rng.random((n, len(LABELS))) ** 2
    y = (rng.random(true_p.shape) < true_p).astype(int)
    raw = true_p**3
    return y, raw


def test_calibrators_improve_brier_out_of_fold():
    y, raw = make_data()
    raw_brier = cal.brier(raw, y).mean()
    for method in cal.METHODS:
        oof = cal.out_of_fold_matrix(method, raw, y, seed=1)
        assert oof.shape == raw.shape
        assert not np.isnan(oof).any()
        assert (oof >= 0).all() and (oof <= 1).all()
        assert cal.brier(oof, y).mean() < raw_brier


def test_out_of_fold_is_seeded():
    y, raw = make_data(600)
    first = cal.out_of_fold("platt", raw[:, 0], y[:, 0], seed=3)
    again = cal.out_of_fold("platt", raw[:, 0], y[:, 0], seed=3)
    other = cal.out_of_fold("platt", raw[:, 0], y[:, 0], seed=4)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_fitted_calibrators_are_monotone_and_survive_json():
    y, raw = make_data(800)
    grid = np.linspace(0, 1, 50)
    for method in cal.METHODS:
        fitted = cal.fit_calibrator(method, raw[:, 0], y[:, 0])
        assert fitted["method"] == method
        restored = json.loads(json.dumps(fitted))
        values = cal.apply_calibrator(restored, grid)
        assert (np.diff(values) >= -1e-12).all()
        assert (values >= 0).all() and (values <= 1).all()


def test_single_class_label_gets_a_constant_calibrator():
    fitted = cal.fit_calibrator("isotonic", np.array([0.1, 0.5, 0.9]), np.array([0, 0, 0]))
    assert fitted == {"method": "constant", "value": 0.0}
    assert np.allclose(cal.apply_calibrator(fitted, np.array([0.2, 0.7])), 0.0)


def test_choose_method_prefers_lower_brier_and_breaks_ties_to_platt():
    assert cal.choose_method({"platt": 0.10, "isotonic": 0.09}) == "isotonic"
    assert cal.choose_method({"platt": 0.09, "isotonic": 0.10}) == "platt"
    assert cal.choose_method({"platt": 0.10, "isotonic": 0.10}) == "platt"


def test_evaluate_system_tables():
    y, raw = make_data(1000)
    table, reliability, chosen, oof = cal.evaluate_system("demo", raw, y, seed=1)

    assert table["label"].tolist() == LABELS + ["mean"]
    assert chosen in cal.METHODS
    assert (table["chosen_method"] == chosen).all()
    mean_row = table[table["label"] == "mean"].iloc[0]
    assert mean_row["brier_raw"] == pytest.approx(table["brier_raw"][:-1].mean())
    assert oof.shape == raw.shape

    assert len(reliability) == len(LABELS) * cal.N_BINS
    per_label = reliability[reliability["label"] == "toxic"]
    assert per_label["n"].sum() == len(y)
    assert per_label["n"].max() - per_label["n"].min() <= 1
    assert per_label["mean_predicted"].is_monotonic_increasing


def test_save_and_apply_calibrators_round_trip(tmp_path):
    y, raw = make_data(600)
    path = cal.calibrator_path("jev", "v2", tmp_path)
    assert path.name == "jev_v2.json"
    assert cal.calibrator_path("distilbert", None, tmp_path).name == "distilbert.json"

    fitted = cal.fit_calibrators("isotonic", raw, y)
    cal.save_calibrators(path, "jev", "isotonic", fitted, "v2")
    saved = cal.load_calibrators(path)

    assert saved["system"] == "jev"
    assert saved["question_version"] == "v2"
    assert set(saved["labels"]) == set(LABELS)

    table = pd.DataFrame(raw, columns=LABELS)
    table.insert(0, "id", [f"r{i}" for i in range(len(table))])
    applied = cal.apply_calibrators(table, saved)
    assert applied["id"].tolist() == table["id"].tolist()
    expected = cal.apply_calibrator(fitted["toxic"], raw[:, 0])
    assert np.allclose(applied["toxic"], expected)

    with pytest.raises(FileNotFoundError):
        cal.load_calibrators(tmp_path / "missing.json")


def test_record_calibration_methods_replaces_the_block_and_keeps_the_rest(tmp_path):
    config = tmp_path / "project.yaml"
    config.write_text("# keep this comment\nseed: 42\njev:\n  model: m\n", encoding="utf-8")

    cal.record_calibration_methods(config, {"logreg_word": "isotonic", "distilbert": "platt", "jev": "platt"})
    cal.record_calibration_methods(config, {"logreg_word": "platt", "distilbert": "platt", "jev": "isotonic"})

    text = config.read_text(encoding="utf-8")
    assert text.startswith("# keep this comment")
    assert text.count("calibration:") == 1
    loaded = yaml.safe_load(text)
    assert loaded["seed"] == 42
    assert loaded["jev"] == {"model": "m"}
    assert loaded["calibration"]["method"] == {
        "logreg_word": "platt",
        "distilbert": "platt",
        "jev": "isotonic",
    }
