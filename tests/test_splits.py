import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import calibration as cal
from policyjev.evaluation import splits

SYSTEM_NAMES = ["logreg_word", "distilbert", "jev"]
METHODS = {"logreg_word": "platt", "distilbert": "isotonic", "jev": "platt"}


def set_tag(monkeypatch, present: bool) -> list[str]:
    """Pretend the protocol-frozen tag does or does not exist; record lookups."""
    seen: list[str] = []

    def fake(tag: str) -> bool:
        seen.append(tag)
        return present

    monkeypatch.setattr(splits, "tag_exists", fake)
    return seen


def make_world(n: int, seed: int):
    """Labels plus three systems' raw scores of varying quality."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    raw = {
        name: np.clip(y * q + rng.random(y.shape) * (1 - q), 0, 1)
        for name, q in zip(SYSTEM_NAMES, (0.2, 0.5, 0.4))
    }
    return y, raw


def labeled_frame(ids: list[str], y: np.ndarray) -> pd.DataFrame:
    """Frame in the loader format."""
    frame = pd.DataFrame(y, columns=LABELS)
    frame.insert(0, "comment_text", [f"text {i}" for i in ids])
    frame.insert(0, "id", ids)
    return frame


def test_test_split_is_refused_without_the_tag_and_never_reads_anything(monkeypatch):
    seen = set_tag(monkeypatch, False)

    def forbidden(*args, **kwargs):
        raise AssertionError("test data must not be read")

    monkeypatch.setattr(splits, "load_test", forbidden)
    monkeypatch.setattr(splits.pd, "read_csv", forbidden)

    with pytest.raises(SystemExit) as info:
        splits.require_split_access("test")
    assert "protocol-frozen" in str(info.value)
    with pytest.raises(SystemExit):
        splits.load_labeled_frame("test")
    with pytest.raises(SystemExit):
        splits.load_split_scores("jev", "test", ["a"], "v2")
    with pytest.raises(SystemExit):
        splits.load_texts("test", ["a"])
    with pytest.raises(SystemExit):
        splits.load_split_scores_bundle("test", methods=METHODS)
    assert set(seen) == {"protocol-frozen"}


def test_validation_never_needs_the_tag(monkeypatch):
    seen = set_tag(monkeypatch, False)
    splits.require_split_access("validation")
    assert seen == []

    with pytest.raises(ValueError):
        splits.require_split_access("holdout")


def test_prediction_paths():
    assert splits.prediction_path("distilbert", "validation", "v2").name == "distilbert_validation.csv"
    assert splits.prediction_path("jev", "validation", "v2").name == "jev_validation_v2.csv"
    assert splits.prediction_path("logreg_word", "test", "v2").name == "logreg_word_test.csv"
    assert splits.prediction_path("jev", "test", "v2").name == "jev_test_v2.csv"


def test_thresholds_and_f1_helpers():
    y = (np.random.default_rng(0).random((200, 6)) < 0.3).astype(int)
    calibrated = y * 0.8 + 0.1

    thresholds, f1 = splits.validation_thresholds(calibrated, y, 0.01)

    assert np.allclose(f1, 1.0)
    assert ((thresholds > 0.1) & (thresholds <= 0.9)).all()
    assert np.allclose(splits.f1_at_thresholds(calibrated, y, thresholds), 1.0)
    assert (splits.f1_at_thresholds(np.full_like(calibrated, 0.5), y, np.full(6, 0.9)) == 0).all()


def test_validation_bundle_uses_out_of_fold_calibration(monkeypatch):
    ids = [f"v{i}" for i in range(500)]
    y, raw = make_world(500, 0)
    monkeypatch.setattr(splits, "load_labeled_frame", lambda split: labeled_frame(ids, y))
    monkeypatch.setattr(
        splits, "load_split_scores", lambda system, split, got_ids, version: raw[system]
    )

    data = splits.load_split_scores_bundle("validation", methods=METHODS)

    assert data.split == "validation"
    assert data.ids == ids
    assert set(data.calibrated) == set(SYSTEM_NAMES)
    for system in SYSTEM_NAMES:
        assert data.calibrated[system].shape == (500, 6)
        assert not np.allclose(data.calibrated[system], data.raw[system])
        assert data.thresholds[system].shape == (6,)
        assert data.validation_f1[system].shape == (6,)


def test_test_bundle_uses_frozen_calibrators_and_thresholds_after_verification(monkeypatch, tmp_path):
    val_ids = [f"v{i}" for i in range(500)]
    test_ids = [f"t{i}" for i in range(300)]
    y_val, raw_val = make_world(500, 0)
    y_test, raw_test = make_world(300, 1)

    for system in SYSTEM_NAMES:
        fitted = cal.fit_calibrators(METHODS[system], raw_val[system], y_val)
        cal.save_calibrators(cal.calibrator_path(system, "v2", tmp_path), system, METHODS[system], fitted, "v2")

    set_tag(monkeypatch, True)
    monkeypatch.setattr(
        splits,
        "load_split_scores",
        lambda system, split, got_ids, version: (raw_val if split == "validation" else raw_test)[system],
    )
    monkeypatch.setattr(splits, "load_labeled_frame", lambda split: labeled_frame(val_ids, y_val))
    validation = splits.load_split_scores_bundle("validation", methods=METHODS)

    table_path = tmp_path / "validation_thresholds.csv"
    splits.write_thresholds_table(validation.thresholds, validation.validation_f1, table_path)
    verified: list = []
    monkeypatch.setattr(splits, "verify_files", lambda paths: verified.extend(paths))
    monkeypatch.setattr(splits, "thresholds_table_path", lambda: table_path)
    monkeypatch.setattr(
        splits, "calibrator_path", lambda system, version: cal.calibrator_path(system, "v2", tmp_path)
    )

    def only_test_rows(split):
        if split == "validation":
            raise AssertionError("the test bundle must not recompute anything from validation")
        return labeled_frame(test_ids, y_test)

    monkeypatch.setattr(splits, "load_labeled_frame", only_test_rows)

    test = splits.load_split_scores_bundle("test", methods=METHODS)

    assert test.split == "test"
    assert test.ids == test_ids
    assert np.array_equal(test.y, y_test)
    assert table_path in verified
    for system in SYSTEM_NAMES:
        saved_path = cal.calibrator_path(system, "v2", tmp_path)
        assert saved_path in verified
        saved = cal.load_calibrators(saved_path)
        expected = cal.apply_calibrators(pd.DataFrame(raw_test[system], columns=LABELS), saved)
        assert np.allclose(test.calibrated[system], expected.to_numpy())
        assert np.allclose(test.thresholds[system], validation.thresholds[system], atol=1e-9)
        assert np.allclose(test.validation_f1[system], validation.validation_f1[system], atol=1e-9)


def test_thresholds_table_round_trip(tmp_path):
    thresholds = {s: np.linspace(0.1, 0.6, 6) for s in ("a", "b")}
    f1 = {s: np.linspace(0.3, 0.8, 6) for s in ("a", "b")}
    path = splits.write_thresholds_table(thresholds, f1, tmp_path / "t.csv")

    loaded_thresholds, loaded_f1 = splits.load_thresholds_table(["b"], path)

    assert np.allclose(loaded_thresholds["b"], thresholds["b"])
    assert np.allclose(loaded_f1["b"], f1["b"])
    with pytest.raises(ValueError):
        splits.load_thresholds_table(["missing"], path)
