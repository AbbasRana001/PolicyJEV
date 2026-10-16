import math
import sys

import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.policy import thresholds as pt


def groups(*specs):
    """Build probabilities and labels from (probability, count, positives) groups."""
    p, y = [], []
    for value, count, positives in specs:
        p += [value] * count
        y += [1] * positives + [0] * (count - positives)
    return np.array(p), np.array(y)


def reference(p, y, recall_target, precision_target, floor=30):
    """Brute-force reference implementation of the selection rule."""
    candidates = sorted(set(p.tolist()))
    total = y.sum()
    high = math.inf
    for t in candidates:
        flagged = (p >= t).sum()
        if flagged >= floor and y[p >= t].sum() / flagged >= precision_target - 1e-12:
            high = t
            break
    low = max(t for t in candidates if y[p >= t].sum() / total >= recall_target - 1e-12)
    return min(low, high), high


def test_hand_example_with_exact_boundary_precision_and_recall():
    p, y = groups((0.9, 30, 27), (0.8, 10, 0), (0.5, 20, 4), (0.1, 40, 1))
    chosen = pt.select_label_thresholds(p, y, 0.90, 0.90)

    assert chosen.high == 0.9
    assert chosen.low == 0.5
    assert chosen.action_enabled
    assert chosen.positives == 32
    assert chosen.flagged_at_high == 30
    assert chosen.precision_at_high == pytest.approx(0.9)
    assert chosen.recall_at_low == pytest.approx(31 / 32)
    assert chosen.review_share == pytest.approx(60 / 100)


def test_ties_use_greater_or_equal():
    p, y = groups((0.9, 30, 27), (0.1, 70, 5))
    chosen = pt.select_label_thresholds(p, y, 0.80, 0.90)

    assert chosen.high == 0.9
    assert chosen.flagged_at_high == 30
    assert chosen.low == 0.9
    assert chosen.recall_at_low == pytest.approx(27 / 32)
    assert chosen.review_share == pytest.approx(0.30)


def test_support_floor_skips_small_high_groups_and_picks_the_lowest_qualifier():
    p, y = groups((0.9, 29, 29), (0.8, 10, 10), (0.5, 20, 4), (0.1, 41, 0))
    chosen = pt.select_label_thresholds(p, y, 0.90, 0.90)

    assert chosen.high == 0.8
    assert chosen.flagged_at_high == 39
    assert chosen.precision_at_high == 1.0


def test_floor_makes_an_otherwise_perfect_top_group_unusable():
    p, y = groups((0.9, 29, 29), (0.1, 71, 0))
    chosen = pt.select_label_thresholds(p, y, 0.90, 0.90)

    assert chosen.high == math.inf
    assert not chosen.action_enabled
    assert chosen.flagged_at_high == 0
    assert math.isnan(chosen.precision_at_high)
    assert chosen.low == 0.9


def test_action_disabled_when_precision_is_never_reached():
    rng = np.random.default_rng(0)
    p = np.round(rng.random(2000), 3)
    y = (rng.random(2000) < 0.3).astype(int)

    chosen = pt.select_label_thresholds(p, y, 0.90, 0.90)

    assert chosen.high == math.inf and not chosen.action_enabled
    assert chosen.recall_at_low >= 0.90


def test_low_above_high_is_clamped_to_high():
    p, y = groups((0.9, 30, 30), (0.3, 3, 0), (0.1, 67, 0))
    chosen = pt.select_label_thresholds(p, y, 0.90, 0.90)

    assert chosen.high == 0.3
    assert chosen.low == 0.3
    assert chosen.recall_at_low == 1.0
    assert chosen.flagged_at_high == 33


def test_support_floor_can_be_changed():
    p, y = groups((0.9, 10, 10), (0.1, 90, 0))
    assert pt.select_label_thresholds(p, y, 0.9, 0.9, support_floor=10).high == 0.9
    assert pt.select_label_thresholds(p, y, 0.9, 0.9, support_floor=11).high == math.inf


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("targets", [(0.80, 0.80), (0.90, 0.90), (0.95, 0.95), (0.80, 0.95)])
def test_matches_a_brute_force_reference_on_random_data(seed, targets):
    rng = np.random.default_rng(seed)
    score = rng.random(1500)
    y = (rng.random(1500) < 0.2 + 0.5 * score).astype(int)
    p = np.round(score**2, 2)

    chosen = pt.select_label_thresholds(p, y, *targets)
    low, high = reference(p, y, *targets)

    assert chosen.low == low
    assert chosen.high == high
    assert chosen.recall_at_low >= targets[0] - 1e-9
    if chosen.action_enabled:
        assert chosen.precision_at_high >= targets[1] - 1e-9
        assert chosen.flagged_at_high >= pt.SUPPORT_FLOOR
        assert chosen.low <= chosen.high


def test_label_without_positives_is_rejected():
    with pytest.raises(ValueError):
        pt.select_label_thresholds(np.array([0.1, 0.2]), np.array([0, 0]), 0.9, 0.9)


def make_table(n: int = 600):
    rng = np.random.default_rng(1)
    y = (rng.random((n, len(LABELS))) < 0.2).astype(int)
    calibrated = {
        system: np.round(np.clip(y * 0.5 + rng.random(y.shape) * 0.5, 0, 1), 3)
        for system in pt.SYSTEMS
    }
    return pt.build_policy_table(calibrated, y), calibrated, y


def test_table_has_every_grid_point_one_primary_and_the_right_columns():
    table, _, _ = make_table()

    assert list(table.columns) == pt.COLUMNS
    assert len(table) == 3 * 9 * len(LABELS)
    for system in pt.SYSTEMS:
        for label in LABELS:
            rows = table[(table["system"] == system) & (table["label"] == label)]
            assert len(rows) == 9
            assert set(zip(rows["recall_target"], rows["precision_target"])) == {
                (r, p) for r in pt.RECALL_GRID for p in pt.PRECISION_GRID
            }
            primary = rows[rows["is_primary"]]
            assert len(primary) == 1
            assert (primary.iloc[0]["recall_target"], primary.iloc[0]["precision_target"]) == (0.9, 0.9)
    assert (table["low"] <= table["high"]).all()
    assert (table["action_enabled"] == np.isfinite(table["high"])).all()
    assert (table.loc[~table["action_enabled"], "validation_flagged_at_high"] == 0).all()


def test_table_values_match_direct_recomputation_after_a_csv_round_trip(tmp_path):
    table, calibrated, y = make_table()
    path = tmp_path / "policy.csv"
    table.to_csv(path, index=False)
    loaded = pd.read_csv(path)

    primary = loaded[loaded["is_primary"] & (loaded["system"] == "jev")].set_index("label")
    for j, label in enumerate(LABELS):
        row = primary.loc[label]
        p = calibrated["jev"][:, j]
        recall = y[p >= row["low"], j].sum() / y[:, j].sum()
        assert recall == pytest.approx(row["validation_recall_at_low"])
        assert row["validation_recall_at_low"] >= 0.90 - 1e-9
        assert (p >= row["low"]).mean() == pytest.approx(row["validation_review_share_label"])
        if row["action_enabled"]:
            flagged = p >= row["high"]
            assert flagged.sum() == row["validation_flagged_at_high"]
            assert y[flagged, j].mean() == pytest.approx(row["validation_precision_at_high"])


def test_infinity_survives_the_csv_round_trip(tmp_path):
    p, y = groups((0.9, 29, 29), (0.1, 71, 0))
    yy = np.tile(y[:, None], (1, len(LABELS)))
    pp = np.tile(p[:, None], (1, len(LABELS)))
    table = pt.build_policy_table({"jev": pp}, yy)
    path = tmp_path / "t.csv"
    table.to_csv(path, index=False)

    loaded = pd.read_csv(path)
    assert (loaded["high"] == math.inf).all()
    assert not loaded["action_enabled"].any()


def test_thresholds_for_returns_label_low_high_for_one_grid_point():
    table, _, _ = make_table()
    chosen = pt.thresholds_for(table, "distilbert", 0.95, 0.80)

    assert chosen.columns.tolist() == ["label", "low", "high"]
    assert chosen["label"].tolist() == LABELS
    expected = table[
        (table["system"] == "distilbert") & (table["recall_target"] == 0.95) & (table["precision_target"] == 0.80)
    ]
    assert chosen["low"].tolist() == expected["low"].tolist()
    with pytest.raises(ValueError):
        pt.thresholds_for(table, "distilbert", 0.5, 0.5)


def test_refuse_test_paths():
    pt.refuse_test_paths(["artifacts/policy/policy_thresholds.csv", "data/processed/split_validation_ids.csv"])
    for bad in ("artifacts/test_policy/out.csv", "TEST/out.csv", "artifacts/predictions/logreg_word_test.csv"):
        with pytest.raises(SystemExit) as info:
            pt.refuse_test_paths(["fine.csv", bad])
        assert "validation inputs only" in str(info.value)


def test_cli_refuses_an_output_path_containing_test_before_loading_anything(monkeypatch):
    from policyjev.evaluation import splits

    monkeypatch.setattr(
        splits, "load_split_scores_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded"))
    )
    monkeypatch.setattr(sys, "argv", ["thresholds", "--out", "artifacts/policy/test_thresholds.csv"])

    with pytest.raises(SystemExit) as info:
        pt.main()
    assert "refusing path" in str(info.value)


def test_cli_writes_the_table_for_validation_inputs(monkeypatch, tmp_path, capsys):
    from policyjev.evaluation import calibration, splits
    from policyjev.evaluation.splits import SplitScores

    table_inputs, calibrated, y = make_table()
    bundle = SplitScores("validation", [f"r{i}" for i in range(len(y))], y, calibrated, calibrated, {}, {})
    seen = []

    def fake_bundle(split, systems, methods):
        seen.append(split)
        return bundle

    monkeypatch.setattr(splits, "load_split_scores_bundle", fake_bundle)
    monkeypatch.setattr(calibration, "configured_methods", lambda: {})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["thresholds", "--out", "out/policy.csv"])

    pt.main()

    written = pd.read_csv(tmp_path / "out" / "policy.csv")
    assert seen == ["validation"]
    assert len(written) == 3 * 9 * len(LABELS)
    assert "wrote" in capsys.readouterr().out
